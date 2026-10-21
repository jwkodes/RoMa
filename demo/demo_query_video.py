"""Match a masked grid in sampled side-by-side stereo video pairs with Tiny RoMa."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch

from romatch import tiny_roma_v1_outdoor
from romatch.query_matcher import TinyRoMaQueryMatcher


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stats(values):
    a = np.asarray(values)
    a = a[np.isfinite(a)]
    return {
        "count": len(a),
        "median": float(np.median(a)) if len(a) else None,
        "p95": float(np.percentile(a, 95)) if len(a) else None,
    }


def render(eyes, points, result, frame_index, timestamp, elapsed):
    h, w = eyes[0].shape[:2]
    canvas = np.zeros((h + 100, 2 * w, 3), np.uint8)
    canvas[:h] = cv2.hconcat(eyes)
    ids = np.flatnonzero(result["valid"])
    if len(ids) > 120:
        ids = ids[np.linspace(0, len(ids) - 1, 120).astype(int)]
    overlay = canvas[:h].copy()
    for i in ids:
        a = tuple(np.rint(points[i]).astype(int))
        b = tuple(np.rint(result["right_points"][i]).astype(int) + [w, 0])
        t = i / max(len(points) - 1, 1)
        color = (255, round(220 * (1 - t)), round(235 * t))
        cv2.line(overlay, a, b, color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.6, canvas[:h], 0.4, 0, dst=canvas[:h])
    for i in ids:
        t = i / max(len(points) - 1, 1)
        color = (255, round(220 * (1 - t)), round(235 * t))
        for eye, xy in enumerate((points[i], result["right_points"][i])):
            p = tuple(np.rint(xy).astype(int) + [eye * w, 0])
            cv2.circle(canvas, p, 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, p, 3, color, -1, cv2.LINE_AA)
    labels = [
        f"Tiny RoMa query wrapper | LEFT / RIGHT | frame {frame_index} | {timestamp:.2f}s",
        f"{result['valid'].sum()}/{len(points)} accepted | show max 120 | forward wrapper {elapsed * 1000:.1f} ms",
        "Confidence >=0.5 | endpoint masks | no vertical filter | no ground truth",
    ]
    for j, label in enumerate(labels):
        cv2.putText(
            canvas,
            label,
            (15, h + 26 + j * 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument(
        "--masks",
        type=Path,
        required=True,
        help="Native-eye left.png/right.png; 255=allow.",
    )
    p.add_argument("--output", type=Path, default=Path("results/tiny-query-video"))
    p.add_argument(
        "--weights-dir",
        type=Path,
        help="Optional local tiny_roma_v1_outdoor.pth, xfeat.pt and xfeat_model.py bundle.",
    )
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--stride", type=int, default=30)
    p.add_argument("--max-frames", type=int)
    args = p.parse_args()
    if args.stride < 1 or (args.max_frames is not None and args.max_frames < 1):
        p.error("stride and max-frames must be positive")
    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    torch.manual_seed(0)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)

    def sync():
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

    hashes = {}
    if args.weights_dir:
        root = args.weights_dir
        names = ["tiny_roma_v1_outdoor.pth", "xfeat.pt", "xfeat_model.py"]
        hashes = {name: digest(root / name) for name in names}
        spec = importlib.util.spec_from_file_location(
            "local_xfeat_model", root / "xfeat_model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        backbone = module.XFeatModel()
        backbone.load_state_dict(
            torch.load(root / "xfeat.pt", map_location="cpu", weights_only=True)
        )
        weights = torch.load(
            root / "tiny_roma_v1_outdoor.pth", map_location="cpu", weights_only=True
        )
        model = tiny_roma_v1_outdoor(device=device, weights=weights, xfeat=backbone)
    else:
        model = tiny_roma_v1_outdoor(device=device)
    matcher = TinyRoMaQueryMatcher(model, confidence_threshold=0.5)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise ValueError("Cannot open input video")
    width, height = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    if width % 2:
        raise ValueError("Expected side-by-side stereo video with even width")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Expected a positive input frame rate")
    size = (960, 540)
    masks = []
    for name in ["left", "right"]:
        mask = cv2.imread(str(args.masks / f"{name}.png"), cv2.IMREAD_UNCHANGED)
        if (
            mask is None
            or mask.shape != (height, width // 2)
            or not np.isin(mask, [0, 255]).all()
        ):
            raise ValueError("Expected native-eye binary masks, 255=allow")
        masks.append(
            cv2.resize(
                mask.astype(np.float32) / 255, size, interpolation=cv2.INTER_AREA
            )
            >= 1
        )
    x, y = np.meshgrid(
        np.linspace(12, size[0] - 13, 40), np.linspace(12, size[1] - 13, 24)
    )
    points = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
    allowed = (
        cv2.remap(
            masks[0].astype(np.float32),
            points[:, 0:1],
            points[:, 1:2],
            cv2.INTER_LINEAR,
        )[:, 0]
        >= 1
    )
    points = points[allowed]
    queries = torch.from_numpy(points).to(device)
    tmasks = [torch.from_numpy(m).to(device) for m in masks]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "pairs").mkdir()
    (args.output / "previews").mkdir()
    for eye, mask in zip(["left", "right"], masks):
        cv2.imwrite(
            str(args.output / f"{eye}-allow-mask.png"), mask.astype(np.uint8) * 255
        )
    manifest = {
        "video": str(args.video.resolve()),
        "video_sha256": digest(args.video),
        "source_size_wh": [width, height],
        "input_eye_size_wh": list(size),
        "internal_eye_size_wh": [size[0] // 32 * 32, size[1] // 32 * 32],
        "source_fps": fps,
        "stride": args.stride,
        "max_source_frames": args.max_frames,
        "queries": len(points),
        "device": device,
        "torch": torch.__version__,
        "precision": "float32",
        "weights_sha256": hashes,
        "mask_sha256": {
            name: digest(args.masks / f"{name}.png") for name in ["left", "right"]
        },
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            name: digest(name)
            for name in ["romatch/models/tiny.py", "romatch/query_matcher.py"]
        },
        "coordinates": "Zero-based pixel centres, normalized=(pixel+0.5)*2/size-1; no vertical filtering.",
        "mask_policy": "Filter queries and endpoints; pixels supplied to model remain unchanged.",
        "timing": "Synchronized wrapper call: one dense inference, query sampling and filtering; excludes image upload, decode, saving and drawing.",
    }
    dump(args.output / "manifest.json", manifest)
    out = imageio_ffmpeg.write_frames(
        str(args.output / "stereo_correspondences.mp4"),
        (size[0] * 2, size[1] + 100),
        fps=fps / args.stride,
        codec="libx264",
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
        quality=5,
        macro_block_size=2,
        output_params=["-movflags", "+faststart"],
    )
    out.send(None)
    rows, pooled_dy, timings = [], [], []
    started = time.perf_counter()
    index = 0
    try:
        with (args.output / "frames.jsonl").open("w") as log:
            while args.max_frames is None or index < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                current = index
                index += 1
                if current % args.stride:
                    continue
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                eyes = [
                    cv2.resize(
                        frame[:, i * width // 2 : (i + 1) * width // 2],
                        size,
                        interpolation=cv2.INTER_AREA,
                    )
                    for i in range(2)
                ]
                tensors = [
                    torch.from_numpy(cv2.cvtColor(eye, cv2.COLOR_BGR2RGB))
                    .permute(2, 0, 1)[None]
                    .to(device)
                    .float()
                    / 255
                    for eye in eyes
                ]
                sync()
                begin = time.perf_counter()
                result = matcher.match(*tensors, queries, *tmasks)
                sync()
                elapsed = time.perf_counter() - begin
                data = {
                    name: getattr(result, name).cpu().numpy()
                    for name in [
                        "right_points",
                        "confidence",
                        "valid",
                    ]
                }
                assert data["right_points"].shape == points.shape
                dy = abs(data["right_points"][:, 1] - points[:, 1])
                keep = data["valid"]
                pooled_dy.extend(dy[keep].tolist())
                timings.append(elapsed)
                row = {
                    "frame": current,
                    "timestamp_s": timestamp,
                    "accepted": int(keep.sum()),
                    "wrapper_seconds": elapsed,
                    "abs_vertical_px": stats(dy[keep]),
                }
                rows.append(row)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                np.savez_compressed(
                    args.output / "pairs" / f"{current:06d}.npz",
                    left_points=points,
                    **data,
                )
                view = render(eyes, points, data, current, timestamp, elapsed)
                out.send(view.tobytes())
                if len(rows) in [1, 18, 34]:
                    cv2.imwrite(
                        str(args.output / "previews" / f"frame-{current:04d}.jpg"), view
                    )
                print(json.dumps(row), flush=True)
    finally:
        cap.release()
        out.close()
    if not rows:
        raise ValueError("No frames processed")
    summary = {
        "samples": len(rows),
        "queries": len(points),
        "accepted_per_pair": stats([r["accepted"] for r in rows]),
        "pooled_abs_vertical_px": stats(pooled_dy),
        "wrapper_seconds": stats(timings),
        "wrapper_seconds_after_first": stats(timings[1:]),
        "total_inference_seconds": sum(timings),
        "decode_inference_render_seconds": time.perf_counter() - started,
        "note": "No ground-truth correspondence accuracy; vertical differences are consistency diagnostics.",
    }
    dump(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
