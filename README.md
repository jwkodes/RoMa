# 
<p align="center">
  <h1 align="center"> <ins>RoMa</ins> 🏛️:<br> Robust Dense Feature Matching <br> ⭐CVPR 2024⭐</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=Ul-vMR0AAAAJ">Johan Edstedt</a>
    ·
    <a href="https://scholar.google.com/citations?user=HS2WuHkAAAAJ">Qiyu Sun</a>
    ·
    <a href="https://scholar.google.com/citations?user=FUE3Wd0AAAAJ">Georg Bökman</a>
    ·
    <a href="https://scholar.google.com/citations?user=6WRQpCQAAAAJ">Mårten Wadenbäck</a>
    ·
    <a href="https://scholar.google.com/citations?user=lkWfR08AAAAJ">Michael Felsberg</a>
  </p>
  <h2 align="center"><p>
    <a href="https://arxiv.org/abs/2305.15404" align="center">Paper</a> | 
    <a href="https://parskatt.github.io/RoMa" align="center">Project Page</a>
  </p></h2>
  <div align="center"></div>
</p>
<br/>
<p align="center">
    <img src="https://github.com/Parskatt/RoMa/assets/22053118/15d8fea7-aa6d-479f-8a93-350d950d006b" alt="example" width=80%>
    <br>
    <em>RoMa is the robust dense feature matcher capable of estimating pixel-dense warps and reliable certainties for almost any image pair.</em>
</p>

## Setup/Install
In your python environment (tested on Linux python 3.12), run:
```bash
uv pip install -e .
```
or 
```bash
uv sync
```
You can also install `romatch` directly as a package from PyPI by
```bash
uv pip install romatch
```
or 
```bash
uv add romatch
```

## Fused local correlation kernel
Include the `--extra fused-local-corr` flag as:
```bash
uv sync --extra fused-local-corr
```
or 
```bash
uv pip install romatch[fused-local-corr]
```
or
```bash
uv add romatch[fused-local-corr]
```



## Demo / How to Use
We provide two demos in the [demos folder](demo).
Here's the gist of it:
```python
from romatch import roma_outdoor
roma_model = roma_outdoor(device=device)
# Match
warp, certainty = roma_model.match(imA_path, imB_path, device=device)
# Sample matches for estimation
matches, certainty = roma_model.sample(warp, certainty)
# Convert to pixel coordinates (RoMa produces matches in [-1,1]x[-1,1])
kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
# Find a fundamental matrix (or anything else of interest)
F, mask = cv2.findFundamentalMat(
    kptsA.cpu().numpy(), kptsB.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
)
```

**New**: You can also match arbitrary keypoints with RoMa. See [match_keypoints](romatch/models/matcher.py) in RegressionMatcher.

## Settings

### Resolution
By default RoMa uses an initial resolution of (560,560) which is then upsampled to (864,864). 
You can change this at construction (see roma_outdoor kwargs).
You can also change this later, by changing the roma_model.w_resized, roma_model.h_resized, and roma_model.upsample_res.

### Sampling
roma_model.sample_thresh controls the thresholding used when sampling matches for estimation. In certain cases a lower or higher threshold may improve results.


## Reproducing Results
The experiments in the paper are provided in the [experiments folder](experiments).

### Training
1. First follow the instructions provided here: https://github.com/Parskatt/DKM for downloading and preprocessing datasets.
2. Run the relevant experiment, e.g.,
```bash
torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d experiments/roma_outdoor.py
```
### Testing
```bash
python experiments/roma_outdoor.py --only_test --benchmark mega-1500
```
## License
All our code except DINOv2 is MIT license.
DINOv2 has an Apache 2 license [DINOv2](https://github.com/facebookresearch/dinov2/blob/main/LICENSE).

## Acknowledgement
Our codebase builds on the code in [DKM](https://github.com/Parskatt/DKM).

## Tiny RoMa
If you find that RoMa is too heavy, you might want to try Tiny RoMa which is built on top of XFeat.
```python
from romatch import tiny_roma_v1_outdoor
tiny_roma_model = tiny_roma_v1_outdoor(device=device)
```
Mega1500:
|  | AUC@5 | AUC@10 | AUC@20 |
|----------|----------|----------|----------|
| XFeat    | 46.4    | 58.9    | 69.2    |
| XFeat*    |  51.9   | 67.2    | 78.9    |
| Tiny RoMa v1    | 56.4 | 69.5 | 79.5     |
| RoMa    |  -   | -    | -    |

Mega-8-Scenes (See DKM):
|  | AUC@5 | AUC@10 | AUC@20 |
|----------|----------|----------|----------|
| XFeat    | -    | -    | -    |
| XFeat*    |  50.1   | 64.4    | 75.2    |
| Tiny RoMa v1    | 57.7 | 70.5 | 79.6     |
| RoMa    |  -   | -    | -    |

IMC22 :'):
|  | mAA@10 |
|----------|----------|
| XFeat    | 42.1    |
| XFeat*    |  -   |
| Tiny RoMa v1    | 42.2 |
| RoMa    |  -   |

## Query-point wrapper for Tiny RoMa (this fork)

`TinyRoMaQueryMatcher` composes an existing Tiny RoMa instance with a query-point
interface. It samples the dense warp at **your existing left-image points** and
returns right-image coordinates in the same order. It does not detect a separate
right-image keypoint set, associate nearest keypoints, or change the model.

```python
import torch
from romatch import tiny_roma_v1_outdoor
from romatch.query_matcher import TinyRoMaQueryMatcher

model = tiny_roma_v1_outdoor(device="mps")  # also CPU or CUDA
matcher = TinyRoMaQueryMatcher(model, confidence_threshold=0.5)
# Images: RGB float32 tensors [1,3,H,W], values in [0,1], on model.device.
# Points: floating tensor [N,2] in original left-image pixel coordinates.
left_points = torch.tensor([[120.25, 80.5], [300.0, 200.0]], device=model.device)
result = matcher.match(left_image, right_image, left_points)
right_points = result.right_points       # [N,2], original right-image pixels
confidence = result.confidence           # [N], forward model score
valid = result.valid                     # [N], combined acceptance mask
```

- All input queries retain their output slot, including invalid queries; use
  `result.valid` before triangulation. Nonfinite/out-of-bounds predictions are
  rejected. Empty query sets return empty results without running the model.
- Each stereo pair uses one forward model call. Default filtering requires
  confidence ≥0.5; this is configurable and is a choice for this fork’s
  experiments, not a prescribed Tiny RoMa threshold. No vertical/epipolar
  filtering is applied.
- Optional `left_mask` / `right_mask` are boolean `[H,W]` tensors on the image
  device, with `True=allowed`. They conservatively filter queries and endpoints;
  pixels given to the model remain unchanged. These are not attention masks.
- Coordinates use zero-based pixel centres: normalize with `(pixel+0.5)*2/size-1`
  and invert with `(normalized+1)*size/2-0.5`. This convention differs by half a
  pixel from upstream's `to_pixel_coordinates()` helper; do not mix conventions.
- `sample_query_points(warp, certainty, points, source_size, target_size)` also
  exposes the sampling step independently. Sizes are `(height,width)`. It takes a
  **directional** `[H,W,4]` warp, not full RoMa's concatenated symmetric output.
- The wrapper has no temporal state, track IDs, keyframe logic or triangulation.
  Load the model once and reuse it. Outputs remain on the input device.
- CPU and Apple MPS query sampling are tested. TensorRT/ONNX export is not
  implemented or validated by this wrapper.

### Stereo video demo

From this repository, run:

```sh
uv sync --extra demo
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --extra demo python -m demo.demo_query_video \
  --video /path/to/stereo_sbs.mp4 \
  --masks /path/to/masks \
  --device mps --stride 30 --output results/tiny-query-video
```

The default model factory downloads official weights if needed. For the pinned
local test bundle, add `--weights-dir weights` (containing
`tiny_roma_v1_outdoor.pth`, `xfeat.pt` and `xfeat_model.py`). Model files and results
are ignored by Git. The optional `demo` extra supplies the H.264 encoder.

The demo splits the encoded stereo pair, resizes each eye to 960×540, and uses a
40×24 grid filtered by the saved masks. It writes a connecting-line video, all
query predictions, validity/confidence arrays, timings and provenance.
Colours identify pairs, not confidence. The selected points are not evidence of
correct disparity without ground truth.

Run the focused tests with:

```sh
uv run python -m unittest discover -s tests -p test_query_matcher.py -v
```

Tests cover pixel-centre conventions, subpixel interpolation, unequal image/field
sizes, image boundaries, masks, confidence rejection, empty inputs,
query ordering and CPU/MPS agreement (MPS check skips if unavailable).

## Reproducibility
There are a few diffs in the current codebase compared to the original repo used to run experiments.

1. The `scale_factor` used in the `match` method now is relative to the original training resolution of `560`. Previosly it was based on the set coarse resolution (which might or might not be `560`).
2. Newer PyTorch, original code used something like `2.1`.
3. Stochastic eval: both RANSAC and the chosen correspondences can affect results in `Mega1500`.
4. Matrix inverse in GP has been replaced with cholesky decomp.

That being said, if diff of results are $>0.5$ there probably is something wrong, please let me know.


## BibTeX
If you find our models useful, please consider citing our paper!
```
@inproceedings{edstedt2024roma,
title={{RoMa: Robust Dense Feature Matching}},
author={Edstedt, Johan and Sun, Qiyu and Bökman, Georg and Wadenbäck, Mårten and Felsberg, Michael},
booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
year={2024}
}
```
