"""Query existing pixel locations in Tiny RoMa's directional dense warp."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class QueryMatches:
    """One result per input query, including rejected queries, in original order.

    Coordinates use zero-based pixel centres. Confidence is the model score,
    not a calibrated probability of correctness.
    """

    right_points: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor


def _inside(points, size):
    h, w = size
    return (
        torch.isfinite(points).all(-1)
        & (points[:, 0] >= 0)
        & (points[:, 0] <= w - 1)
        & (points[:, 1] >= 0)
        & (points[:, 1] <= h - 1)
    )


def _sample(field, points, size):
    """Sample HWC data with align_corners=False and zero-based pixel centres."""
    scale = points.new_tensor((size[1], size[0]))
    grid = 2 * (points.float() + 0.5) / scale - 1
    # Invalid queries are rejected separately; never send NaNs into grid_sample.
    grid = torch.nan_to_num(grid, nan=0.0, posinf=2.0, neginf=-2.0)
    # MPS does not support grid_sample's border padding. Clamping to the
    # outer pixel centres gives the same border behaviour on every backend.
    bounds = points.new_tensor((1 - 1 / field.shape[1], 1 - 1 / field.shape[0]))
    grid = torch.maximum(torch.minimum(grid, bounds), -bounds)
    # A replicated one-pixel rim avoids zero-padding leakage from float32
    # rounding exactly at the border. Adjust normalized coordinates for the rim.
    padded = F.pad(field.float().permute(2, 0, 1)[None], (1, 1, 1, 1), mode="replicate")
    grid = grid * points.new_tensor(
        (field.shape[1] / (field.shape[1] + 2), field.shape[0] / (field.shape[0] + 2))
    )
    sampled = F.grid_sample(
        padded,
        grid[None, None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled[0, :, 0].T


@torch.inference_mode()
def sample_query_points(warp, certainty, left_points, source_size, target_size):
    """Sample a single-direction Tiny RoMa warp at supplied left pixel points.

    warp: [Hf,Wf,4], with normalized target coordinates in its last two channels.
    certainty: [Hf,Wf]. left_points: floating [N,2] on the same device.
    source_size and target_size are (height,width) of the supplied images,
    independent of the field resolution. Full RoMa's concatenated symmetric
    output must NOT be passed here. No nearest-keypoint association is performed.
    """
    if warp.ndim != 3 or warp.shape[-1] != 4 or min(warp.shape[:2]) < 1:
        raise ValueError("Expected directional warp [H,W,4].")
    if certainty.shape != warp.shape[:2]:
        raise ValueError("Certainty must have shape [H,W] matching the warp.")
    if (
        left_points.ndim != 2
        or left_points.shape[1] != 2
        or not left_points.is_floating_point()
    ):
        raise ValueError("Expected floating left_points [N,2].")
    if any(len(s) != 2 or min(s) < 1 for s in (source_size, target_size)):
        raise ValueError("Image sizes must be positive (height,width).")
    if not (warp.device == certainty.device == left_points.device):
        raise ValueError("Warp, certainty and query points must share a device.")
    points = left_points.float()
    if len(points) == 0:
        return QueryMatches(
            points.clone(),
            points.new_empty(0),
            torch.empty(0, dtype=torch.bool, device=points.device),
        )
    target_norm = _sample(warp[..., 2:], points, source_size)
    right = (target_norm + 1) * points.new_tensor(
        (target_size[1], target_size[0])
    ) / 2 - 0.5
    # Snap only floating-point roundoff at the image boundary (e.g. identity
    # mappings can produce -1e-6 instead of zero after normalization).
    limit = points.new_tensor((target_size[1] - 1, target_size[0] - 1))
    eps = 4 * torch.finfo(right.dtype).eps * max(target_size)
    right = torch.where((right < 0) & (right >= -eps), 0.0, right)
    right = torch.where((right > limit) & (right <= limit + eps), limit, right)
    confidence = _sample(certainty[..., None], points, source_size)[:, 0]
    valid = (
        _inside(points, source_size)
        & _inside(right, target_size)
        & torch.isfinite(confidence)
    )
    return QueryMatches(right, confidence, valid)


class TinyRoMaQueryMatcher:
    """Compose an existing TinyRoMa with query sampling and optional filtering.

    Inputs are RGB floating tensors [1,3,H,W] in [0,1], on model.device;
    query coordinates [N,2] use the original left image's zero-based pixel centres.
    Masks are bool [H,W], True=allowed, on the same device. They filter endpoints
    only, never alter model input pixels. No temporal state or point IDs are kept.
    """

    def __init__(self, model, confidence_threshold=0.5):
        if (
            not math.isfinite(confidence_threshold)
            or not 0 <= confidence_threshold <= 1
        ):
            raise ValueError("confidence_threshold must be in [0,1].")
        self.model = model
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _mask_valid(mask, points, size):
        if mask is None:
            return _inside(points, size)
        # Conservative bilinear mask sampling, allowing float32 roundoff only.
        return _inside(points, size) & (
            _sample(mask[..., None], points, size)[:, 0] >= 1 - 1e-6
        )

    @torch.inference_mode()
    def match(
        self, left_image, right_image, left_points, left_mask=None, right_mask=None
    ):
        for image, mask in ((left_image, left_mask), (right_image, right_mask)):
            if (
                image.ndim != 4
                or image.shape[:2] != (1, 3)
                or min(image.shape[-2:]) < 32
            ):
                raise ValueError("Expected one RGB image [1,3,H,W], H and W >=32.")
            if image.dtype != torch.float32 or image.device != torch.device(
                self.model.device
            ):
                raise ValueError(
                    "Images must be float32 on model.device, with values in [0,1]."
                )
            if mask is not None and (
                mask.shape != image.shape[-2:]
                or mask.dtype != torch.bool
                or mask.device != image.device
            ):
                raise ValueError(
                    "Masks must be bool [H,W] on the image device; True=allowed."
                )
        if (
            left_points.ndim != 2
            or left_points.shape[1] != 2
            or not left_points.is_floating_point()
        ):
            raise ValueError("Expected floating left_points [N,2].")
        if left_points.device != left_image.device:
            raise ValueError("Query points must be on the image device.")
        if len(left_points) == 0:
            empty = left_points.new_empty(0, dtype=torch.float32)
            return QueryMatches(
                left_points.float().clone(),
                empty,
                torch.empty(0, dtype=torch.bool, device=left_points.device),
            )
        left_size, right_size = left_image.shape[-2:], right_image.shape[-2:]
        warp, certainty = self.model.match(left_image, right_image, batched=False)
        forward = sample_query_points(
            warp, certainty, left_points, left_size, right_size
        )
        valid = (
            forward.valid
            & (forward.confidence >= self.confidence_threshold)
            & self._mask_valid(left_mask, left_points, left_size)
            & self._mask_valid(right_mask, forward.right_points, right_size)
        )
        return QueryMatches(forward.right_points, forward.confidence, valid)
