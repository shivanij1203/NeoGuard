"""Image preprocessing helpers for the N-CNN.

This file owns two boundaries:
  - Face crop from MediaPipe landmarks, used as the occlusion and presence
    gate before the CNN ever sees a frame.
  - Image to tensor, used inside the inference path.

Normalization stats are placeholders. The model trains from scratch, so
ImageNet mean and std are wrong here. The defaults in config (zero mean,
unit std) are a no-op until real stats are computed from the training set.

Input contracts are checked explicitly. We do not infer dtype intent from
value heuristics, because a float frame that happens to carry 0 to 255 values
would silently feed the model wildly out-of-range input.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

# Lazy OpenCV import so unit tests that only exercise the tensor path do not
# pay the cv2 import cost when not needed. cv2 is already a project dep.
try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - cv2 is in requirements
    cv2 = None


# Allowed RGB image input contracts:
#   uint8 array with values in [0, 255], or
#   float32 / float64 array already scaled to [0, 1].
# Anything else is rejected, including float arrays carrying 0 to 255.
_FLOAT_RANGE_TOL = 1e-3


def _to_float01(rgb: np.ndarray) -> np.ndarray:
    """Validate the input contract and return a float32 array in [0, 1]."""
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float32) / 255.0
    if np.issubdtype(rgb.dtype, np.floating):
        if rgb.size == 0:
            return rgb.astype(np.float32)
        lo = float(rgb.min())
        hi = float(rgb.max())
        if lo < -_FLOAT_RANGE_TOL or hi > 1.0 + _FLOAT_RANGE_TOL:
            raise ValueError(
                "float RGB input must already be scaled to [0, 1]; got range "
                f"[{lo:.3f}, {hi:.3f}]. Pass a uint8 array for 0 to 255 input."
            )
        return rgb.astype(np.float32)
    raise ValueError(
        f"unsupported dtype {rgb.dtype}; expected uint8 in [0, 255] or float in [0, 1]"
    )


def crop_face_from_landmarks(
    frame_bgr: np.ndarray,
    landmarks_px: np.ndarray,
    margin_ratio: float,
    min_face_size_px: int,
) -> Optional[np.ndarray]:
    """Crop the face from a BGR frame using a MediaPipe pixel-landmark bbox.

    Inputs:
        frame_bgr: (H, W, 3) uint8 BGR frame as produced by OpenCV.
        landmarks_px: (N, 2) or (N, 3) array of landmark pixel coordinates.
        margin_ratio: extra padding around the landmark bbox, as a fraction
            of the bbox side.
        min_face_size_px: reject bboxes smaller than this on either side.

    Returns:
        (H', W', 3) uint8 RGB crop, or None if the face is too small,
        partially off-frame, or otherwise unfit for the CNN.

    Behaviour for partial visibility: the gate errs toward None. If any
    landmark falls outside the frame, or if the margin-padded bbox spills
    over a frame edge, the function returns None and the caller is expected
    to let audio carry that window.

    TODO: landmark-visibility based occlusion check for NICU-realistic
    occluders (CPAP straps and tape over the mouth, phototherapy mask over
    the eyes, prone position). The bbox-only gate catches missing or tiny
    faces but not partial occlusion of the pain regions (brow, mouth,
    nasolabial fold). When that lands, use MediaPipe's per-landmark
    visibility or a presence score on the AU-relevant landmark groups,
    and return None when any pain-relevant region is occluded.
    """
    if frame_bgr is None or landmarks_px is None:
        return None
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        return None
    if landmarks_px.ndim != 2 or landmarks_px.shape[1] < 2 or landmarks_px.shape[0] == 0:
        return None
    if margin_ratio < 0:
        raise ValueError("margin_ratio must be >= 0")
    if min_face_size_px < 1:
        raise ValueError("min_face_size_px must be >= 1")

    h, w = frame_bgr.shape[:2]
    xs = landmarks_px[:, 0]
    ys = landmarks_px[:, 1]

    # Partial-visibility check: any landmark outside the frame disqualifies
    # this window. Audio handles it.
    if (
        xs.min() < 0
        or ys.min() < 0
        or xs.max() > w - 1
        or ys.max() > h - 1
    ):
        return None

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    if bbox_w <= 0 or bbox_h <= 0:
        return None

    pad_x = bbox_w * margin_ratio
    pad_y = bbox_h * margin_ratio
    x0 = int(round(x_min - pad_x))
    y0 = int(round(y_min - pad_y))
    x1 = int(round(x_max + pad_x))
    y1 = int(round(y_max + pad_y))

    # Margin-padded box must sit entirely within the frame. Spilling over an
    # edge means the face is near the boundary; treat as partial and skip.
    if x0 < 0 or y0 < 0 or x1 > w - 1 or y1 > h - 1:
        return None

    crop_w = x1 - x0 + 1
    crop_h = y1 - y0 + 1
    if crop_w < min_face_size_px or crop_h < min_face_size_px:
        return None

    if cv2 is None:
        raise RuntimeError("cv2 is required for crop_face_from_landmarks")

    crop_bgr = frame_bgr[y0 : y1 + 1, x0 : x1 + 1]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return crop_rgb


def face_crop_to_tensor(
    face_crop_rgb: np.ndarray,
    input_size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """Resize an RGB face crop to (input_size, input_size), scale to [0, 1],
    standardize per channel, and return a (1, 3, H, W) float32 tensor.

    Input contract:
        uint8 array in [0, 255], or float array already in [0, 1]. A float
        array carrying 0 to 255 values is rejected, not silently divided.
    """
    if face_crop_rgb is None:
        raise ValueError("face_crop_rgb is None")
    if face_crop_rgb.ndim != 3 or face_crop_rgb.shape[2] != 3:
        raise ValueError(
            f"expected (H, W, 3) RGB array, got shape {face_crop_rgb.shape}"
        )
    if cv2 is None:
        raise RuntimeError("cv2 is required for face_crop_to_tensor")
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean and std must each have length 3")
    if face_crop_rgb.dtype != np.uint8 and not np.issubdtype(face_crop_rgb.dtype, np.floating):
        raise ValueError(
            f"unsupported dtype {face_crop_rgb.dtype}; expected uint8 in [0, 255] "
            "or float in [0, 1]"
        )

    resized = cv2.resize(
        face_crop_rgb, (input_size, input_size), interpolation=cv2.INTER_AREA
    )
    arr = _to_float01(resized)

    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean_arr) / std_arr

    # HWC to CHW, add batch dim.
    chw = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)
    return tensor
