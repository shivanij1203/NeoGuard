"""Tests for the preprocess module: face crop from landmarks (the gate) and
image-to-tensor with an explicit input contract.

Pipeline tests on synthetic frames only. No metric here should be interpreted
as model performance.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.ncnn.preprocess import (
    crop_face_from_landmarks,
    face_crop_to_tensor,
)


def _fake_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """A BGR frame with a unique color per pixel so we can verify the BGR to
    RGB conversion downstream of the crop."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _fake_landmarks(cx: float, cy: float, half: float, n: int = 468) -> np.ndarray:
    """A grid of n points spanning a square of half-side `half` around (cx, cy).
    Returns shape (n, 3) like MediaPipe's pixel landmarks."""
    side = int(np.ceil(np.sqrt(n)))
    xs = np.linspace(cx - half, cx + half, side)
    ys = np.linspace(cy - half, cy + half, side)
    pts = np.array([(x, y, 0.0) for y in ys for x in xs])[:n]
    return pts


@pytest.mark.unit
def test_crop_returns_rgb_with_margin_and_expected_size():
    frame = _fake_frame(240, 320)
    landmarks = _fake_landmarks(cx=160, cy=120, half=40)
    crop = crop_face_from_landmarks(
        frame, landmarks, margin_ratio=0.25, min_face_size_px=16
    )
    assert crop is not None
    h, w = crop.shape[:2]
    # bbox is roughly 80x80; with 25% margin each side, expect about 120x120.
    assert 110 <= h <= 130
    assert 110 <= w <= 130
    assert crop.shape[2] == 3
    assert crop.dtype == np.uint8


@pytest.mark.unit
def test_crop_swaps_bgr_to_rgb():
    """A frame whose B channel is a constant and R channel is zero should
    come out of the crop with the constant in the R channel."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[..., 0] = 200  # B
    frame[..., 1] = 0    # G
    frame[..., 2] = 50   # R
    landmarks = _fake_landmarks(cx=50, cy=50, half=20)
    crop = crop_face_from_landmarks(
        frame, landmarks, margin_ratio=0.0, min_face_size_px=8
    )
    assert crop is not None
    # In the RGB crop, channel 0 must be the original R (50) and channel 2
    # must be the original B (200).
    assert int(crop[..., 0].mean()) == 50
    assert int(crop[..., 2].mean()) == 200


@pytest.mark.unit
def test_crop_returns_none_when_face_too_small():
    frame = _fake_frame(240, 320)
    landmarks = _fake_landmarks(cx=160, cy=120, half=5)
    crop = crop_face_from_landmarks(
        frame, landmarks, margin_ratio=0.1, min_face_size_px=64
    )
    assert crop is None


@pytest.mark.unit
def test_crop_returns_none_when_any_landmark_off_frame():
    """Partial visibility errs toward None. Audio handles occluded windows."""
    frame = _fake_frame(240, 320)
    landmarks = _fake_landmarks(cx=10, cy=120, half=40)  # left edge spillover
    crop = crop_face_from_landmarks(
        frame, landmarks, margin_ratio=0.1, min_face_size_px=16
    )
    assert crop is None


@pytest.mark.unit
def test_crop_returns_none_when_margin_padding_spills_off_frame():
    """Margin pushing the bbox over the edge counts as partial too."""
    frame = _fake_frame(240, 320)
    # bbox tight to top edge: half=40 centered at y=45 means y_min=5, top
    # edge at 0. Any positive margin spills.
    landmarks = _fake_landmarks(cx=160, cy=45, half=40)
    crop = crop_face_from_landmarks(
        frame, landmarks, margin_ratio=0.5, min_face_size_px=16
    )
    assert crop is None


@pytest.mark.unit
def test_crop_returns_none_for_degenerate_inputs():
    frame = _fake_frame()
    assert crop_face_from_landmarks(None, np.zeros((10, 2)), 0.1, 8) is None  # type: ignore[arg-type]
    assert crop_face_from_landmarks(frame, None, 0.1, 8) is None  # type: ignore[arg-type]
    # 1D landmarks
    assert crop_face_from_landmarks(frame, np.array([1.0, 2.0]), 0.1, 8) is None
    # Empty landmarks
    assert crop_face_from_landmarks(frame, np.zeros((0, 2)), 0.1, 8) is None


@pytest.mark.unit
def test_crop_validates_parameters():
    frame = _fake_frame()
    landmarks = _fake_landmarks(cx=160, cy=120, half=40)
    with pytest.raises(ValueError):
        crop_face_from_landmarks(frame, landmarks, margin_ratio=-0.1, min_face_size_px=8)
    with pytest.raises(ValueError):
        crop_face_from_landmarks(frame, landmarks, margin_ratio=0.1, min_face_size_px=0)


@pytest.mark.unit
def test_to_tensor_accepts_uint8_in_0_255():
    crop = np.random.randint(0, 256, size=(60, 60, 3), dtype=np.uint8)
    t = face_crop_to_tensor(crop, 120, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert t.shape == (1, 3, 120, 120)
    assert t.dtype == torch.float32
    assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0


@pytest.mark.unit
def test_to_tensor_accepts_float_in_0_1():
    crop = np.random.rand(60, 60, 3).astype(np.float32)
    t = face_crop_to_tensor(crop, 120, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert t.shape == (1, 3, 120, 120)
    assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0


@pytest.mark.unit
def test_to_tensor_rejects_float_in_0_255():
    """The contract fix. A float frame in 0 to 255 must not silently pass."""
    crop = (np.random.rand(60, 60, 3) * 255.0).astype(np.float32)
    with pytest.raises(ValueError):
        face_crop_to_tensor(crop, 120, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


@pytest.mark.unit
def test_to_tensor_rejects_negative_float():
    crop = np.full((60, 60, 3), -0.1, dtype=np.float32)
    with pytest.raises(ValueError):
        face_crop_to_tensor(crop, 120, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


@pytest.mark.unit
def test_to_tensor_rejects_unsupported_dtype():
    crop = np.zeros((60, 60, 3), dtype=np.int32)
    with pytest.raises(ValueError):
        face_crop_to_tensor(crop, 120, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


@pytest.mark.unit
def test_to_tensor_applies_normalization():
    crop = np.full((60, 60, 3), 128, dtype=np.uint8)
    t = face_crop_to_tensor(crop, 120, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # (128/255 - 0.5) / 0.5 ~= 0.004
    assert float(t.mean()) == pytest.approx((128 / 255 - 0.5) / 0.5, abs=1e-3)
