"""Tests for NCNNFacialPainClassifier.

Real MediaPipe is heavy and slow for unit tests; we monkeypatch the face
detector and the crop helper to drive the classifier through its branches
deterministically. The model itself is exercised end-to-end on a synthetic
crop, so the inference path is real.
"""
from __future__ import annotations

import numpy as np
import pytest

from ml import ncnn_classifier as ncnn_mod
from ml.ncnn_classifier import NCNNFacialPainClassifier
from ml.ncnn.scale import prob_to_score
from ml.ood_gate import (
    CHIN,
    FOREHEAD,
    LEFT_EYE_INNER,
    LEFT_TEMPLE,
    RIGHT_EYE_INNER,
    RIGHT_TEMPLE,
)


@pytest.fixture(scope="module")
def classifier() -> NCNNFacialPainClassifier:
    return NCNNFacialPainClassifier()


def _fake_frame() -> np.ndarray:
    return np.random.randint(0, 256, size=(240, 320, 3), dtype=np.uint8)


def _landmarks(aspect: float, ipd_ratio: float) -> np.ndarray:
    """Synthetic 468-point mesh with a target face aspect and inter-eye ratio.

    Only the six indices the input-validity gate reads are positioned; the rest
    sit at the face centre. The crop helper is monkeypatched in these tests, so
    the exact crop geometry does not matter, only that the gate sees a valid
    (or, where intended, invalid) face shape.
    """
    cx, cy, width = 160.0, 120.0, 100.0
    pts = np.full((468, 3), [cx, cy, 0.0], dtype=np.float64)
    pts[LEFT_TEMPLE] = (cx - width / 2.0, cy, 0.0)
    pts[RIGHT_TEMPLE] = (cx + width / 2.0, cy, 0.0)
    height = aspect * width
    pts[FOREHEAD] = (cx, cy - height / 2.0, 0.0)
    pts[CHIN] = (cx, cy + height / 2.0, 0.0)
    ipd = ipd_ratio * width
    pts[LEFT_EYE_INNER] = (cx - ipd / 2.0, cy - 20.0, 0.0)
    pts[RIGHT_EYE_INNER] = (cx + ipd / 2.0, cy - 20.0, 0.0)
    return pts


def _fake_landmarks() -> np.ndarray:
    """Infant-like geometry that passes the input-validity gate."""
    return _landmarks(aspect=1.10, ipd_ratio=0.45)


@pytest.mark.unit
def test_no_face_returns_absent_contract(classifier, monkeypatch):
    """When MediaPipe finds no face, the contract is all-None plus
    face_detected=False. Critically, prob_pain is None, not 0.0."""
    monkeypatch.setattr(classifier.face_detector, "detect", lambda frame: None)
    out = classifier.predict(_fake_frame(), compute_uncertainty=True)
    assert out["face_detected"] is False
    assert out["prob_pain"] is None
    assert out["uncertainty"] is None
    assert out["facial_score"] is None
    assert isinstance(out["frame_to_score_ms"], float)
    assert out["frame_to_score_ms"] >= 0.0


@pytest.mark.unit
def test_face_detected_but_crop_rejected_is_absent(classifier, monkeypatch):
    """Gate rejecting the crop (occluded, too small) reads as absent
    upstream, same shape as no-face. This is the err-toward-None policy."""
    monkeypatch.setattr(
        classifier.face_detector,
        "detect",
        lambda frame: {"landmarks_px": _fake_landmarks(), "frame_shape": frame.shape[:2]},
    )
    monkeypatch.setattr(ncnn_mod, "crop_face_from_landmarks", lambda *a, **k: None)
    out = classifier.predict(_fake_frame(), compute_uncertainty=True)
    assert out["face_detected"] is False
    assert out["prob_pain"] is None
    assert out["uncertainty"] is None
    assert out["facial_score"] is None


@pytest.mark.unit
def test_face_detected_runs_full_inference_path(classifier, monkeypatch):
    """End-to-end through the model. With compute_uncertainty=True we expect
    a real uncertainty number out of MC dropout."""
    monkeypatch.setattr(
        classifier.face_detector,
        "detect",
        lambda frame: {"landmarks_px": _fake_landmarks(), "frame_shape": frame.shape[:2]},
    )
    fake_crop = np.random.randint(0, 256, size=(96, 96, 3), dtype=np.uint8)
    monkeypatch.setattr(ncnn_mod, "crop_face_from_landmarks", lambda *a, **k: fake_crop)

    out = classifier.predict(_fake_frame(), compute_uncertainty=True)

    assert out["face_detected"] is True
    assert 0.0 <= out["prob_pain"] <= 1.0
    assert out["uncertainty"] is not None and out["uncertainty"] >= 0.0
    assert out["facial_score"] == prob_to_score(out["prob_pain"])
    assert out["frame_to_score_ms"] > 0.0


@pytest.mark.unit
def test_compute_uncertainty_false_returns_none_uncertainty(classifier, monkeypatch):
    """The cheap-frame path must skip MC dropout and return uncertainty=None.
    The caller is expected to keep its cached value."""
    monkeypatch.setattr(
        classifier.face_detector,
        "detect",
        lambda frame: {"landmarks_px": _fake_landmarks(), "frame_shape": frame.shape[:2]},
    )
    fake_crop = np.random.randint(0, 256, size=(96, 96, 3), dtype=np.uint8)
    monkeypatch.setattr(ncnn_mod, "crop_face_from_landmarks", lambda *a, **k: fake_crop)

    out = classifier.predict(_fake_frame(), compute_uncertainty=False)

    assert out["face_detected"] is True
    assert out["prob_pain"] is not None
    assert out["uncertainty"] is None
    assert out["facial_score"] == prob_to_score(out["prob_pain"])


@pytest.mark.unit
def test_out_of_distribution_face_is_flagged_before_inference(classifier, monkeypatch):
    """An adult-like face trips the input-validity gate: the classifier returns
    out_of_distribution=True with no score, and never reaches the crop or the
    model. The crop helper is set to raise so we prove the short-circuit."""
    monkeypatch.setattr(
        classifier.face_detector,
        "detect",
        lambda frame: {"landmarks_px": _landmarks(aspect=1.45, ipd_ratio=0.34),
                       "frame_shape": frame.shape[:2]},
    )

    def _boom(*_a, **_k):
        raise AssertionError("crop must not run for an out-of-distribution face")

    monkeypatch.setattr(ncnn_mod, "crop_face_from_landmarks", _boom)

    out = classifier.predict(_fake_frame(), compute_uncertainty=True)
    assert out["out_of_distribution"] is True
    assert out["ood_reason"] == "subject_not_infant"
    assert out["face_detected"] is True
    assert out["prob_pain"] is None
    assert out["facial_score"] is None


@pytest.mark.unit
def test_prob_pain_is_deterministic_on_cheap_path(classifier, monkeypatch):
    """Same crop in, same prob_pain out, when compute_uncertainty=False.
    Confirms the live score is reproducible even though MC is stochastic."""
    monkeypatch.setattr(
        classifier.face_detector,
        "detect",
        lambda frame: {"landmarks_px": _fake_landmarks(), "frame_shape": frame.shape[:2]},
    )
    fake_crop = np.random.RandomState(0).randint(0, 256, size=(96, 96, 3)).astype(np.uint8)
    monkeypatch.setattr(ncnn_mod, "crop_face_from_landmarks", lambda *a, **k: fake_crop)

    a = classifier.predict(_fake_frame(), compute_uncertainty=False)
    b = classifier.predict(_fake_frame(), compute_uncertainty=False)
    assert a["prob_pain"] == b["prob_pain"]
