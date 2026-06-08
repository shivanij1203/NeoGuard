"""Out-of-distribution input gate for the facial pain pipeline.

A general input-validity check, deliberately independent of the N-CNN and of
any pain feature extraction. The facial model has no "not sure" output, so a
non-infant face is rejected here on cheap face geometry before scoring, rather
than letting the model extrapolate on a subject it was never meant to see.

Two ratios are computed from MediaPipe face landmarks:
    face_aspect_ratio    = forehead-to-chin distance / temple-to-temple distance
    ipd_face_width_ratio = inter-eye-corner distance / temple-to-temple distance
A face trips the gate when the aspect ratio exceeds the configured max or the
inter-eye ratio falls below the configured min. Either condition alone is
enough.

The thresholds live in config (ood_infant_aspect_max, ood_infant_ipd_min) and
are unvalidated heuristics; see the warning there. This module does no AU
feature extraction and no pain inference, and imports neither. Landmarks come
from FaceDetector.detect().

Research prototype, not a medical device.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# MediaPipe Face Mesh landmark indices used for the geometry ratios.
FOREHEAD = 10
CHIN = 152
LEFT_TEMPLE = 234
RIGHT_TEMPLE = 454
# Inner eye corners. Their separation is used as an inter-pupillary proxy; the
# pupils themselves are not landmarked.
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362

# Reason string surfaced on the payload when the gate trips.
REASON_NOT_INFANT = "subject_not_infant"


@dataclass(frozen=True)
class OODResult:
    """Outcome of the input-validity gate for one set of landmarks."""

    out_of_distribution: bool
    reason: Optional[str]
    face_aspect_ratio: float
    ipd_face_width_ratio: float


def _dist(landmarks: np.ndarray, idx1: int, idx2: int) -> float:
    """Euclidean distance between two landmarks in pixel space (x, y only)."""
    return float(
        np.sqrt(
            (landmarks[idx1][0] - landmarks[idx2][0]) ** 2
            + (landmarks[idx1][1] - landmarks[idx2][1]) ** 2
        )
    )


def face_geometry(landmarks: np.ndarray) -> tuple[float, float]:
    """Return (face_aspect_ratio, ipd_face_width_ratio) from landmarks.

    Both ratios are normalized by temple-to-temple face width. A degenerate
    (sub-pixel) face width is floored at 1.0 so the ratios stay finite.
    """
    face_height = _dist(landmarks, FOREHEAD, CHIN)
    face_width = _dist(landmarks, LEFT_TEMPLE, RIGHT_TEMPLE)
    safe_width = max(face_width, 1.0)
    aspect = face_height / safe_width
    ipd = _dist(landmarks, LEFT_EYE_INNER, RIGHT_EYE_INNER)
    ipd_ratio = ipd / safe_width
    return aspect, ipd_ratio


def evaluate_input_validity(
    landmarks: np.ndarray,
    aspect_max: float,
    ipd_min: float,
) -> OODResult:
    """Decide whether a detected face is out of distribution (not an infant).

    Trips when aspect > aspect_max OR ipd_ratio < ipd_min. Thresholds are
    passed in by the caller (from config) so this stays a pure function that is
    trivial to unit test on synthetic geometry. The comparisons are strict, so
    a face sitting exactly on a threshold is treated as in distribution.
    """
    aspect, ipd_ratio = face_geometry(landmarks)
    is_ood = aspect > aspect_max or ipd_ratio < ipd_min
    return OODResult(
        out_of_distribution=is_ood,
        reason=REASON_NOT_INFANT if is_ood else None,
        face_aspect_ratio=aspect,
        ipd_face_width_ratio=ipd_ratio,
    )
