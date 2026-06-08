"""Unit tests for the out-of-distribution input gate.

All landmark geometry here is synthetic: points are placed by hand to produce
a target face aspect ratio and inter-eye ratio. There is no real subject data
and nothing here measures model accuracy. The tests check the gate logic only:
an infant-like geometry passes, an adult-like geometry trips each threshold
independently, and the threshold boundary is handled with strict comparisons.

Research prototype, not a medical device.
"""
from __future__ import annotations

import numpy as np
import pytest

from ml.ood_gate import (
    CHIN,
    FOREHEAD,
    LEFT_EYE_INNER,
    LEFT_TEMPLE,
    REASON_NOT_INFANT,
    RIGHT_EYE_INNER,
    RIGHT_TEMPLE,
    evaluate_input_validity,
    face_geometry,
)

# Mirrors the documented config defaults so the tests read against the same
# numbers the gate ships with. These are unvalidated heuristics, not tuned.
ASPECT_MAX = 1.18
IPD_MIN = 0.40

MESH_POINTS = 468


def _landmarks(aspect: float, ipd_ratio: float, face_width: float = 100.0) -> np.ndarray:
    """Build a synthetic landmark array with a target aspect and inter-eye ratio.

    Temples set the face width along x, forehead and chin set the height along
    y, and the inner eye corners set the inter-eye distance. Every other point
    is left at the origin; the gate only reads these six indices.
    """
    pts = np.zeros((MESH_POINTS, 3), dtype=np.float64)
    cx, cy = 50.0, 50.0
    half_w = face_width / 2.0
    pts[LEFT_TEMPLE] = (cx - half_w, cy, 0.0)
    pts[RIGHT_TEMPLE] = (cx + half_w, cy, 0.0)

    height = aspect * face_width
    pts[FOREHEAD] = (cx, cy - height / 2.0, 0.0)
    pts[CHIN] = (cx, cy + height / 2.0, 0.0)

    ipd = ipd_ratio * face_width
    pts[LEFT_EYE_INNER] = (cx - ipd / 2.0, cy - 20.0, 0.0)
    pts[RIGHT_EYE_INNER] = (cx + ipd / 2.0, cy - 20.0, 0.0)
    return pts


@pytest.mark.unit
def test_face_geometry_recovers_constructed_ratios():
    aspect, ipd_ratio = face_geometry(_landmarks(aspect=1.10, ipd_ratio=0.45))
    assert aspect == pytest.approx(1.10)
    assert ipd_ratio == pytest.approx(0.45)


@pytest.mark.unit
def test_infant_like_geometry_passes():
    result = evaluate_input_validity(
        _landmarks(aspect=1.10, ipd_ratio=0.45), aspect_max=ASPECT_MAX, ipd_min=IPD_MIN
    )
    assert result.out_of_distribution is False
    assert result.reason is None


@pytest.mark.unit
def test_adult_aspect_trips_the_gate():
    # Tall, narrow adult face: aspect above the max, inter-eye ratio normal.
    result = evaluate_input_validity(
        _landmarks(aspect=1.40, ipd_ratio=0.45), aspect_max=ASPECT_MAX, ipd_min=IPD_MIN
    )
    assert result.out_of_distribution is True
    assert result.reason == REASON_NOT_INFANT
    assert result.face_aspect_ratio > ASPECT_MAX


@pytest.mark.unit
def test_adult_ipd_trips_the_gate():
    # Close-set eyes relative to face width: inter-eye ratio below the min,
    # aspect normal. Confirms either threshold alone is sufficient.
    result = evaluate_input_validity(
        _landmarks(aspect=1.10, ipd_ratio=0.35), aspect_max=ASPECT_MAX, ipd_min=IPD_MIN
    )
    assert result.out_of_distribution is True
    assert result.reason == REASON_NOT_INFANT
    assert result.ipd_face_width_ratio < IPD_MIN


@pytest.mark.unit
def test_aspect_boundary_is_strict():
    """A face sitting exactly on the aspect threshold is in distribution;
    a hair above it trips. Uses the constructed ratio as the threshold so the
    comparison is exact and not subject to float-literal drift."""
    pts = _landmarks(aspect=1.10, ipd_ratio=0.45)
    aspect, _ = face_geometry(pts)

    on_boundary = evaluate_input_validity(pts, aspect_max=aspect, ipd_min=0.0)
    assert on_boundary.out_of_distribution is False

    just_over = evaluate_input_validity(pts, aspect_max=aspect - 0.01, ipd_min=0.0)
    assert just_over.out_of_distribution is True


@pytest.mark.unit
def test_ipd_boundary_is_strict():
    pts = _landmarks(aspect=1.10, ipd_ratio=0.45)
    _, ipd_ratio = face_geometry(pts)

    on_boundary = evaluate_input_validity(pts, aspect_max=10.0, ipd_min=ipd_ratio)
    assert on_boundary.out_of_distribution is False

    just_under = evaluate_input_validity(pts, aspect_max=10.0, ipd_min=ipd_ratio + 0.01)
    assert just_under.out_of_distribution is True


@pytest.mark.unit
def test_degenerate_face_does_not_raise_and_is_flagged():
    """A sub-pixel face floors the width at 1.0 rather than dividing by zero.
    Its inter-eye ratio collapses to zero, so the gate flags it."""
    pts = np.zeros((MESH_POINTS, 3), dtype=np.float64)
    result = evaluate_input_validity(pts, aspect_max=ASPECT_MAX, ipd_min=IPD_MIN)
    assert result.out_of_distribution is True
