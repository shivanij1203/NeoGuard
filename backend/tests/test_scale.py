"""Tests for the shared probability-to-score mapping.

This is the single source the classifier and the dashboard band both use.
If it breaks here, both surfaces break consistently rather than drifting.
"""
from __future__ import annotations

import pytest

from ml.ncnn.scale import SCORE_MAX, SCORE_MIN, prob_to_score


@pytest.mark.unit
def test_prob_zero_maps_to_score_min():
    assert prob_to_score(0.0) == SCORE_MIN


@pytest.mark.unit
def test_prob_one_maps_to_score_max():
    assert prob_to_score(1.0) == SCORE_MAX


@pytest.mark.unit
def test_prob_to_score_is_monotonic():
    samples = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    out = [prob_to_score(p) for p in samples]
    assert out == sorted(out)


@pytest.mark.unit
def test_prob_to_score_rejects_none():
    with pytest.raises(ValueError):
        prob_to_score(None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_prob_to_score_rejects_out_of_range():
    with pytest.raises(ValueError):
        prob_to_score(-0.01)
    with pytest.raises(ValueError):
        prob_to_score(1.01)
