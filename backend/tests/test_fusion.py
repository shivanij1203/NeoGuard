"""Tests for uncertainty-weighted fusion."""
from __future__ import annotations

import math

import pytest

from ml.fusion import (
    STATUS_AUDIO_ONLY,
    STATUS_FACIAL_ONLY,
    STATUS_FRESH,
    STATUS_UNAVAILABLE,
    fuse_uncertainty_weighted,
)


@pytest.mark.unit
def test_both_absent_emits_unavailable_with_none_composite():
    out = fuse_uncertainty_weighted(
        facial_score=None, uncertainty=None, audio_score=None,
        base_facial_weight=0.7, base_audio_weight=0.3,
    )
    assert out["composite_score"] is None
    assert out["signal_status"] == STATUS_UNAVAILABLE


@pytest.mark.unit
def test_facial_only_carries_full():
    """Audio absent: facial score carries at full weight, not facial_weight
    times the score."""
    out = fuse_uncertainty_weighted(
        facial_score=6.0, uncertainty=0.1, audio_score=None,
        base_facial_weight=0.7, base_audio_weight=0.3,
    )
    assert out["composite_score"] == 6.0
    assert out["signal_status"] == STATUS_FACIAL_ONLY
    assert out["weights_used"] == {"facial": 1.0, "audio": 0.0}


@pytest.mark.unit
def test_audio_only_carries_full():
    out = fuse_uncertainty_weighted(
        facial_score=None, uncertainty=None, audio_score=8.0,
        base_facial_weight=0.7, base_audio_weight=0.3,
    )
    assert out["composite_score"] == 8.0
    assert out["signal_status"] == STATUS_AUDIO_ONLY
    assert out["weights_used"] == {"facial": 0.0, "audio": 1.0}


@pytest.mark.unit
def test_low_uncertainty_preserves_facial_weight_ratio():
    """At uncertainty=0, effective facial weight is the configured weight.
    With facial=0.7 audio=0.3, both 6.0 and 8.0, expect 0.7*6 + 0.3*8 = 6.6."""
    out = fuse_uncertainty_weighted(
        facial_score=6.0, uncertainty=0.0, audio_score=8.0,
        base_facial_weight=0.7, base_audio_weight=0.3,
    )
    assert math.isclose(out["composite_score"], 6.6, abs_tol=1e-6)
    assert out["signal_status"] == STATUS_FRESH


@pytest.mark.unit
def test_high_uncertainty_shifts_composite_toward_audio():
    """At uncertainty=1 the facial weight collapses to zero, so audio carries
    the whole composite. Status stays FRESH because both signals were
    provided this frame; the discount lives in the weights."""
    out = fuse_uncertainty_weighted(
        facial_score=2.0, uncertainty=1.0, audio_score=9.0,
        base_facial_weight=0.7, base_audio_weight=0.3,
    )
    assert math.isclose(out["composite_score"], 9.0, abs_tol=1e-6)
    assert out["signal_status"] == STATUS_FRESH
    assert out["weights_used"]["facial"] == 0.0
    assert out["weights_used"]["audio"] == 1.0


@pytest.mark.unit
def test_uncertainty_above_one_is_clipped():
    """A pathological uncertainty > 1 must not produce a negative facial
    weight or a composite outside [0, 10]."""
    out = fuse_uncertainty_weighted(
        facial_score=2.0, uncertainty=2.0, audio_score=9.0,
        base_facial_weight=0.7, base_audio_weight=0.3,
    )
    assert math.isclose(out["composite_score"], 9.0, abs_tol=1e-6)
    assert 0.0 <= out["weights_used"]["facial"] <= 1.0


@pytest.mark.unit
def test_renormalization_makes_weights_sum_to_one():
    out = fuse_uncertainty_weighted(
        facial_score=5.0, uncertainty=0.5, audio_score=5.0,
        base_facial_weight=0.7, base_audio_weight=0.3,
    )
    total = out["weights_used"]["facial"] + out["weights_used"]["audio"]
    assert math.isclose(total, 1.0, abs_tol=1e-3)


@pytest.mark.unit
def test_none_uncertainty_treated_as_no_penalty():
    """When facial is present but uncertainty is None, do not silently
    discount the facial signal. Treat as zero uncertainty."""
    full = fuse_uncertainty_weighted(
        facial_score=6.0, uncertainty=0.0, audio_score=8.0,
        base_facial_weight=0.7, base_audio_weight=0.3,
    )
    none_unc = fuse_uncertainty_weighted(
        facial_score=6.0, uncertainty=None, audio_score=8.0,
        base_facial_weight=0.7, base_audio_weight=0.3,
    )
    assert full["composite_score"] == none_unc["composite_score"]


@pytest.mark.unit
def test_negative_base_weights_rejected():
    with pytest.raises(ValueError):
        fuse_uncertainty_weighted(
            facial_score=5.0, uncertainty=0.0, audio_score=5.0,
            base_facial_weight=-0.1, base_audio_weight=0.3,
        )


@pytest.mark.unit
def test_zero_base_weights_rejected():
    with pytest.raises(ValueError):
        fuse_uncertainty_weighted(
            facial_score=5.0, uncertainty=0.0, audio_score=5.0,
            base_facial_weight=0.0, base_audio_weight=0.0,
        )
