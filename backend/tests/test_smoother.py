"""Tests for the EMA smoother.

Pure function over (prior, new, alpha). The contract that matters most for
this project is "hold on absent": a None new_value must not collapse the
smoothed signal toward zero.
"""
from __future__ import annotations

import math

import pytest

from ml.smoother import ema_update


@pytest.mark.unit
def test_cold_start_returns_new_value():
    assert ema_update(prior_smoothed=None, new_value=0.7, alpha=0.5) == 0.7


@pytest.mark.unit
def test_absent_input_holds_prior():
    """None new value preserves the prior smoothed value exactly. Absent is
    not zero."""
    assert ema_update(prior_smoothed=0.4, new_value=None, alpha=0.5) == 0.4


@pytest.mark.unit
def test_absent_input_with_no_prior_returns_none():
    assert ema_update(prior_smoothed=None, new_value=None, alpha=0.5) is None


@pytest.mark.unit
def test_alpha_one_means_no_smoothing():
    assert ema_update(prior_smoothed=0.2, new_value=0.9, alpha=1.0) == 0.9


@pytest.mark.unit
def test_alpha_small_pulls_toward_prior():
    out = ema_update(prior_smoothed=0.2, new_value=0.9, alpha=0.1)
    assert math.isclose(out, 0.1 * 0.9 + 0.9 * 0.2)


@pytest.mark.unit
def test_invalid_alpha_rejected():
    with pytest.raises(ValueError):
        ema_update(prior_smoothed=0.5, new_value=0.5, alpha=0.0)
    with pytest.raises(ValueError):
        ema_update(prior_smoothed=0.5, new_value=0.5, alpha=-0.1)
    with pytest.raises(ValueError):
        ema_update(prior_smoothed=0.5, new_value=0.5, alpha=1.1)


@pytest.mark.unit
def test_repeated_application_converges_toward_input():
    """Feeding the same value repeatedly should drive the smoothed value
    toward that input."""
    s = None
    for _ in range(100):
        s = ema_update(s, new_value=0.8, alpha=0.2)
    assert math.isclose(s, 0.8, abs_tol=1e-6)


@pytest.mark.unit
def test_absent_run_does_not_decay_the_smoothed_value():
    """Many None inputs in a row must not change the smoothed value at all,
    which is the difference between holding and averaging-in zero."""
    s = 0.55
    for _ in range(50):
        s = ema_update(s, new_value=None, alpha=0.3)
    assert s == 0.55
