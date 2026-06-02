"""Tests for FacialStreamState, the per-connection MC dropout cadence state."""
from __future__ import annotations

import pytest

from ml.ncnn.stream_state import FacialStreamState


@pytest.mark.unit
def test_initial_state_demands_uncertainty_refresh():
    state = FacialStreamState()
    assert state.should_refresh_uncertainty(interval_frames=30)


@pytest.mark.unit
def test_refresh_runs_again_at_the_interval():
    """First frame refreshes (no cache yet). Then frames 1..N-1 use the cache.
    Frame N (counter == interval) refreshes again. With interval=4 the
    refresh frames in the first cycle are counter values 0 and 4."""
    state = FacialStreamState()
    refresh_frames: list[int] = []
    for _ in range(8):
        if state.should_refresh_uncertainty(interval_frames=4):
            refresh_frames.append(state.frame_count)
            state = state.advanced(new_uncertainty=0.1)
        else:
            state = state.advanced(new_uncertainty=None)
    assert refresh_frames == [0, 4]


@pytest.mark.unit
def test_advanced_increments_frame_count_immutably():
    state = FacialStreamState()
    new = state.advanced(new_uncertainty=0.2)
    assert state.frame_count == 0
    assert new.frame_count == 1


@pytest.mark.unit
def test_advanced_with_none_preserves_prior_cached_uncertainty():
    state = FacialStreamState(frame_count=5, cached_uncertainty=0.3)
    new = state.advanced(new_uncertainty=None)
    assert new.cached_uncertainty == 0.3
    assert new.frame_count == 6


@pytest.mark.unit
def test_advanced_with_value_overwrites_cache():
    state = FacialStreamState(frame_count=5, cached_uncertainty=0.3)
    new = state.advanced(new_uncertainty=0.05)
    assert new.cached_uncertainty == 0.05


@pytest.mark.unit
def test_invalid_interval_is_rejected():
    state = FacialStreamState()
    with pytest.raises(ValueError):
        state.should_refresh_uncertainty(interval_frames=0)


@pytest.mark.unit
def test_advanced_holds_smoothed_prob_when_none():
    state = FacialStreamState(smoothed_prob_pain=0.42)
    new = state.advanced(new_smoothed_prob=None)
    assert new.smoothed_prob_pain == 0.42


@pytest.mark.unit
def test_advanced_overwrites_smoothed_prob_when_provided():
    state = FacialStreamState(smoothed_prob_pain=0.42)
    new = state.advanced(new_smoothed_prob=0.7)
    assert new.smoothed_prob_pain == 0.7


@pytest.mark.unit
def test_advanced_records_fresh_composite_with_frame_stamp():
    state = FacialStreamState(frame_count=4)
    new = state.advanced(new_fresh_composite=5.5)
    assert new.last_composite_score == 5.5
    assert new.last_fresh_composite_frame == 5  # the new frame_count value


@pytest.mark.unit
def test_advanced_holds_stale_composite_when_none():
    state = FacialStreamState(
        frame_count=4,
        last_composite_score=3.3,
        last_fresh_composite_frame=2,
    )
    new = state.advanced(new_fresh_composite=None)
    assert new.last_composite_score == 3.3
    assert new.last_fresh_composite_frame == 2
    assert new.frame_count == 5
