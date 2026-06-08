"""Per-stream state for the N-CNN facial pipeline.

Owns two things that should not live in a global registry:
  - frame_count, used to throttle the K-pass MC dropout uncertainty refresh.
  - cached_uncertainty, the last MC-dropout standard deviation, reused on
    frames where MC is not refreshed.

A separate FacialStreamState belongs to each WebSocket connection. Per the
plan, this is per-connection, not keyed by patient_id in a global dict. If a
genuine multi-viewer requirement appears later, lift the state into a small
registry with an eviction policy then.

TODO: eviction policy if state ever moves out of the per-connection scope so
a long-running service does not accumulate state forever.

Research prototype, not a medical device.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class FacialStreamState:
    """Immutable per-connection state for the facial pipeline.

    Holds MC-dropout cadence, the EMA-smoothed pain probability, and the last
    fresh composite score so a both-modalities-absent frame can hold a stale
    value rather than fabricating zero. All updates produce a new instance.
    """

    frame_count: int = 0
    cached_uncertainty: Optional[float] = None
    smoothed_prob_pain: Optional[float] = None
    last_composite_score: Optional[float] = None
    last_fresh_composite_frame: Optional[int] = None

    def should_refresh_uncertainty(self, interval_frames: int) -> bool:
        """True when this frame should run the K-pass MC dropout refresh.
        Always refresh on the very first frame, then every interval_frames."""
        if interval_frames < 1:
            raise ValueError("interval_frames must be >= 1")
        if self.cached_uncertainty is None:
            return True
        return (self.frame_count % interval_frames) == 0

    def advanced(
        self,
        new_uncertainty: Optional[float] = None,
        new_smoothed_prob: Optional[float] = None,
        new_fresh_composite: Optional[float] = None,
    ) -> "FacialStreamState":
        """Return a new state with the frame counter incremented.

        Each field follows the same hold-on-None semantics:
          - new_uncertainty=None preserves the cached uncertainty.
          - new_smoothed_prob=None preserves the prior smoothed probability,
            because EMA's "hold on absent" lives in the smoother itself; the
            caller passes None when the smoother returned None.
          - new_fresh_composite=None preserves the prior composite and the
            frame number at which it was last fresh, so a stale-hold layer
            can compute its age.
        """
        next_uncertainty = (
            new_uncertainty
            if new_uncertainty is not None
            else self.cached_uncertainty
        )
        next_smoothed = (
            new_smoothed_prob
            if new_smoothed_prob is not None
            else self.smoothed_prob_pain
        )
        new_frame_count = self.frame_count + 1
        if new_fresh_composite is not None:
            next_composite = new_fresh_composite
            next_fresh_frame = new_frame_count
        else:
            next_composite = self.last_composite_score
            next_fresh_frame = self.last_fresh_composite_frame
        return replace(
            self,
            frame_count=new_frame_count,
            cached_uncertainty=next_uncertainty,
            smoothed_prob_pain=next_smoothed,
            last_composite_score=next_composite,
            last_fresh_composite_frame=next_fresh_frame,
        )
