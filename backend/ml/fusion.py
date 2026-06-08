"""Uncertainty-weighted fusion of the facial and audio pain signals.

Inputs:
    facial_score: smoothed facial pain on the 0-to-10 dashboard scale, or
        None when there is no usable facial signal.
    uncertainty: MC dropout standard deviation on the pain probability, or
        None. Clipped to [0, 1] for weighting; None is treated as 0
        (no penalty) so a missing uncertainty value does not silently
        discount the facial signal.
    audio_score: cry analyzer score on the 0-to-10 scale, or None.
    base_facial_weight, base_audio_weight: configured fusion weights.

Fusion rule when both modalities are present:
    effective_facial_weight = (1 - clip(uncertainty, 0, 1)) * base_facial_weight
    denom = effective_facial_weight + base_audio_weight
    composite = (effective_facial_weight * facial_score + base_audio_weight * audio_score) / denom
High facial uncertainty shifts the composite toward audio. Low facial
uncertainty preserves the facial signal.

When only one modality is present, it carries at full weight. When both are
absent, the function returns a None composite with signal_status="unavailable"
and the caller decides whether to hold the prior value as stale.

The pain_label band is computed by the caller via the same get_pain_label
function the dashboard uses, so the score the classifier produces and the
band the dashboard shows cannot drift apart.

TODO(phase 7): hysteresis episode detector for alerting. Alerts should not
fire on a single elevated frame. A clinician-facing alert needs sustained
elevation across some time window, with separate enter and exit thresholds
so the alert state does not chatter near the band boundary. This file
intentionally does not emit alerts; alerting is deferred until a real
threshold can be set against trial data.

Research prototype, not a medical device.
"""
from __future__ import annotations

from typing import Optional


# Signal status taxonomy. Surface on the payload so a UI can render the
# distinction without re-deriving it from None checks.
STATUS_FRESH = "fresh"               # both modalities contributed
STATUS_FACIAL_ONLY = "facial_only"   # audio absent, facial carried
STATUS_AUDIO_ONLY = "audio_only"     # facial absent, audio carried
STATUS_STALE = "stale"               # both absent, holding prior composite
STATUS_UNAVAILABLE = "unavailable"   # both absent, no prior composite to hold
# Set upstream by the out-of-distribution gate, not emitted by
# fuse_uncertainty_weighted. A non-infant face hard-stops the whole composite
# before fusion, so it is distinct from a facial-only absence.
STATUS_OUT_OF_DISTRIBUTION = "out_of_distribution"


def fuse_uncertainty_weighted(
    facial_score: Optional[float],
    uncertainty: Optional[float],
    audio_score: Optional[float],
    base_facial_weight: float,
    base_audio_weight: float,
) -> dict:
    """Compute one fused composite. Caller is responsible for any
    stale-hold logic when this returns STATUS_UNAVAILABLE."""
    if base_facial_weight < 0 or base_audio_weight < 0:
        raise ValueError("base weights must be >= 0")
    if base_facial_weight == 0 and base_audio_weight == 0:
        raise ValueError("at least one base weight must be positive")

    facial_present = facial_score is not None
    audio_present = audio_score is not None

    if not facial_present and not audio_present:
        return {
            "composite_score": None,
            "signal_status": STATUS_UNAVAILABLE,
            "weights_used": {"facial": 0.0, "audio": 0.0},
        }

    if facial_present and not audio_present:
        return {
            "composite_score": round(float(facial_score), 2),
            "signal_status": STATUS_FACIAL_ONLY,
            "weights_used": {"facial": 1.0, "audio": 0.0},
        }

    if audio_present and not facial_present:
        return {
            "composite_score": round(float(audio_score), 2),
            "signal_status": STATUS_AUDIO_ONLY,
            "weights_used": {"facial": 0.0, "audio": 1.0},
        }

    # Both present. Uncertainty weighting.
    # Unknown uncertainty (None) currently defaults to full facial trust
    # (treated as 0), so a missing MC reading does not silently discount the
    # facial signal. Revisit if MC ever fails to refresh on a long run.
    u = uncertainty if uncertainty is not None else 0.0
    u_clipped = max(0.0, min(1.0, float(u)))
    effective_facial = (1.0 - u_clipped) * base_facial_weight
    denom = effective_facial + base_audio_weight
    if denom <= 0:
        # Edge: facial discounted to zero and audio weight is zero. Treat as
        # audio carrying, even though the configured weight is zero, so we
        # never silently emit zero.
        return {
            "composite_score": round(float(audio_score), 2),
            "signal_status": STATUS_AUDIO_ONLY,
            "weights_used": {"facial": 0.0, "audio": 1.0},
        }

    w_facial = effective_facial / denom
    w_audio = base_audio_weight / denom
    composite = w_facial * float(facial_score) + w_audio * float(audio_score)
    return {
        "composite_score": round(composite, 2),
        "signal_status": STATUS_FRESH,
        "weights_used": {"facial": round(w_facial, 4), "audio": round(w_audio, 4)},
    }
