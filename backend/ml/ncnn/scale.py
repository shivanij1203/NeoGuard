"""Single source of truth for the probability-to-score mapping.

Both the classifier wrapper and the dashboard band must use the same mapping.
If they drift, the number a clinician sees and the number internal fusion
operates on stop being the same number. Define it once here and import from
both places.

Research prototype, not a medical device.
"""
from __future__ import annotations


SCORE_MIN = 0.0
SCORE_MAX = 10.0


def prob_to_score(prob_pain: float) -> float:
    """Map a calibrated pain probability in [0, 1] to a 0-to-10 dashboard
    score. Linear mapping for now; Phase 5 reuses this same function for the
    dashboard band so the score and band cannot diverge.

    TODO: revisit the mapping once we have calibration data. A linear scale is
    a placeholder, not a clinical scale.
    """
    if prob_pain is None:
        raise ValueError("prob_pain is None; absent facial signal should not be scored")
    if prob_pain < 0.0 or prob_pain > 1.0:
        raise ValueError(f"prob_pain must be in [0, 1], got {prob_pain}")
    return round(prob_pain * SCORE_MAX, 2)
