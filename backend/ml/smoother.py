"""Exponential moving average for the calibrated pain probability.

The smoother holds on absent frames. A None input does not get folded into
the average and does not collapse the smoothed value toward zero; it returns
the prior smoothed value unchanged. This matches the project's "absent is
not zero" contract: a missing facial signal is missing, not "no pain".

Pure function over (prior, new, alpha). State is held by the caller in
FacialStreamState. No globals.

Research prototype, not a medical device.
"""
from __future__ import annotations

from typing import Optional


def ema_update(
    prior_smoothed: Optional[float],
    new_value: Optional[float],
    alpha: float,
) -> Optional[float]:
    """One EMA step.

    Returns:
        prior_smoothed when new_value is None (hold on absence),
        new_value when prior_smoothed is None (cold start),
        alpha * new_value + (1 - alpha) * prior_smoothed otherwise.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if new_value is None:
        return prior_smoothed
    if prior_smoothed is None:
        return float(new_value)
    return alpha * float(new_value) + (1.0 - alpha) * float(prior_smoothed)
