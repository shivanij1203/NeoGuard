"""Scoring layer: N-CNN facial classifier wiring, EMA temporal smoothing on
prob_pain, and uncertainty-weighted fusion with audio.

Both-modalities-absent policy: do not fabricate a composite. If a prior
fresh composite exists on the stream, hold it tagged stale with an age in
frames so the dashboard can render the staleness. With no prior composite,
report signal_status unavailable and composite_score None.

TODO(phase 7): hysteresis-based episode detection for alerting, with separate
enter and exit thresholds across a time window so a single elevated frame
does not page anyone. Alerts are deferred until trial data sets the
thresholds; no alerting is wired here.

Research prototype, not a medical device.
"""
from __future__ import annotations

import base64
import logging
from typing import Optional

import cv2
import numpy as np

from config import settings
from ml.cry_analyzer import CryAnalyzer
from ml.fusion import (
    STATUS_STALE,
    STATUS_UNAVAILABLE,
    fuse_uncertainty_weighted,
)
from ml.ncnn.scale import prob_to_score
from ml.ncnn.stream_state import FacialStreamState
from ml.ncnn_classifier import NCNNFacialPainClassifier
from ml.smoother import ema_update

logger = logging.getLogger(__name__)

# Pain label used when no composite is available. Keeps the dashboard contract
# from breaking on None and avoids fabricating a "No Pain" label for an
# absent signal.
UNAVAILABLE_LABEL = {"level": "Signal Unavailable", "color": "#9ca3af", "severity": -1}

# Singleton instances.
_facial_classifier: Optional[NCNNFacialPainClassifier] = None
_cry_analyzer: Optional[CryAnalyzer] = None


def get_facial_classifier() -> NCNNFacialPainClassifier:
    global _facial_classifier
    if _facial_classifier is None:
        _facial_classifier = NCNNFacialPainClassifier()
    return _facial_classifier


def get_cry_analyzer() -> CryAnalyzer:
    global _cry_analyzer
    if _cry_analyzer is None:
        _cry_analyzer = CryAnalyzer()
    return _cry_analyzer


def get_pain_label(score: Optional[float]) -> dict:
    """Human-readable pain band for the dashboard.

    Reuses the same 0-to-10 scale as prob_to_score, so the band the dashboard
    shows is computed from the same number the classifier produced. Returns
    the unavailable label when there is no composite score.
    """
    if score is None:
        return UNAVAILABLE_LABEL
    if score <= 1:
        return {"level": "No Pain", "color": "#22c55e", "severity": 0}
    if score <= 3:
        return {"level": "Mild Discomfort", "color": "#eab308", "severity": 1}
    if score <= 6:
        return {"level": "Moderate Pain", "color": "#f97316", "severity": 2}
    return {"level": "Severe Pain", "color": "#ef4444", "severity": 3}


def _alert_level(score: Optional[float], status: str) -> str:
    """Display band derived from the fused composite.

    No alerting fires here. This is a display string only, the same as today.
    The hysteresis episode detector (TODO phase 7) is what should actually
    drive nurse-facing alerts; this string is for the dashboard only.
    """
    if status == STATUS_STALE:
        return "stale"
    if status == STATUS_UNAVAILABLE or score is None:
        return "unavailable"
    if score >= settings.pain_urgent_threshold:
        return "severe"
    if score >= settings.pain_alert_threshold:
        return "moderate"
    return "none"


def _empty_payload(stream_state: FacialStreamState) -> tuple[dict, FacialStreamState]:
    """Used for keep-alive frames with no data. Does not advance state."""
    return (
        {
            "composite_score": None,
            "alert_level": "unavailable",
            "facial_score": None,
            "audio_score": None,
            "pain_label": UNAVAILABLE_LABEL,
            "face_detected": False,
            "prob_pain": None,
            "prob_pain_smoothed": None,
            "uncertainty": None,
            "signal_status": STATUS_UNAVAILABLE,
            "stale": False,
            "stale_age_frames": None,
            "fusion_weights": {"facial": 0.0, "audio": 0.0},
            "frame_to_score_ms": None,
            "cry_detected": False,
            "cry_type": "no_cry",
        },
        stream_state,
    )


async def process_frame_data(
    data: dict | None,
    patient_id: int,
    stream_state: Optional[FacialStreamState] = None,
) -> tuple[dict, FacialStreamState]:
    """Process one WebSocket frame/audio message.

    Per-connection state (cadence, smoother, last composite) is threaded
    through by the caller. Returns the result payload and the updated state.
    Absent-is-not-zero: facial fields stay None when there is no usable face,
    the smoother holds its prior value rather than averaging in a zero, and
    a both-modalities-absent frame holds the prior composite tagged stale.
    """
    if stream_state is None:
        stream_state = FacialStreamState()

    if data is None:
        return _empty_payload(stream_state)

    facial_result: Optional[dict] = None
    audio_result: Optional[dict] = None

    if "frame" in data:
        try:
            frame_bytes = base64.b64decode(data["frame"])
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                classifier = get_facial_classifier()
                should_refresh = stream_state.should_refresh_uncertainty(
                    settings.ncnn_mc_interval_frames
                )
                facial_result = classifier.predict(
                    frame, compute_uncertainty=should_refresh
                )
        except Exception as e:
            logger.error(f"Error processing frame: {e}")

    if "audio" in data:
        try:
            audio_bytes = base64.b64decode(data["audio"])
            analyzer = get_cry_analyzer()
            audio_result = analyzer.predict_from_bytes(audio_bytes)
        except Exception as e:
            logger.error(f"Error processing audio: {e}")

    face_detected = bool(facial_result and facial_result.get("face_detected"))
    raw_prob_pain = facial_result.get("prob_pain") if facial_result else None
    fresh_uncertainty = facial_result.get("uncertainty") if facial_result else None
    frame_to_score_ms = (
        facial_result.get("frame_to_score_ms") if facial_result else None
    )
    audio_score = audio_result.get("audio_score") if audio_result else None

    # EMA smoother on prob_pain. None input holds the prior smoothed value,
    # so an absent-face frame never drags the smoothed signal toward zero.
    smoothed_prob = ema_update(
        prior_smoothed=stream_state.smoothed_prob_pain,
        new_value=raw_prob_pain,
        alpha=settings.ema_smoother_alpha,
    )

    # Facial 0-to-10 score via the single shared mapping.
    facial_score_for_fusion: Optional[float]
    if face_detected and smoothed_prob is not None:
        facial_score_for_fusion = prob_to_score(smoothed_prob)
    else:
        facial_score_for_fusion = None

    effective_uncertainty = (
        fresh_uncertainty
        if fresh_uncertainty is not None
        else stream_state.cached_uncertainty
    )
    fusion = fuse_uncertainty_weighted(
        facial_score=facial_score_for_fusion,
        uncertainty=effective_uncertainty if face_detected else None,
        audio_score=audio_score,
        base_facial_weight=settings.facial_weight,
        base_audio_weight=settings.audio_weight,
    )

    composite_score = fusion["composite_score"]
    signal_status = fusion["signal_status"]
    stale = False
    stale_age_frames: Optional[int] = None
    fresh_composite_for_state: Optional[float] = composite_score

    # Both-modalities-absent: hold prior composite tagged stale if we have
    # one, with an age in frames so the dashboard can render the staleness.
    if signal_status == STATUS_UNAVAILABLE and stream_state.last_composite_score is not None:
        if stream_state.last_fresh_composite_frame is not None:
            stale_age_frames = (
                stream_state.frame_count
                + 1
                - stream_state.last_fresh_composite_frame
            )
        composite_score = stream_state.last_composite_score
        stale = True
        signal_status = STATUS_STALE
        fresh_composite_for_state = None
    elif signal_status == STATUS_UNAVAILABLE:
        # No prior composite to hold. Stay unavailable.
        fresh_composite_for_state = None

    pain_label = get_pain_label(composite_score)
    alert_level = _alert_level(composite_score, signal_status)

    new_state = stream_state.advanced(
        new_uncertainty=fresh_uncertainty,
        new_smoothed_prob=smoothed_prob,
        new_fresh_composite=fresh_composite_for_state,
    )

    return (
        {
            "composite_score": composite_score,
            "alert_level": alert_level,
            "facial_score": facial_score_for_fusion,
            "audio_score": audio_score,
            "pain_label": pain_label,
            "face_detected": face_detected,
            "prob_pain": raw_prob_pain,
            "prob_pain_smoothed": new_state.smoothed_prob_pain,
            "uncertainty": (
                new_state.cached_uncertainty if face_detected else None
            ),
            "signal_status": signal_status,
            "stale": stale,
            "stale_age_frames": stale_age_frames,
            "fusion_weights": fusion["weights_used"],
            "frame_to_score_ms": frame_to_score_ms,
            "cry_detected": audio_result.get("cry_detected", False) if audio_result else False,
            "cry_type": audio_result.get("cry_type", "no_cry") if audio_result else "no_cry",
        },
        new_state,
    )
