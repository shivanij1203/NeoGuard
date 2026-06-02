"""Scoring layer: N-CNN facial classifier wiring and per-connection cadence.

This phase wires the WebSocket path to the N-CNN classifier and threads a
per-connection FacialStreamState so the K-pass MC dropout can be throttled.
Composite scoring is an interim weighted combination; a later phase replaces
it with uncertainty-weighted fusion and EMA smoothing.

Absent is not zero. When there is no usable face, facial fields stay None
rather than collapsing to a confident "no pain". When both modalities are
absent, the composite is None, not 0.0.

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
from ml.ncnn.scale import prob_to_score
from ml.ncnn.stream_state import FacialStreamState
from ml.ncnn_classifier import NCNNFacialPainClassifier

logger = logging.getLogger(__name__)

# Pain label used when no composite is available. Avoids fabricating a
# "No Pain" label for an absent signal.
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


def compute_composite_score(
    facial_score: Optional[float],
    audio_score: Optional[float],
) -> dict:
    """Interim composite on the 0-to-10 scale.

    Both modalities present: weighted combination. A single modality carries
    at full weight, including audio, so an interim audio-only window is not
    deflated by the facial weight. Both absent: composite is None, never 0.0.
    A later phase replaces this with uncertainty-weighted fusion.
    """
    facial_w = settings.facial_weight
    audio_w = settings.audio_weight

    if facial_score is not None and audio_score is not None:
        composite = round(
            float(np.clip(facial_w * facial_score + audio_w * audio_score, 0, 10)), 2
        )
    elif facial_score is not None:
        composite = round(float(np.clip(facial_score, 0, 10)), 2)
    elif audio_score is not None:
        composite = round(float(np.clip(audio_score, 0, 10)), 2)
    else:
        composite = None

    return {
        "composite_score": composite,
        "alert_level": _alert_level(composite),
        "facial_score": facial_score,
        "audio_score": audio_score,
    }


def _alert_level(score: Optional[float]) -> str:
    """Display band derived from the composite. No alerting fires here; this
    is a display string only."""
    if score is None:
        return "unavailable"
    if score >= settings.pain_urgent_threshold:
        return "severe"
    if score >= settings.pain_alert_threshold:
        return "moderate"
    return "none"


def get_pain_label(score: Optional[float]) -> dict:
    """Human-readable pain band. Returns the unavailable label on None."""
    if score is None:
        return UNAVAILABLE_LABEL
    if score <= 1:
        return {"level": "No Pain", "color": "#22c55e", "severity": 0}
    if score <= 3:
        return {"level": "Mild Discomfort", "color": "#eab308", "severity": 1}
    if score <= 6:
        return {"level": "Moderate Pain", "color": "#f97316", "severity": 2}
    return {"level": "Severe Pain", "color": "#ef4444", "severity": 3}


def _empty_payload(stream_state: FacialStreamState) -> tuple[dict, FacialStreamState]:
    """Keep-alive frame with no data. Does not advance state."""
    return (
        {
            "composite_score": None,
            "alert_level": "unavailable",
            "facial_score": None,
            "audio_score": None,
            "pain_label": UNAVAILABLE_LABEL,
            "face_detected": False,
            "prob_pain": None,
            "uncertainty": None,
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

    Per-connection state (cadence, cached uncertainty) is threaded through by
    the caller. Returns the result payload and the updated state.
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

    facial_score = (
        prob_to_score(raw_prob_pain)
        if (face_detected and raw_prob_pain is not None)
        else None
    )

    composite = compute_composite_score(facial_score, audio_score)
    pain_label = get_pain_label(composite["composite_score"])

    new_state = stream_state.advanced(new_uncertainty=fresh_uncertainty)

    return (
        {
            **composite,
            "pain_label": pain_label,
            "face_detected": face_detected,
            "prob_pain": raw_prob_pain,
            "uncertainty": new_state.cached_uncertainty if face_detected else None,
            "frame_to_score_ms": frame_to_score_ms,
            "cry_detected": audio_result.get("cry_detected", False) if audio_result else False,
            "cry_type": audio_result.get("cry_type", "no_cry") if audio_result else "no_cry",
        },
        new_state,
    )
