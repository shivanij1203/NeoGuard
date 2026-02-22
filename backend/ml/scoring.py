import numpy as np
import base64
import cv2
import logging
from datetime import datetime

from config import settings
from ml.pain_classifier import FacialPainClassifier
from ml.cry_analyzer import CryAnalyzer

logger = logging.getLogger(__name__)

# Singleton instances
_facial_classifier: FacialPainClassifier | None = None
_cry_analyzer: CryAnalyzer | None = None


def get_facial_classifier() -> FacialPainClassifier:
    global _facial_classifier
    if _facial_classifier is None:
        _facial_classifier = FacialPainClassifier()
    return _facial_classifier


def get_cry_analyzer() -> CryAnalyzer:
    global _cry_analyzer
    if _cry_analyzer is None:
        _cry_analyzer = CryAnalyzer()
    return _cry_analyzer


def compute_composite_score(
    facial_score: float | None,
    audio_score: float | None,
) -> dict:
    """
    Compute composite pain score using weighted combination.
    NIPS-inspired scale: 0-10.

    Weights: facial 70%, audio 30% (when both available).
    If only one modality available, use it alone.
    """
    facial_w = settings.facial_weight
    audio_w = settings.audio_weight

    if facial_score is not None and audio_score is not None:
        composite = facial_w * facial_score + audio_w * audio_score
    elif facial_score is not None:
        composite = facial_score
    elif audio_score is not None:
        composite = audio_score
    else:
        composite = 0.0

    composite = round(np.clip(composite, 0, 10), 2)

    # Determine alert level
    if composite >= settings.pain_urgent_threshold:
        alert_level = "severe"
    elif composite >= settings.pain_alert_threshold:
        alert_level = "moderate"
    else:
        alert_level = "none"

    return {
        "composite_score": composite,
        "alert_level": alert_level,
        "facial_score": facial_score,
        "audio_score": audio_score,
    }


def get_pain_label(score: float) -> dict:
    """Get human-readable pain level info from score."""
    if score <= 1:
        return {"level": "No Pain", "color": "#22c55e", "severity": 0}
    elif score <= 3:
        return {"level": "Mild Discomfort", "color": "#eab308", "severity": 1}
    elif score <= 6:
        return {"level": "Moderate Pain", "color": "#f97316", "severity": 2}
    else:
        return {"level": "Severe Pain", "color": "#ef4444", "severity": 3}


async def process_frame_data(data: dict | None, patient_id: int) -> dict:
    """
    Process incoming frame/audio data from WebSocket.
    Called by the WebSocket handler for real-time scoring.
    """
    if data is None:
        return compute_composite_score(None, None)

    facial_result = None
    audio_result = None

    # Process video frame if present
    if "frame" in data:
        try:
            frame_bytes = base64.b64decode(data["frame"])
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                classifier = get_facial_classifier()
                facial_result = classifier.predict(frame)
        except Exception as e:
            logger.error(f"Error processing frame: {e}")

    # Process audio chunk if present
    if "audio" in data:
        try:
            audio_bytes = base64.b64decode(data["audio"])
            analyzer = get_cry_analyzer()
            audio_result = analyzer.predict_from_bytes(audio_bytes)
        except Exception as e:
            logger.error(f"Error processing audio: {e}")

    facial_score = facial_result["facial_score"] if facial_result and facial_result.get("face_detected") else None
    audio_score = audio_result["audio_score"] if audio_result else None

    composite = compute_composite_score(facial_score, audio_score)
    pain_label = get_pain_label(composite["composite_score"])

    result = {
        **composite,
        "pain_label": pain_label,
        "face_detected": facial_result.get("face_detected", False) if facial_result else False,
        "cry_detected": audio_result.get("cry_detected", False) if audio_result else False,
        "cry_type": audio_result.get("cry_type", "no_cry") if audio_result else "no_cry",
    }

    # Include feature details if face detected
    if facial_result and facial_result.get("face_detected"):
        features = facial_result.get("features", {})
        result["brow_furrow"] = features.get("brow_eye_dist_norm")
        result["eye_squeeze"] = features.get("avg_ear")
        result["nasolabial_furrow"] = features.get("nose_lip_dist_norm")
        result["mouth_stretch"] = features.get("mouth_aspect_ratio")

    return result
