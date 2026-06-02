"""REST analyze endpoints. Stateless one-shot scoring.

The WebSocket path owns cadence. These endpoints do not carry state across
calls; each call stands on its own and runs MC dropout every time because
there is no stream cadence to throttle against.

Research prototype, not a medical device.
"""
from __future__ import annotations

import base64
import logging
from datetime import datetime

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel

from ml.ncnn.scale import prob_to_score
from ml.scoring import (
    UNAVAILABLE_LABEL,
    compute_composite_score,
    get_cry_analyzer,
    get_facial_classifier,
    get_pain_label,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analyze", tags=["analyze"])


class FrameRequest(BaseModel):
    frame: str  # base64-encoded JPEG
    patient_id: int | None = None


class AnalysisResponse(BaseModel):
    face_detected: bool
    facial_score: float | None
    audio_score: float | None
    composite_score: float | None
    alert_level: str
    pain_label: dict
    features: dict | None
    cry_detected: bool
    cry_type: str
    timestamp: str
    landmarks: list[list[float]] | None = None
    prob_pain: float | None = None
    uncertainty: float | None = None
    frame_to_score_ms: float | None = None


@router.post("/frame", response_model=AnalysisResponse)
async def analyze_frame(request: FrameRequest):
    """Analyze a single video frame. Stateless, one frame in, one frame out."""
    try:
        frame_bytes = base64.b64decode(request.frame)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return _empty_response()

        classifier = get_facial_classifier()
        result = classifier.predict(frame, compute_uncertainty=True)

        face_detected = bool(result.get("face_detected"))
        prob_pain = result.get("prob_pain")
        facial_score = (
            prob_to_score(prob_pain)
            if (face_detected and prob_pain is not None)
            else None
        )

        composite = compute_composite_score(facial_score, None)
        label = get_pain_label(composite["composite_score"])

        landmarks_list = None
        if result.get("landmarks") is not None:
            landmarks_list = result["landmarks"][:, :2].tolist()

        return AnalysisResponse(
            face_detected=face_detected,
            facial_score=facial_score,
            audio_score=None,
            composite_score=composite["composite_score"],
            alert_level=composite["alert_level"],
            pain_label=label,
            features=None,
            cry_detected=False,
            cry_type="no_cry",
            timestamp=datetime.utcnow().isoformat(),
            landmarks=landmarks_list,
            prob_pain=prob_pain,
            uncertainty=result.get("uncertainty"),
            frame_to_score_ms=result.get("frame_to_score_ms"),
        )

    except Exception as e:
        logger.error(f"Frame analysis error: {e}")
        return _empty_response()


@router.post("/audio")
async def analyze_audio(file: UploadFile = File(...)):
    """Analyze an audio clip for cry classification."""
    try:
        contents = await file.read()
        analyzer = get_cry_analyzer()
        result = analyzer.predict_from_bytes(contents)

        audio_score = result["audio_score"] if result["cry_detected"] else None
        composite = compute_composite_score(None, audio_score)
        label = get_pain_label(composite["composite_score"])

        return {
            **result,
            "composite_score": composite["composite_score"],
            "alert_level": composite["alert_level"],
            "pain_label": label,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Audio analysis error: {e}")
        return {
            "cry_detected": False,
            "cry_type": "no_cry",
            "audio_score": None,
            "composite_score": None,
            "alert_level": "unavailable",
            "error": str(e),
        }


def _empty_response() -> AnalysisResponse:
    return AnalysisResponse(
        face_detected=False,
        facial_score=None,
        audio_score=None,
        composite_score=None,
        alert_level="unavailable",
        pain_label=UNAVAILABLE_LABEL,
        features=None,
        cry_detected=False,
        cry_type="no_cry",
        timestamp=datetime.utcnow().isoformat(),
        landmarks=None,
        prob_pain=None,
        uncertainty=None,
        frame_to_score_ms=None,
    )
