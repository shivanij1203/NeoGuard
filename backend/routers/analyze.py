"""REST analyze endpoints. Stateless one-shot scoring.

The WebSocket path owns smoothing and cadence. These endpoints intentionally
do not smooth across calls; each call stands on its own. MC dropout runs on
every call because there is no stream cadence to throttle against.

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

from config import settings
from ml.fusion import (
    STATUS_OUT_OF_DISTRIBUTION,
    STATUS_UNAVAILABLE,
    fuse_uncertainty_weighted,
)
from ml.ncnn.scale import prob_to_score
from ml.scoring import (
    SUBJECT_NOT_RECOGNIZED_LABEL,
    UNAVAILABLE_LABEL,
    _alert_level,
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
    signal_status: str = STATUS_UNAVAILABLE
    ood_reason: str | None = None
    fusion_weights: dict = {"facial": 0.0, "audio": 0.0}


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

        # Out-of-distribution hard stop. A non-infant face yields no score on
        # the stateless path either; mirror the WebSocket contract.
        if result.get("out_of_distribution"):
            landmarks_list = None
            if result.get("landmarks") is not None:
                landmarks_list = result["landmarks"][:, :2].tolist()
            return AnalysisResponse(
                face_detected=True,
                facial_score=None,
                audio_score=None,
                composite_score=None,
                alert_level=_alert_level(None, STATUS_OUT_OF_DISTRIBUTION),
                pain_label=SUBJECT_NOT_RECOGNIZED_LABEL,
                features=None,
                cry_detected=False,
                cry_type="no_cry",
                timestamp=datetime.utcnow().isoformat(),
                landmarks=landmarks_list,
                prob_pain=None,
                uncertainty=None,
                frame_to_score_ms=result.get("frame_to_score_ms"),
                signal_status=STATUS_OUT_OF_DISTRIBUTION,
                ood_reason=result.get("ood_reason"),
                fusion_weights={"facial": 0.0, "audio": 0.0},
            )

        face_detected = bool(result.get("face_detected"))
        prob_pain = result.get("prob_pain")
        uncertainty = result.get("uncertainty")

        # Stateless path: no smoother. Facial score is prob_to_score on the
        # raw prob_pain.
        if face_detected and prob_pain is not None:
            facial_score = prob_to_score(prob_pain)
        else:
            facial_score = None

        fusion = fuse_uncertainty_weighted(
            facial_score=facial_score,
            uncertainty=uncertainty if face_detected else None,
            audio_score=None,
            base_facial_weight=settings.facial_weight,
            base_audio_weight=settings.audio_weight,
        )

        label = get_pain_label(fusion["composite_score"])
        alert = _alert_level(fusion["composite_score"], fusion["signal_status"])

        landmarks_list = None
        if result.get("landmarks") is not None:
            landmarks_list = result["landmarks"][:, :2].tolist()

        return AnalysisResponse(
            face_detected=face_detected,
            facial_score=facial_score,
            audio_score=None,
            composite_score=fusion["composite_score"],
            alert_level=alert,
            pain_label=label,
            features=result.get("features"),
            cry_detected=False,
            cry_type="no_cry",
            timestamp=datetime.utcnow().isoformat(),
            landmarks=landmarks_list,
            prob_pain=prob_pain,
            uncertainty=uncertainty,
            frame_to_score_ms=result.get("frame_to_score_ms"),
            signal_status=fusion["signal_status"],
            fusion_weights=fusion["weights_used"],
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
        fusion = fuse_uncertainty_weighted(
            facial_score=None,
            uncertainty=None,
            audio_score=audio_score,
            base_facial_weight=settings.facial_weight,
            base_audio_weight=settings.audio_weight,
        )
        label = get_pain_label(fusion["composite_score"])
        alert = _alert_level(fusion["composite_score"], fusion["signal_status"])

        return {
            **result,
            "composite_score": fusion["composite_score"],
            "alert_level": alert,
            "pain_label": label,
            "signal_status": fusion["signal_status"],
            "fusion_weights": fusion["weights_used"],
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
            "signal_status": STATUS_UNAVAILABLE,
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
        signal_status=STATUS_UNAVAILABLE,
        fusion_weights={"facial": 0.0, "audio": 0.0},
    )
