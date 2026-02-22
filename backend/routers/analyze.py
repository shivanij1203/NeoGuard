import base64
import logging
import numpy as np
import cv2
from datetime import datetime

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

from ml.scoring import get_facial_classifier, get_cry_analyzer, compute_composite_score, get_pain_label

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analyze", tags=["analyze"])


class FrameRequest(BaseModel):
    frame: str  # base64-encoded JPEG
    patient_id: int | None = None


class AnalysisResponse(BaseModel):
    face_detected: bool
    facial_score: float | None
    audio_score: float | None
    composite_score: float
    alert_level: str
    pain_label: dict
    features: dict | None
    cry_detected: bool
    cry_type: str
    timestamp: str
    # Landmark data for overlay rendering
    landmarks: list[list[float]] | None = None


@router.post("/frame", response_model=AnalysisResponse)
async def analyze_frame(request: FrameRequest):
    """
    Analyze a single video frame for facial pain indicators.
    Accepts base64-encoded JPEG image.
    Returns pain score, features, and landmark coordinates for overlay.
    """
    try:
        frame_bytes = base64.b64decode(request.frame)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return _empty_response()

        classifier = get_facial_classifier()
        result = classifier.predict(frame)

        facial_score = result["facial_score"] if result["face_detected"] else None
        composite = compute_composite_score(facial_score, None)
        label = get_pain_label(composite["composite_score"])

        # Convert landmarks to serializable format
        landmarks_list = None
        if result.get("landmarks") is not None:
            landmarks_list = result["landmarks"][:, :2].tolist()  # x, y only

        return AnalysisResponse(
            face_detected=result["face_detected"],
            facial_score=facial_score,
            audio_score=None,
            composite_score=composite["composite_score"],
            alert_level=composite["alert_level"],
            pain_label=label,
            features=result.get("features"),
            cry_detected=False,
            cry_type="no_cry",
            timestamp=datetime.utcnow().isoformat(),
            landmarks=landmarks_list,
        )

    except Exception as e:
        logger.error(f"Frame analysis error: {e}")
        return _empty_response()


@router.post("/audio")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze an audio clip for cry classification.
    Accepts WAV/MP3 file upload.
    """
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
            "audio_score": 0.0,
            "composite_score": 0.0,
            "alert_level": "none",
            "error": str(e),
        }


def _empty_response():
    return AnalysisResponse(
        face_detected=False,
        facial_score=None,
        audio_score=None,
        composite_score=0.0,
        alert_level="none",
        pain_label={"level": "No Pain", "color": "#22c55e", "severity": 0},
        features=None,
        cry_detected=False,
        cry_type="no_cry",
        timestamp=datetime.utcnow().isoformat(),
        landmarks=None,
    )
