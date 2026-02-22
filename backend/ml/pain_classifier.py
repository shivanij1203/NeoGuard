import numpy as np
import joblib
import logging
from pathlib import Path

from ml.face_detector import FaceDetector
from ml.feature_extractor import FeatureExtractor
from config import settings

logger = logging.getLogger(__name__)


class FacialPainClassifier:
    """
    Classifies facial pain from MediaPipe Face Mesh landmarks.

    Pipeline: Frame → FaceDetector → FeatureExtractor → Classifier → Pain Score (0-10)

    Uses a trained XGBoost/RandomForest model on geometric AU-proxy features.
    Falls back to rule-based scoring if no trained model is available.
    """

    def __init__(self):
        self.face_detector = FaceDetector(
            min_detection_confidence=settings.min_detection_confidence,
            min_tracking_confidence=settings.min_tracking_confidence,
        )
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self._load_model()

    def _load_model(self):
        model_path = settings.facial_model_path
        if model_path.exists():
            self.model = joblib.load(model_path)
            logger.info(f"Loaded facial pain model from {model_path}")
        else:
            logger.warning(f"No trained model at {model_path}, using rule-based scoring")

    def predict(self, frame: np.ndarray) -> dict:
        """
        Predict pain score from a BGR video frame.
        Returns dict with score, features, and detection info.
        """
        detection = self.face_detector.detect(frame)

        if detection is None:
            return {
                "face_detected": False,
                "facial_score": 0.0,
                "features": {},
                "landmarks": None,
            }

        features = self.feature_extractor.extract(detection["landmarks_px"])
        feature_array = self.feature_extractor.features_to_array(features)

        if self.model is not None:
            # Use trained model
            score = float(self.model.predict(feature_array.reshape(1, -1))[0])
            score = np.clip(score, 0, 10)
        else:
            # Rule-based fallback using AU-proxy thresholds
            score = self._rule_based_score(features)

        return {
            "face_detected": True,
            "facial_score": round(score, 2),
            "features": features,
            "landmarks": detection["landmarks_px"],
        }

    def predict_with_overlay(self, frame: np.ndarray) -> tuple[dict, np.ndarray]:
        """Predict and return annotated frame with face mesh overlay."""
        detection = self.face_detector.detect(frame)

        if detection is None:
            return {
                "face_detected": False,
                "facial_score": 0.0,
                "features": {},
                "landmarks": None,
            }, frame

        features = self.feature_extractor.extract(detection["landmarks_px"])
        feature_array = self.feature_extractor.features_to_array(features)

        if self.model is not None:
            score = float(self.model.predict(feature_array.reshape(1, -1))[0])
            score = np.clip(score, 0, 10)
        else:
            score = self._rule_based_score(features)

        annotated_frame = self.face_detector.draw_landmarks(frame, detection)

        return {
            "face_detected": True,
            "facial_score": round(score, 2),
            "features": features,
            "landmarks": detection["landmarks_px"],
        }, annotated_frame

    def _rule_based_score(self, features: dict) -> float:
        """
        Rule-based pain scoring using NIPS-inspired thresholds.
        Each AU contributes 0-2 points, total scaled to 0-10.
        """
        score = 0.0

        # AU4: Brow furrow — low brow-eye distance = pain
        brow_dist = features.get("brow_eye_dist_norm", 0.1)
        if brow_dist < 0.04:
            score += 2.5
        elif brow_dist < 0.06:
            score += 1.5

        # Inner brow pinch
        inner_brow = features.get("inner_brow_dist_norm", 0.2)
        if inner_brow < 0.12:
            score += 1.0

        # AU6+7 / AU43: Eye squeeze — low EAR = pain
        ear = features.get("avg_ear", 0.3)
        if ear < 0.15:
            score += 2.5  # Eyes tightly shut
        elif ear < 0.22:
            score += 1.5

        # AU9+10: Nasolabial furrow — short nose-lip distance = pain
        nose_lip = features.get("nose_lip_dist_norm", 0.1)
        if nose_lip < 0.04:
            score += 1.5
        elif nose_lip < 0.06:
            score += 0.5

        # AU27: Mouth stretch — high MAR = cry face
        mar = features.get("mouth_aspect_ratio", 0.0)
        if mar > 0.6:
            score += 2.0
        elif mar > 0.4:
            score += 1.0

        # Eye asymmetry bonus
        asymmetry = features.get("eye_asymmetry", 0.0)
        if asymmetry > 0.3:
            score += 0.5

        return min(score, 10.0)

    def close(self):
        self.face_detector.close()
