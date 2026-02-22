import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract AU-proxy geometric features from MediaPipe Face Mesh landmarks.

    Maps facial geometry to neonatal pain Action Units:
    - AU4:  Brow lowering (furrowed brow)
    - AU6+7: Eye squeeze
    - AU9+10: Nasolabial furrow / upper lip raise
    - AU43: Eye closure
    - AU27: Mouth stretch (cry face)
    """

    # Landmark indices (MediaPipe Face Mesh 468-point model)
    # Brow
    LEFT_BROW_INNER = 107
    LEFT_BROW_OUTER = 70
    RIGHT_BROW_INNER = 336
    RIGHT_BROW_OUTER = 300
    LEFT_BROW_MID = 105
    RIGHT_BROW_MID = 334

    # Eyes
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    LEFT_EYE_LEFT = 33
    LEFT_EYE_RIGHT = 133
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    RIGHT_EYE_LEFT = 362
    RIGHT_EYE_RIGHT = 263

    # Nose / Nasolabial
    NOSE_TIP = 1
    NOSE_BRIDGE = 6

    # Mouth
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    MOUTH_LEFT = 78
    MOUTH_RIGHT = 308
    UPPER_LIP_TOP = 0
    LOWER_LIP_BOTTOM = 17

    # Reference points for normalization
    FOREHEAD = 10
    CHIN = 152
    LEFT_TEMPLE = 234
    RIGHT_TEMPLE = 454

    def extract(self, landmarks: np.ndarray) -> dict:
        """
        Extract pain-relevant features from 468 facial landmarks.
        Input: landmarks array of shape (468, 3) in pixel coordinates.
        Returns: dict of feature names → float values.
        """
        # Compute face normalization distance (forehead to chin)
        face_height = self._dist(landmarks, self.FOREHEAD, self.CHIN)
        face_width = self._dist(landmarks, self.LEFT_TEMPLE, self.RIGHT_TEMPLE)

        if face_height < 1 or face_width < 1:
            logger.warning("Face too small for reliable feature extraction")
            return self._empty_features()

        features = {}

        # === AU4: Brow Furrow ===
        # Measure brow-to-eye distance (decreases when brow furrows)
        left_brow_eye_dist = self._dist(landmarks, self.LEFT_BROW_MID, self.LEFT_EYE_TOP)
        right_brow_eye_dist = self._dist(landmarks, self.RIGHT_BROW_MID, self.RIGHT_EYE_TOP)
        avg_brow_eye_dist = (left_brow_eye_dist + right_brow_eye_dist) / 2
        features["brow_eye_dist_norm"] = avg_brow_eye_dist / face_height

        # Inner brow distance (decreases when brows pull together)
        inner_brow_dist = self._dist(landmarks, self.LEFT_BROW_INNER, self.RIGHT_BROW_INNER)
        features["inner_brow_dist_norm"] = inner_brow_dist / face_width

        # Brow slope (steepens during furrow)
        left_brow_slope = (landmarks[self.LEFT_BROW_INNER][1] - landmarks[self.LEFT_BROW_OUTER][1]) / max(
            abs(landmarks[self.LEFT_BROW_INNER][0] - landmarks[self.LEFT_BROW_OUTER][0]), 1
        )
        right_brow_slope = (landmarks[self.RIGHT_BROW_INNER][1] - landmarks[self.RIGHT_BROW_OUTER][1]) / max(
            abs(landmarks[self.RIGHT_BROW_INNER][0] - landmarks[self.RIGHT_BROW_OUTER][0]), 1
        )
        features["brow_slope_avg"] = (abs(left_brow_slope) + abs(right_brow_slope)) / 2

        # === AU6+7 & AU43: Eye Squeeze / Closure ===
        left_ear = self._eye_aspect_ratio(
            landmarks, self.LEFT_EYE_TOP, self.LEFT_EYE_BOTTOM,
            self.LEFT_EYE_LEFT, self.LEFT_EYE_RIGHT
        )
        right_ear = self._eye_aspect_ratio(
            landmarks, self.RIGHT_EYE_TOP, self.RIGHT_EYE_BOTTOM,
            self.RIGHT_EYE_LEFT, self.RIGHT_EYE_RIGHT
        )
        features["left_ear"] = left_ear
        features["right_ear"] = right_ear
        features["avg_ear"] = (left_ear + right_ear) / 2

        # === AU9+10: Nasolabial Furrow ===
        # Nose tip to upper lip distance (decreases during pain)
        nose_lip_dist = self._dist(landmarks, self.NOSE_TIP, self.UPPER_LIP_TOP)
        features["nose_lip_dist_norm"] = nose_lip_dist / face_height

        # Nose bridge angle change
        nose_bridge_to_tip = self._dist(landmarks, self.NOSE_BRIDGE, self.NOSE_TIP)
        features["nose_length_norm"] = nose_bridge_to_tip / face_height

        # === AU27: Mouth Stretch (Cry) ===
        mouth_height = self._dist(landmarks, self.MOUTH_TOP, self.MOUTH_BOTTOM)
        mouth_width = self._dist(landmarks, self.MOUTH_LEFT, self.MOUTH_RIGHT)
        features["mouth_aspect_ratio"] = mouth_height / max(mouth_width, 1)
        features["mouth_height_norm"] = mouth_height / face_height
        features["mouth_width_norm"] = mouth_width / face_width

        # Full lip stretch
        lip_stretch = self._dist(landmarks, self.UPPER_LIP_TOP, self.LOWER_LIP_BOTTOM)
        features["lip_stretch_norm"] = lip_stretch / face_height

        # === Composite indicators ===
        # Face symmetry (pain often asymmetric)
        left_eye_area = left_ear * self._dist(landmarks, self.LEFT_EYE_LEFT, self.LEFT_EYE_RIGHT)
        right_eye_area = right_ear * self._dist(landmarks, self.RIGHT_EYE_LEFT, self.RIGHT_EYE_RIGHT)
        features["eye_asymmetry"] = abs(left_eye_area - right_eye_area) / max(left_eye_area + right_eye_area, 1)

        return features

    def get_feature_names(self) -> list[str]:
        """Return ordered list of feature names for model input."""
        return [
            "brow_eye_dist_norm", "inner_brow_dist_norm", "brow_slope_avg",
            "left_ear", "right_ear", "avg_ear",
            "nose_lip_dist_norm", "nose_length_norm",
            "mouth_aspect_ratio", "mouth_height_norm", "mouth_width_norm", "lip_stretch_norm",
            "eye_asymmetry",
        ]

    def features_to_array(self, features: dict) -> np.ndarray:
        """Convert feature dict to ordered numpy array for model input."""
        return np.array([features[name] for name in self.get_feature_names()])

    def _dist(self, landmarks: np.ndarray, idx1: int, idx2: int) -> float:
        """Euclidean distance between two landmarks (2D, ignoring z)."""
        return float(np.sqrt(
            (landmarks[idx1][0] - landmarks[idx2][0]) ** 2 +
            (landmarks[idx1][1] - landmarks[idx2][1]) ** 2
        ))

    def _eye_aspect_ratio(self, landmarks: np.ndarray, top: int, bottom: int, left: int, right: int) -> float:
        """
        Eye Aspect Ratio (EAR) — measures eye openness.
        Low EAR = eye squeeze/closure (pain indicator).
        """
        vertical = self._dist(landmarks, top, bottom)
        horizontal = self._dist(landmarks, left, right)
        return vertical / max(horizontal, 1)

    def _empty_features(self) -> dict:
        return {name: 0.0 for name in self.get_feature_names()}
