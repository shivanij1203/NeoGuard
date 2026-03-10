"""
Video Feature Extractor — NeoGuard Inference Pipeline
======================================================
Step 1: OpenCV crops face from frame
Step 2: MediaPipe maps 468 facial landmarks
Step 3: Extract 13 Action Unit (AU) features (clinically linked to pain)

AU features extracted:
  AU1  - Inner brow raise       (forehead tension)
  AU4  - Brow lowering L/R      (forehead → eye compression)
  AU6  - Cheek raise L/R        (eye-to-mouth distance)
  AU9  - Nose wrinkle           (nose bridge scrunch)
  AU20 - Lip corner stretch     (mouth corners pull sideways)
  AU25 - Lips part              (mouth open / cry shape)
  AU43 - Eye closure L/R        (lid compression)
  + Inter-pupil distance (normalization reference, used internally)

All distances are normalized by inter-pupil distance to be
scale-invariant across different face sizes and camera distances.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional


# MediaPipe landmark indices for pain-relevant facial regions
# Reference: MediaPipe Face Mesh topology
LANDMARKS = {
    # Brow region
    "left_inner_brow":  107,
    "right_inner_brow": 336,
    "left_outer_brow":  70,
    "right_outer_brow": 300,

    # Eye corners
    "left_eye_inner":   133,
    "left_eye_outer":   33,
    "right_eye_inner":  362,
    "right_eye_outer":  263,

    # Eyelids (upper/lower for closure)
    "left_upper_lid":   159,
    "left_lower_lid":   145,
    "right_upper_lid":  386,
    "right_lower_lid":  374,

    # Pupil centers (approximated)
    "left_pupil":       468,  # iris center (requires refine_landmarks=True)
    "right_pupil":      473,

    # Nose
    "nose_bridge":      6,
    "nose_tip":         1,
    "nose_left_wing":   218,
    "nose_right_wing":  438,

    # Mouth
    "mouth_left":       61,
    "mouth_right":      291,
    "upper_lip_center": 13,
    "lower_lip_center": 14,
}


class FaceFeatureExtractor:
    """
    Extracts 13 pain-relevant AU features from a face image.

    Usage:
        extractor = FaceFeatureExtractor()
        features = extractor.extract(frame)
        # features: np.ndarray of shape (13,) or None if no face detected
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,    # enables iris landmarks (468, 473)
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _get_point(self, landmarks, key: str, img_w: int, img_h: int) -> np.ndarray:
        idx = LANDMARKS[key]
        lm = landmarks.landmark[idx]
        return np.array([lm.x * img_w, lm.y * img_h])

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _crop_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Crop face region from frame using Haar cascade. Returns None if no face."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40)
        )
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        # Pad the crop by 20% for better MediaPipe coverage
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + w + pad_x)
        y2 = min(frame.shape[0], y + h + pad_y)
        return frame[y1:y2, x1:x2]

    def extract(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 13 AU features from a BGR video frame.

        Args:
            frame: BGR image as numpy array (from OpenCV)

        Returns:
            np.ndarray of shape (13,) — normalized AU feature vector
            None if no face detected
        """
        # Step 1: Crop face
        face_crop = self._crop_face(frame)
        if face_crop is None:
            # Fallback: try on full frame
            face_crop = frame

        img_h, img_w = face_crop.shape[:2]
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Step 2: MediaPipe 468 landmarks
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0]

        def pt(key):
            return self._get_point(lm, key, img_w, img_h)

        # Step 3: Normalization reference — inter-pupil distance
        try:
            interp = self._dist(pt("left_pupil"), pt("right_pupil"))
        except Exception:
            # Fallback: eye corner distance
            interp = self._dist(pt("left_eye_outer"), pt("right_eye_outer"))

        if interp < 1e-6:
            return None

        def normed(a, b):
            return self._dist(pt(a), pt(b)) / interp

        # Step 4: 13 AU features
        features = np.array([
            # AU1  Inner brow raise — gap between inner brows
            normed("left_inner_brow",  "right_inner_brow"),

            # AU4  Brow lowering L — inner brow compressed toward eye
            normed("left_inner_brow",  "left_eye_inner"),

            # AU4  Brow lowering R
            normed("right_inner_brow", "right_eye_inner"),

            # AU2  Outer brow raise L
            normed("left_outer_brow",  "left_eye_outer"),

            # AU2  Outer brow raise R
            normed("right_outer_brow", "right_eye_outer"),

            # AU6  Cheek raise L — eye corner to mouth corner
            normed("left_eye_inner",   "mouth_left"),

            # AU6  Cheek raise R
            normed("right_eye_inner",  "mouth_right"),

            # AU9  Nose wrinkle — nose wing spread
            normed("nose_left_wing",   "nose_right_wing"),

            # AU9  Nose bridge scrunch — bridge to tip compression
            normed("nose_bridge",      "nose_tip"),

            # AU20 Lip corner stretch — mouth width
            normed("mouth_left",       "mouth_right"),

            # AU25 Lips part — vertical mouth opening
            normed("upper_lip_center", "lower_lip_center"),

            # AU43 Eye closure L — lid gap
            normed("left_upper_lid",   "left_lower_lid"),

            # AU43 Eye closure R
            normed("right_upper_lid",  "right_lower_lid"),
        ], dtype=np.float32)

        return features

    def extract_from_file(self, image_path: str) -> Optional[np.ndarray]:
        """Extract features from an image file path."""
        frame = cv2.imread(image_path)
        if frame is None:
            return None
        return self.extract(frame)

    def extract_from_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Extract features from raw image bytes."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        return self.extract(frame)

    def close(self):
        self.face_mesh.close()
