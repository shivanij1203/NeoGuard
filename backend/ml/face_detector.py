import cv2
import numpy as np
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """MediaPipe Face Mesh wrapper for extracting 468 facial landmarks."""

    # Key landmark indices for neonatal pain Action Units
    # Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    LANDMARKS = {
        # Brow landmarks (AU4 - brow lowering)
        "left_brow": [70, 63, 105, 66, 107],
        "right_brow": [336, 296, 334, 293, 300],
        # Eye landmarks (AU6+7 - eye squeeze, AU43 - eye closure)
        "left_eye": [33, 160, 158, 133, 153, 144],
        "right_eye": [362, 385, 387, 263, 373, 380],
        # Nose landmarks (AU9+10 - nasolabial furrow)
        "nose_tip": [1],
        "nose_bridge": [6],
        # Lip/mouth landmarks (AU27 - mouth stretch)
        "upper_lip": [13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82],
        "lower_lip": [14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, 95, 88, 178, 87],
        "mouth_top": [13],
        "mouth_bottom": [14],
        "mouth_left": [78],
        "mouth_right": [308],
        # Upper/lower eyelid for EAR calculation
        "left_eye_top": [159],
        "left_eye_bottom": [145],
        "left_eye_left": [33],
        "left_eye_right": [133],
        "right_eye_top": [386],
        "right_eye_bottom": [374],
        "right_eye_left": [362],
        "right_eye_right": [263],
        # Forehead reference
        "forehead": [10],
        # Chin
        "chin": [152],
    }

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        logger.info("FaceDetector initialized with MediaPipe Face Mesh")

    def detect(self, frame: np.ndarray) -> dict | None:
        """
        Detect face and extract 468 landmarks from a BGR frame.
        Returns dict with normalized and pixel landmarks, or None if no face found.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]

        # Convert to numpy arrays
        landmarks_norm = np.array(
            [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
        )
        landmarks_px = landmarks_norm.copy()
        landmarks_px[:, 0] *= w
        landmarks_px[:, 1] *= h

        return {
            "landmarks_norm": landmarks_norm,
            "landmarks_px": landmarks_px,
            "face_landmarks": face_landmarks,
            "frame_shape": (h, w),
        }

    def draw_landmarks(self, frame: np.ndarray, detection: dict) -> np.ndarray:
        """Draw face mesh landmarks on frame for visualization."""
        annotated = frame.copy()
        self.mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=detection["face_landmarks"],
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        return annotated

    def get_landmark_points(self, detection: dict, landmark_key: str) -> np.ndarray:
        """Get pixel coordinates for a named landmark group."""
        indices = self.LANDMARKS[landmark_key]
        return detection["landmarks_px"][indices]

    def close(self):
        self.face_mesh.close()
