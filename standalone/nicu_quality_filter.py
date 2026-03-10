"""
NICU Image Quality Filter - Standalone Module
==============================================

A lightweight quality pre-screening tool for NICU neonatal pain research.
Addresses challenges from published research:
1. Dark images (pixel intensity ≤25) are unusable
2. Occlusion from medical equipment blocks faces

Usage:
    from nicu_quality_filter import QualityFilter

    filter = QualityFilter()

    # Single image
    result = filter.check(image)
    if result['usable']:
        # proceed with annotation/training

    # Video - get only usable frames
    usable_frames = filter.filter_video("video.mp4")

Requirements:
    pip install opencv-python numpy mediapipe

References:
    - "Accurate Neonatal Face Detection for Improved Pain Classification
       in the Challenging NICU Setting" (IEEE Access, 2024)

Author: [Your Name] - USF RPAL Collaboration
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Generator
from dataclasses import dataclass


@dataclass
class QualityResult:
    """Simple result container"""
    usable: bool
    usability: str  # "usable", "marginal", "unusable"
    score: float  # 0-100
    brightness: float
    is_too_dark: bool
    face_detected: bool
    face_status: str  # "detected", "uncertain", "not_detected"
    issues: List[str]


class QualityFilter:
    """
    Lightweight quality filter for NICU images.

    Designed to integrate easily into existing Python ML pipelines.
    No web framework dependencies - just OpenCV and NumPy.
    """

    # Threshold from IEEE Access 2024 paper
    DARK_THRESHOLD = 25  # Pixel intensity ≤25 = unusable

    def __init__(self, use_face_detection: bool = True):
        """
        Args:
            use_face_detection: Enable face detection (requires mediapipe)
        """
        self.use_face_detection = use_face_detection
        self.face_detector = None

        if use_face_detection:
            try:
                import mediapipe as mp
                self.face_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=0.3
                )
            except ImportError:
                print("Warning: mediapipe not installed. Face detection disabled.")
                self.use_face_detection = False

    def check(self, image: np.ndarray) -> QualityResult:
        """
        Check if a single image/frame is usable.

        Args:
            image: BGR image as numpy array (OpenCV format)

        Returns:
            QualityResult with usability assessment
        """
        issues = []

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 1. Brightness check (from research paper)
        brightness = np.mean(gray)
        is_too_dark = brightness <= self.DARK_THRESHOLD

        if is_too_dark:
            issues.append(f"Too dark (brightness: {brightness:.1f}, threshold: {self.DARK_THRESHOLD})")

        # 2. Face detection (with fallback to Haar cascade)
        face_detected = True  # Default if detection disabled
        face_confidence = 1.0
        face_status = "detected"  # detected, uncertain, not_detected

        if self.use_face_detection:
            face_detected = False
            face_confidence = 0.0

            # Try MediaPipe first
            if self.face_detector:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_detector.process(rgb)

                if results.detections:
                    face_confidence = results.detections[0].score[0]
                    if face_confidence >= 0.5:
                        face_detected = True
                        face_status = "detected"
                    elif face_confidence >= 0.2:
                        face_detected = True
                        face_status = "uncertain"

            # Fallback to Haar cascade if MediaPipe didn't detect
            if not face_detected:
                haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
                if len(faces) > 0:
                    face_detected = True
                    face_confidence = 0.35
                    face_status = "uncertain"

            if not face_detected:
                face_status = "not_detected"
                issues.append("No face detected (possibly occluded)")

        # Calculate score
        score = 100.0
        if is_too_dark:
            score -= 50
        if face_status == "not_detected":
            score -= 40
        elif face_status == "uncertain":
            score -= 15  # Less harsh for uncertain

        score = max(0, min(100, score))

        # Determine usability
        if is_too_dark or face_status == "not_detected":
            usability = "unusable"
            usable = False
        elif face_status == "uncertain":
            usability = "marginal"
            usable = True  # Still usable, just needs review
        else:
            usability = "usable"
            usable = True

        return QualityResult(
            usable=usable,
            usability=usability,
            score=score,
            brightness=brightness,
            is_too_dark=is_too_dark,
            face_detected=face_detected,
            face_status=face_status,
            issues=issues
        )

    def check_batch(self, images: List[np.ndarray]) -> List[QualityResult]:
        """Check multiple images at once."""
        return [self.check(img) for img in images]

    def filter_video(self, video_path: str, fps: float = 1.0) -> Generator[Tuple[int, np.ndarray, QualityResult], None, None]:
        """
        Filter video frames, yielding only usable ones.

        Args:
            video_path: Path to video file
            fps: Frames to extract per second (default: 1)

        Yields:
            Tuple of (frame_number, frame_image, quality_result)
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps) if fps < video_fps else 1

        frame_num = 0
        extracted = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % frame_interval == 0:
                result = self.check(frame)
                yield (extracted, frame, result)
                extracted += 1

            frame_num += 1

        cap.release()

    def get_usable_frames(self, video_path: str, fps: float = 1.0) -> List[Tuple[int, np.ndarray]]:
        """
        Get only usable frames from a video.

        Args:
            video_path: Path to video file
            fps: Frames to extract per second

        Returns:
            List of (frame_number, frame_image) tuples for usable frames only
        """
        usable = []
        for frame_num, frame, result in self.filter_video(video_path, fps):
            if result.usable:
                usable.append((frame_num, frame))
        return usable

    def analyze_video(self, video_path: str, fps: float = 1.0) -> Dict:
        """
        Analyze entire video and return summary statistics.

        Args:
            video_path: Path to video file
            fps: Frames to extract per second

        Returns:
            Dictionary with summary stats
        """
        results = list(self.filter_video(video_path, fps))

        total = len(results)
        usable_count = sum(1 for _, _, r in results if r.usability == "usable")
        marginal_count = sum(1 for _, _, r in results if r.usability == "marginal")
        unusable_count = sum(1 for _, _, r in results if r.usability == "unusable")
        too_dark = sum(1 for _, _, r in results if r.is_too_dark)
        no_face = sum(1 for _, _, r in results if r.face_status == "not_detected")

        return {
            'total_frames': total,
            'usable': usable_count,
            'marginal': marginal_count,
            'unusable': unusable_count,
            'usable_percentage': round((usable_count + marginal_count) / total * 100, 1) if total > 0 else 0,
            'issues': {
                'too_dark': too_dark,
                'no_face_detected': no_face
            },
            'usable_indices': [i for i, _, r in results if r.usability in ["usable", "marginal"]]
        }


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    import sys

    qf = QualityFilter()

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        print(f"Analyzing: {video_path}")

        summary = qf.analyze_video(video_path)

        print(f"\nResults:")
        print(f"  Total frames: {summary['total_frames']}")
        print(f"  Usable:       {summary['usable']} frames")
        print(f"  Marginal:     {summary['marginal']} frames")
        print(f"  Unusable:     {summary['unusable']} frames")
        print(f"  Annotation-ready: {summary['usable_percentage']}%")
        print(f"\nIssues:")
        print(f"  Too dark:     {summary['issues']['too_dark']}")
        print(f"  No face:      {summary['issues']['no_face_detected']}")
    else:
        print("NICU Quality Filter - Standalone Module")
        print("=" * 40)
        print("\nUsage: python nicu_quality_filter.py <video_path>")
        print("\nOr import in your code:")
        print("  from nicu_quality_filter import QualityFilter")
        print("  qf = QualityFilter()")
        print("  result = qf.check(image)")
        print("  if result.usable:")
        print("      # proceed with annotation")
