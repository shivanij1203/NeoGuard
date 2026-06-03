#!/usr/bin/env python3
"""
NeoGuard Live Webcam Demo
=========================
Runs the N-CNN facial pipeline on a webcam feed with a live HUD.

Research prototype, not a medical device. The N-CNN ships with random
weights until a checkpoint is trained under subject-wise CV, so the score
displayed here is not meaningful and must not be interpreted clinically.
This demo is for end-to-end pipeline sanity checks (gate, latency, fusion
shape) only.

Usage:
    cd backend
    source venv/bin/activate
    python demo_webcam.py
"""

import time

import cv2
import numpy as np

from ml.face_detector import FaceDetector
from ml.ncnn_classifier import NCNNFacialPainClassifier
from ml.scoring import get_pain_label


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def draw_hud(
    frame: np.ndarray,
    face_detected: bool,
    facial_score: float | None,
    prob_pain: float | None,
    uncertainty: float | None,
    latency_ms: float | None,
    fps: float,
) -> np.ndarray:
    """Top-bar HUD with score, prob, uncertainty, latency, FPS."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    if facial_score is None:
        score_text = "Score: --"
        color_bgr = (150, 150, 150)
    else:
        label = get_pain_label(facial_score)
        score_text = f"Score: {facial_score:.1f}/10  ({label['level']})"
        color_bgr = hex_to_bgr(label["color"])

    cv2.putText(frame, score_text, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

    prob_text = "prob_pain: --" if prob_pain is None else f"prob_pain: {prob_pain:.3f}"
    unc_text = "uncertainty: --" if uncertainty is None else f"uncertainty: {uncertainty:.3f}"
    lat_text = "latency: --" if latency_ms is None else f"latency: {latency_ms:.1f} ms"

    cv2.putText(frame, f"{prob_text}    {unc_text}    {lat_text}",
                (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    status_color = (0, 200, 0) if face_detected else (0, 0, 200)
    status_text = "FACE DETECTED" if face_detected else "NO FACE / OCCLUDED"
    cv2.putText(frame, status_text, (15, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.putText(frame, "Research prototype, not a medical device.",
                (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    return frame


def main() -> None:
    print("=" * 60)
    print("  NeoGuard Live Webcam Demo (N-CNN, random weights)")
    print("  Research prototype, not a medical device.")
    print("  Press 'q' to quit")
    print("=" * 60)

    classifier = NCNNFacialPainClassifier()
    face_detector_for_overlay = FaceDetector(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps = 0.0
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = classifier.predict(frame, compute_uncertainty=True)

        # Draw MediaPipe mesh for visualization only. The N-CNN does not use
        # the mesh; this is the same overlay the dashboard renders.
        annotated = frame.copy()
        detection = face_detector_for_overlay.detect(frame)
        if detection is not None:
            annotated = face_detector_for_overlay.draw_landmarks(annotated, detection)

        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = frame_count / elapsed
        if elapsed > 2:
            start_time = time.time()
            frame_count = 0

        annotated = draw_hud(
            annotated,
            face_detected=result.get("face_detected", False),
            facial_score=result.get("facial_score"),
            prob_pain=result.get("prob_pain"),
            uncertainty=result.get("uncertainty"),
            latency_ms=result.get("frame_to_score_ms"),
            fps=fps,
        )

        cv2.imshow("NeoGuard - Live Pain Monitor", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    classifier.close()
    face_detector_for_overlay.close()
    print("\nDemo ended.")


if __name__ == "__main__":
    main()
