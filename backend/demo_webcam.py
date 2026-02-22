#!/usr/bin/env python3
"""
NeoGuard Live Webcam Demo
=========================
Runs MediaPipe Face Mesh on webcam feed with real-time pain score overlay.
Press 'q' to quit.

Usage:
    cd backend
    source venv/bin/activate
    python demo_webcam.py
"""

import cv2
import numpy as np
import time

from ml.face_detector import FaceDetector
from ml.feature_extractor import FeatureExtractor
from ml.pain_classifier import FacialPainClassifier
from ml.scoring import get_pain_label


def draw_pain_overlay(frame, score, features, face_detected, fps):
    """Draw pain score HUD overlay on frame."""
    h, w = frame.shape[:2]
    label = get_pain_label(score)

    # Semi-transparent panel at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Pain score
    color_bgr = hex_to_bgr(label["color"])
    cv2.putText(frame, f"Pain Score: {score:.1f}/10", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)
    cv2.putText(frame, label["level"], (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 1)

    # Score bar
    bar_x, bar_y, bar_w, bar_h = w - 220, 15, 200, 20
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    fill_w = int(bar_w * min(score / 10, 1.0))
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color_bgr, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Face detection status
    status_color = (0, 200, 0) if face_detected else (0, 0, 200)
    status_text = "FACE DETECTED" if face_detected else "NO FACE"
    cv2.putText(frame, status_text, (w - 220, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

    # Feature details panel (bottom-left)
    if face_detected and features:
        overlay2 = frame.copy()
        panel_h = 160
        cv2.rectangle(overlay2, (0, h - panel_h), (260, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

        y_offset = h - panel_h + 20
        feature_display = [
            ("Brow Furrow", features.get("brow_eye_dist_norm", 0), 0.04, 0.08, True),
            ("Eye Squeeze (EAR)", features.get("avg_ear", 0), 0.15, 0.35, True),
            ("Nasolabial", features.get("nose_lip_dist_norm", 0), 0.04, 0.08, True),
            ("Mouth Open (MAR)", features.get("mouth_aspect_ratio", 0), 0.1, 0.6, False),
            ("Eye Asymmetry", features.get("eye_asymmetry", 0), 0.05, 0.3, False),
        ]

        cv2.putText(frame, "AU-Proxy Features:", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += 20

        for name, val, low, high, inverted in feature_display:
            # Determine intensity (green=ok, red=pain)
            if inverted:
                ratio = max(0, min(1, (high - val) / (high - low)))
            else:
                ratio = max(0, min(1, (val - low) / (high - low)))

            r = int(ratio * 255)
            g = int((1 - ratio) * 255)
            bar_color = (0, g, r)

            cv2.putText(frame, f"{name}: {val:.3f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

            # Mini bar
            bx = 180
            bw = 70
            cv2.rectangle(frame, (bx, y_offset - 8), (bx + bw, y_offset), (40, 40, 40), -1)
            fill = int(bw * ratio)
            if fill > 0:
                cv2.rectangle(frame, (bx, y_offset - 8), (bx + fill, y_offset), bar_color, -1)

            y_offset += 22

    return frame


def draw_key_landmarks(frame, landmarks_px):
    """Draw key pain-relevant landmarks highlighted on the face."""
    if landmarks_px is None:
        return frame

    # Key landmark groups with colors
    groups = {
        "brow": ([70, 63, 105, 66, 107, 336, 296, 334, 293, 300], (0, 200, 255)),   # Orange - brow
        "eyes": ([33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380], (255, 200, 0)),  # Cyan - eyes
        "nose": ([1, 6, 0], (0, 255, 200)),         # Teal - nose
        "mouth": ([13, 14, 78, 308], (255, 0, 200)),  # Pink - mouth
    }

    for group_name, (indices, color) in groups.items():
        for idx in indices:
            if idx < len(landmarks_px):
                x, y = int(landmarks_px[idx][0]), int(landmarks_px[idx][1])
                cv2.circle(frame, (x, y), 3, color, -1)

    return frame


def hex_to_bgr(hex_color):
    """Convert hex color string to BGR tuple."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def main():
    print("=" * 50)
    print("  NeoGuard Live Webcam Demo")
    print("  Press 'q' to quit")
    print("=" * 50)

    classifier = FacialPainClassifier()
    print("ML models loaded.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps = 0
    frame_count = 0
    start_time = time.time()

    print("Camera opened. Starting analysis...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run pain analysis
        result, annotated = classifier.predict_with_overlay(frame)

        # Draw key landmarks highlighted
        if result["face_detected"] and result.get("landmarks") is not None:
            annotated = draw_key_landmarks(annotated, result["landmarks"])

        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = frame_count / elapsed

        # Reset FPS counter every 2 seconds
        if elapsed > 2:
            start_time = time.time()
            frame_count = 0

        # Draw HUD overlay
        annotated = draw_pain_overlay(
            annotated,
            result["facial_score"],
            result.get("features", {}),
            result["face_detected"],
            fps,
        )

        # Show frame
        cv2.imshow("NeoGuard - Live Pain Monitor", annotated)

        # Print score periodically
        if frame_count % 30 == 0 and result["face_detected"]:
            label = get_pain_label(result["facial_score"])
            features = result.get("features", {})
            print(
                f"Score: {result['facial_score']:.1f} ({label['level']}) | "
                f"EAR: {features.get('avg_ear', 0):.3f} | "
                f"MAR: {features.get('mouth_aspect_ratio', 0):.3f} | "
                f"Brow: {features.get('brow_eye_dist_norm', 0):.3f} | "
                f"FPS: {fps:.0f}"
            )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    classifier.close()
    print("\nDemo ended.")


if __name__ == "__main__":
    main()
