"""NCNNFacialPainClassifier: facial pain pipeline backed by the N-CNN.

Routes a BGR frame through MediaPipe (presence and occlusion gate), the
landmark-bbox crop, and the N-CNN. Returns a contract dict that distinguishes
"no facial signal" from "confident no pain".

Contract dict:
    {
        "face_detected": bool,
        "prob_pain": float | None,    # None when face_detected=False
        "uncertainty": float | None,  # None when not computed this call
        "facial_score": float | None, # None when face_detected=False; else
                                      # prob_to_score(prob_pain) in [0, 10]
        "landmarks": ndarray | None,
        "frame_to_score_ms": float,
    }

Absent is not zero. A None facial_score means the CNN did not score this
frame; Phase 5 fusion treats that differently from prob_pain near zero.

The model ships with random weights until a subject-wise CV training run
produces a checkpoint. This is intentional for the scaffolding phase and is
logged at init so it is loud. No metric returned here should be interpreted
as model performance.

Research prototype, not a medical device.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import torch

from config import settings
from ml.face_detector import FaceDetector
from ml.ncnn.calibration import TemperatureScaler
from ml.ncnn.inference import (
    predict_logits,
    predict_with_mc_dropout,
)
from ml.ncnn.model import NCNN
from ml.ncnn.preprocess import (
    crop_face_from_landmarks,
    face_crop_to_tensor,
)
from ml.ncnn.scale import prob_to_score

logger = logging.getLogger(__name__)


class NCNNFacialPainClassifier:
    """Facial pain pipeline backed by the N-CNN. Stateless across frames.

    Cadence state (frame count, cached uncertainty) lives outside this class
    in a per-connection FacialStreamState. Call predict(..., compute_uncertainty)
    with the cadence decision made by the caller.
    """

    def __init__(self) -> None:
        self.face_detector = FaceDetector(
            min_detection_confidence=settings.min_detection_confidence,
            min_tracking_confidence=settings.min_tracking_confidence,
        )
        self.model = NCNN(
            dropout=settings.ncnn_dropout,
            dense_width=settings.ncnn_dense_width,
        )
        # Materialize LazyLinear so the inference path is deterministic from
        # the first real frame.
        with torch.no_grad():
            self.model(torch.zeros(1, 3, settings.ncnn_input_size, settings.ncnn_input_size))
        self.model.eval()
        # Default scaler at T=1 is a no-op; a real T is fit on a held-out
        # subject-disjoint fold once data lands. See ml/ncnn/calibration.py.
        self.temperature_scaler = TemperatureScaler(initial_temperature=1.0)
        logger.warning(
            "NCNNFacialPainClassifier initialized with RANDOM weights. "
            "Predictions are not meaningful until a checkpoint is loaded. "
            "Research prototype, not a medical device."
        )

    def predict(
        self,
        frame_bgr: np.ndarray,
        compute_uncertainty: bool,
    ) -> dict:
        """Score one BGR frame.

        Always runs one deterministic forward pass for prob_pain. Runs the
        K-pass MC dropout for uncertainty only when compute_uncertainty is
        True; otherwise returns uncertainty=None and the caller is expected
        to keep its cached value.
        """
        t0 = time.perf_counter()

        detection = self.face_detector.detect(frame_bgr)
        if detection is None:
            return self._absent_result(start_time=t0)

        crop_rgb = crop_face_from_landmarks(
            frame_bgr,
            detection["landmarks_px"],
            margin_ratio=settings.ncnn_crop_margin_ratio,
            min_face_size_px=settings.ncnn_min_face_size_px,
        )
        if crop_rgb is None:
            # Gate rejected the crop (too small, partial, occluded by
            # bbox-overflow). Treat as absent facial signal.
            return self._absent_result(start_time=t0, landmarks=detection["landmarks_px"])

        tensor = face_crop_to_tensor(
            crop_rgb,
            input_size=settings.ncnn_input_size,
            mean=settings.ncnn_norm_mean,
            std=settings.ncnn_norm_std,
        )

        logits = predict_logits(self.model, tensor)
        with torch.no_grad():
            logits = self.temperature_scaler(logits)
        prob_pain = float(
            torch.softmax(logits, dim=1)[0, settings.ncnn_pain_class_index].item()
        )

        uncertainty: Optional[float] = None
        if compute_uncertainty:
            _, uncertainty = predict_with_mc_dropout(
                self.model,
                tensor,
                n_passes=settings.ncnn_mc_passes,
                pain_class_index=settings.ncnn_pain_class_index,
                temperature_scaler=self.temperature_scaler,
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        facial_score = prob_to_score(prob_pain)
        logger.info(
            "frame_to_score_ms=%.1f face_detected=True prob_pain=%.4f "
            "uncertainty=%s mc_refresh=%s",
            latency_ms,
            prob_pain,
            f"{uncertainty:.4f}" if uncertainty is not None else "cached",
            compute_uncertainty,
        )
        return {
            "face_detected": True,
            "prob_pain": prob_pain,
            "uncertainty": uncertainty,
            "facial_score": facial_score,
            "landmarks": detection["landmarks_px"],
            "frame_to_score_ms": round(latency_ms, 2),
        }

    def _absent_result(
        self,
        start_time: float,
        landmarks: Optional[np.ndarray] = None,
    ) -> dict:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "frame_to_score_ms=%.1f face_detected=False prob_pain=None",
            latency_ms,
        )
        return {
            "face_detected": False,
            "prob_pain": None,
            "uncertainty": None,
            "facial_score": None,
            "landmarks": landmarks,
            "frame_to_score_ms": round(latency_ms, 2),
        }

    def close(self) -> None:
        self.face_detector.close()
