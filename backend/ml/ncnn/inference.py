"""Inference helpers for the N-CNN.

Two cadences, kept as separate callables so the FastAPI loop can run the
cheap deterministic path every frame and the K-pass MC dropout path on a
throttle:

  - predict_logits: single deterministic forward pass on a fully eval()
    model. Same input gives the same output.
  - predict_with_mc_dropout: keep only dropout layers in train mode, run K
    passes, return mean and standard deviation of the pain-class probability.

predict_pain wraps these for the public API. It accepts a face crop, returns
{"prob_pain", "uncertainty"}, and can either compute uncertainty fresh or
accept a cached value so the caller can throttle MC dropout without
restructuring this module.

Checkpoint loading: the model uses LazyLinear, so a checkpoint cannot be
loaded until the lazy layer has been materialized. Call load_ncnn_state_dict
rather than nn.Module.load_state_dict directly; it runs one dummy forward
first. LazyLinear will be frozen to a fixed Linear once we lock the input
resolution after data lands.

Research prototype, not a medical device.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ml.ncnn.calibration import TemperatureScaler
from ml.ncnn.model import NCNN
from ml.ncnn.preprocess import face_crop_to_tensor


def predict_logits(model: NCNN, x: torch.Tensor) -> torch.Tensor:
    """Single deterministic forward pass. The model is forced into eval mode
    so dropout is off and weights produce the same output for the same input.
    Returns raw logits, shape (N, num_classes)."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return model(x)
    finally:
        if was_training:
            model.train()


def _enable_dropout_only(model: nn.Module) -> None:
    """Flip every nn.Dropout layer into train mode while leaving conv and
    linear weights in eval. This is the MC dropout convention: weights are
    fixed, sampling happens only through dropout masks."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def predict_with_mc_dropout(
    model: NCNN,
    x: torch.Tensor,
    n_passes: int,
    pain_class_index: int = 1,
    temperature_scaler: Optional[TemperatureScaler] = None,
) -> tuple[float, float]:
    """Run K stochastic forward passes with only dropout in train mode.

    Returns (mean_prob_pain, std_prob_pain). Batch dimension must be 1.
    Pure NumPy returns rather than tensors so callers do not need torch.
    """
    if n_passes < 1:
        raise ValueError("n_passes must be >= 1")
    if x.shape[0] != 1:
        raise ValueError("MC dropout helper expects batch size 1")

    was_training = model.training
    _enable_dropout_only(model)
    try:
        probs = []
        with torch.no_grad():
            for _ in range(n_passes):
                logits = model(x)
                if temperature_scaler is not None:
                    logits = temperature_scaler(logits)
                p = torch.softmax(logits, dim=1)[0, pain_class_index]
                probs.append(float(p.item()))
    finally:
        if was_training:
            model.train()
        else:
            model.eval()

    arr = np.asarray(probs, dtype=np.float64)
    mean = float(arr.mean())
    # ddof=0 keeps the value well defined for n_passes = 1 (returns 0.0).
    std = float(arr.std(ddof=0))
    return mean, std


def predict_pain(
    face_crop_rgb: np.ndarray,
    model: NCNN,
    input_size: int,
    norm_mean: Sequence[float],
    norm_std: Sequence[float],
    pain_class_index: int = 1,
    temperature_scaler: Optional[TemperatureScaler] = None,
    cached_uncertainty: Optional[float] = None,
    mc_passes: Optional[int] = None,
) -> dict:
    """Public per-frame API.

    Always runs one deterministic pass for prob_pain. Uncertainty is either
    taken from cached_uncertainty (so the caller can throttle MC dropout)
    or computed fresh by running mc_passes MC-dropout passes. Exactly one
    of cached_uncertainty or mc_passes must be provided.

    Returns:
        {"prob_pain": float in [0, 1], "uncertainty": float >= 0}
    """
    if (cached_uncertainty is None) == (mc_passes is None):
        raise ValueError(
            "predict_pain requires exactly one of cached_uncertainty or mc_passes"
        )

    tensor = face_crop_to_tensor(face_crop_rgb, input_size, norm_mean, norm_std)

    logits = predict_logits(model, tensor)
    if temperature_scaler is not None:
        with torch.no_grad():
            logits = temperature_scaler(logits)
    prob_pain = float(torch.softmax(logits, dim=1)[0, pain_class_index].item())

    if cached_uncertainty is not None:
        uncertainty = float(cached_uncertainty)
    else:
        _, uncertainty = predict_with_mc_dropout(
            model,
            tensor,
            n_passes=int(mc_passes),  # type: ignore[arg-type]
            pain_class_index=pain_class_index,
            temperature_scaler=temperature_scaler,
        )

    return {"prob_pain": prob_pain, "uncertainty": uncertainty}


def load_ncnn_state_dict(model: NCNN, state_dict: dict, input_size: int) -> None:
    """Load weights into an NCNN that uses LazyLinear.

    LazyLinear parameters are uninitialized until the first forward pass.
    PyTorch 2.2 happens to auto-materialize them from a state dict that
    carries concrete shapes, but that is a version-dependent convenience.
    This helper makes the materialization explicit: run one dummy forward at
    the expected input size first, then load_state_dict. The contract is the
    same regardless of PyTorch version.

    Once the architecture is frozen post-data, swap LazyLinear for a fixed
    nn.Linear with the explicit feature count and this helper can go.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, input_size, input_size)
        _ = model(dummy)
    model.load_state_dict(state_dict)
    if was_training:
        model.train()
