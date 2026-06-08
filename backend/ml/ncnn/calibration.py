"""Temperature scaling for the N-CNN logits.

Raw softmax outputs from a CNN are typically overconfident. Temperature scaling
is a single-parameter post-hoc calibration: divide the logits by a learned
scalar T, then softmax. At T = 1 it is a no-op. T > 1 flattens the distribution
(less confident), T < 1 sharpens it.

The scalar T is fit on a held-out validation fold by minimizing NLL on the
saved logits. This file ships the module and a documented fit method, but the
fit is not called anywhere yet. We do not have data, and we do not invent it.

Research prototype, not a medical device. No metric returned here should be
interpreted as model performance.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    """Single learned scalar applied to logits before softmax.

    Use:
        scaler = TemperatureScaler()
        calibrated_logits = scaler(raw_logits)
        probs = torch.softmax(calibrated_logits, dim=1)
    """

    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        if initial_temperature <= 0:
            raise ValueError("temperature must be positive")
        # Stored as a learnable parameter so it survives load_state_dict and
        # so a future fit step can optimise it with an LBFGS pass.
        self.temperature = nn.Parameter(torch.tensor(float(initial_temperature)))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Guard against a parameter update driving T to zero or negative.
        # Clamping keeps the calibration well defined without blocking gradients
        # in the typical positive regime.
        safe_t = self.temperature.clamp(min=1e-3)
        return logits / safe_t

    @torch.no_grad()
    def set_temperature(self, value: float) -> None:
        if value <= 0:
            raise ValueError("temperature must be positive")
        self.temperature.fill_(float(value))

    def fit(
        self,
        validation_logits: torch.Tensor,
        validation_labels: torch.Tensor,
        max_iter: int = 50,
    ) -> float:
        """Fit T by minimizing NLL on a held-out subject-wise validation fold.

        Intentionally not wired into the pipeline yet. We have no data, so any
        fit done before subject-wise CV would be meaningless or worse.

        TODO: call this only on logits collected from a subject-disjoint
        validation fold. A frame-level split here is a bug, same as anywhere
        else in the pipeline.
        """
        validation_logits = validation_logits.detach()
        validation_labels = validation_labels.detach()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(self(validation_logits), validation_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.detach().clamp(min=1e-3))
