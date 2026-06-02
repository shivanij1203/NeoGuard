"""Tests for temperature scaling.

Pipeline tests only. No metric here should be interpreted as model performance.
"""
from __future__ import annotations

import pytest
import torch

from ml.ncnn.calibration import TemperatureScaler


@pytest.mark.unit
def test_temperature_one_is_identity():
    scaler = TemperatureScaler(initial_temperature=1.0)
    logits = torch.tensor([[2.0, -1.0], [0.5, 0.7]])
    out = scaler(logits)
    assert torch.allclose(out, logits)


@pytest.mark.unit
def test_temperature_above_one_reduces_confidence():
    """Higher T flattens the softmax. The max probability should drop."""
    logits = torch.tensor([[3.0, -1.0]])
    cold = TemperatureScaler(initial_temperature=1.0)
    hot = TemperatureScaler(initial_temperature=4.0)

    p_cold = torch.softmax(cold(logits), dim=1).max().item()
    p_hot = torch.softmax(hot(logits), dim=1).max().item()
    assert p_hot < p_cold


@pytest.mark.unit
def test_temperature_below_one_sharpens_confidence():
    logits = torch.tensor([[1.5, 0.0]])
    base = TemperatureScaler(initial_temperature=1.0)
    sharp = TemperatureScaler(initial_temperature=0.25)

    p_base = torch.softmax(base(logits), dim=1).max().item()
    p_sharp = torch.softmax(sharp(logits), dim=1).max().item()
    assert p_sharp > p_base


@pytest.mark.unit
def test_set_temperature_updates_parameter():
    scaler = TemperatureScaler()
    scaler.set_temperature(2.5)
    assert pytest.approx(float(scaler.temperature.item()), rel=1e-6) == 2.5


@pytest.mark.unit
def test_invalid_temperature_is_rejected():
    with pytest.raises(ValueError):
        TemperatureScaler(initial_temperature=0.0)
    scaler = TemperatureScaler()
    with pytest.raises(ValueError):
        scaler.set_temperature(-1.0)
