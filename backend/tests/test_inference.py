"""Tests for N-CNN inference helpers: deterministic pass, MC dropout, and
predict_pain. Plus checkpoint loading against a LazyLinear model.

Pipeline tests only on synthetic tensors and crops. No metric here should be
interpreted as model performance.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.ncnn.calibration import TemperatureScaler
from ml.ncnn.inference import (
    load_ncnn_state_dict,
    predict_logits,
    predict_pain,
    predict_with_mc_dropout,
)
from ml.ncnn.model import NCNN


def _make_model(seed: int = 0) -> NCNN:
    torch.manual_seed(seed)
    model = NCNN()
    # Materialize the LazyLinear so weights are deterministic from here on.
    with torch.no_grad():
        _ = model(torch.zeros(1, 3, 120, 120))
    return model


@pytest.mark.unit
def test_predict_logits_is_deterministic_for_same_input():
    """Single pass through eval() model must be repeatable. This locks in
    that only dropout flips to train mode in the MC path; conv and linear
    weights stay fixed."""
    model = _make_model()
    x = torch.randn(1, 3, 120, 120)
    a = predict_logits(model, x)
    b = predict_logits(model, x)
    assert torch.equal(a, b)


@pytest.mark.unit
def test_predict_logits_returns_expected_shape():
    model = _make_model()
    x = torch.randn(4, 3, 120, 120)
    out = predict_logits(model, x)
    assert out.shape == (4, 2)


@pytest.mark.unit
def test_mc_dropout_returns_variance_across_passes():
    """K passes with dropout active should produce some spread on probability
    of the pain class. With dropout=0.5 and a random init, this is reliable
    even on synthetic input."""
    model = _make_model()
    x = torch.randn(1, 3, 120, 120)
    mean, std = predict_with_mc_dropout(model, x, n_passes=20)
    assert 0.0 <= mean <= 1.0
    assert std >= 0.0
    assert std > 0.0, "MC dropout produced zero variance, dropout is not active"


@pytest.mark.unit
def test_mc_dropout_does_not_leave_model_in_train_mode():
    """The helper must restore eval state after running so subsequent
    deterministic passes are not contaminated."""
    model = _make_model()
    model.eval()
    x = torch.randn(1, 3, 120, 120)
    _ = predict_with_mc_dropout(model, x, n_passes=3)
    assert not model.training
    # And every individual dropout layer should also be back in eval.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            assert not module.training


@pytest.mark.unit
def test_mc_dropout_weights_stay_fixed_across_passes():
    """Confirm only dropout is stochastic. Compare a parameter snapshot before
    and after the K-pass run."""
    model = _make_model()
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
    _ = predict_with_mc_dropout(
        model, torch.randn(1, 3, 120, 120), n_passes=10
    )
    for k, v in model.state_dict().items():
        assert torch.equal(snapshot[k], v), f"weight {k} changed during MC dropout"


@pytest.mark.unit
def test_predict_pain_returns_contract_dict_with_mc_passes():
    model = _make_model()
    crop = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    result = predict_pain(
        crop,
        model,
        input_size=120,
        norm_mean=(0.0, 0.0, 0.0),
        norm_std=(1.0, 1.0, 1.0),
        mc_passes=8,
    )
    assert set(result.keys()) == {"prob_pain", "uncertainty"}
    assert 0.0 <= result["prob_pain"] <= 1.0
    assert result["uncertainty"] >= 0.0


@pytest.mark.unit
def test_predict_pain_accepts_cached_uncertainty():
    model = _make_model()
    crop = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    result = predict_pain(
        crop,
        model,
        input_size=120,
        norm_mean=(0.0, 0.0, 0.0),
        norm_std=(1.0, 1.0, 1.0),
        cached_uncertainty=0.123,
    )
    assert result["uncertainty"] == pytest.approx(0.123)
    assert 0.0 <= result["prob_pain"] <= 1.0


@pytest.mark.unit
def test_predict_pain_requires_exactly_one_uncertainty_source():
    model = _make_model()
    crop = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        predict_pain(
            crop, model, input_size=120,
            norm_mean=(0.0, 0.0, 0.0), norm_std=(1.0, 1.0, 1.0),
        )
    with pytest.raises(ValueError):
        predict_pain(
            crop, model, input_size=120,
            norm_mean=(0.0, 0.0, 0.0), norm_std=(1.0, 1.0, 1.0),
            cached_uncertainty=0.1, mc_passes=5,
        )


@pytest.mark.unit
def test_predict_pain_with_temperature_scaler():
    model = _make_model()
    crop = np.random.randint(0, 256, size=(48, 48, 3), dtype=np.uint8)
    scaler = TemperatureScaler(initial_temperature=2.5)
    result = predict_pain(
        crop,
        model,
        input_size=120,
        norm_mean=(0.0, 0.0, 0.0),
        norm_std=(1.0, 1.0, 1.0),
        temperature_scaler=scaler,
        cached_uncertainty=0.0,
    )
    assert 0.0 <= result["prob_pain"] <= 1.0


@pytest.mark.unit
def test_load_state_dict_after_dummy_forward_materializes_lazy_linear():
    """A fresh NCNN with LazyLinear cannot accept a state dict cold. The
    helper runs a dummy forward first; the state dict from a materialized
    twin then loads cleanly."""
    src = _make_model(seed=42)
    state = {k: v.detach().clone() for k, v in src.state_dict().items()}

    dst = NCNN()
    load_ncnn_state_dict(dst, state, input_size=120)

    # After load, both models must produce the same logits on the same input.
    x = torch.randn(1, 3, 120, 120)
    a = predict_logits(src, x)
    b = predict_logits(dst, x)
    assert torch.allclose(a, b, atol=1e-6)


@pytest.mark.unit
def test_load_state_dict_helper_matches_native_cold_load():
    """PyTorch 2.2 auto-materializes LazyLinear from a state dict that already
    carries concrete shapes, so cold load happens to work today. The helper
    makes the materialization explicit, which is the contract we want
    regardless of PyTorch version. Check the two paths agree."""
    src = _make_model(seed=7)
    state = {k: v.detach().clone() for k, v in src.state_dict().items()}

    via_helper = NCNN()
    load_ncnn_state_dict(via_helper, state, input_size=120)

    cold = NCNN()
    cold.load_state_dict(state)

    x = torch.randn(1, 3, 120, 120)
    out_helper = predict_logits(via_helper, x)
    out_cold = predict_logits(cold, x)
    assert torch.allclose(out_helper, out_cold, atol=1e-6)
