"""Shape and structural tests for the N-CNN model.

These are pipeline tests on synthetic tensors. They do not exercise any
trained weights and do not produce any accuracy number. Per project rules,
no metric reported here should be interpreted as model performance.
"""
from __future__ import annotations

import pytest
import torch

from ml.ncnn.model import NCNN


@pytest.mark.unit
def test_forward_returns_logits_for_two_classes():
    model = NCNN().eval()
    x = torch.randn(2, 3, 120, 120)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2), f"expected (2, 2) logits, got {tuple(out.shape)}"
    assert out.dtype == torch.float32


@pytest.mark.unit
def test_each_branch_produces_expected_spatial_size_and_channels():
    """Branches must agree on spatial size at the concat point. 120x120 input
    with one 2x2 pool per branch should land at 60x60."""
    model = NCNN().eval()
    x = torch.randn(1, 3, 120, 120)
    with torch.no_grad():
        branches = model.branch_outputs(x)

    assert branches["generic"].shape == (1, 32, 60, 60)
    assert branches["deep"].shape == (1, 64, 60, 60)
    assert branches["prominent"].shape == (1, 32, 60, 60)


@pytest.mark.unit
def test_all_three_branches_feed_the_merge_via_forward_hook():
    """Patch the merge layer to capture its input, confirm channels = 32+64+32."""
    model = NCNN().eval()
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, inputs, _output):
        captured["merge_in"] = inputs[0]

    handle = model.merge[0].register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(torch.randn(1, 3, 120, 120))
    finally:
        handle.remove()

    assert "merge_in" in captured, "merge conv never received an input"
    merge_in = captured["merge_in"]
    assert merge_in.shape == (1, 128, 60, 60), (
        f"expected concat to be (1, 128, 60, 60), got {tuple(merge_in.shape)}"
    )


@pytest.mark.unit
def test_branches_actually_contribute_to_the_merged_tensor():
    """Zeroing any one branch's output should change the merged tensor.
    Confirms no branch is being silently dropped by the concat."""
    model = NCNN().eval()
    x = torch.randn(1, 3, 120, 120)

    with torch.no_grad():
        l = model.generic(x)
        c = model.deep(x)
        r = model.prominent(x)
        baseline = torch.cat([l, c, r], dim=1)
        no_generic = torch.cat([torch.zeros_like(l), c, r], dim=1)
        no_deep = torch.cat([l, torch.zeros_like(c), r], dim=1)
        no_prominent = torch.cat([l, c, torch.zeros_like(r)], dim=1)

    assert not torch.allclose(baseline, no_generic)
    assert not torch.allclose(baseline, no_deep)
    assert not torch.allclose(baseline, no_prominent)


@pytest.mark.unit
def test_each_branch_has_nonzero_trainable_params():
    model = NCNN()
    for name in ("generic", "deep", "prominent", "merge"):
        params = sum(p.numel() for p in getattr(model, name).parameters() if p.requires_grad)
        assert params > 0, f"branch {name} has no trainable params"


@pytest.mark.unit
def test_lazy_linear_resolves_after_one_forward_pass():
    """LazyLinear should specialize on the first forward. After that, the
    architecture is frozen and a fixed-size Linear can be substituted later."""
    model = NCNN()
    # Before any forward, LazyLinear has no in_features.
    with torch.no_grad():
        _ = model(torch.randn(1, 3, 120, 120))
    # After the forward, fc1 should be a fully initialized Linear with a
    # concrete in_features value.
    assert isinstance(model.fc1, torch.nn.Linear)
    assert getattr(model.fc1, "in_features", None) is not None
    assert model.fc1.in_features > 0
