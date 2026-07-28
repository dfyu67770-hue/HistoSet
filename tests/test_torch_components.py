from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")


def test_tiny_histoset_forward():
    import torch

    from histoset.torch_models import TinyHistoSetUNet

    model = TinyHistoSetUNet(base_channels=4)
    output = model(torch.zeros(1, 3, 32, 32))
    assert output["explanation_logits"].shape == (1, 10, 32, 32)
    assert output["pattern_logits"].shape == (1, 4, 32, 32)


def test_hierarchy_loss_components():
    import torch

    from histoset.torch_losses import hierarchy_consistency_loss, remap_explanation_to_pattern, soft_targets

    counts = torch.ones(1, 10, 8, 8)
    foreground = torch.ones(1, 8, 8, dtype=torch.bool)
    target = soft_targets(counts, foreground)
    assert target.shape == counts.shape
    explanation_logits = torch.zeros(1, 10, 8, 8)
    pattern_probability = remap_explanation_to_pattern(torch.softmax(explanation_logits, dim=1))
    assert pattern_probability.shape == (1, 4, 8, 8)
    pattern_logits = torch.zeros(1, 4, 8, 8)
    assert hierarchy_consistency_loss(explanation_logits, pattern_logits, foreground).item() >= 0
