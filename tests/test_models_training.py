from __future__ import annotations

import pytest
import torch

from finland_geoai.models.factory import create_model
from finland_geoai.training.losses import WeightedCrossEntropy, WeightedCrossEntropyDice


@pytest.mark.parametrize("name,base", [("unet", 8), ("tiny_deeplab", 8)])
def test_forward_backward_smoke(name: str, base: int) -> None:
    torch.manual_seed(42)
    model = create_model(name, 4, 6, base)
    inputs = torch.randn(2, 4, 32, 32)
    targets = torch.randint(0, 6, (2, 32, 32))
    logits = model(inputs)
    loss = WeightedCrossEntropyDice(torch.ones(6))(logits, targets)
    loss.backward()
    assert logits.shape == (2, 6, 32, 32)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_weighted_cross_entropy_handles_all_one_class_patch() -> None:
    logits = torch.randn(1, 6, 8, 8, requires_grad=True)
    target = torch.zeros((1, 8, 8), dtype=torch.long)
    loss = WeightedCrossEntropy(torch.ones(6))(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
