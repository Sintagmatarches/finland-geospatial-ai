"""Defensible class-imbalance-aware segmentation objectives."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int) -> torch.Tensor:
    valid = targets != ignore_index
    safe_targets = torch.where(valid, targets, 0)
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(safe_targets, logits.shape[1]).permute(0, 3, 1, 2).float()
    valid_channels = valid.unsqueeze(1)
    probabilities = probabilities * valid_channels
    one_hot = one_hot * valid_channels
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    present = denominator > 0
    dice = (2 * intersection[present] + 1e-6) / (denominator[present] + 1e-6)
    return 1 - dice.mean()


class WeightedCrossEntropyDice(nn.Module):
    class_weights: torch.Tensor

    def __init__(self, class_weights: torch.Tensor, ignore_index: int = 255) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(
            logits, targets, weight=self.class_weights, ignore_index=self.ignore_index
        )
        return cross_entropy + soft_dice_loss(logits, targets, self.ignore_index)


class WeightedCrossEntropy(nn.Module):
    class_weights: torch.Tensor

    def __init__(self, class_weights: torch.Tensor, ignore_index: int = 255) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
        )
