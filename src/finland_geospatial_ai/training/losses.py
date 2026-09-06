"""Losses for imbalanced multi-class land-cover segmentation."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    classes = logits.shape[1]
    valid = target != ignore_index
    safe_target = target.masked_fill(~valid, 0)
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(safe_target, classes).permute(0, 3, 1, 2).float()
    valid_channels = valid[:, None]
    probabilities = probabilities * valid_channels
    one_hot = one_hot * valid_channels
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    present = one_hot.sum(dim=(0, 2, 3)) > 0
    dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    return 1 - dice[present].mean()


def segmentation_loss(
    logits: torch.Tensor, target: torch.Tensor, ignore_index: int, dice_weight: float
) -> torch.Tensor:
    cross_entropy = F.cross_entropy(logits, target, ignore_index=ignore_index)
    return (1 - dice_weight) * cross_entropy + dice_weight * soft_dice_loss(
        logits, target, ignore_index
    )
