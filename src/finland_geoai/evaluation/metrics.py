"""Single segmentation metric implementation shared by training and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def confusion_matrix(
    prediction: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255
) -> torch.Tensor:
    prediction = prediction.detach().view(-1).to(torch.int64)
    target = target.detach().view(-1).to(torch.int64)
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    encoded = target[valid] * num_classes + prediction[valid]
    return torch.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def metrics_from_confusion(matrix: torch.Tensor) -> dict[str, Any]:
    values = matrix.to(torch.float64)
    true_positive = values.diag()
    support = values.sum(dim=1)
    predicted = values.sum(dim=0)
    union = support + predicted - true_positive
    iou = torch.where(union > 0, true_positive / union, torch.nan)
    precision = torch.where(predicted > 0, true_positive / predicted, torch.nan)
    recall = torch.where(support > 0, true_positive / support, torch.nan)
    dice_denominator = support + predicted
    dice = torch.where(dice_denominator > 0, 2 * true_positive / dice_denominator, torch.nan)
    total = values.sum()
    def finite_list(tensor: torch.Tensor) -> list[float | None]:
        return [float(value) if torch.isfinite(value) else None for value in tensor]

    return {
        "miou": float(torch.nanmean(iou)),
        "macro_dice": float(torch.nanmean(dice)),
        "pixel_accuracy": float(true_positive.sum() / total) if total else float("nan"),
        "per_class_iou": finite_list(iou),
        "per_class_dice": finite_list(dice),
        "per_class_precision": finite_list(precision),
        "per_class_recall": finite_list(recall),
        "support": support.to(torch.int64).tolist(),
        "confusion_matrix": matrix.to(torch.int64).tolist(),
    }


def expected_calibration_error(
    probabilities: np.ndarray, targets: np.ndarray, bins: int = 15
) -> tuple[float, list[dict[str, float]]]:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correctness = prediction == targets
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    rows: list[dict[str, float]] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (confidence > lower) & (confidence <= upper)
        if not selected.any():
            continue
        accuracy = float(correctness[selected].mean())
        mean_confidence = float(confidence[selected].mean())
        fraction = float(selected.mean())
        ece += fraction * abs(accuracy - mean_confidence)
        rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "accuracy": accuracy,
                "confidence": mean_confidence,
                "fraction": fraction,
            }
        )
    return float(ece), rows
