"""Transparent pixel, boundary and calibration metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import binary_dilation


@dataclass
class SegmentationMetrics:
    confusion: np.ndarray

    @classmethod
    def empty(cls, num_classes: int) -> SegmentationMetrics:
        return cls(np.zeros((num_classes, num_classes), dtype=np.int64))

    def update(self, prediction: np.ndarray, target: np.ndarray, ignore_index: int) -> None:
        valid = target != ignore_index
        encoded = target[valid] * self.confusion.shape[0] + prediction[valid]
        self.confusion += np.bincount(encoded, minlength=self.confusion.size).reshape(
            self.confusion.shape
        )

    def compute(self) -> dict[str, object]:
        true_positive = np.diag(self.confusion).astype(float)
        support = self.confusion.sum(axis=1).astype(float)
        predicted = self.confusion.sum(axis=0).astype(float)
        union = support + predicted - true_positive
        iou = np.divide(true_positive, union, out=np.zeros_like(union), where=union > 0)
        recall = np.divide(
            true_positive, support, out=np.zeros_like(support), where=support > 0
        )
        precision = np.divide(
            true_positive, predicted, out=np.zeros_like(predicted), where=predicted > 0
        )
        dice_denominator = support + predicted
        dice = np.divide(
            2 * true_positive,
            dice_denominator,
            out=np.zeros_like(dice_denominator),
            where=dice_denominator > 0,
        )
        accuracy = true_positive.sum() / max(self.confusion.sum(), 1)
        return {
            "pixel_accuracy": float(accuracy),
            "mean_iou": float(np.mean(iou)),
            "macro_dice": float(np.mean(dice)),
            "mean_class_recall": float(np.mean(recall)),
            "per_class_iou": iou.tolist(),
            "per_class_dice": dice.tolist(),
            "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(),
            "confusion_matrix": self.confusion.tolist(),
        }


def boundary_f1(
    prediction: np.ndarray, target: np.ndarray, ignore_index: int, tolerance: int = 3
) -> float:
    valid = target != ignore_index

    def edges(values: np.ndarray) -> np.ndarray:
        result = np.zeros_like(valid)
        result[:, 1:] |= valid[:, 1:] & valid[:, :-1] & (values[:, 1:] != values[:, :-1])
        result[1:, :] |= valid[1:, :] & valid[:-1, :] & (values[1:, :] != values[:-1, :])
        return result

    predicted_edges, target_edges = edges(prediction), edges(target)
    if not predicted_edges.any() or not target_edges.any():
        return float(predicted_edges.any() == target_edges.any())
    precision = (
        predicted_edges & binary_dilation(target_edges, iterations=tolerance)
    ).sum() / predicted_edges.sum()
    recall = (
        target_edges & binary_dilation(predicted_edges, iterations=tolerance)
    ).sum() / target_edges.sum()
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def calibration_bins(
    probabilities: torch.Tensor, target: torch.Tensor, ignore_index: int, bins: int = 15
) -> dict[str, object]:
    confidence_tensor, prediction = probabilities.max(dim=1)
    valid = target != ignore_index
    confidence = confidence_tensor[valid].detach().cpu().numpy()
    correct = (prediction[valid] == target[valid]).detach().cpu().numpy()
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (confidence > lower) & (confidence <= upper)
        count = int(selected.sum())
        if count:
            accuracy = float(correct[selected].mean())
            mean_confidence = float(confidence[selected].mean())
            ece += count / max(len(confidence), 1) * abs(accuracy - mean_confidence)
        else:
            accuracy = mean_confidence = 0.0
        rows.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "accuracy": accuracy,
                "confidence": mean_confidence,
            }
        )
    return {"expected_calibration_error": float(ece), "bins": rows}
