from __future__ import annotations

import numpy as np
import pytest
import torch

from finland_geoai.evaluation.metrics import (
    confusion_matrix,
    expected_calibration_error,
    metrics_from_confusion,
)


def test_known_segmentation_metrics_and_ignore() -> None:
    target = torch.tensor([[0, 0], [1, 255]])
    prediction = torch.tensor([[0, 1], [1, 0]])
    matrix = confusion_matrix(prediction, target, num_classes=2)
    assert matrix.tolist() == [[1, 1], [0, 1]]
    metrics = metrics_from_confusion(matrix)
    assert metrics["miou"] == pytest.approx((0.5 + 0.5) / 2)
    assert metrics["pixel_accuracy"] == pytest.approx(2 / 3)


def test_ece_is_zero_for_matching_bin_accuracy() -> None:
    probabilities = np.array([[0.75, 0.25], [0.75, 0.25], [0.75, 0.25], [0.75, 0.25]])
    targets = np.array([0, 0, 0, 1])
    ece, rows = expected_calibration_error(probabilities, targets, bins=4)
    assert ece == pytest.approx(0.0)
    assert rows[0]["accuracy"] == pytest.approx(0.75)
