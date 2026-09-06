from __future__ import annotations

import numpy as np
import pytest
import torch

from finland_geospatial_ai.evaluation.metrics import (
    SegmentationMetrics,
    boundary_f1,
    calibration_bins,
)
from finland_geospatial_ai.models import create_model
from finland_geospatial_ai.training.losses import segmentation_loss


def test_unet_preserves_spatial_shape() -> None:
    model = create_model(
        {"architecture": "unet", "in_channels": 3, "num_classes": 5, "base_channels": 4}
    )
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 5, 32, 32)


def test_segformer_adapter_preserves_spatial_shape_without_network() -> None:
    model = create_model(
        {
            "architecture": "segformer_b0",
            "backbone": "nvidia/mit-b0",
            "num_classes": 3,
            "pretrained": False,
        }
    )
    model.eval()
    with torch.inference_mode():
        assert model(torch.randn(1, 3, 32, 32)).shape == (1, 3, 32, 32)


def test_loss_backpropagates_with_ignore_pixels() -> None:
    logits = torch.randn(1, 3, 8, 8, requires_grad=True)
    target = torch.randint(0, 3, (1, 8, 8))
    target[:, 0] = 255
    loss = segmentation_loss(logits, target, 255, 0.5)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_known_confusion_and_iou() -> None:
    metrics = SegmentationMetrics.empty(2)
    metrics.update(np.array([[0, 1], [1, 0]]), np.array([[0, 0], [1, 255]]), 255)
    result = metrics.compute()
    assert result["confusion_matrix"] == [[1, 1], [0, 1]]
    assert result["mean_iou"] == pytest.approx(0.5)


def test_boundary_f1_is_one_for_exact_match() -> None:
    values = np.zeros((16, 16), dtype=np.uint8)
    values[:, 8:] = 1
    assert boundary_f1(values, values, 255) == pytest.approx(1.0)


def test_perfect_predictions_are_well_calibrated() -> None:
    probabilities = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    target = torch.tensor([[[0, 1]]])
    assert calibration_bins(probabilities, target, 255)[
        "expected_calibration_error"
    ] == pytest.approx(0.0)
