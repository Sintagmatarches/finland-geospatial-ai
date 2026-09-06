"""One explicit model factory shared by training and inference."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from finland_geospatial_ai.models.unet import UNet


class SegFormerAdapter(nn.Module):
    def __init__(self, backbone: str, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerModel

        if pretrained:
            config = SegformerConfig.from_pretrained(backbone, num_labels=num_classes)
            self.model = SegformerForSemanticSegmentation(config)
            self.model.segformer = SegformerModel.from_pretrained(backbone)
        else:
            config = SegformerConfig(num_labels=num_classes, num_channels=3)
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=image).logits
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


def create_model(config: dict[str, Any]) -> nn.Module:
    architecture = config["architecture"]
    if architecture == "unet":
        return UNet(
            int(config.get("in_channels", 3)),
            int(config["num_classes"]),
            int(config.get("base_channels", 24)),
        )
    if architecture == "segformer_b0":
        return SegFormerAdapter(
            str(config.get("backbone", "nvidia/mit-b0")),
            int(config["num_classes"]),
            bool(config.get("pretrained", True)),
        )
    raise ValueError(f"Unknown architecture: {architecture}")
