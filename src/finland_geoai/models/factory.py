"""Model construction from versioned experiment configuration."""

from __future__ import annotations

from torch import nn

from finland_geoai.models.deeplab import TinyDeepLabV3
from finland_geoai.models.unet import CompactUNet


def create_model(name: str, in_channels: int, num_classes: int, base_channels: int) -> nn.Module:
    if name == "unet":
        return CompactUNet(in_channels, num_classes, base_channels)
    if name == "tiny_deeplab":
        return TinyDeepLabV3(in_channels, num_classes, base_channels)
    raise ValueError(f"Unknown model architecture: {name}")
