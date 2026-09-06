"""Compact U-Net with no hidden framework abstractions."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 24) -> None:
        super().__init__()
        channels = [base_channels * factor for factor in (1, 2, 4, 8)]
        self.encoders = nn.ModuleList(
            [
                ConvBlock(in_channels, channels[0]),
                ConvBlock(channels[0], channels[1]),
                ConvBlock(channels[1], channels[2]),
                ConvBlock(channels[2], channels[3]),
            ]
        )
        self.pool = nn.MaxPool2d(2)
        self.bridge = ConvBlock(channels[3], channels[3] * 2)
        decoder_inputs = [channels[3] * 2, channels[3], channels[2], channels[1]]
        decoder_outputs = [channels[3], channels[2], channels[1], channels[0]]
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(source, target, 2, stride=2)
                for source, target in zip(decoder_inputs, decoder_outputs, strict=True)
            ]
        )
        self.decoders = nn.ModuleList([ConvBlock(target * 2, target) for target in decoder_outputs])
        self.head = nn.Conv2d(channels[0], num_classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        skips = []
        features = image
        for encoder in self.encoders:
            features = encoder(features)
            skips.append(features)
            features = self.pool(features)
        features = self.bridge(features)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips), strict=True):
            features = upconv(features)
            if features.shape[-2:] != skip.shape[-2:]:
                features = F.interpolate(
                    features, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            features = decoder(torch.cat((features, skip), dim=1))
        return self.head(features)
