"""Lightweight DeepLabV3-style network with depthwise encoding and ASPP context."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SeparableBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPP(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        branch_channels = channels // 2
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, branch_channels, 1, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                ),
                *[
                    nn.Sequential(
                        nn.Conv2d(
                            channels,
                            branch_channels,
                            3,
                            padding=dilation,
                            dilation=dilation,
                            bias=False,
                        ),
                        nn.BatchNorm2d(branch_channels),
                        nn.ReLU(inplace=True),
                    )
                    for dilation in (2, 4, 6)
                ],
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(inputs) for branch in self.branches], dim=1))


class TinyDeepLabV3(nn.Module):
    """A small atrous-context model materially different from the U-Net baseline."""

    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 24) -> None:
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.encoder = nn.Sequential(
            SeparableBlock(c, c * 2, stride=2),
            SeparableBlock(c * 2, c * 3, stride=2),
            SeparableBlock(c * 3, c * 4),
            SeparableBlock(c * 4, c * 4),
        )
        self.aspp = ASPP(c * 4)
        self.classifier = nn.Sequential(
            SeparableBlock(c * 4, c * 2),
            nn.Conv2d(c * 2, num_classes, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.aspp(self.encoder(self.stem(inputs)))
        logits = self.classifier(features)
        return F.interpolate(logits, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
