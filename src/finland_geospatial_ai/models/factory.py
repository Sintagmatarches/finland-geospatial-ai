"""One explicit model factory shared by training and inference."""

from __future__ import annotations

import re
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
            config = SegformerConfig()
            config.num_labels = num_classes
            config.num_channels = 3
            config._attn_implementation = "eager"
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=image).logits
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


def create_model(config: dict[str, Any], *, for_inference: bool = False) -> nn.Module:
    architecture = config["architecture"]
    if architecture == "unet":
        return UNet(
            int(config.get("in_channels", 3)),
            int(config["num_classes"]),
            int(config.get("base_channels", 24)),
        )
    if architecture == "segformer_b0":
        backbone = str(config.get("backbone", "nvidia/mit-b0"))
        if for_inference and backbone != "nvidia/mit-b0":
            raise ValueError("Offline inference requires the declared SegFormer-B0 architecture")
        return SegFormerAdapter(
            backbone,
            int(config["num_classes"]),
            bool(config.get("pretrained", True)) and not for_inference,
        )
    raise ValueError(f"Unknown architecture: {architecture}")


def load_compatible_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Strictly load the published Transformers 4 SegFormer checkpoint on v5.

    Transformers 5 renamed SegFormer modules. These replacements mirror the
    upstream conversion map and do not alter tensors or tolerate missing keys.
    """
    if any(".segformer.encoder." in key for key in state_dict):
        converted: dict[str, torch.Tensor] = {}
        replacements = (
            (r"encoder\.patch_embeddings\.(\d+)\.", r"stages.\1.patch_embeddings."),
            (r"encoder\.block\.(\d+)\.", r"stages.\1.blocks."),
            (r"encoder\.layer_norm\.(\d+)", r"stages.\1.layer_norm"),
            (r"attention\.self\.query", "attention.q_proj"),
            (r"attention\.self\.key", "attention.k_proj"),
            (r"attention\.self\.value", "attention.v_proj"),
            (r"attention\.self\.sr", "attention.sequence_reduction.sequence_reduction"),
            (r"attention\.self\.layer_norm", "attention.sequence_reduction.layer_norm"),
            (r"attention\.output\.dense", "attention.o_proj"),
            (r"mlp\.dense1", "mlp.fc1"),
            (r"mlp\.dense2", "mlp.fc2"),
            (r"layer_norm_1", "layernorm_before"),
            (r"layer_norm_2", "layernorm_after"),
            (r"decode_head\.linear_c", "decode_head.linear_projections"),
        )
        for key, tensor in state_dict.items():
            for pattern, replacement in replacements:
                key = re.sub(pattern, replacement, key)
            if key in converted:
                raise ValueError(f"Checkpoint conversion produced duplicate key: {key}")
            converted[key] = tensor
        state_dict = converted
    model.load_state_dict(state_dict, strict=True)
