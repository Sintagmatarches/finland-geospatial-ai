from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from finland_geoai.data.dataset import GeoPatchDataset, apply_dihedral


def test_geometric_augmentation_keeps_image_and_mask_aligned() -> None:
    mask = torch.arange(16).reshape(4, 4)
    image = mask.float().unsqueeze(0)
    for operation in range(8):
        transformed_image, transformed_mask = apply_dihedral(image, mask, operation)
        assert torch.equal(transformed_image[0], transformed_mask.float())


def test_dataset_and_dataloader_smoke(tiny_dataset: Path) -> None:
    dataset = GeoPatchDataset(tiny_dataset, "train", data_root=tiny_dataset.parent / "data")
    images, masks, metadata = next(iter(DataLoader(dataset, batch_size=1)))
    assert images.shape == (1, 4, 8, 8)
    assert masks.shape == (1, 8, 8)
    assert metadata["split"] == ["train"]


def test_missing_raster_fails_clearly(tiny_dataset: Path) -> None:
    manifest = json.loads(tiny_dataset.read_text())
    manifest["patches"][0]["path"] = "missing.npz"
    tiny_dataset.write_text(json.dumps(manifest))
    dataset = GeoPatchDataset(tiny_dataset, "train", data_root=tiny_dataset.parent / "data")
    with pytest.raises(FileNotFoundError, match="Missing patch"):
        dataset[0]
