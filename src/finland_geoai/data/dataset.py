"""Explicit PyTorch Dataset and paired image/mask augmentation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def apply_dihedral(
    image: torch.Tensor, mask: torch.Tensor, operation: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one of eight square symmetries identically to image and mask."""
    operation %= 8
    if operation >= 4:
        image = torch.flip(image, dims=(-1,))
        mask = torch.flip(mask, dims=(-1,))
    rotations = operation % 4
    if rotations:
        image = torch.rot90(image, rotations, dims=(-2, -1))
        mask = torch.rot90(mask, rotations, dims=(-2, -1))
    return image, mask


class GeoPatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    def __init__(
        self,
        manifest_path: Path | str,
        split: str,
        augment: bool = False,
        data_root: Path | str = "data/processed/v1",
        band_indices: list[int] | None = None,
        seed: int = 42,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.records = [p for p in self.manifest["patches"] if p["split"] == split]
        if not self.records:
            raise ValueError(f"No patches found for split={split!r}")
        self.data_root = Path(data_root)
        self.band_indices = band_indices or list(range(len(self.manifest["bands"])))
        self.augment = augment
        self.seed = seed
        normalization = self.manifest["normalization"]
        self.scale = float(normalization["scale"])
        self.mean = torch.tensor(normalization["mean"], dtype=torch.float32)[
            self.band_indices, None, None
        ]
        self.std = torch.tensor(normalization["std"], dtype=torch.float32)[
            self.band_indices, None, None
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        record = self.records[index]
        path = self.data_root / record["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing patch: {path}")
        try:
            with np.load(path, allow_pickle=False) as sample:
                image_np = sample["image"][self.band_indices].astype(np.float32)
                mask_np = sample["mask"].astype(np.int64)
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"Invalid patch archive: {path}") from exc
        if image_np.ndim != 3 or mask_np.shape != image_np.shape[-2:]:
            raise ValueError(f"Image/mask shape mismatch in {path}")
        image = (torch.from_numpy(image_np) / self.scale - self.mean) / self.std
        mask = torch.from_numpy(mask_np)
        if self.augment:
            # Each worker receives an independent torch seed; index keeps smoke runs deterministic.
            operation = random.Random(torch.initial_seed() + self.seed + index).randrange(8)
            image, mask = apply_dihedral(image, mask, operation)
        metadata = {"id": record["id"], "aoi_id": record["aoi_id"], "split": record["split"]}
        return image.contiguous(), mask.contiguous(), metadata
