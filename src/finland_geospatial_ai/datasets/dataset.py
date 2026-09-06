"""Interview-readable custom PyTorch Dataset for aligned GeoTIFF patches."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

from finland_geospatial_ai.datasets.manifest import DatasetManifest
from finland_geospatial_ai.geospatial.raster import assert_aligned


def apply_dihedral(
    image: torch.Tensor, mask: torch.Tensor, boundary: torch.Tensor, operation: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one of the eight square symmetries identically to all spatial tensors."""
    operation %= 8
    if operation >= 4:
        image = torch.flip(image, dims=(-1,))
        mask = torch.flip(mask, dims=(-1,))
        boundary = torch.flip(boundary, dims=(-1,))
    turns = operation % 4
    if turns:
        image = torch.rot90(image, turns, dims=(-2, -1))
        mask = torch.rot90(mask, turns, dims=(-2, -1))
        boundary = torch.rot90(boundary, turns, dims=(-2, -1))
    return image, mask, boundary


class NLSPatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    def __init__(
        self,
        manifest_path: Path | str,
        split: str,
        *,
        data_root: Path | str | None = None,
        augment: bool = False,
        use_boundary_ignore: bool = True,
        seed: int = 3067,
        brightness: float = 0.0,
        contrast: float = 0.0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = DatasetManifest.model_validate_json(
            self.manifest_path.read_text(encoding="utf-8")
        )
        self.records = [record for record in self.manifest.patches if record.split == split]
        if not self.records:
            raise ValueError(f"No patches found for split={split!r}")
        self.data_root = Path(data_root) if data_root else self.manifest_path.parents[2]
        self.augment = augment
        self.use_boundary_ignore = use_boundary_ignore
        self.ignore_index = self.manifest.ignore_index
        self.seed = seed
        self.brightness = brightness
        self.contrast = contrast
        normalization = self.manifest.normalization
        self.scale = float(normalization["scale"])
        self.mean = torch.tensor(normalization["mean"], dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(normalization["std"], dtype=torch.float32)[:, None, None]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        record = self.records[index]
        image_path = self.data_root / record.image_path
        mask_path = self.data_root / record.mask_path
        boundary_path = self.data_root / record.boundary_path
        if not image_path.is_file() or not mask_path.is_file() or not boundary_path.is_file():
            raise FileNotFoundError(f"Missing patch artifact for {record.id}")
        try:
            with (
                rasterio.open(image_path) as image_source,
                rasterio.open(mask_path) as mask_source,
                rasterio.open(boundary_path) as boundary_source,
            ):
                assert_aligned(
                    (image_source.height, image_source.width),
                    image_source.transform,
                    image_source.crs,
                    (mask_source.height, mask_source.width),
                    mask_source.transform,
                    mask_source.crs,
                )
                assert_aligned(
                    (image_source.height, image_source.width),
                    image_source.transform,
                    image_source.crs,
                    (boundary_source.height, boundary_source.width),
                    boundary_source.transform,
                    boundary_source.crs,
                )
                image = torch.from_numpy(image_source.read().astype(np.float32))
                mask = torch.from_numpy(mask_source.read(1).astype(np.int64))
                boundary = torch.from_numpy(boundary_source.read(1).astype(bool))
        except rasterio.errors.RasterioError as exc:
            raise ValueError(f"Unreadable patch artifact for {record.id}") from exc
        image = image / self.scale
        if self.augment:
            image, mask, boundary = apply_dihedral(image, mask, boundary, random.randrange(8))
            if self.brightness:
                image = image + random.uniform(-self.brightness, self.brightness)
            if self.contrast:
                factor = 1.0 + random.uniform(-self.contrast, self.contrast)
                channel_mean = image.mean(dim=(-2, -1), keepdim=True)
                image = (image - channel_mean) * factor + channel_mean
            image = image.clamp(0, 1)
        if self.use_boundary_ignore:
            mask = mask.clone()
            mask[boundary] = self.ignore_index
        image = (image - self.mean) / self.std
        metadata = {
            "id": record.id,
            "split": record.split,
            "source_map_sheet": record.source_map_sheet,
            "bounds": record.bounds,
            "boundary": boundary,
        }
        return image.contiguous(), mask.contiguous(), metadata


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(
    dataset: NLSPatchDataset,
    *,
    batch_size: int,
    training: bool,
    num_workers: int,
    seed: int,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=training and len(dataset) >= batch_size,
    )
