from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from finland_geoai.models.factory import create_model


@pytest.fixture
def class_map() -> dict[str, dict[str, object]]:
    names = ["forest", "shrub_grass", "cropland", "built_up", "other_natural", "water"]
    return {
        str(index): {"name": name, "color": [index * 30, 100, 200 - index * 20]}
        for index, name in enumerate(names)
    }


@pytest.fixture
def checkpoint_path(tmp_path: Path, class_map: dict[str, dict[str, object]]) -> Path:
    model = create_model("unet", 4, 6, 8)
    path = tmp_path / "model.pt"
    torch.save(
        {
            "artifact_schema": "finland-geoai-model/v1",
            "state_dict": model.state_dict(),
            "model_config": {
                "model": "unet",
                "in_channels": 4,
                "num_classes": 6,
                "base_channels": 8,
            },
            "band_indices": [0, 1, 2, 3],
            "normalization": {
                "scale": 10000.0,
                "mean": [0.1, 0.1, 0.1, 0.1],
                "std": [0.05, 0.05, 0.05, 0.05],
            },
            "classes": class_map,
            "dataset_version": "test/v1",
            "temperature": 1.0,
        },
        path,
    )
    return path


def geotiff_bytes(
    bands: int = 4,
    width: int = 64,
    height: int = 64,
    crs: str | None = "EPSG:3067",
    resolution: float = 10.0,
    nodata: int | None = None,
) -> bytes:
    data = np.full((bands, height, width), 1000, dtype=np.uint16)
    if nodata is not None:
        data[:] = nodata
    with MemoryFile() as memory_file:
        with memory_file.open(
            driver="GTiff",
            width=width,
            height=height,
            count=bands,
            dtype="uint16",
            crs=crs,
            transform=from_origin(385000, 6680000, resolution, resolution),
            nodata=nodata,
        ) as destination:
            destination.write(data)
        return memory_file.read()


@pytest.fixture
def tiny_dataset(tmp_path: Path, class_map: dict[str, dict[str, object]]) -> Path:
    data_root = tmp_path / "data"
    patches = []
    for split, x_offset in (("train", 0), ("val", 1000), ("test", 2000)):
        relative = Path(split) / "aoi" / f"{split}.npz"
        path = data_root / relative
        path.parent.mkdir(parents=True)
        image = np.arange(4 * 8 * 8, dtype=np.uint16).reshape(4, 8, 8)
        mask = np.arange(8 * 8, dtype=np.uint8).reshape(8, 8) % 6
        np.savez_compressed(path, image=image, mask=mask, valid=np.ones((8, 8), dtype=bool))
        patches.append(
            {
                "id": split,
                "path": str(relative).replace("\\", "/"),
                "aoi_id": f"{split}-aoi",
                "split": split,
                "bounds": [x_offset, 0, x_offset + 80, 80],
                "sha256": "unused",
            }
        )
    manifest = {
        "schema_version": "test/v1",
        "split_version": "test-split/v1",
        "bands": ["blue", "green", "red", "nir"],
        "patch_size": 8,
        "ignore_index": 255,
        "normalization": {
            "scale": 10000.0,
            "mean": [0, 0, 0, 0],
            "std": [1, 1, 1, 1],
        },
        "class_mapping": {"classes": class_map},
        "class_counts_by_split": {
            split: {str(i): 1 for i in range(6)} for split in ("train", "val", "test")
        },
        "patches": patches,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path
