from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from safetensors.torch import save_file

from finland_geoai.models.factory import create_model as create_v1_model
from finland_geospatial_ai.models import create_model


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_raster(
    path: Path, data: np.ndarray, transform: object, nodata: int | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = data.shape[0] if data.ndim == 3 else 1
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[-1],
        height=data.shape[-2],
        count=count,
        dtype=data.dtype,
        crs="EPSG:3067",
        transform=transform,
        nodata=nodata,
    ) as destination:
        destination.write(data if data.ndim == 3 else data[None])


@pytest.fixture
def tiny_manifest(tmp_path: Path) -> Path:
    records = []
    classes = [
        {
            "id": 0,
            "name": "other_land",
            "color": [1, 2, 3],
            "source_layers": [],
            "rule": "residual",
        },
        {
            "id": 1,
            "name": "water",
            "color": [4, 5, 6],
            "source_layers": ["lake_part"],
            "rule": "water",
        },
    ]
    for split_index, split in enumerate(("train", "val", "test")):
        transform = from_origin(1000 + split_index * 1000, 7000000, 0.5, 0.5)
        image = np.full((3, 32, 32), 100 + split_index, dtype=np.uint8)
        mask = np.indices((32, 32)).sum(axis=0).astype(np.uint8) % 2
        boundary = np.zeros((32, 32), dtype=np.uint8)
        image_path = tmp_path / "data" / "processed" / split / "image.tif"
        mask_path = tmp_path / "data" / "processed" / split / "mask.tif"
        boundary_path = tmp_path / "data" / "processed" / split / "boundary.tif"
        write_raster(image_path, image, transform)
        write_raster(mask_path, mask, transform, 255)
        write_raster(boundary_path, boundary, transform)
        bounds = rasterio.transform.array_bounds(32, 32, transform)
        records.append(
            {
                "id": split,
                "split": split,
                "source_map_sheet": f"sheet-{split}",
                "source_orthophoto": f"{split}.tif",
                "orthophoto_year": 2025,
                "image_path": image_path.relative_to(tmp_path).as_posix(),
                "mask_path": mask_path.relative_to(tmp_path).as_posix(),
                "boundary_path": boundary_path.relative_to(tmp_path).as_posix(),
                "bounds": list(bounds),
                "transform": list(transform)[:6],
                "width": 32,
                "height": 32,
                "valid_fraction": 1.0,
                "class_counts": {"0": 512, "1": 512},
                "image_sha256": digest(image_path),
                "mask_sha256": digest(mask_path),
                "boundary_sha256": digest(boundary_path),
            }
        )
    manifest = {
        "schema_version": "finland-geospatial-ai-dataset/v2",
        "dataset_version": "test-v1",
        "split_version": "test-spatial-v1",
        "class_map_version": "test-classes-v1",
        "generated_at": "2026-09-01T00:00:00+00:00",
        "generation_commit": "test",
        "config_sha256": "0" * 64,
        "crs": "EPSG:3067",
        "aoi": {
            "type": "Polygon",
            "coordinates": [
                [[900, 6999900], [4000, 6999900], [4000, 7000100], [900, 7000100], [900, 6999900]]
            ],
        },
        "native_pixel_size_m": 0.5,
        "bands": ["red", "green", "blue"],
        "patch_size_px": 32,
        "patch_ground_size_m": 16.0,
        "ignore_index": 255,
        "boundary_ignore_width_m": 1.5,
        "sources": {"licence": "CC BY 4.0"},
        "classes": classes,
        "precedence": ["other_land", "water"],
        "normalization": {"scale": 255.0, "mean": [0.4, 0.4, 0.4], "std": [0.2, 0.2, 0.2]},
        "patch_counts": {"train": 1, "val": 1, "test": 1},
        "class_counts_by_split": {
            split: {"0": 512, "1": 512} for split in ("train", "val", "test")
        },
        "ignored_pixel_fraction_by_split": {split: 0.0 for split in ("train", "val", "test")},
        "patches": records,
    }
    manifest_path = tmp_path / "data" / "manifests" / "test-v1.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    split_path = tmp_path / "data" / "splits" / "test-spatial-v1.geojson"
    split_path.parent.mkdir(parents=True)
    features = []
    for record in records:
        left, bottom, right, top = record["bounds"]
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "split": record["split"],
                    "map_sheet": record["source_map_sheet"],
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [left, bottom],
                            [right, bottom],
                            [right, top],
                            [left, top],
                            [left, bottom],
                        ]
                    ],
                },
            }
        )
    split_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": features,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


@pytest.fixture
def tiny_model(tmp_path: Path) -> Path:
    model_config = {"architecture": "unet", "in_channels": 3, "num_classes": 2, "base_channels": 4}
    model = create_model(model_config)
    weights = tmp_path / "tiny.safetensors"
    save_file(model.state_dict(), weights)
    metadata = {
        "schema_version": "finland-geospatial-ai-model/v2",
        "experiment_id": "tiny",
        "model": model_config,
        "dataset_version": "test-v1",
        "split_version": "test-spatial-v1",
        "class_map_version": "test-classes-v1",
        "normalization": {"scale": 255.0, "mean": [0.4, 0.4, 0.4], "std": [0.2, 0.2, 0.2]},
        "classes": [{"id": 0, "name": "other_land"}, {"id": 1, "name": "water"}],
        "ignore_index": 255,
        "native_pixel_size_m": 0.5,
        "weights": weights.name,
    }
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


@pytest.fixture
def class_map() -> dict[str, dict[str, object]]:
    names = ["forest", "shrub_grass", "cropland", "built_up", "other_natural", "water"]
    return {
        str(index): {"name": name, "color": [index * 30, 100, 200 - index * 20]}
        for index, name in enumerate(names)
    }


@pytest.fixture
def checkpoint_path(tmp_path: Path, class_map: dict[str, dict[str, object]]) -> Path:
    model = create_v1_model("unet", 4, 6, 8)
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
                "path": relative.as_posix(),
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
