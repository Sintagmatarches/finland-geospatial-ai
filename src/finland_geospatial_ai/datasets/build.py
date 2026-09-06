"""Build a versioned, spatially separated NLS patch dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml
from rasterio.windows import Window
from shapely.geometry import box, mapping

from finland_geospatial_ai.datasets.manifest import DatasetManifest
from finland_geospatial_ai.geospatial.raster import inspect_raster
from finland_geospatial_ai.labels.rasterize import LabelRasterizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "uncommitted"


def _snapshot_date(path: Path, configured: str | None) -> str:
    if configured:
        return configured
    match = re.search(r"(20\d{6})T", path.name)
    if not match:
        raise ValueError("Topographic Database snapshot date is absent from config and filename")
    value = match.group(1)
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def _find_sources(raw_dir: Path, sheets: set[str]) -> tuple[dict[str, Path], Path]:
    rasters = [
        path for path in raw_dir.rglob("*") if path.suffix.lower() in {".jp2", ".tif", ".tiff"}
    ]
    by_sheet: dict[str, Path] = {}
    for sheet in sheets:
        matches = [path for path in rasters if sheet.lower() in path.name.lower()]
        if len(matches) != 1:
            raise ValueError(f"Expected one orthophoto containing {sheet}, found {len(matches)}")
        by_sheet[sheet] = matches[0]
    geopackages = list(raw_dir.rglob("*.gpkg"))
    if len(geopackages) != 1:
        raise ValueError(f"Expected one Topographic Database GeoPackage, found {len(geopackages)}")
    return by_sheet, geopackages[0]


def _candidate_windows(
    source: rasterio.io.DatasetReader, size: int, stride: int, edge_px: int
) -> list[Window]:
    windows: list[Window] = []
    for row in range(edge_px, source.height - edge_px - size + 1, stride):
        for col in range(edge_px, source.width - edge_px - size + 1, stride):
            windows.append(Window(col, row, size, size))
    return windows


def _select_candidates(
    candidates: list[dict[str, Any]],
    count: int,
    num_classes: int,
    rare_fraction: float,
    seed: int,
    balance_sheets: bool = True,
) -> list[dict[str, Any]]:
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} valid patches are available; {count} requested")
    sheets = sorted({item["sheet"] for item in candidates})
    if balance_sheets and len(sheets) > 1:
        quotient, extra_sheets = divmod(count, len(sheets))
        balanced = []
        for index, sheet in enumerate(sheets):
            sheet_count = quotient + int(index < extra_sheets)
            balanced.extend(
                _select_candidates(
                    [item for item in candidates if item["sheet"] == sheet],
                    sheet_count,
                    num_classes,
                    rare_fraction,
                    seed + index,
                    balance_sheets=False,
                )
            )
        return sorted(balanced, key=lambda item: item["id"])
    selected: dict[str, dict[str, Any]] = {}
    pixels = int(candidates[0]["width"] * candidates[0]["height"])
    for class_id in range(1, num_classes):
        ranked = sorted(
            candidates,
            key=lambda item: (item["class_counts"].get(str(class_id), 0), item["id"]),
            reverse=True,
        )
        qualifying = [
            item
            for item in ranked
            if item["class_counts"].get(str(class_id), 0) / pixels >= rare_fraction
        ]
        for item in qualifying[: max(1, count // (num_classes * 4))]:
            selected[item["id"]] = item
    rng = random.Random(seed)
    remaining_candidates = [item for item in candidates if item["id"] not in selected]
    rng.shuffle(remaining_candidates)
    for item in remaining_candidates:
        if len(selected) >= count:
            break
        selected[item["id"]] = item
    return sorted(selected.values(), key=lambda item: item["id"])[:count]


def _write_patch(path: Path, array: np.ndarray, profile: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(array if array.ndim == 3 else array[None])


def build_dataset(config_path: Path, root: Path) -> Path:
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    raw_dir = root / config["paths"]["raw_dir"]
    processed_dir = root / config["paths"]["processed_dir"]
    manifest_path = root / config["paths"]["manifest"]
    split_path = root / config["paths"]["split_manifest"]
    split_sheets = {
        sheet: split
        for split, definition in config["splits"].items()
        for sheet in definition["map_sheets"]
    }
    imagery, gpkg = _find_sources(raw_dir, set(split_sheets))
    raster_metadata = {
        sheet: inspect_raster(
            path, expected_bands=3, expected_dtype="uint8", expected_resolution_m=0.5
        )
        for sheet, path in imagery.items()
    }
    rasterizer = LabelRasterizer(gpkg, config)
    size = int(config["patch"]["size_px"])
    stride = int(config["patch"]["stride_px"])
    boundary_px = round(
        float(config["patch"]["boundary_ignore_width_m"])
        / float(config["source"]["orthophoto"]["native_pixel_size_m"])
    )
    candidates_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    split_features = []
    for sheet, split in sorted(split_sheets.items()):
        path = imagery[sheet]
        edge_px = round(config["splits"][split]["edge_buffer_m"] / 0.5)
        with rasterio.open(path) as source:
            safe_bounds = box(*source.bounds).buffer(-config["splits"][split]["edge_buffer_m"])
            split_features.append(
                {
                    "type": "Feature",
                    "properties": {"split": split, "map_sheet": sheet},
                    "geometry": mapping(safe_bounds),
                }
            )
            for window in _candidate_windows(source, size, stride, edge_px):
                valid = source.dataset_mask(window=window) > 0
                valid_fraction = float(valid.mean())
                if valid_fraction < config["patch"]["minimum_valid_fraction"]:
                    continue
                transform = source.window_transform(window)
                mask, boundary = rasterizer.rasterize_window(
                    transform, size, size, valid, boundary_px, use_boundary_ignore=False
                )
                patch_id = f"{split}-{sheet}-{int(window.row_off):05d}-{int(window.col_off):05d}"
                class_counts = {
                    str(class_id): int((mask == class_id).sum())
                    for class_id in range(len(config["classes"]))
                }
                item = {
                    "id": patch_id,
                    "split": split,
                    "sheet": sheet,
                    "width": size,
                    "height": size,
                    "valid_fraction": valid_fraction,
                    "class_counts": class_counts,
                    "row": int(window.row_off),
                    "col": int(window.col_off),
                }
                candidates_by_split[split].append(item)

    selected: list[dict[str, Any]] = []
    for offset, split in enumerate(("train", "val", "test")):
        selected.extend(
            _select_candidates(
                candidates_by_split[split],
                int(config["patch"]["samples"][split]),
                len(config["classes"]),
                float(config["patch"]["rare_class_min_fraction"]),
                int(config["seed"]) + offset,
            )
        )

    records = []
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq = np.zeros(3, dtype=np.float64)
    train_pixels = 0
    class_counts_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    ignored_counts: Counter[str] = Counter()
    pixel_counts: Counter[str] = Counter()
    image_dtype: np.dtype[Any] | None = None
    for item in selected:
        split = item["split"]
        with rasterio.open(imagery[item["sheet"]]) as source:
            window = Window(item["col"], item["row"], size, size)
            image = source.read(window=window)
            valid = source.dataset_mask(window=window) > 0
            transform = source.window_transform(window)
            source_profile = source.profile
        mask, boundary_bool = rasterizer.rasterize_window(
            transform, size, size, valid, boundary_px, use_boundary_ignore=False
        )
        boundary = boundary_bool.astype(np.uint8)
        image_dtype = image.dtype
        image_path = processed_dir / split / "images" / f"{item['id']}.tif"
        mask_path = processed_dir / split / "masks" / f"{item['id']}.tif"
        boundary_path = processed_dir / split / "boundaries" / f"{item['id']}.tif"
        image_profile = {
            **source_profile,
            "driver": "GTiff",
            "width": size,
            "height": size,
            "count": 3,
            "transform": transform,
            "compress": "deflate",
            "tiled": True,
        }
        mask_profile = {
            "driver": "GTiff",
            "width": size,
            "height": size,
            "count": 1,
            "dtype": "uint8",
            "crs": "EPSG:3067",
            "transform": transform,
            "nodata": 255,
            "compress": "deflate",
            "tiled": True,
        }
        _write_patch(image_path, image, image_profile)
        _write_patch(mask_path, mask, mask_profile)
        _write_patch(boundary_path, boundary, {**mask_profile, "nodata": None})
        if split == "train":
            scale = (
                float(np.iinfo(image.dtype).max) if np.issubdtype(image.dtype, np.integer) else 1.0
            )
            values = image[:, valid].astype(np.float64) / scale
            channel_sum += values.sum(axis=1)
            channel_sq += np.square(values).sum(axis=1)
            train_pixels += values.shape[1]
        class_counts_by_split[split].update(item["class_counts"])
        ignored_counts[split] += int((boundary_bool | ~valid).sum())
        pixel_counts[split] += size * size
        bounds = rasterio.transform.array_bounds(size, size, transform)
        records.append(
            {
                "id": item["id"],
                "split": split,
                "source_map_sheet": item["sheet"],
                "source_orthophoto": imagery[item["sheet"]].name,
                "orthophoto_year": config["source"]["orthophoto"]["acquisition_year"],
                "image_path": image_path.relative_to(root).as_posix(),
                "mask_path": mask_path.relative_to(root).as_posix(),
                "boundary_path": boundary_path.relative_to(root).as_posix(),
                "bounds": tuple(float(value) for value in bounds),
                "transform": tuple(transform)[:6],
                "width": size,
                "height": size,
                "valid_fraction": item["valid_fraction"],
                "class_counts": item["class_counts"],
                "image_sha256": sha256(image_path),
                "mask_sha256": sha256(mask_path),
                "boundary_sha256": sha256(boundary_path),
            }
        )
    mean = channel_sum / train_pixels
    std = np.sqrt(np.maximum(channel_sq / train_pixels - mean**2, 1e-12))
    if image_dtype is None:
        raise ValueError("No patches were selected")
    scale = float(np.iinfo(image_dtype).max) if np.issubdtype(image_dtype, np.integer) else 1.0
    acquisition_receipt = raw_dir / "acquisition-receipt.json"
    vector_counts = {layer: len(frame) for layer, frame in rasterizer.frames.items()}
    manifest = DatasetManifest.model_validate(
        {
            "schema_version": "finland-geospatial-ai-dataset/v2",
            "dataset_version": config["dataset_version"],
            "split_version": config["split_version"],
            "class_map_version": config["class_map_version"],
            "generated_at": datetime.now(UTC).isoformat(),
            "generation_commit": _git_commit(),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "crs": "EPSG:3067",
            "aoi": mapping(box(*config["source"]["topographic_database"]["bbox"])),
            "native_pixel_size_m": 0.5,
            "bands": config["source"]["orthophoto"]["bands"],
            "patch_size_px": size,
            "patch_ground_size_m": config["patch"]["ground_size_m"],
            "ignore_index": config["patch"]["ignore_index"],
            "boundary_ignore_width_m": config["patch"]["boundary_ignore_width_m"],
            "sources": {
                "licensor": config["source"]["licensor"],
                "licence": config["source"]["licence"],
                "acquisition_method": "NLS OGC API Processes",
                "orthophoto_product": config["source"]["orthophoto"],
                "topographic_product": config["source"]["topographic_database"],
                "orthophotos": {
                    sheet: {
                        "file": path.name,
                        "sha256": sha256(path),
                        "metadata": raster_metadata[sheet].to_dict(),
                    }
                    for sheet, path in imagery.items()
                },
                "topographic_database": {
                    "file": gpkg.name,
                    "sha256": sha256(gpkg),
                    "snapshot_date": _snapshot_date(
                        gpkg, config["source"]["topographic_database"].get("snapshot_date")
                    ),
                },
                "topographic_layer_feature_counts": vector_counts,
                "acquisition_receipt": (
                    {"file": acquisition_receipt.name, "sha256": sha256(acquisition_receipt)}
                    if acquisition_receipt.is_file()
                    else None
                ),
            },
            "classes": config["classes"],
            "precedence": config["precedence"],
            "normalization": {"scale": scale, "mean": mean.tolist(), "std": std.tolist()},
            "patch_counts": dict(Counter(item["split"] for item in selected)),
            "class_counts_by_split": {
                split: dict(counts) for split, counts in class_counts_by_split.items()
            },
            "ignored_pixel_fraction_by_split": {
                split: ignored_counts[split] / pixel_counts[split] for split in pixel_counts
            },
            "patches": records,
        }
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "name": config["split_version"],
                "crs": {"type": "name", "properties": {"name": "EPSG:3067"}},
                "features": split_features,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data/nls_l324.yaml"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(build_dataset(args.config.resolve(), args.root.resolve()))


if __name__ == "__main__":
    main()
