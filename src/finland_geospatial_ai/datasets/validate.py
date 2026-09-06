"""Fail-fast validation for the versioned NLS dataset contract."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry

from finland_geospatial_ai.datasets.build import sha256
from finland_geospatial_ai.datasets.manifest import DatasetManifest
from finland_geospatial_ai.geospatial.raster import assert_aligned, inspect_raster


def validate_dataset(manifest_path: Path, root: Path | None = None) -> dict[str, object]:
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    root = root or manifest_path.parents[2]
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    split_boxes: dict[str, list[BaseGeometry]] = {"train": [], "val": [], "test": []}
    allowed = {item.id for item in manifest.classes} | {manifest.ignore_index}
    actual_counts: Counter[str] = Counter()
    for record in manifest.patches:
        if record.id in seen_ids:
            errors.append(f"duplicate patch id: {record.id}")
        seen_ids.add(record.id)
        if record.image_sha256 in seen_images:
            errors.append(f"duplicate image content: {record.id}")
        seen_images.add(record.image_sha256)
        paths = [root / record.image_path, root / record.mask_path, root / record.boundary_path]
        if not all(path.is_file() for path in paths):
            errors.append(f"missing artifacts: {record.id}")
            continue
        expected_hashes = [record.image_sha256, record.mask_sha256, record.boundary_sha256]
        for path, expected in zip(paths, expected_hashes, strict=True):
            if sha256(path) != expected:
                errors.append(f"hash mismatch: {path}")
        try:
            inspect_raster(
                paths[0], expected_bands=3, expected_dtype="uint8", expected_resolution_m=0.5
            )
            with (
                rasterio.open(paths[0]) as image,
                rasterio.open(paths[1]) as mask,
                rasterio.open(paths[2]) as boundary,
            ):
                actual_bounds = tuple(float(value) for value in image.bounds)
                if tuple(record.transform) != tuple(image.transform)[:6]:
                    errors.append(f"manifest transform mismatch: {record.id}")
                if not np.allclose(record.bounds, actual_bounds, atol=1e-6):
                    errors.append(f"manifest bounds mismatch: {record.id}")
                if (record.height, record.width) != (image.height, image.width):
                    errors.append(f"manifest dimensions mismatch: {record.id}")
                assert_aligned(
                    (image.height, image.width),
                    image.transform,
                    image.crs,
                    (mask.height, mask.width),
                    mask.transform,
                    mask.crs,
                )
                assert_aligned(
                    (image.height, image.width),
                    image.transform,
                    image.crs,
                    (boundary.height, boundary.width),
                    boundary.transform,
                    boundary.crs,
                )
                values = set(np.unique(mask.read(1)).tolist())
                if values - allowed:
                    errors.append(f"undeclared mask values {values - allowed}: {record.id}")
                boundary_values = set(np.unique(boundary.read(1)).tolist())
                if not boundary_values <= {0, 1}:
                    errors.append(f"non-binary boundary mask: {record.id}")
        except Exception as exc:
            errors.append(f"invalid raster contract for {record.id}: {exc}")
        split_boxes[record.split].append(box(*record.bounds))
        actual_counts[record.split] += 1
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if any(a.intersects(b) for a in split_boxes[left] for b in split_boxes[right]):
            errors.append(f"spatial leakage between {left} and {right}")
    if dict(actual_counts) != manifest.patch_counts:
        errors.append(f"patch count mismatch: {dict(actual_counts)} != {manifest.patch_counts}")
    for split in ("train", "val", "test"):
        counts = manifest.class_counts_by_split.get(split, {})
        missing = [item.name for item in manifest.classes if counts.get(str(item.id), 0) == 0]
        if missing:
            errors.append(f"{split} has no pixels for classes: {missing}")
    split_manifest = root / "data" / "splits" / f"{manifest.split_version}.geojson"
    if split_manifest.is_file():
        data = json.loads(split_manifest.read_text(encoding="utf-8"))
        declared = {feature["properties"]["split"] for feature in data["features"]}
        if declared != {"train", "val", "test"}:
            errors.append("split GeoJSON does not declare train, val and test")
        for feature in data["features"]:
            if shape(feature["geometry"]).is_empty:
                errors.append("empty split geometry")
        safe_areas = {
            (feature["properties"]["split"], feature["properties"].get("map_sheet")): shape(
                feature["geometry"]
            )
            for feature in data["features"]
        }
        for record in manifest.patches:
            area = safe_areas.get((record.split, record.source_map_sheet))
            if area is None:
                errors.append(
                    "patch has no matching split/map-sheet declaration: "
                    f"{record.id} ({record.split}, {record.source_map_sheet})"
                )
            elif not area.covers(box(*record.bounds)):
                errors.append(f"patch falls outside buffered split area: {record.id}")
    else:
        errors.append(f"missing split manifest: {split_manifest}")
    if errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(errors))
    return {
        "dataset_version": manifest.dataset_version,
        "patches": len(manifest.patches),
        "splits": dict(actual_counts),
        "crs": manifest.crs,
        "pixel_size_m": manifest.native_pixel_size_m,
        "status": "valid",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.manifest.resolve(), args.root), indent=2))


if __name__ == "__main__":
    main()
