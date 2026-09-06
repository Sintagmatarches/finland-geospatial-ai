"""Build aligned Sentinel-2/WorldCover patches without downloading full source rasters."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rasterio.enums import Resampling

from finland_geoai.geo.raster import grid_from_wgs84_bbox, patch_bounds, read_aligned

STAC_ASSET_NAMES = {"blue": "blue", "green": "green", "red": "red", "nir": "nir"}
INVALID_SCL = {0, 1, 3, 8, 9, 10, 11}


def _read_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "finland-geospatial-ai/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed config host
        return json.load(response)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted"


def _source_to_target_map(config: dict[str, Any]) -> np.ndarray:
    lookup = np.full(256, int(config["class_mapping"]["ignore_index"]), dtype=np.uint8)
    for target, details in config["class_mapping"]["classes"].items():
        for source_code in details["source_codes"]:
            lookup[int(source_code)] = int(target)
    return lookup


def _worldcover_url(config: dict[str, Any], tile: str) -> str:
    base = config["worldcover"]["base_url"].rstrip("/")
    return f"{base}/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"


def build(config_path: Path, output_root: Path, manifest_path: Path) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    patch_size = int(config["patch_size"])
    stride = int(config["stride"])
    ignore_index = int(config["class_mapping"]["ignore_index"])
    class_lookup = _source_to_target_map(config)
    output_root.mkdir(parents=True, exist_ok=True)

    patches: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    counts_by_split: dict[str, Counter[int]] = defaultdict(Counter)
    stats_sum = np.zeros(len(config["bands"]), dtype=np.float64)
    stats_sq = np.zeros(len(config["bands"]), dtype=np.float64)
    stats_count = 0

    for aoi in config["aois"]:
        item_url = (
            f"{config['stac']['endpoint'].rstrip('/')}/collections/"
            f"{config['stac']['collection']}/items/{aoi['stac_item_id']}"
        )
        item = _read_json(item_url)
        cloud_cover = float(item["properties"].get("eo:cloud_cover", 100.0))
        if cloud_cover > float(config["stac"]["max_cloud_cover"]):
            raise ValueError(f"{aoi['id']}: STAC item exceeds configured cloud threshold")
        grid = grid_from_wgs84_bbox(
            tuple(aoi["bbox_wgs84"]), config["target_crs"], float(config["resolution_m"])
        )

        band_urls = [item["assets"][STAC_ASSET_NAMES[name]]["href"] for name in config["bands"]]
        # Reflectance is continuous: bilinear interpolation is appropriate during reprojection.
        imagery = np.stack(
            [read_aligned(url, grid, Resampling.bilinear) for url in band_urls], axis=0
        ).astype(np.uint16)
        # Scene Classification and land-cover masks are categorical: nearest-neighbour only.
        scl_url = item["assets"]["scl"]["href"]
        scl = read_aligned(scl_url, grid, Resampling.nearest).astype(np.uint8)
        valid = ~np.isin(scl, list(INVALID_SCL))

        source_labels: np.ndarray = np.zeros((grid.height, grid.width), dtype=np.uint8)
        worldcover_urls = [_worldcover_url(config, tile) for tile in aoi["worldcover_tiles"]]
        for label_url in worldcover_urls:
            candidate = read_aligned(label_url, grid, Resampling.nearest).astype(np.uint8)
            source_labels = np.where(
                (source_labels == 0) & (candidate != 0), candidate, source_labels
            )
        labels = class_lookup[source_labels]
        valid &= labels != ignore_index
        labels[~valid] = ignore_index

        aoi_dir = output_root / aoi["split"] / aoi["id"]
        aoi_dir.mkdir(parents=True, exist_ok=True)
        for row in range(0, grid.height - patch_size + 1, stride):
            for col in range(0, grid.width - patch_size + 1, stride):
                patch_valid = valid[row : row + patch_size, col : col + patch_size]
                valid_fraction = float(patch_valid.mean())
                if valid_fraction < float(config["min_valid_fraction"]):
                    continue
                patch_image = imagery[:, row : row + patch_size, col : col + patch_size]
                patch_mask = labels[row : row + patch_size, col : col + patch_size]
                if not np.any(patch_valid):
                    continue
                patch_id = f"{aoi['id']}-r{row:05d}-c{col:05d}"
                relative_path = Path(aoi["split"]) / aoi["id"] / f"{patch_id}.npz"
                path = output_root / relative_path
                np.savez_compressed(
                    path,
                    image=patch_image,
                    mask=patch_mask,
                    valid=patch_valid,
                    crs=np.array(config["target_crs"]),
                    transform=np.asarray(
                        (
                            grid.transform.a,
                            grid.transform.b,
                            grid.transform.c + col * grid.resolution,
                            grid.transform.d,
                            grid.transform.e,
                            grid.transform.f - row * grid.resolution,
                        ),
                        dtype=np.float64,
                    ),
                )
                class_counts = Counter(int(x) for x in patch_mask[patch_valid].ravel())
                counts_by_split[aoi["split"]].update(class_counts)
                if aoi["split"] == "train":
                    pixels = patch_image[:, patch_valid].astype(np.float64) / 10000.0
                    stats_sum += pixels.sum(axis=1)
                    stats_sq += np.square(pixels).sum(axis=1)
                    stats_count += pixels.shape[1]
                patches.append(
                    {
                        "id": patch_id,
                        "path": str(relative_path).replace("\\", "/"),
                        "aoi_id": aoi["id"],
                        "split": aoi["split"],
                        "bounds": patch_bounds(grid.transform, row, col, patch_size),
                        "valid_fraction": round(valid_fraction, 6),
                        "class_counts": {str(k): v for k, v in sorted(class_counts.items())},
                        "sha256": _sha256(path),
                    }
                )
        source_records.append(
            {
                "aoi_id": aoi["id"],
                "split": aoi["split"],
                "bbox_wgs84": aoi["bbox_wgs84"],
                "target_bounds": list(grid.bounds),
                "stac_item_id": item["id"],
                "acquisition_datetime": item["properties"]["datetime"],
                "cloud_cover_percent": cloud_cover,
                "source_epsg": item["properties"].get("proj:epsg"),
                "imagery_urls": dict(zip(config["bands"], band_urls, strict=True)),
                "scl_url": scl_url,
                "worldcover_urls": worldcover_urls,
            }
        )

    if stats_count == 0:
        raise ValueError("No valid training pixels were produced")
    means = stats_sum / stats_count
    stds = np.sqrt(np.maximum(stats_sq / stats_count - np.square(means), 1e-12))
    split_patch_counts = Counter(patch["split"] for patch in patches)
    manifest: dict[str, Any] = {
        "schema_version": config["schema_version"],
        "split_version": config["split_version"],
        "generated_at": datetime.now(UTC).isoformat(),
        "code_commit": _git_commit(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "target_crs": config["target_crs"],
        "resolution_m": config["resolution_m"],
        "patch_size": patch_size,
        "stride": stride,
        "bands": config["bands"],
        "imagery_source": "Copernicus Sentinel-2 Level-2A via Earth Search/AWS COGs",
        "stac_collection": config["stac"]["collection"],
        "label_source": "ESA WorldCover 2021 v200",
        "resampling": {"reflectance": "bilinear", "scl": "nearest", "labels": "nearest"},
        "ignore_index": ignore_index,
        "class_mapping": config["class_mapping"],
        "normalization": {
            "scale": 10000.0,
            "mean": means.round(8).tolist(),
            "std": stds.round(8).tolist(),
            "fit_split": "train",
            "pixel_count": stats_count,
        },
        "sources": source_records,
        "patch_counts": dict(sorted(split_patch_counts.items())),
        "class_counts_by_split": {
            split: {str(k): v for k, v in sorted(counts.items())}
            for split, counts in sorted(counts_by_split.items())
        },
        "patches": patches,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/v1"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/dataset-manifest-v1.json")
    )
    args = parser.parse_args()
    manifest = build(args.config, args.output_root, args.manifest)
    print(json.dumps({"patch_counts": manifest["patch_counts"]}, indent=2))


if __name__ == "__main__":
    main()
