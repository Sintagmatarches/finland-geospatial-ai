"""Create deterministic, synthetic assets for Docker API validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from safetensors.torch import save_file

from finland_geospatial_ai.models import create_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/smoke"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        "architecture": "unet",
        "in_channels": 3,
        "num_classes": 2,
        "base_channels": 4,
    }
    model = create_model(model_config)
    weights = args.output_dir / "smoke.safetensors"
    save_file(model.state_dict(), weights)
    metadata = {
        "schema_version": "finland-geospatial-ai-model/v2",
        "experiment_id": "docker-smoke-only",
        "model": model_config,
        "dataset_version": "synthetic-smoke-only",
        "split_version": "synthetic-smoke-only",
        "class_map_version": "synthetic-smoke-only",
        "normalization": {"scale": 255.0, "mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
        "classes": [{"id": 0, "name": "other_land"}, {"id": 1, "name": "water"}],
        "ignore_index": 255,
        "native_pixel_size_m": 0.5,
        "weights": weights.name,
    }
    (args.output_dir / "selected-model.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    image = np.full((3, 32, 32), 127, dtype=np.uint8)
    with rasterio.open(
        args.output_dir / "input.tif",
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=3,
        dtype="uint8",
        crs="EPSG:3067",
        transform=from_origin(400000, 6700000, 0.5, 0.5),
    ) as destination:
        destination.write(image)


if __name__ == "__main__":
    main()
