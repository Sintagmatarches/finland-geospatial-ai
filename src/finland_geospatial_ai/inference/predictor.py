"""Overlap-tiled prediction with probability blending and GeoTIFF preservation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from safetensors.torch import load_file

from finland_geospatial_ai.geospatial.raster import inspect_raster
from finland_geospatial_ai.models import create_model


class Predictor:
    def __init__(self, metadata_path: Path | str, device: str | None = None) -> None:
        self.metadata_path = Path(metadata_path)
        self.metadata: dict[str, Any] = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        classes = self.metadata.get("classes", [])
        if len(classes) != int(self.metadata["model"]["num_classes"]):
            raise ValueError("Model metadata class count does not match the architecture")
        self.model = create_model(self.metadata["model"])
        weights = self.metadata_path.parent / self.metadata["weights"]
        expected_hash = self.metadata.get("weights_sha256")
        if expected_hash:
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            if digest != expected_hash:
                raise ValueError("Model checkpoint SHA-256 does not match metadata")
        self.model.load_state_dict(load_file(weights), strict=True)
        self.model.to(self.device).eval()
        normalization = self.metadata["normalization"]
        if not all(len(normalization[key]) == 3 for key in ("mean", "std")):
            raise ValueError("Model normalization metadata must contain three channels")
        if any(float(value) <= 0 for value in normalization["std"]):
            raise ValueError("Model normalization standard deviations must be positive")
        self.scale = float(normalization["scale"])
        self.mean = torch.tensor(normalization["mean"], dtype=torch.float32)[None, :, None, None]
        self.std = torch.tensor(normalization["std"], dtype=torch.float32)[None, :, None, None]

    def predict(
        self,
        input_path: Path | str,
        output_path: Path | str,
        tile_size: int = 256,
        overlap: int = 64,
        confidence_path: Path | str | None = None,
    ) -> Path:
        input_path, output_path = Path(input_path), Path(output_path)
        inspect_raster(
            input_path, expected_bands=3, expected_dtype="uint8", expected_resolution_m=0.5
        )
        if overlap < 0 or overlap >= tile_size:
            raise ValueError("overlap must be non-negative and smaller than tile_size")
        stride = tile_size - overlap
        with rasterio.open(input_path) as source:
            valid = source.dataset_mask() > 0
            if not valid.any():
                raise ValueError("Raster has no valid pixels")
            scores = np.zeros(
                (self.metadata["model"]["num_classes"], source.height, source.width),
                dtype=np.float32,
            )
            weights = np.zeros((source.height, source.width), dtype=np.float32)
            rows = list(range(0, max(source.height - tile_size, 0) + 1, stride))
            cols = list(range(0, max(source.width - tile_size, 0) + 1, stride))
            rows.append(max(source.height - tile_size, 0))
            cols.append(max(source.width - tile_size, 0))
            taper = np.outer(np.hanning(tile_size), np.hanning(tile_size)).astype(np.float32)
            taper = np.maximum(taper, 1e-3)
            with torch.inference_mode():
                for row in sorted(set(rows)):
                    for col in sorted(set(cols)):
                        height = min(tile_size, source.height - row)
                        width = min(tile_size, source.width - col)
                        image = source.read(window=Window(col, row, width, height))
                        padded = np.zeros((3, tile_size, tile_size), dtype=image.dtype)
                        padded[:, :height, :width] = image
                        tensor = torch.from_numpy(padded.astype(np.float32))[None] / self.scale
                        tensor = ((tensor - self.mean) / self.std).to(self.device)
                        probability = (
                            self.model(tensor).softmax(dim=1).cpu().numpy()[0, :, :height, :width]
                        )
                        blend = taper[:height, :width]
                        scores[:, row : row + height, col : col + width] += probability * blend
                        weights[row : row + height, col : col + width] += blend
            probabilities = scores / np.maximum(weights, 1e-6)
            prediction = probabilities.argmax(axis=0).astype(np.uint8)
            prediction[~valid] = 255
            profile = source.profile.copy()
            profile.update(driver="GTiff", count=1, dtype="uint8", nodata=255, compress="deflate")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(output_path, "w", **profile) as destination:
                destination.write(prediction, 1)
                destination.update_tags(
                    model_experiment=self.metadata["experiment_id"],
                    dataset_version=self.metadata["dataset_version"],
                    class_map_version=self.metadata["class_map_version"],
                )
            if confidence_path is not None:
                confidence_path = Path(confidence_path)
                confidence_path.parent.mkdir(parents=True, exist_ok=True)
                confidence_profile = source.profile.copy()
                confidence_profile.update(
                    driver="GTiff", count=1, dtype="float32", nodata=np.nan, compress="deflate"
                )
                confidence = probabilities.max(axis=0).astype(np.float32)
                confidence[~valid] = np.nan
                with rasterio.open(confidence_path, "w", **confidence_profile) as destination:
                    destination.write(confidence, 1)
                    destination.update_tags(
                        model_experiment=self.metadata["experiment_id"], quantity="max_softmax"
                    )
        return output_path
