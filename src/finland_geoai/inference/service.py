"""Validated, CPU-only GeoTIFF inference with georeferencing-preserving outputs."""

from __future__ import annotations

import base64
import hashlib
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.io import MemoryFile

from finland_geoai.models.factory import create_model


class InferenceInputError(ValueError):
    """A safe client-visible raster validation error."""


class SegmentationService:
    def __init__(self, model_path: Path | str) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
        try:
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
            if checkpoint.get("artifact_schema") != "finland-geoai-model/v1":
                raise ValueError("unsupported artifact schema")
            model_config = checkpoint["model_config"]
            self.model = create_model(
                model_config["model"],
                model_config["in_channels"],
                model_config["num_classes"],
                model_config["base_channels"],
            )
            self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        except Exception as exc:
            raise ValueError("Corrupt or incompatible model artifact") from exc
        self.model.eval()
        self.band_indices = checkpoint["band_indices"]
        self.normalization = checkpoint["normalization"]
        self.classes = checkpoint["classes"]
        self.temperature = float(checkpoint.get("temperature", 1.0))
        self.dataset_version = checkpoint["dataset_version"]
        self.model_version = hashlib.sha256(self.model_path.read_bytes()).hexdigest()[:12]
        self.patch_size = 64

    def _read(self, payload: bytes) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if not payload:
            raise InferenceInputError("Uploaded file is empty")
        try:
            with MemoryFile(payload) as memory_file:
                with memory_file.open() as source:
                    if source.count != len(self.band_indices):
                        raise InferenceInputError(
                            f"Expected {len(self.band_indices)} bands in blue, green, red, "
                            "NIR order; "
                            f"received {source.count}"
                        )
                    if source.width != self.patch_size or source.height != self.patch_size:
                        raise InferenceInputError(
                            f"Expected {self.patch_size}x{self.patch_size} pixels; "
                            f"received {source.width}x{source.height}"
                        )
                    if source.crs is None or source.crs.to_epsg() != 3067:
                        raise InferenceInputError("Expected CRS EPSG:3067")
                    x_resolution = abs(float(source.transform.a))
                    y_resolution = abs(float(source.transform.e))
                    if not (abs(x_resolution - 10.0) < 1e-6 and abs(y_resolution - 10.0) < 1e-6):
                        raise InferenceInputError("Expected 10 metre square pixels")
                    if source.dtypes[0] not in {"uint16", "int16", "float32", "float64"}:
                        raise InferenceInputError(f"Unsupported dtype: {source.dtypes[0]}")
                    image = source.read().astype(np.float32)
                    valid = source.dataset_mask() > 0
                    if source.nodata is not None:
                        valid &= ~np.any(image == float(source.nodata), axis=0)
                    if not valid.any():
                        raise InferenceInputError("Raster contains no valid pixels")
                    profile = {
                        "crs": source.crs,
                        "transform": source.transform,
                        "width": source.width,
                        "height": source.height,
                    }
        except InferenceInputError:
            raise
        except rasterio.errors.RasterioError as exc:
            raise InferenceInputError("Malformed or unsupported GeoTIFF") from exc
        return image, valid, profile

    def predict(self, payload: bytes) -> dict[str, Any]:
        image, valid, profile = self._read(payload)
        scale = float(self.normalization["scale"])
        mean = np.asarray(self.normalization["mean"], dtype=np.float32)[:, None, None]
        std = np.asarray(self.normalization["std"], dtype=np.float32)[:, None, None]
        normalized = (image / scale - mean) / std
        tensor = torch.from_numpy(normalized).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tensor)[0]
            calibrated = torch.softmax(logits / self.temperature, dim=0)
            prediction = logits.argmax(dim=0).numpy().astype(np.uint8)
            entropy_tensor = -(calibrated * calibrated.clamp_min(1e-8).log()).sum(dim=0)
            entropy = entropy_tensor.numpy() / math.log(calibrated.shape[0])
        prediction[~valid] = 255
        colors = np.asarray(
            [self.classes[str(index)]["color"] for index in range(6)], dtype=np.uint8
        )
        color_mask = np.zeros((*prediction.shape, 4), dtype=np.uint8)
        for index, color in enumerate(colors):
            color_mask[prediction == index, :3] = color
            color_mask[prediction == index, 3] = 255
        png_buffer = io.BytesIO()
        Image.fromarray(color_mask).save(png_buffer, format="PNG")
        geotiff_buffer = io.BytesIO()
        with MemoryFile() as memory_file:
            with memory_file.open(
                driver="GTiff",
                width=profile["width"],
                height=profile["height"],
                count=1,
                dtype="uint8",
                crs=profile["crs"],
                transform=profile["transform"],
                nodata=255,
                compress="deflate",
            ) as destination:
                destination.write(prediction, 1)
            geotiff_buffer.write(memory_file.read())
        valid_predictions = prediction[valid]
        fractions = {
            self.classes[str(index)]["name"]: float((valid_predictions == index).mean())
            for index in range(6)
        }
        return {
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "crs": "EPSG:3067",
            "transform": list(profile["transform"])[:6],
            "class_legend": [
                {
                    "id": index,
                    "name": self.classes[str(index)]["name"],
                    "color": self.classes[str(index)]["color"],
                }
                for index in range(6)
            ],
            "class_fractions": fractions,
            "mean_predictive_entropy": float(entropy[valid].mean()),
            "calibration_note": (
                "Temperature was fit on Tampere validation but worsened held-out Oulu ECE; "
                "probabilities are uncertainty signals, not calibrated probabilities."
            ),
            "mask_png_base64": base64.b64encode(png_buffer.getvalue()).decode("ascii"),
            "mask_geotiff_base64": base64.b64encode(geotiff_buffer.getvalue()).decode("ascii"),
        }
