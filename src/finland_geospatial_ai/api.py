"""Small production API for native-resolution NLS GeoTIFF inference."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Annotated

import numpy as np
import rasterio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from finland_geospatial_ai.geospatial.raster import inspect_raster
from finland_geospatial_ai.inference import Predictor

app = FastAPI(title="Finland Geospatial AI", version="0.2.1")
_predictor: Predictor | None = None
_inference_lock = threading.Lock()
_model_lock = threading.Lock()


def get_predictor() -> Predictor:
    global _predictor
    with _model_lock:
        if _predictor is None:
            model_path = os.environ.get("GEOAI_MODEL_METADATA")
            if not model_path:
                raise RuntimeError("GEOAI_MODEL_METADATA is not configured")
            _predictor = Predictor(model_path)
    return _predictor


@app.get("/health")
def health() -> dict[str, str]:
    try:
        predictor = get_predictor()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Model unavailable") from exc
    return {"status": "ready", "experiment_id": predictor.metadata["experiment_id"]}


@app.get("/live")
async def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/model")
def model_card() -> dict[str, object]:
    predictor = get_predictor()
    return {
        "experiment_id": predictor.metadata["experiment_id"],
        "dataset_version": predictor.metadata["dataset_version"],
        "native_pixel_size_m": predictor.metadata["native_pixel_size_m"],
        "classes": predictor.metadata["classes"],
    }


@app.post("/predict")
def predict(file: Annotated[UploadFile, File(...)]) -> FileResponse:
    suffix = Path(file.filename or "input.tif").suffix.lower()
    if suffix not in {".tif", ".tiff", ".jp2"}:
        raise HTTPException(status_code=415, detail="Upload a GeoTIFF or JPEG2000 orthophoto")
    if not _inference_lock.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="Inference busy", headers={"Retry-After": "1"})
    try:
        return _predict_locked(file, suffix)
    finally:
        _inference_lock.release()


def _predict_locked(file: UploadFile, suffix: str) -> FileResponse:
    directory = Path(tempfile.mkdtemp(prefix="geoai-request-"))
    input_path = directory / f"input{suffix}"
    output_path = directory / "prediction.tif"
    confidence_path = directory / "confidence.tif"
    package_path = directory / "prediction-package.zip"
    try:
        uploaded_bytes = 0
        maximum_bytes = int(os.environ.get("GEOAI_MAX_UPLOAD_BYTES", str(256 * 1024 * 1024)))
        with input_path.open("wb") as stream:
            while chunk := file.file.read(1024 * 1024):
                uploaded_bytes += len(chunk)
                if uploaded_bytes > maximum_bytes:
                    raise HTTPException(
                        status_code=413, detail="Uploaded raster exceeds size limit"
                    )
                stream.write(chunk)
        raster = inspect_raster(input_path, expected_bands=3, expected_dtype="uint8")
        maximum_pixels = int(os.environ.get("GEOAI_MAX_PIXELS", "4194304"))
        if raster.width * raster.height > maximum_pixels or raster.width > 16384:
            raise HTTPException(status_code=413, detail="Decoded raster exceeds pixel limit")
        try:
            predictor = get_predictor()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Model unavailable") from exc
        predictor.predict(input_path, output_path, confidence_path=confidence_path)
        with (
            rasterio.open(output_path) as mask_source,
            rasterio.open(confidence_path) as confidence_source,
        ):
            mask = mask_source.read(1)
            confidence = confidence_source.read(1)
            pixel_area_m2 = abs(mask_source.transform.a * mask_source.transform.e)
        valid_pixels = mask != 255
        valid_count = max(int(valid_pixels.sum()), 1)
        classes = predictor.metadata["classes"]
        summary = {
            "model_version": predictor.metadata["experiment_id"],
            "dataset_version": predictor.metadata["dataset_version"],
            "width": int(mask.shape[1]),
            "height": int(mask.shape[0]),
            "crs": "EPSG:3067",
            "class_legend": classes,
            "classes": {
                item["name"]: {
                    "pixels": int(((mask == item["id"]) & valid_pixels).sum()),
                    "fraction": float(((mask == item["id"]) & valid_pixels).sum() / valid_count),
                    "area_m2": float(((mask == item["id"]) & valid_pixels).sum() * pixel_area_m2),
                }
                for item in classes
            },
            "uncertainty": {
                "mean_max_softmax": float(confidence[valid_pixels].mean()),
                "p10_max_softmax": float(np.quantile(confidence[valid_pixels], 0.1)),
            },
        }
        summary_path = directory / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(output_path, output_path.name)
            archive.write(confidence_path, confidence_path.name)
            archive.write(summary_path, summary_path.name)
    except HTTPException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    except (ValueError, rasterio.errors.RasterioError) as exc:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(status_code=422, detail="Raster violates inference contract") from exc
    except Exception as exc:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(status_code=500, detail="Inference failed") from exc
    return FileResponse(
        package_path,
        media_type="application/zip",
        filename="prediction-package.zip",
        background=BackgroundTask(shutil.rmtree, directory, ignore_errors=True),
    )
