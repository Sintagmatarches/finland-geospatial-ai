from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from conftest import write_raster
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

import finland_geospatial_ai.api as api_module
from finland_geospatial_ai.inference import Predictor


def test_tiled_prediction_preserves_georeferencing(tmp_path: Path, tiny_model: Path) -> None:
    input_path = tmp_path / "input.tif"
    output_path = tmp_path / "output.tif"
    transform = from_origin(400000, 6700000, 0.5, 0.5)
    write_raster(input_path, np.full((3, 48, 48), 120, dtype=np.uint8), transform)
    Predictor(tiny_model, device="cpu").predict(input_path, output_path, tile_size=32, overlap=8)
    with rasterio.open(output_path) as result:
        assert result.crs.to_epsg() == 3067
        assert result.transform == transform
        assert result.shape == (48, 48)
        assert result.tags()["model_experiment"] == "tiny"


def test_inference_rejects_wrong_resolution(tmp_path: Path, tiny_model: Path) -> None:
    input_path = tmp_path / "wrong-resolution.tif"
    write_raster(
        input_path,
        np.full((3, 32, 32), 120, dtype=np.uint8),
        from_origin(400000, 6700000, 1.0, 1.0),
    )
    with pytest.raises(ValueError, match="Expected 0.5 m"):
        Predictor(tiny_model, device="cpu").predict(input_path, tmp_path / "output.tif")


def test_inference_rejects_wrong_dtype(tmp_path: Path, tiny_model: Path) -> None:
    input_path = tmp_path / "wrong-dtype.tif"
    write_raster(
        input_path,
        np.full((3, 32, 32), 120, dtype=np.uint16),
        from_origin(400000, 6700000, 0.5, 0.5),
    )
    with pytest.raises(ValueError, match="Expected uint8"):
        Predictor(tiny_model, device="cpu").predict(input_path, tmp_path / "output.tif")


def test_inference_rejects_all_nodata(tmp_path: Path, tiny_model: Path) -> None:
    input_path = tmp_path / "all-nodata.tif"
    write_raster(
        input_path,
        np.zeros((3, 32, 32), dtype=np.uint8),
        from_origin(400000, 6700000, 0.5, 0.5),
        nodata=0,
    )
    with pytest.raises(ValueError, match="no valid pixels"):
        Predictor(tiny_model, device="cpu").predict(input_path, tmp_path / "output.tif")


def test_model_metadata_mismatch_is_rejected(tmp_path: Path, tiny_model: Path) -> None:
    metadata = json.loads(tiny_model.read_text(encoding="utf-8"))
    metadata["classes"] = metadata["classes"][:1]
    bad_metadata = tmp_path / "bad-metadata.json"
    bad_metadata.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="class count"):
        Predictor(bad_metadata, device="cpu")


def test_api_health_and_model_metadata(tiny_model: Path) -> None:
    os.environ["GEOAI_MODEL_METADATA"] = str(tiny_model)
    api_module._predictor = None
    client = TestClient(api_module.app)
    assert client.get("/health").json() == {"status": "ready", "experiment_id": "tiny"}
    response = client.get("/model")
    assert response.status_code == 200
    assert response.json()["native_pixel_size_m"] == 0.5


def test_api_rejects_non_raster_extension(tiny_model: Path) -> None:
    os.environ["GEOAI_MODEL_METADATA"] = str(tiny_model)
    api_module._predictor = None
    response = TestClient(api_module.app).post(
        "/predict", files={"file": ("bad.txt", b"bad", "text/plain")}
    )
    assert response.status_code == 415


def test_api_enforces_upload_limit(tiny_model: Path) -> None:
    os.environ["GEOAI_MODEL_METADATA"] = str(tiny_model)
    os.environ["GEOAI_MAX_UPLOAD_BYTES"] = "2"
    api_module._predictor = None
    response = TestClient(api_module.app).post(
        "/predict", files={"file": ("large.tif", b"123", "image/tiff")}
    )
    os.environ.pop("GEOAI_MAX_UPLOAD_BYTES")
    assert response.status_code == 413


def test_api_returns_geotiffs_and_summary(tmp_path: Path, tiny_model: Path) -> None:
    input_path = tmp_path / "upload.tif"
    write_raster(
        input_path,
        np.full((3, 32, 32), 120, dtype=np.uint8),
        from_origin(400000, 6700000, 0.5, 0.5),
    )
    os.environ["GEOAI_MODEL_METADATA"] = str(tiny_model)
    api_module._predictor = None
    response = TestClient(api_module.app).post(
        "/predict", files={"file": ("upload.tif", input_path.read_bytes(), "image/tiff")}
    )
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as package:
        assert set(package.namelist()) == {"prediction.tif", "confidence.tif", "summary.json"}
        summary = json.loads(package.read("summary.json"))
        assert summary["width"] == 32
        assert sum(item["fraction"] for item in summary["classes"].values()) == 1.0
