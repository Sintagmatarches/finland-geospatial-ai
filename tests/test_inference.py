from __future__ import annotations

import base64
from pathlib import Path

import pytest
import rasterio
from conftest import geotiff_bytes
from fastapi.testclient import TestClient
from rasterio.io import MemoryFile

from finland_geoai.inference.api import create_app
from finland_geoai.inference.service import InferenceInputError, SegmentationService


def test_cpu_api_and_geotiff_georeferencing(checkpoint_path: Path) -> None:
    client = TestClient(create_app(checkpoint_path))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["device"] == "cpu"
    response = client.post(
        "/predict",
        files={"file": ("patch.tif", geotiff_bytes(), "image/tiff")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["class_legend"]) == 6
    assert sum(payload["class_fractions"].values()) == pytest.approx(1.0)
    output = base64.b64decode(payload["mask_geotiff_base64"])
    with MemoryFile(output) as memory_file, memory_file.open() as result:
        assert result.crs.to_epsg() == 3067
        assert result.transform == rasterio.transform.from_origin(385000, 6680000, 10, 10)
        assert result.count == 1
        assert result.nodata == 255


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"bands": 3}, "Expected 4 bands"),
        ({"width": 32}, "Expected 64x64"),
        ({"crs": "EPSG:4326"}, "Expected CRS EPSG:3067"),
        ({"resolution": 20}, "Expected 10 metre"),
        ({"nodata": 0}, "no valid pixels"),
    ],
)
def test_invalid_rasters_are_rejected(
    checkpoint_path: Path, kwargs: dict[str, object], message: str
) -> None:
    service = SegmentationService(checkpoint_path)
    with pytest.raises(InferenceInputError, match=message):
        service.predict(geotiff_bytes(**kwargs))


def test_malformed_upload_is_rejected(checkpoint_path: Path) -> None:
    client = TestClient(create_app(checkpoint_path))
    response = client.post(
        "/predict", files={"file": ("bad.tif", b"not a raster", "image/tiff")}
    )
    assert response.status_code == 422


def test_corrupt_checkpoint_fails_health(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    path.write_bytes(b"bad checkpoint")
    response = TestClient(create_app(path)).get("/health")
    assert response.status_code == 503
