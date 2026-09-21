from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
import torch
from conftest import write_raster
from fastapi.testclient import TestClient
from rasterio.transform import from_origin
from safetensors.torch import save_file

import finland_geospatial_ai.api as api
import finland_geospatial_ai.inference.predictor as predictor_module
from finland_geospatial_ai.inference import Predictor
from finland_geospatial_ai.models import create_model

ROOT = Path(__file__).resolve().parents[1]


def test_pretrained_checkpoint_loads_offline_without_hub_access(tiny_model: Path) -> None:
    metadata = json.loads(tiny_model.read_text())
    config = {
        "architecture": "segformer_b0",
        "num_classes": len(metadata["classes"]),
        "backbone": "nvidia/mit-b0",
        "pretrained": True,
    }
    weights = tiny_model.parent / "offline.safetensors"
    save_file(create_model(config, for_inference=True).state_dict(), weights)
    metadata.update(
        model=config,
        weights=weights.name,
        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
    )
    tiny_model.write_text(json.dumps(metadata))
    with (
        patch(
            "transformers.SegformerConfig.from_pretrained", side_effect=AssertionError("network")
        ),
        patch("transformers.SegformerModel.from_pretrained", side_effect=AssertionError("network")),
    ):
        predictor = Predictor(tiny_model, device="cpu")
        assert predictor.metadata["model"]["pretrained"] is True
    weights.write_bytes(b"tampered checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        Predictor(tiny_model, device="cpu")


def test_published_checkpoint_loads_strictly_with_supported_runtime() -> None:
    """Catch library upgrades that silently change the released SegFormer state schema."""
    with (
        patch(
            "transformers.SegformerConfig.from_pretrained",
            side_effect=AssertionError("network"),
        ),
        patch(
            "transformers.SegformerModel.from_pretrained",
            side_effect=AssertionError("network"),
        ),
    ):
        predictor = Predictor(ROOT / "artifacts/models/selected-model.json", device="cpu")
    assert predictor.metadata["experiment_id"] == "E3-segformer-b0-rgb"


@pytest.mark.parametrize(
    "height,width,overlap", [(17, 19, 8), (48, 48, 8), (65, 47, 8), (64, 33, 0)]
)
def test_row_buffer_matches_dense_blending(
    tmp_path: Path, tiny_model: Path, height: int, width: int, overlap: int
) -> None:
    """Compare to the previous full-canvas algorithm, including shifted edge tiles."""
    predictor = Predictor(tiny_model, device="cpu")
    data = np.random.default_rng(2).integers(1, 255, (3, height, width), dtype=np.uint8)
    data[:, :2, :3] = 0
    source = tmp_path / "image.tif"
    write_raster(source, data, from_origin(400000, 6700000, 0.5, 0.5), nodata=0)
    size = 32
    scores = np.zeros((predictor.metadata["model"]["num_classes"], height, width), np.float32)
    counts = np.zeros((height, width), np.float32)
    taper = np.maximum(np.outer(np.hanning(size), np.hanning(size)).astype(np.float32), 1e-3)
    axes = [
        sorted(set([*range(0, max(n - size, 0) + 1, size - overlap), max(n - size, 0)]))
        for n in (height, width)
    ]
    with torch.inference_mode():
        for row in axes[0]:
            for col in axes[1]:
                h, w = min(size, height - row), min(size, width - col)
                padded = np.zeros((3, size, size), np.uint8)
                padded[:, :h, :w] = data[:, row : row + h, col : col + w]
                tensor = torch.from_numpy(padded.astype(np.float32))[None] / predictor.scale
                tensor = (tensor - predictor.mean) / predictor.std
                probabilities = predictor.model(tensor).softmax(1).numpy()[0, :, :h, :w]
                scores[:, row : row + h, col : col + w] += probabilities * taper[:h, :w]
                counts[row : row + h, col : col + w] += taper[:h, :w]
    expected = scores / counts
    output, confidence = tmp_path / "output.tif", tmp_path / "confidence.tif"
    with patch(
        "finland_geospatial_ai.inference.predictor._score_buffers",
        wraps=predictor_module._score_buffers,
    ) as allocation:
        predictor.predict(
            source, output, tile_size=size, overlap=overlap, confidence_path=confidence
        )
    # Allocation regression: the blending buffer must never use the raster height.
    assert allocation.call_args.args[1] <= size
    assert allocation.call_args.args[2] == width
    with rasterio.open(output) as mask, rasterio.open(confidence) as certainty:
        valid = mask.read(1) != 255
        np.testing.assert_array_equal(mask.read(1)[valid], expected.argmax(0)[valid])
        np.testing.assert_allclose(certainty.read(1)[valid], expected.max(0)[valid], atol=1e-7)
        assert not valid[:2, :3].any()
        assert np.isnan(certainty.read(1)[:2, :3]).all()


def test_api_limits_decoded_pixels_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "image.tif"
    write_raster(source, np.ones((3, 32, 32), np.uint8), from_origin(400000, 6700000, 0.5, 0.5))
    monkeypatch.setenv("GEOAI_MAX_PIXELS", "1023")
    with patch.object(api, "get_predictor") as model:
        response = TestClient(api.app).post(
            "/predict", files={"file": ("image.tif", source.read_bytes())}
        )
    assert response.status_code == 413
    model.assert_not_called()


def test_api_busy_returns_retry_without_loading_model() -> None:
    with api._inference_lock:
        response = TestClient(api.app).post("/predict", files={"file": ("image.tif", b"data")})
    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"


def test_inference_failure_cleans_temporary_directory_and_hides_details(
    tmp_path: Path, tiny_model: Path
) -> None:
    source = tmp_path / "image.tif"
    write_raster(source, np.ones((3, 32, 32), np.uint8), from_origin(400000, 6700000, 0.5, 0.5))
    directory = tmp_path / "request"
    directory.mkdir()
    with (
        patch.object(api.tempfile, "mkdtemp", return_value=str(directory)),
        patch.object(api, "get_predictor", return_value=Predictor(tiny_model, device="cpu")),
        patch.object(Predictor, "predict", side_effect=RuntimeError("private server path")),
    ):
        response = TestClient(api.app).post(
            "/predict", files={"file": ("image.tif", source.read_bytes())}
        )
    assert response.status_code == 500
    assert "private" not in response.text
    assert not directory.exists()
    assert not api._inference_lock.locked()


def test_malformed_raster_does_not_load_model() -> None:
    with patch.object(api, "get_predictor") as model:
        response = TestClient(api.app).post(
            "/predict", files={"file": ("bad.tif", b"not a raster")}
        )
    assert response.status_code == 422
    model.assert_not_called()
