from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from finland_geospatial_ai.datasets.dataset import (
    NLSPatchDataset,
    apply_dihedral,
    create_dataloader,
)
from finland_geospatial_ai.datasets.manifest import DatasetManifest
from finland_geospatial_ai.datasets.validate import validate_dataset


def test_manifest_and_files_validate(tiny_manifest: Path) -> None:
    result = validate_dataset(tiny_manifest)
    assert result["status"] == "valid"
    assert result["splits"] == {"train": 1, "val": 1, "test": 1}


def test_dataset_reads_normalized_aligned_patch(tiny_manifest: Path) -> None:
    image, mask, metadata = NLSPatchDataset(tiny_manifest, "train")[0]
    assert image.shape == (3, 32, 32)
    assert mask.shape == (32, 32)
    assert image.dtype == torch.float32
    assert mask.dtype == torch.int64
    assert set(mask.unique().tolist()) == {0, 1}
    assert metadata["source_map_sheet"] == "sheet-train"


def test_validation_dataloader_is_deterministic(tiny_manifest: Path) -> None:
    dataset = NLSPatchDataset(tiny_manifest, "val")
    first = next(
        iter(create_dataloader(dataset, batch_size=1, training=False, num_workers=0, seed=3067))
    )
    second = next(
        iter(create_dataloader(dataset, batch_size=1, training=False, num_workers=0, seed=3067))
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_dihedral_keeps_image_mask_correspondence() -> None:
    base = torch.arange(16).reshape(4, 4)
    image, mask, boundary = apply_dihedral(base[None].float(), base, base.bool(), 5)
    assert torch.equal(image[0].long(), mask)
    assert torch.equal(boundary, mask.bool())


def test_manifest_rejects_duplicate_class_ids(tiny_manifest: Path) -> None:
    data = json.loads(tiny_manifest.read_text(encoding="utf-8"))
    data["classes"][1]["id"] = 0
    with pytest.raises(ValueError, match="unique"):
        DatasetManifest.model_validate(data)


def test_validator_detects_hash_tampering(tiny_manifest: Path) -> None:
    data = json.loads(tiny_manifest.read_text(encoding="utf-8"))
    data["patches"][0]["image_sha256"] = "f" * 64
    tiny_manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_dataset(tiny_manifest)


def test_validator_detects_duplicate_and_spatial_leakage(tiny_manifest: Path) -> None:
    data = json.loads(tiny_manifest.read_text(encoding="utf-8"))
    data["patches"][1]["bounds"] = data["patches"][0]["bounds"]
    data["patches"].append(data["patches"][0])
    data["patch_counts"]["train"] = 2
    tiny_manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate patch id") as error:
        validate_dataset(tiny_manifest)
    assert "spatial leakage" in str(error.value)


def test_validator_rejects_undeclared_map_sheet(tiny_manifest: Path) -> None:
    data = json.loads(tiny_manifest.read_text(encoding="utf-8"))
    data["patches"][0]["source_map_sheet"] = "unknown-sheet"
    tiny_manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="no matching split/map-sheet declaration"):
        validate_dataset(tiny_manifest)
