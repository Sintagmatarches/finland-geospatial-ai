from __future__ import annotations

import json
from pathlib import Path

from finland_geoai.data.validate import boxes_overlap, validate_manifest


def test_touching_boxes_do_not_overlap() -> None:
    assert not boxes_overlap([0, 0, 10, 10], [10, 0, 20, 10])
    assert boxes_overlap([0, 0, 10, 10], [9, 0, 20, 10])


def test_held_out_aoi_manifest_passes(tiny_dataset: Path) -> None:
    manifest = json.loads(tiny_dataset.read_text())
    result = validate_manifest(manifest)
    assert result["ok"]


def test_train_test_spatial_overlap_is_blocking(tiny_dataset: Path) -> None:
    manifest = json.loads(tiny_dataset.read_text())
    manifest["patches"][2]["bounds"] = [40, 0, 120, 80]
    result = validate_manifest(manifest)
    assert not result["ok"]
    assert any("spatial overlap" in error for error in result["errors"])


def test_duplicate_patch_id_is_blocking(tiny_dataset: Path) -> None:
    manifest = json.loads(tiny_dataset.read_text())
    manifest["patches"][1]["id"] = manifest["patches"][0]["id"]
    assert not validate_manifest(manifest)["ok"]
