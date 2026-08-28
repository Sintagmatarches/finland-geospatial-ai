"""Blocking dataset contract and spatial leakage checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def boxes_overlap(a: list[float], b: list[float]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def validate_manifest(
    manifest: dict[str, Any], data_root: Path | None = None, verify_files: bool = False
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    patches = manifest.get("patches", [])
    ids = [patch["id"] for patch in patches]
    if len(ids) != len(set(ids)):
        errors.append("duplicate patch id")
    required_splits = {"train", "val", "test"}
    splits = {patch["split"] for patch in patches}
    if not required_splits.issubset(splits):
        errors.append(f"missing split(s): {sorted(required_splits - splits)}")
    aoi_splits: dict[str, set[str]] = {}
    for patch in patches:
        aoi_splits.setdefault(patch["aoi_id"], set()).add(patch["split"])
    leaked_aois = [aoi for aoi, assigned in aoi_splits.items() if len(assigned) != 1]
    if leaked_aois:
        errors.append(f"AOIs assigned to multiple splits: {leaked_aois}")
    for i, left in enumerate(patches):
        for right in patches[i + 1 :]:
            if left["split"] != right["split"] and boxes_overlap(left["bounds"], right["bounds"]):
                errors.append(f"spatial overlap: {left['id']} vs {right['id']}")
                break
        if errors and errors[-1].startswith("spatial overlap"):
            break
    all_classes = {str(key) for key in manifest["class_mapping"]["classes"]}
    for split, counts in manifest.get("class_counts_by_split", {}).items():
        missing = all_classes - set(counts)
        if missing:
            warnings.append(f"{split} has no pixels for classes {sorted(missing)}")
    if verify_files:
        if data_root is None:
            raise ValueError("data_root is required when verify_files=True")
        seen_hashes: Counter[str] = Counter()
        expected_size = int(manifest["patch_size"])
        for patch in patches:
            path = data_root / patch["path"]
            if not path.is_file():
                errors.append(f"missing raster patch: {path}")
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != patch["sha256"]:
                errors.append(f"checksum mismatch: {patch['id']}")
            seen_hashes[digest] += 1
            try:
                with np.load(path, allow_pickle=False) as sample:
                    expected_image_shape = (
                        len(manifest["bands"]),
                        expected_size,
                        expected_size,
                    )
                    if sample["image"].shape != expected_image_shape:
                        errors.append(f"wrong band count or dimensions: {patch['id']}")
                    if sample["mask"].shape != (expected_size, expected_size):
                        errors.append(f"mask dimensions mismatch: {patch['id']}")
                    if not np.any(sample["valid"]):
                        errors.append(f"empty patch: {patch['id']}")
            except (OSError, ValueError, KeyError) as exc:
                errors.append(f"unreadable patch {patch['id']}: {exc}")
        duplicate_content = [digest for digest, count in seen_hashes.items() if count > 1]
        if duplicate_content:
            errors.append(f"duplicate patch content hashes: {len(duplicate_content)}")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "patch_count": len(patches)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/v1"))
    parser.add_argument("--skip-files", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = validate_manifest(manifest, args.data_root, verify_files=not args.skip_files)
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
