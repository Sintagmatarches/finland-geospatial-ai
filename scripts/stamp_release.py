"""Stamp source provenance into the manifest and compact release artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("commit", help="Full source commit used to build/train the release")
    args = parser.parse_args()
    invalid_character = any(
        character not in "0123456789abcdef" for character in args.commit
    )
    if len(args.commit) != 40 or invalid_character:
        raise ValueError("commit must be a full lowercase SHA-1")

    manifest_path = Path("artifacts/dataset-manifest-v1.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["code_commit"] not in {"uncommitted", args.commit}:
        raise ValueError("Refusing to replace an existing different dataset code commit")
    manifest["code_commit"] = args.commit
    write_json(manifest_path, manifest)

    model_path = Path("artifacts/final-model.pt")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if checkpoint["artifact_schema"] != "finland-geoai-model/v1":
        raise ValueError("Unexpected model artifact schema")
    checkpoint["dataset_sha256"] = sha256(manifest_path)
    checkpoint["training_code_commit"] = args.commit
    torch.save(checkpoint, model_path)

    metrics_path = Path("artifacts/test-metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["final_model_sha256"] = sha256(model_path)
    write_json(metrics_path, metrics)

    lock_path = Path("artifacts/test-evaluation-lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["test_metrics_sha256"] = sha256(metrics_path)
    write_json(lock_path, lock)
    print(
        json.dumps(
            {
                "code_commit": args.commit,
                "dataset_sha256": checkpoint["dataset_sha256"],
                "final_model_sha256": metrics["final_model_sha256"],
                "test_metrics_sha256": lock["test_metrics_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
