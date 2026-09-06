"""Select a final model using validation evidence only."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def select_model(metadata_paths: list[Path], output_path: Path) -> Path:
    if not metadata_paths:
        raise ValueError("At least one candidate model is required")
    candidates: list[dict[str, Any]] = []
    dataset_versions: set[str] = set()
    split_versions: set[str] = set()
    manifest_hashes: set[str] = set()
    class_map_versions: set[str] = set()
    for path in metadata_paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != "finland-geospatial-ai-model/v2":
            raise ValueError(f"Unsupported model metadata schema: {path}")
        weights = path.parent / metadata["weights"]
        if not weights.is_file():
            raise ValueError(f"Missing candidate weights: {weights}")
        actual_weights_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
        if actual_weights_hash != metadata.get("weights_sha256"):
            raise ValueError(f"Candidate weights hash mismatch: {weights}")
        dataset_versions.add(metadata["dataset_version"])
        split_versions.add(metadata["split_version"])
        manifest_hashes.add(metadata["dataset_manifest_sha256"])
        class_map_versions.add(metadata["class_map_version"])
        candidates.append(
            {
                "experiment_id": metadata["experiment_id"],
                "mlflow_run_id": metadata["mlflow_run_id"],
                "metadata_path": path.as_posix(),
                "validation_mean_iou": metadata["best_validation_mean_iou"],
                "validation_macro_dice": metadata["best_validation_metrics"]["macro_dice"],
            }
        )
    if any(len(values) != 1 for values in (
        dataset_versions,
        split_versions,
        manifest_hashes,
        class_map_versions,
    )):
        raise ValueError("Candidate models do not share one dataset, split and class-map contract")
    selected = max(
        candidates, key=lambda item: (item["validation_mean_iou"], item["validation_macro_dice"])
    )
    artifact = {
        "schema_version": "finland-geospatial-ai-model-selection/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_version": next(iter(dataset_versions)),
        "split_version": next(iter(split_versions)),
        "dataset_manifest_sha256": next(iter(manifest_hashes)),
        "class_map_version": next(iter(class_map_versions)),
        "primary_metric": "validation_mean_iou",
        "tie_breaker": "validation_macro_dice",
        "test_metrics_consulted": False,
        "candidates": candidates,
        "selected_experiment_id": selected["experiment_id"],
        "selected_metadata_path": selected["metadata_path"],
        "packaged_metadata_path": (
            output_path.parent / "models" / "selected-model.json"
        ).as_posix(),
        "reason": "Highest validation mean IoU; macro Dice is the declared tie-breaker.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    selected_metadata = json.loads(Path(selected["metadata_path"]).read_text(encoding="utf-8"))
    selected_metadata["selection_artifact"] = output_path.as_posix()
    packaged_path = output_path.parent / "models" / "selected-model.json"
    packaged_path.parent.mkdir(parents=True, exist_ok=True)
    packaged_path.write_text(json.dumps(selected_metadata, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model-selection.json"))
    args = parser.parse_args()
    print(select_model(args.metadata, args.output))


if __name__ == "__main__":
    main()
