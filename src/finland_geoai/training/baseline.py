"""Training-distribution majority baseline evaluated on validation only."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from finland_geoai.evaluation.metrics import metrics_from_confusion


def main() -> None:
    manifest_path = Path("artifacts/dataset-manifest-v1.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_counts = manifest["class_counts_by_split"]["train"]
    majority = max(range(6), key=lambda index: int(train_counts[str(index)]))
    val_counts = manifest["class_counts_by_split"]["val"]
    matrix = torch.zeros((6, 6), dtype=torch.int64)
    for target, count in val_counts.items():
        matrix[int(target), majority] = int(count)
    metrics = metrics_from_confusion(matrix)
    result = {
        "experiment_id": "E0",
        "question": "Do neural models outperform a training-majority sanity baseline?",
        "model": "majority_class",
        "majority_class": majority,
        "bands": [],
        "best_validation_miou": metrics["miou"],
        "validation_metrics": metrics,
        "checkpoint": None,
        "device": "none",
    }
    summary_path = Path("artifacts/experiment-runs.json")
    summaries = json.loads(summary_path.read_text()) if summary_path.exists() else []
    summaries = [entry for entry in summaries if entry["experiment_id"] != "E0"]
    summaries.append(result)
    summary_path.write_text(
        json.dumps(sorted(summaries, key=lambda item: item["experiment_id"]), indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
