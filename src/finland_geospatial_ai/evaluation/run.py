"""Evaluate a frozen model once on sealed held-out geography."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import spearmanr

from finland_geospatial_ai.datasets.dataset import NLSPatchDataset, create_dataloader
from finland_geospatial_ai.datasets.manifest import DatasetManifest
from finland_geospatial_ai.evaluation.metrics import SegmentationMetrics, boundary_f1
from finland_geospatial_ai.models import create_model


def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1)


def _safe_boolean_mean(values: list[np.ndarray]) -> float | None:
    combined = np.concatenate(values)
    return float(combined.mean()) if combined.size else None


def evaluate(
    metadata_path: Path,
    manifest_path: Path,
    output_dir: Path,
    split: str = "test",
    device_name: str = "auto",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    test_lock = output_dir / "test-evaluation-lock.json"
    if split == "test" and test_lock.exists():
        existing = json.loads(test_lock.read_text(encoding="utf-8"))
        raise RuntimeError(
            "Sealed test evaluation is already locked by "
            f"{existing.get('experiment_id', 'an earlier run')}"
        )
    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if metadata["dataset_version"] != manifest.dataset_version:
        raise ValueError("Model and dataset versions differ")
    weights = metadata_path.parent / metadata["weights"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation requested but CUDA is unavailable")
    device = torch.device(device_name)
    model = create_model(metadata["model"])
    model.load_state_dict(load_file(weights), strict=True)
    model.to(device).eval()
    dataset = NLSPatchDataset(manifest_path, split, use_boundary_ignore=False)
    loader = create_dataloader(dataset, batch_size=1, training=False, num_workers=0, seed=3067)
    global_metrics = SegmentationMetrics.empty(len(manifest.classes))
    by_sheet: dict[str, SegmentationMetrics] = defaultdict(
        lambda: SegmentationMetrics.empty(len(manifest.classes))
    )
    patch_rows = []
    confidences: list[np.ndarray] = []
    entropies: list[np.ndarray] = []
    correctness: list[np.ndarray] = []
    target_classes: list[np.ndarray] = []
    boundary_correct: list[np.ndarray] = []
    interior_correct: list[np.ndarray] = []
    sheet_confidence: dict[str, list[np.ndarray]] = defaultdict(list)
    sheet_entropy: dict[str, list[np.ndarray]] = defaultdict(list)
    sheet_targets: dict[str, list[np.ndarray]] = defaultdict(list)
    elapsed = 0.0
    with torch.inference_mode():
        for image, target, batch_metadata in loader:
            image = image.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            logits = model(image)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - started
            probabilities = logits.softmax(dim=1)
            prediction = probabilities.argmax(dim=1).cpu().numpy()[0]
            raw_truth = target.numpy()[0]
            boundary = batch_metadata["boundary"].numpy()[0].astype(bool)
            truth = raw_truth.copy()
            truth[boundary] = manifest.ignore_index
            patch_metrics = SegmentationMetrics.empty(len(manifest.classes))
            patch_metrics.update(prediction, truth, manifest.ignore_index)
            global_metrics.update(prediction, truth, manifest.ignore_index)
            sheet = batch_metadata["source_map_sheet"][0]
            by_sheet[sheet].update(prediction, truth, manifest.ignore_index)
            valid = truth != manifest.ignore_index
            confidence = probabilities.max(dim=1).values.cpu().numpy()[0]
            entropy = _entropy(probabilities).cpu().numpy()[0]
            confidences.append(confidence[valid])
            entropies.append(entropy[valid])
            correctness.append((prediction[valid] == truth[valid]).astype(np.uint8))
            target_classes.append(truth[valid])
            sheet_confidence[sheet].append(confidence[valid])
            sheet_entropy[sheet].append(entropy[valid])
            sheet_targets[sheet].append(truth[valid])
            raw_valid = raw_truth != manifest.ignore_index
            boundary_correct.append(
                prediction[boundary & raw_valid] == raw_truth[boundary & raw_valid]
            )
            interior_correct.append(
                prediction[(~boundary) & raw_valid] == raw_truth[(~boundary) & raw_valid]
            )
            computed = patch_metrics.compute()
            patch_rows.append(
                {
                    "patch_id": batch_metadata["id"][0],
                    "map_sheet": sheet,
                    "mean_iou": computed["mean_iou"],
                    "pixel_accuracy": computed["pixel_accuracy"],
                    "boundary_f1": boundary_f1(
                        prediction, raw_truth, manifest.ignore_index, tolerance=3
                    ),
                    "mean_entropy": float(entropy[valid].mean()),
                    "mean_confidence": float(confidence[valid].mean()),
                }
            )
    confidence_values = np.concatenate(confidences)
    entropy_values = np.concatenate(entropies)
    correct_values = np.concatenate(correctness)
    target_values = np.concatenate(target_classes)
    calibration = []
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:], strict=True):
        selected = (confidence_values > lower) & (confidence_values <= upper)
        count = int(selected.sum())
        accuracy = float(correct_values[selected].mean()) if count else 0.0
        confidence = float(confidence_values[selected].mean()) if count else 0.0
        ece += count / len(confidence_values) * abs(accuracy - confidence)
        calibration.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "accuracy": accuracy,
                "confidence": confidence,
            }
        )
    order = np.argsort(-confidence_values)
    risk_coverage = []
    for coverage in np.linspace(0.1, 1.0, 10):
        retained = order[: max(1, round(len(order) * coverage))]
        risk_coverage.append(
            {"coverage": float(coverage), "error_rate": float(1 - correct_values[retained].mean())}
        )
    per_class_uncertainty = {}
    for item in manifest.classes:
        selected = target_values == item.id
        per_class_uncertainty[item.name] = {
            "pixels": int(selected.sum()),
            "accuracy": float(correct_values[selected].mean()),
            "mean_confidence": float(confidence_values[selected].mean()),
            "mean_entropy": float(entropy_values[selected].mean()),
        }
    entropy_error_correlation = spearmanr(
        entropy_values, 1 - correct_values, nan_policy="omit"
    ).statistic
    correlation = (
        float(entropy_error_correlation) if np.isfinite(entropy_error_correlation) else None
    )
    result = {
        "schema_version": "finland-geospatial-ai-evaluation/v2",
        "experiment_id": metadata["experiment_id"],
        "dataset_version": manifest.dataset_version,
        "split_version": manifest.split_version,
        "split": split,
        "sealed_test": split == "test",
        "classes": [item.name for item in manifest.classes],
        "global": global_metrics.compute(),
        "boundary_f1_mean": float(np.mean([row["boundary_f1"] for row in patch_rows])),
        "expected_calibration_error": float(ece),
        "calibration_bins": calibration,
        "risk_coverage": risk_coverage,
        "mean_entropy": float(np.mean([row["mean_entropy"] for row in patch_rows])),
        "entropy_error_spearman": correlation,
        "per_class_uncertainty": per_class_uncertainty,
        "boundary_pixel_accuracy": _safe_boolean_mean(boundary_correct),
        "interior_pixel_accuracy": _safe_boolean_mean(interior_correct),
        "per_map_sheet": {
            sheet: {
                **value.compute(),
                "mean_confidence": float(np.concatenate(sheet_confidence[sheet]).mean()),
                "mean_entropy": float(np.concatenate(sheet_entropy[sheet]).mean()),
                "class_pixel_counts": {
                    item.name: int((np.concatenate(sheet_targets[sheet]) == item.id).sum())
                    for item in manifest.classes
                },
            }
            for sheet, value in by_sheet.items()
        },
        "runtime": {
            "device": str(device),
            "patches": len(dataset),
            "seconds": elapsed,
            "milliseconds_per_patch": 1000 * elapsed / len(dataset),
            "megapixels_per_second": len(dataset) * manifest.patch_size_px**2 / 1e6 / elapsed,
        },
    }
    result_path = output_dir / f"{metadata['experiment_id']}-{split}-metrics.json"
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if split == "test":
        final_artifact = {
            **metadata,
            "schema_version": "finland-geospatial-ai-final-model/v2",
            "final_test_evaluation": result,
            "evaluation_artifact": result_path.as_posix(),
        }
        (output_dir / "final-model-artifact.json").write_text(
            json.dumps(final_artifact, indent=2, allow_nan=False), encoding="utf-8"
        )
        lock = {
            "schema_version": "finland-geospatial-ai-test-lock/v2",
            "created_at": datetime.now(UTC).isoformat(),
            "experiment_id": metadata["experiment_id"],
            "dataset_version": manifest.dataset_version,
            "split_version": manifest.split_version,
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "evaluation_artifact": result_path.as_posix(),
            "evaluation_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "sealed_test_evaluations": 1,
        }
        test_lock.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    with (output_dir / f"{metadata['experiment_id']}-{split}-patches.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=patch_rows[0].keys())
        writer.writeheader()
        writer.writerows(patch_rows)
    return result_path


def majority_baseline(manifest_path: Path, output_path: Path, split: str = "val") -> Path:
    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported baseline split: {split}")
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    train_counts = manifest.class_counts_by_split["train"]
    majority = int(max(train_counts, key=lambda key: train_counts[key]))
    metrics = SegmentationMetrics.empty(len(manifest.classes))
    dataset = NLSPatchDataset(manifest_path, split, use_boundary_ignore=True)
    for index in range(len(dataset)):
        _image, target, _metadata = dataset[index]
        truth = target.numpy()
        metrics.update(np.full_like(truth, majority), truth, manifest.ignore_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": "finland-geospatial-ai-evaluation/v2",
                "experiment_id": "B0-majority-class",
                "majority_class_id": majority,
                "dataset_version": manifest.dataset_version,
                "split": split,
                "sealed_test": split == "test",
                "global": metrics.compute(),
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--majority-baseline", action="store_true")
    args = parser.parse_args()
    if args.majority_baseline:
        print(
            majority_baseline(
                args.manifest,
                args.output_dir / f"B0-majority-class-{args.split}-metrics.json",
                args.split,
            )
        )
    elif args.metadata:
        print(evaluate(args.metadata, args.manifest, args.output_dir, args.split, args.device))
    else:
        parser.error("--metadata is required unless --majority-baseline is used")


if __name__ == "__main__":
    main()
