"""One-shot held-out evaluation, validation-fit calibration and deterministic error analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch.nn import functional as F
from torch.utils.data import DataLoader

from finland_geoai.data.dataset import GeoPatchDataset
from finland_geoai.evaluation.metrics import (
    confusion_matrix,
    expected_calibration_error,
    metrics_from_confusion,
)
from finland_geoai.models.factory import create_model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"Corrupt or incompatible checkpoint: {path}") from exc
    required = {"artifact_schema", "state_dict", "model_config", "dataset_sha256", "classes"}
    missing = required - set(checkpoint)
    if missing or checkpoint["artifact_schema"] != "finland-geoai-model/v1":
        raise ValueError(f"Incompatible model metadata; missing={sorted(missing)}")
    return checkpoint


def _collect(
    model: torch.nn.Module, dataset: GeoPatchDataset, batch_size: int = 16
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    patch_rows: list[dict[str, Any]] = []
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    model.eval()
    offset = 0
    with torch.no_grad():
        for images, masks, metadata in loader:
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)
            entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1)
            entropy /= math.log(logits.shape[1])
            for index in range(images.shape[0]):
                valid = masks[index] != 255
                matrix = confusion_matrix(predictions[index], masks[index], logits.shape[1])
                metrics = metrics_from_confusion(matrix)
                patch_rows.append(
                    {
                        "index": offset + index,
                        "id": metadata["id"][index],
                        "aoi_id": metadata["aoi_id"][index],
                        "miou": metrics["miou"],
                        "mean_entropy": float(entropy[index][valid].mean()),
                    }
                )
            offset += images.shape[0]
            valid = masks != 255
            all_logits.append(logits.permute(0, 2, 3, 1)[valid].cpu())
            all_targets.append(masks[valid].cpu())
    return patch_rows, torch.cat(all_logits), torch.cat(all_targets)


def _fit_temperature(logits: torch.Tensor, targets: torch.Tensor, seed: int = 42) -> float:
    generator = torch.Generator().manual_seed(seed)
    if logits.shape[0] > 250_000:
        indices = torch.randperm(logits.shape[0], generator=generator)[:250_000]
        logits = logits[indices]
        targets = targets[indices]
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_temperature.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 10.0))


def _calibration(
    logits: torch.Tensor, targets: torch.Tensor, temperature: float
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    raw = torch.softmax(logits, dim=1).numpy()
    calibrated = torch.softmax(logits / temperature, dim=1).numpy()
    target_array = targets.numpy()
    raw_ece, raw_bins = expected_calibration_error(raw, target_array)
    calibrated_ece, calibrated_bins = expected_calibration_error(calibrated, target_array)
    return (
        {
            "temperature": temperature,
            "fit_split": "val",
            "raw_ece": raw_ece,
            "calibrated_ece": calibrated_ece,
            "raw_nll": float(F.cross_entropy(logits, targets)),
            "calibrated_nll": float(F.cross_entropy(logits / temperature, targets)),
            "raw_bins": raw_bins,
            "calibrated_bins": calibrated_bins,
        },
        raw,
        calibrated,
    )


def _risk_coverage(probabilities: np.ndarray, targets: np.ndarray) -> list[dict[str, float]]:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    order = np.argsort(-confidence)
    rows = []
    for coverage in np.linspace(0.1, 1.0, 10):
        count = max(1, int(len(order) * coverage))
        selected = order[:count]
        rows.append(
            {
                "coverage": float(coverage),
                "error_rate": float(1.0 - correct[selected].mean()),
                "minimum_confidence": float(confidence[selected].min()),
            }
        )
    return rows


def _plot_confusion(matrix: list[list[int]], names: list[str], path: Path) -> None:
    values = np.asarray(matrix, dtype=np.float64)
    normalized = values / np.maximum(values.sum(axis=1, keepdims=True), 1)
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(names)), names, rotation=35, ha="right")
    axis.set_yticks(range(len(names)), names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Reference")
    axis.set_title("Held-out Oulu confusion matrix (row-normalized)")
    for row in range(len(names)):
        for col in range(len(names)):
            axis.text(col, row, f"{normalized[row, col]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_reliability(calibration: dict[str, Any], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="perfect")
    for key, label, color in (
        ("raw_bins", "raw", "#d62728"),
        ("calibrated_bins", "temperature-scaled", "#1f77b4"),
    ):
        rows = calibration[key]
        axis.plot(
            [row["confidence"] for row in rows],
            [row["accuracy"] for row in rows],
            marker="o",
            label=label,
            color=color,
        )
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Pixel accuracy")
    axis.set_title("Held-out Oulu reliability")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_risk_coverage(rows: list[dict[str, float]], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(
        [row["coverage"] for row in rows],
        [row["error_rate"] for row in rows],
        marker="o",
        color="#5b3cc4",
    )
    axis.set(xlabel="Coverage (most confident pixels retained)", ylabel="Error rate")
    axis.set_title("Does uncertainty identify difficult pixels?")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _rgb(raw_image: np.ndarray) -> np.ndarray:
    rgb = raw_image[[2, 1, 0]].transpose(1, 2, 0).astype(np.float32)
    low, high = np.percentile(rgb, (2, 98))
    return np.clip((rgb - low) / max(high - low, 1), 0, 1)


def _plot_cases(
    model: torch.nn.Module,
    dataset: GeoPatchDataset,
    rows: list[dict[str, Any]],
    names: list[str],
    colors: list[list[int]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    by_iou = sorted(rows, key=lambda row: row["miou"])
    selections = {
        "worst_iou": by_iou[0],
        "median_iou": by_iou[len(by_iou) // 2],
        "best_iou": by_iou[-1],
        "highest_entropy": max(rows, key=lambda row: row["mean_entropy"]),
    }
    cmap = ListedColormap(np.asarray(colors) / 255.0)
    output: list[dict[str, Any]] = []
    model.eval()
    for criterion, row in selections.items():
        image, target, _metadata = dataset[row["index"]]
        with torch.no_grad():
            logits = model(image.unsqueeze(0))[0]
            probabilities = torch.softmax(logits, dim=0)
            prediction = logits.argmax(dim=0)
            entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=0)
            entropy /= math.log(logits.shape[0])
        record = dataset.records[row["index"]]
        with np.load(dataset.data_root / record["path"], allow_pickle=False) as sample:
            raw_image = sample["image"]
        target_np = target.numpy()
        prediction_np = prediction.numpy()
        error = (prediction_np != target_np) & (target_np != 255)
        fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
        axes[0].imshow(_rgb(raw_image))
        axes[0].set_title("Sentinel-2 RGB")
        axes[1].imshow(target_np, cmap=cmap, vmin=0, vmax=len(names) - 1)
        axes[1].set_title("WorldCover reference")
        axes[2].imshow(prediction_np, cmap=cmap, vmin=0, vmax=len(names) - 1)
        axes[2].set_title("Prediction")
        axes[3].imshow(_rgb(raw_image))
        axes[3].imshow(np.ma.masked_where(~error, error), cmap="Reds", alpha=0.7)
        axes[3].set_title("Error overlay")
        uncertainty = axes[4].imshow(entropy.numpy(), cmap="magma", vmin=0, vmax=1)
        axes[4].set_title("Predictive entropy")
        for axis in axes:
            axis.axis("off")
        fig.colorbar(uncertainty, ax=axes[4], fraction=0.046)
        fig.suptitle(f"{criterion}: {row['id']} · patch mIoU {row['miou']:.3f}")
        fig.tight_layout()
        path = output_dir / f"error-case-{criterion}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        output.append({"criterion": criterion, **row, "figure": str(path).replace("\\", "/")})
    return output


def evaluate(checkpoint_path: Path, manifest_path: Path, figures_dir: Path) -> dict[str, Any]:
    lock_path = Path("artifacts/test-evaluation-lock.json")
    if lock_path.exists():
        raise RuntimeError(
            "Protected test set has already been evaluated. Version the split and document "
            "a defect before another final evaluation."
        )
    checkpoint = _load_checkpoint(checkpoint_path)
    manifest_digest = _sha256(manifest_path)
    if checkpoint["dataset_sha256"] != manifest_digest:
        raise ValueError("Checkpoint dataset digest does not match the supplied manifest")
    model_config = checkpoint["model_config"]
    model = create_model(
        model_config["model"],
        model_config["in_channels"],
        model_config["num_classes"],
        model_config["base_channels"],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    val_dataset = GeoPatchDataset(
        manifest_path, "val", augment=False, band_indices=checkpoint["band_indices"]
    )
    test_dataset = GeoPatchDataset(
        manifest_path, "test", augment=False, band_indices=checkpoint["band_indices"]
    )
    _val_rows, val_logits, val_targets = _collect(model, val_dataset)
    temperature = _fit_temperature(val_logits, val_targets)
    # Test logits are collected exactly once after model and calibration selection are frozen.
    test_rows, test_logits, test_targets = _collect(model, test_dataset)
    test_matrix = confusion_matrix(test_logits.argmax(dim=1), test_targets, 6)
    test_metrics = metrics_from_confusion(test_matrix)
    calibration, _raw, calibrated = _calibration(test_logits, test_targets, temperature)
    risk_coverage = _risk_coverage(calibrated, test_targets.numpy())
    names = [checkpoint["classes"][str(index)]["name"] for index in range(6)]
    colors = [checkpoint["classes"][str(index)]["color"] for index in range(6)]
    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_confusion(test_metrics["confusion_matrix"], names, figures_dir / "confusion-matrix.png")
    _plot_reliability(calibration, figures_dir / "reliability.png")
    _plot_risk_coverage(risk_coverage, figures_dir / "risk-coverage.png")
    selected_cases = _plot_cases(model, test_dataset, test_rows, names, colors, figures_dir)

    final_checkpoint = dict(checkpoint)
    final_checkpoint["temperature"] = temperature
    final_checkpoint["test_metrics"] = test_metrics
    final_checkpoint["test_evaluated_at"] = datetime.now(UTC).isoformat()
    final_path = Path("artifacts/final-model.pt")
    torch.save(final_checkpoint, final_path)
    result = {
        "evaluation_schema": "finland-geoai-evaluation/v1",
        "evaluated_at": final_checkpoint["test_evaluated_at"],
        "selection": {
            "checkpoint": str(checkpoint_path).replace("\\", "/"),
            "validation_miou": checkpoint["validation_metrics"]["miou"],
            "best_epoch": checkpoint["best_epoch"],
            "rule": "highest validation mIoU across declared E1-E4; test unopened",
        },
        "test_geography": "Oulu held-out AOI",
        "test_metrics": test_metrics,
        "calibration": calibration,
        "risk_coverage": risk_coverage,
        "error_cases": selected_cases,
        "final_model": str(final_path).replace("\\", "/"),
        "final_model_sha256": _sha256(final_path),
    }
    metrics_path = Path("artifacts/test-metrics.json")
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lock = {
        "split_version": checkpoint["split_version"],
        "checkpoint_sha256": _sha256(checkpoint_path),
        "test_metrics_sha256": _sha256(metrics_path),
        "evaluated_at": result["evaluated_at"],
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/dataset-manifest-v1.json")
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--figures", action="store_true", help="Retained for Makefile ergonomics")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.artifact, args.manifest, args.figures_dir), indent=2))


if __name__ == "__main__":
    main()
