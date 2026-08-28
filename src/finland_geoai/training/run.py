"""Readable PyTorch training loop with validation-only selection and real MLflow tracking."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from finland_geoai.data.dataset import GeoPatchDataset
from finland_geoai.evaluation.metrics import confusion_matrix, metrics_from_confusion
from finland_geoai.models.factory import create_model
from finland_geoai.training.losses import WeightedCrossEntropy, WeightedCrossEntropyDice


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _class_weights(manifest: dict[str, Any], num_classes: int) -> torch.Tensor:
    counts = manifest["class_counts_by_split"]["train"]
    frequencies = torch.tensor([float(counts.get(str(i), 0)) for i in range(num_classes)])
    if torch.any(frequencies == 0):
        raise ValueError("A training class has zero support")
    weights = 1.0 / torch.sqrt(frequencies / frequencies.sum())
    return weights / weights.mean()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    num_classes: int,
    optimizer: AdamW | None,
) -> tuple[float, dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_batches = 0
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for images, masks, _metadata in loader:
        images = images.to(device)
        masks = masks.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_function(logits, masks)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_batches += 1
        matrix += confusion_matrix(logits.argmax(dim=1).cpu(), masks.cpu(), num_classes)
    return total_loss / total_batches, metrics_from_confusion(matrix)


def train(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    device = select_device(config["device"])
    manifest_path = Path(config["dataset_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    band_indices = config.get("band_indices")
    train_dataset = GeoPatchDataset(
        manifest_path,
        "train",
        augment=config["augmentation"] == "dihedral",
        band_indices=band_indices,
        seed=int(config["seed"]),
    )
    validation_dataset = GeoPatchDataset(
        manifest_path, "val", augment=False, band_indices=band_indices
    )
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
    )
    model = create_model(
        config["model"],
        int(config["in_channels"]),
        int(config["num_classes"]),
        int(config["base_channels"]),
    ).to(device)
    weights = _class_weights(manifest, int(config["num_classes"])).to(device)
    if config["loss"] == "weighted_ce_dice":
        loss_function = WeightedCrossEntropyDice(weights, int(manifest["ignore_index"])).to(device)
    elif config["loss"] == "weighted_ce":
        loss_function = WeightedCrossEntropy(weights, int(manifest["ignore_index"])).to(device)
    else:
        raise ValueError(f"Unknown loss: {config['loss']}")
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_miou = -1.0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    dataset_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
    mlflow.set_experiment(config["mlflow_experiment"])
    with mlflow.start_run(run_name=f"{config['experiment_id']}-{config['model']}") as run:
        mlflow.log_params(
            {
                **{
                    key: value
                    for key, value in config.items()
                    if isinstance(value, str | int | float)
                },
                "bands": ",".join(manifest["bands"][i] for i in train_dataset.band_indices),
                "dataset_version": manifest["schema_version"],
                "split_version": manifest["split_version"],
                "device_resolved": str(device),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            }
        )
        mlflow.log_artifact(str(config_path), artifact_path="config")
        for epoch in range(1, int(config["epochs"]) + 1):
            train_loss, train_metrics = _run_epoch(
                model, train_loader, loss_function, device, int(config["num_classes"]), optimizer
            )
            with torch.no_grad():
                val_loss, val_metrics = _run_epoch(
                    model,
                    validation_loader,
                    loss_function,
                    device,
                    int(config["num_classes"]),
                    None,
                )
            row = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_miou": train_metrics["miou"],
                "val_loss": val_loss,
                "val_miou": val_metrics["miou"],
                "val_macro_dice": val_metrics["macro_dice"],
            }
            history.append(row)
            mlflow.log_metrics(
                {key: value for key, value in row.items() if key != "epoch"}, step=epoch
            )
            print(json.dumps(row))
            if val_metrics["miou"] > best_miou + 1e-6:
                best_miou = val_metrics["miou"]
                stale_epochs = 0
                torch.save(
                    {
                        "artifact_schema": "finland-geoai-model/v1",
                        "state_dict": model.state_dict(),
                        "model_config": {
                            key: config[key]
                            for key in ("model", "in_channels", "num_classes", "base_channels")
                        },
                        "band_indices": train_dataset.band_indices,
                        "normalization": manifest["normalization"],
                        "classes": manifest["class_mapping"]["classes"],
                        "ignore_index": manifest["ignore_index"],
                        "dataset_sha256": dataset_digest,
                        "dataset_version": manifest["schema_version"],
                        "split_version": manifest["split_version"],
                        "best_epoch": epoch,
                        "validation_metrics": val_metrics,
                        "torch_version": torch.__version__,
                        "device_trained": str(device),
                        "mlflow_run_id": run.info.run_id,
                    },
                    output_path,
                )
            else:
                stale_epochs += 1
                if stale_epochs >= int(config["patience"]):
                    break
        history_path = output_path.with_suffix(".history.json")
        history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(history_path), artifact_path="training")
        mlflow.log_artifact(str(output_path), artifact_path="checkpoint")
        result = {
            "experiment_id": config["experiment_id"],
            "question": config["question"],
            "model": config["model"],
            "bands": [manifest["bands"][i] for i in train_dataset.band_indices],
            "best_validation_miou": best_miou,
            "epochs_completed": len(history),
            "mlflow_run_id": run.info.run_id,
            "checkpoint": str(output_path).replace("\\", "/"),
            "device": str(device),
        }
    summary_path = Path("artifacts/experiment-runs.json")
    summaries = json.loads(summary_path.read_text()) if summary_path.exists() else []
    summaries = [entry for entry in summaries if entry["experiment_id"] != result["experiment_id"]]
    summaries.append(result)
    ordered_summaries = sorted(summaries, key=lambda x: x["experiment_id"])
    summary_path.write_text(json.dumps(ordered_summaries, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(train(args.config), indent=2))


if __name__ == "__main__":
    main()
