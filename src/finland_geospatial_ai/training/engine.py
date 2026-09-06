"""Train one declared experiment and log every decision to MLflow."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any, cast

import mlflow
import numpy as np
import torch
import yaml
from safetensors.torch import save_file

from finland_geospatial_ai.datasets.dataset import NLSPatchDataset, create_dataloader
from finland_geospatial_ai.datasets.manifest import DatasetManifest
from finland_geospatial_ai.evaluation.metrics import SegmentationMetrics
from finland_geospatial_ai.models import create_model
from finland_geospatial_ai.training.losses import segmentation_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "uncommitted"


def _evaluate_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader[Any],
    device: torch.device,
    num_classes: int,
    ignore_index: int,
) -> dict[str, object]:
    model.eval()
    metrics = SegmentationMetrics.empty(num_classes)
    with torch.inference_mode():
        for image, target, _metadata in loader:
            prediction = model(image.to(device)).argmax(dim=1).cpu().numpy()
            for predicted_item, target_item in zip(prediction, target.numpy(), strict=True):
                metrics.update(predicted_item, target_item, ignore_index)
    return metrics.compute()


def train(config_path: Path, manifest_path: Path, output_dir: Path, tracking_uri: str) -> Path:
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    training = config["training"]
    seed = int(training["seed"])
    training_boundary_ignore = bool(config.get("use_boundary_ignore", True))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = NLSPatchDataset(
        manifest_path,
        "train",
        augment=bool(training["augmentation"]["dihedral"]),
        use_boundary_ignore=training_boundary_ignore,
        seed=seed,
        brightness=float(training["augmentation"]["brightness"]),
        contrast=float(training["augmentation"]["contrast"]),
    )
    validation_dataset = NLSPatchDataset(
        manifest_path,
        "val",
        use_boundary_ignore=True,
        seed=seed,
    )
    train_loader = create_dataloader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        training=True,
        num_workers=int(training["num_workers"]),
        seed=seed,
    )
    validation_loader = create_dataloader(
        validation_dataset,
        batch_size=int(training["batch_size"]),
        training=False,
        num_workers=int(training["num_workers"]),
        seed=seed,
    )
    model = create_model(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"]) and device.type == "cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"{config['experiment_id']}.safetensors"
    metadata_path = output_dir / f"{config['experiment_id']}.json"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("finland-geospatial-ai-nls")
    best_iou = -1.0
    best_validation: dict[str, object] = {}
    stale_epochs = 0
    started = time.perf_counter()
    with mlflow.start_run(run_name=config["experiment_id"]) as run:
        mlflow.log_params(
            {
                "experiment_id": config["experiment_id"],
                "architecture": config["model"]["architecture"],
                "dataset_version": manifest.dataset_version,
                "split_version": manifest.split_version,
                "class_map_version": manifest.class_map_version,
                "native_pixel_size_m": manifest.native_pixel_size_m,
                "training_boundary_ignore": training_boundary_ignore,
                "validation_boundary_ignore": True,
                "seed": seed,
                "device": str(device),
                "train_patches": len(train_dataset),
                "val_patches": len(validation_dataset),
            }
        )
        mlflow.log_artifact(str(config_path), artifact_path="configuration")
        mlflow.log_artifact(str(manifest_path), artifact_path="dataset")
        for epoch in range(int(training["epochs"])):
            model.train()
            running_loss = 0.0
            for image, target, _metadata in train_loader:
                image, target = image.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=scaler.is_enabled(),
                ):
                    logits = model(image)
                    loss = segmentation_loss(
                        logits,
                        target,
                        manifest.ignore_index,
                        float(training["dice_weight"]),
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += float(loss.detach())
            validation = _evaluate_loader(
                model,
                validation_loader,
                device,
                len(manifest.classes),
                manifest.ignore_index,
            )
            epoch_loss = running_loss / max(len(train_loader), 1)
            validation_iou = cast(float, validation["mean_iou"])
            mlflow.log_metrics(
                {"train_loss": epoch_loss, "val_mean_iou": validation_iou}, step=epoch
            )
            print(f"epoch={epoch + 1} loss={epoch_loss:.4f} val_mIoU={validation_iou:.4f}")
            if validation_iou > best_iou:
                best_iou = validation_iou
                best_validation = validation
                stale_epochs = 0
                save_file(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    checkpoint,
                )
            else:
                stale_epochs += 1
                if stale_epochs >= int(training["patience"]):
                    break
        metadata = {
            "schema_version": "finland-geospatial-ai-model/v2",
            "experiment_id": config["experiment_id"],
            "question": config["question"],
            "model": config["model"],
            "experiment_config_sha256": _file_sha256(config_path),
            "dataset_manifest": manifest_path.as_posix(),
            "dataset_manifest_sha256": _file_sha256(manifest_path),
            "dataset_version": manifest.dataset_version,
            "split_version": manifest.split_version,
            "class_map_version": manifest.class_map_version,
            "normalization": manifest.normalization,
            "classes": [item.model_dump() for item in manifest.classes],
            "ignore_index": manifest.ignore_index,
            "native_pixel_size_m": manifest.native_pixel_size_m,
            "training_boundary_ignore": training_boundary_ignore,
            "validation_boundary_ignore": True,
            "crs": manifest.crs,
            "bands": manifest.bands,
            "patch_size_px": manifest.patch_size_px,
            "best_validation_mean_iou": best_iou,
            "best_validation_metrics": best_validation,
            "training_seconds": time.perf_counter() - started,
            "mlflow_run_id": run.info.run_id,
            "weights": checkpoint.name,
            "weights_sha256": _file_sha256(checkpoint),
            "training_commit": _git_commit(),
            "runtime": {
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "libraries": {
                    package: importlib.metadata.version(package)
                    for package in (
                        "torchvision",
                        "transformers",
                        "rasterio",
                        "geopandas",
                        "shapely",
                        "pyproj",
                        "mlflow",
                        "numpy",
                    )
                },
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(checkpoint), artifact_path="model")
        mlflow.log_artifact(str(metadata_path), artifact_path="model")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()
    print(train(args.config, args.manifest, args.output_dir, args.tracking_uri))


if __name__ == "__main__":
    main()
