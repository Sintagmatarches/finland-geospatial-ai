"""Generate visual evidence directly from evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap

from finland_geospatial_ai.datasets.manifest import DatasetManifest
from finland_geospatial_ai.inference import Predictor


def _save_metrics_figures(metrics: dict[str, Any], output_dir: Path) -> None:
    classes = metrics["classes"]
    global_metrics = metrics["global"]
    plt.figure(figsize=(8, 4.5))
    plt.bar(classes, global_metrics["per_class_iou"], color="#2f7d6d")
    plt.ylim(0, 1)
    plt.ylabel("IoU")
    plt.title("Sealed-test IoU by class")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "per-class-iou.png", dpi=180)
    plt.close()

    confusion = np.asarray(global_metrics["confusion_matrix"], dtype=float)
    confusion /= np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(confusion, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="Fraction of true class")
    plt.xticks(range(len(classes)), classes, rotation=35, ha="right")
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("Predicted")
    plt.ylabel("Reference")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion-matrix.png", dpi=180)
    plt.close()

    bins = metrics["calibration_bins"]
    observed = [row["accuracy"] for row in bins if row["count"]]
    expected = [row["confidence"] for row in bins if row["count"]]
    plt.figure(figsize=(5.5, 5.5))
    plt.plot([0, 1], [0, 1], "--", color="0.5", label="Perfect")
    plt.plot(expected, observed, marker="o", label="Model")
    plt.xlabel("Mean confidence")
    plt.ylabel("Pixel accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "reliability.png", dpi=180)
    plt.close()

    risk = metrics["risk_coverage"]
    plt.figure(figsize=(6, 4.5))
    plt.plot([row["coverage"] for row in risk], [row["error_rate"] for row in risk], marker="o")
    plt.xlabel("Retained pixel coverage")
    plt.ylabel("Error rate")
    plt.tight_layout()
    plt.savefig(output_dir / "risk-coverage.png", dpi=180)
    plt.close()

    geography = metrics["per_map_sheet"]
    plt.figure(figsize=(6, 4.5))
    plt.bar(geography.keys(), [row["mean_iou"] for row in geography.values()], color="#486f9e")
    plt.ylim(0, 1)
    plt.ylabel("Mean IoU")
    plt.xlabel("Held-out NLS map sheet")
    plt.tight_layout()
    plt.savefig(output_dir / "geographic-generalization.png", dpi=180)
    plt.close()


def generate_report(
    metrics_path: Path,
    patches_path: Path,
    manifest_path: Path,
    metadata_path: Path,
    output_dir: Path,
) -> Path:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_metrics_figures(metrics, output_dir)
    with patches_path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    ordered = sorted(rows, key=lambda row: float(row["mean_iou"]))
    examples = {
        "worst": ordered[0],
        "median": ordered[len(ordered) // 2],
        "best": ordered[-1],
        "highest-entropy": max(rows, key=lambda row: float(row["mean_entropy"])),
    }
    by_id = {record.id: record for record in manifest.patches}
    root = manifest_path.parents[2]
    predictor = Predictor(metadata_path, device="cpu")
    colors = np.asarray([item.color for item in manifest.classes], dtype=float) / 255
    cmap = ListedColormap(colors)
    cmap.set_bad("#d9d9d9")
    with tempfile.TemporaryDirectory(prefix="geoai-report-") as temporary:
        temporary_path = Path(temporary)
        for name, row in examples.items():
            record = by_id[row["patch_id"]]
            prediction_path = temporary_path / f"{name}-prediction.tif"
            confidence_path = temporary_path / f"{name}-confidence.tif"
            predictor.predict(
                root / record.image_path, prediction_path, confidence_path=confidence_path
            )
            with rasterio.open(root / record.image_path) as source:
                image = np.moveaxis(source.read(), 0, -1).astype(float)
                image /= np.iinfo(source.dtypes[0]).max
            with rasterio.open(root / record.mask_path) as source:
                raw_truth = source.read(1)
            with rasterio.open(root / record.boundary_path) as source:
                boundary = source.read(1).astype(bool)
            with rasterio.open(prediction_path) as source:
                prediction = source.read(1)
            with rasterio.open(confidence_path) as source:
                confidence = source.read(1)
            evaluated = (~boundary) & (raw_truth != manifest.ignore_index)
            truth = np.ma.masked_where(~evaluated, raw_truth)
            error = (prediction != raw_truth) & evaluated
            figure, axes = plt.subplots(1, 5, figsize=(18, 4))
            axes[0].imshow(np.clip(image, 0, 1))
            axes[0].set_title("0.5 m orthophoto")
            axes[1].imshow(truth, cmap=cmap, vmin=0, vmax=len(colors) - 1)
            axes[1].set_title("NLS reference (boundary masked)")
            axes[2].imshow(prediction, cmap=cmap, vmin=0, vmax=len(colors) - 1)
            axes[2].set_title("Prediction")
            axes[3].imshow(error, cmap="Reds", vmin=0, vmax=1)
            axes[3].set_title("Error")
            axes[4].imshow(1 - confidence, cmap="magma", vmin=0, vmax=1)
            axes[4].set_title("Uncertainty")
            for axis in axes:
                axis.axis("off")
            figure.suptitle(f"{name}: {record.id}")
            figure.tight_layout()
            figure.savefig(output_dir / f"error-case-{name}.png", dpi=180)
            plt.close(figure)
    index = output_dir / "visual-evidence.json"
    index.write_text(
        json.dumps(
            {
                "metrics": metrics_path.as_posix(),
                "examples": examples,
                "generated_files": sorted(path.name for path in output_dir.glob("*.png")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--patches", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/figures"))
    args = parser.parse_args()
    print(
        generate_report(args.metrics, args.patches, args.manifest, args.metadata, args.output_dir)
    )


if __name__ == "__main__":
    main()
