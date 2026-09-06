"""Render evidence-backed experiment, error-analysis and model-card reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from finland_geospatial_ai.datasets.manifest import DatasetManifest


def _metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def generate_reports(
    selection_path: Path,
    test_metrics_path: Path,
    baseline_path: Path,
    manifest_path: Path,
    patch_metrics_path: Path,
    output_dir: Path,
) -> list[Path]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    test = json.loads(test_metrics_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    with patch_metrics_path.open(encoding="utf-8") as stream:
        patch_rows = list(csv.DictReader(stream))
    output_dir.mkdir(parents=True, exist_ok=True)
    global_metrics = test["global"]
    class_rows = [
        (
            item.name,
            global_metrics["per_class_iou"][item.id],
            global_metrics["per_class_dice"][item.id],
            global_metrics["per_class_precision"][item.id],
            global_metrics["per_class_recall"][item.id],
        )
        for item in manifest.classes
    ]
    experiment = [
        "# Generated experiment report",
        "",
        f"Selected experiment: `{selection['selected_experiment_id']}`.",
        "Selection used validation evidence only; sealed-test metrics were not consulted.",
        "",
        "## Candidate validation evidence",
        "",
        "| Experiment | MLflow run | Validation mIoU | Validation macro Dice |",
        "|---|---|---:|---:|",
    ]
    experiment.extend(
        (
            f"| {row['experiment_id']} | `{row['mlflow_run_id']}` | "
            f"{row['validation_mean_iou']:.4f} | "
            f"{row['validation_macro_dice']:.4f} |"
        )
        for row in selection["candidates"]
    )
    experiment.extend(
        [
            "",
            "## Sealed-test result",
            "",
            f"- Mean IoU: **{global_metrics['mean_iou']:.4f}**",
            f"- Macro Dice: **{global_metrics['macro_dice']:.4f}**",
            f"- Pixel accuracy: {global_metrics['pixel_accuracy']:.4f}",
            f"- Validation majority-baseline mIoU (context only): "
            f"{baseline['global']['mean_iou']:.4f}",
            f"- Boundary F1 (3 px): {test['boundary_f1_mean']:.4f}",
            f"- Expected calibration error: {test['expected_calibration_error']:.4f}",
            f"- Entropy/error Spearman correlation: {_metric(test['entropy_error_spearman'])}",
            "",
            "## Per-class result",
            "",
            "| Class | IoU | Dice | Precision | Recall |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    experiment.extend(
        f"| {name} | {iou:.4f} | {dice:.4f} | {precision:.4f} | {recall:.4f} |"
        for name, iou, dice, precision, recall in class_rows
    )
    experiment.extend(
        [
            "",
            "## Geographic generalisation",
            "",
            "| Held-out map sheet | mIoU | Mean confidence | Mean entropy |",
            "|---|---:|---:|---:|",
        ]
    )
    experiment.extend(
        (
            f"| {sheet} | {row['mean_iou']:.4f} | "
            f"{row['mean_confidence']:.4f} | {row['mean_entropy']:.4f} |"
        )
        for sheet, row in test["per_map_sheet"].items()
    )
    experiment.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "The evidence applies to the declared L324 blocks, 2025 RGB orthophotos, the recorded "
            "Topographic Database snapshot, and the five-class mapping. It does not establish "
            "Finland-wide performance. `other_land` is a heterogeneous residual class, not forest.",
        ]
    )
    experiment_path = output_dir / "experiment-report.md"
    experiment_path.write_text("\n".join(experiment) + "\n", encoding="utf-8")

    ordered = sorted(patch_rows, key=lambda row: float(row["mean_iou"]))
    error_lines = [
        "# Generated visual error analysis",
        "",
        "The table identifies measured cases used by the generated orthophoto/reference/"
        "prediction/error/uncertainty panels.",
        "",
        "| Patch | Map sheet | mIoU | Boundary F1 | Entropy | Confidence |",
        "|---|---|---:|---:|---:|---:|",
    ]
    error_lines.extend(
        (
            f"| `{row['patch_id']}` | {row['map_sheet']} | "
            f"{float(row['mean_iou']):.4f} | {float(row['boundary_f1']):.4f} | "
            f"{float(row['mean_entropy']):.4f} | "
            f"{float(row['mean_confidence']):.4f} |"
        )
        for row in [*ordered[:3], ordered[len(ordered) // 2], *ordered[-3:]]
    )
    error_lines.extend(
        [
            "",
            f"Boundary-pixel accuracy: {_metric(test['boundary_pixel_accuracy'])}; "
            f"interior-pixel accuracy: {_metric(test['interior_pixel_accuracy'])}.",
            f"The boundary band is {manifest.boundary_ignore_width_m:g} m wide on each side of "
            "class transitions and is excluded from primary IoU/Dice metrics.",
            "Higher boundary error is consistent with positional uncertainty only if that measured "
            "difference is present; inspect the generated panels before assigning a cause.",
        ]
    )
    error_path = output_dir / "error-analysis.md"
    error_path.write_text("\n".join(error_lines) + "\n", encoding="utf-8")

    model_card = [
        "# Generated final model card",
        "",
        f"- Model: `{selection['selected_experiment_id']}`",
        f"- Dataset: `{manifest.dataset_version}`",
        f"- Split: `{manifest.split_version}`",
        f"- Input: RGB, {manifest.native_pixel_size_m} m, {manifest.crs}",
        f"- Patch: {manifest.patch_size_px} px ({manifest.patch_ground_size_m:g} m)",
        f"- Final test mIoU: {global_metrics['mean_iou']:.4f}",
        f"- Final test macro Dice: {global_metrics['macro_dice']:.4f}",
        f"- ECE: {test['expected_calibration_error']:.4f}",
        "",
        "## Intended use",
        "",
        "Analyst-assisted research and bounded offline mapping on compatible NLS-style imagery.",
        "Not for cadastral, safety-critical, enforcement, or legal land-use decisions.",
        "",
        "## Known limitations",
        "",
        "Labels inherit vector generalisation and temporal mismatch. The study covers a bounded "
        "south-west Finland region. Confidence does not guarantee correctness, and `other_land` "
        "contains several visually distinct surfaces.",
    ]
    model_path = output_dir / "model-card.md"
    model_path.write_text("\n".join(model_card) + "\n", encoding="utf-8")
    return [experiment_path, error_path, model_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--test-metrics", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--patch-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports"))
    args = parser.parse_args()
    for path in generate_reports(
        args.selection,
        args.test_metrics,
        args.baseline,
        args.manifest,
        args.patch_metrics,
        args.output_dir,
    ):
        print(path)


if __name__ == "__main__":
    main()
