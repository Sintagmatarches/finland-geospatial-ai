from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import yaml
from conftest import write_raster
from rasterio.transform import from_origin
from shapely.geometry import box

from finland_geospatial_ai.datasets.build import build_dataset
from finland_geospatial_ai.datasets.report import generate_dataset_report
from finland_geospatial_ai.datasets.validate import validate_dataset
from finland_geospatial_ai.evaluation.report import generate_report
from finland_geospatial_ai.evaluation.run import evaluate, majority_baseline
from finland_geospatial_ai.evaluation.selection import select_model
from finland_geospatial_ai.evaluation.summarize import generate_reports
from finland_geospatial_ai.training.engine import train


def test_source_to_manifest_pipeline(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw" / "tiny"
    origins = {"S1": 1000, "S2": 2000, "S3": 3000}
    water = []
    for sheet, left in origins.items():
        transform = from_origin(left, 7000000, 0.5, 0.5)
        write_raster(
            raw_dir / f"orthophoto-{sheet}.tif",
            np.full((3, 32, 32), 100, dtype=np.uint8),
            transform,
        )
        water.append(box(left, 6999992, left + 4, 7000000))
    raw_dir.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"id": [1, 2, 3]}, geometry=water, crs="EPSG:3067").to_file(
        raw_dir / "topodb.gpkg", layer="lake_part", driver="GPKG"
    )
    config = {
        "dataset_version": "tiny-v1",
        "split_version": "tiny-splits-v1",
        "class_map_version": "tiny-classes-v1",
        "seed": 3067,
        "source": {
            "licensor": "National Land Survey of Finland",
            "licence": "CC BY 4.0",
            "orthophoto": {
                "native_pixel_size_m": 0.5,
                "bands": ["red", "green", "blue"],
                "acquisition_year": 2025,
            },
            "topographic_database": {
                "bbox": [900, 6999900, 3100, 7000100],
                "snapshot_date": "2026-01-01",
            },
        },
        "patch": {
            "size_px": 16,
            "ground_size_m": 8.0,
            "stride_px": 16,
            "minimum_valid_fraction": 1.0,
            "boundary_ignore_width_m": 1.0,
            "ignore_index": 255,
            "samples": {"train": 1, "val": 1, "test": 1},
            "rare_class_min_fraction": 0.01,
        },
        "splits": {
            "train": {"map_sheets": ["S1"], "edge_buffer_m": 0},
            "val": {"map_sheets": ["S2"], "edge_buffer_m": 0},
            "test": {"map_sheets": ["S3"], "edge_buffer_m": 0},
        },
        "classes": [
            {
                "id": 0,
                "name": "other_land",
                "color": [1, 2, 3],
                "source_layers": [],
                "rule": "residual",
            },
            {
                "id": 1,
                "name": "water",
                "color": [4, 5, 6],
                "source_layers": ["lake_part"],
                "rule": "water",
            },
        ],
        "precedence": ["other_land", "water"],
        "paths": {
            "raw_dir": "data/raw/tiny",
            "processed_dir": "data/processed/tiny",
            "manifest": "data/manifests/tiny-v1.json",
            "split_manifest": "data/splits/tiny-splits-v1.geojson",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest_path = build_dataset(config_path, tmp_path)
    result = validate_dataset(manifest_path)
    report = generate_dataset_report(manifest_path, tmp_path / "reports")
    assert result["patches"] == 3
    assert result["pixel_size_m"] == 0.5
    assert json.loads(report.read_text(encoding="utf-8"))["duplicate_check"] == "passed"


def test_training_baseline_and_sealed_evaluation(tmp_path: Path, tiny_manifest: Path) -> None:
    config = {
        "experiment_id": "integration-unet",
        "question": "Can the integration model complete one epoch?",
        "model": {"architecture": "unet", "in_channels": 3, "num_classes": 2, "base_channels": 4},
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "patience": 1,
            "num_workers": 0,
            "seed": 3067,
            "amp": False,
            "loss": "cross_entropy_dice",
            "dice_weight": 0.5,
            "augmentation": {"dihedral": True, "brightness": 0.0, "contrast": 0.0},
        },
    }
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    model_dir = tmp_path / "models"
    metadata = train(config_path, tiny_manifest, model_dir, (tmp_path / "mlruns").as_uri())
    selection = select_model([metadata], tmp_path / "model-selection.json")
    evaluation_dir = tmp_path / "evaluation"
    metrics = evaluate(metadata, tiny_manifest, evaluation_dir)
    baseline = majority_baseline(tiny_manifest, evaluation_dir / "baseline.json")
    visual_index = generate_report(
        metrics,
        evaluation_dir / "integration-unet-test-patches.csv",
        tiny_manifest,
        metadata,
        tmp_path / "figures",
    )
    generated_reports = generate_reports(
        selection,
        metrics,
        baseline,
        tiny_manifest,
        evaluation_dir / "integration-unet-test-patches.csv",
        tmp_path / "generated-reports",
    )
    assert json.loads(metrics.read_text(encoding="utf-8"))["sealed_test"] is True
    with pytest.raises(RuntimeError, match="already locked"):
        evaluate(metadata, tiny_manifest, evaluation_dir)
    assert (evaluation_dir / "final-model-artifact.json").is_file()
    baseline_evidence = json.loads(baseline.read_text(encoding="utf-8"))
    assert baseline_evidence["experiment_id"] == "B0-majority-class"
    assert baseline_evidence["split"] == "val"
    assert baseline_evidence["sealed_test"] is False
    assert json.loads(selection.read_text(encoding="utf-8"))["test_metrics_consulted"] is False
    assert json.loads(visual_index.read_text(encoding="utf-8"))["generated_files"]
    assert all(path.is_file() for path in generated_reports)
