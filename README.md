# Finland Geospatial AI

**Semantic segmentation of real Finnish Earth Observation imagery with PyTorch.** Four-band Sentinel-2 Level-2A imagery is aligned to ESA WorldCover at 10 m, split by held-out geography, trained on CPU, evaluated once on Oulu, and served as a validated GeoTIFF API.

[![CI](https://github.com/Sintagmatarches/finland-geospatial-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Sintagmatarches/finland-geospatial-ai/actions/workflows/ci.yml)

![Deterministically selected median held-out prediction with error and entropy](reports/figures/error-case-median_iou.png?v=20260828-geospatial-ai-v1)

| Evidence | Verified result |
| --- | --- |
| Real data | 850 non-overlapping 64×64 patches; Sentinel-2 L2A + WorldCover 2021 v200 |
| Spatial validation | Helsinki + Lahti train; Tampere validation; Oulu protected test |
| Models actually trained | majority baseline, compact U-Net RGB, compact U-Net RGB+NIR, TinyDeepLabV3 RGB+NIR, loss ablation |
| Final held-out result | **0.3982 mIoU**, **0.4743 macro Dice**, 0.8056 pixel accuracy on Oulu |
| Experiment finding | NIR improved U-Net validation mIoU 0.4037 → 0.4512; TinyDeepLab underperformed at 0.3867 |
| Uncertainty | most-confident 70% of test pixels had 10.25% error vs 19.44% at full coverage |
| Calibration | temperature scaling failed geographically: ECE 0.0406 → 0.0886; it is not claimed as an improvement |
| Delivery | real local MLflow runs, 1.9 MB versioned model, CPU FastAPI, georeferenced GeoTIFF output, Docker, CI |

## What is being predicted?

Each 640 m × 640 m patch is segmented into forest, shrub/grass, cropland, built-up, other natural land, and water. Source classes are never silently discarded: the exact mapping and class support live in the versioned [dataset manifest](artifacts/dataset-manifest-v1.json) and [dataset card](reports/dataset-card.md).

The result is deliberately uneven. Test IoU is 0.893 for water, 0.760 for forest and 0.563 for built-up, but only 0.091 for shrub/grass, 0.082 for cropland and 0.000 for rare `other_natural`. This is a bounded portfolio experiment—not a Finland-wide operational land-cover product.

![Held-out class confusions](reports/figures/confusion-matrix.png?v=20260828-geospatial-ai-v1)

## Reproduce

Python 3.12 is the recorded environment. Raw and processed imagery is not committed; the builder reads bounded windows from public COGs.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
make data
make validate-data
python -m finland_geoai.training.baseline
python -m finland_geoai.training.run --config configs/unet_rgb.yaml
python -m finland_geoai.training.run --config configs/unet.yaml
python -m finland_geoai.training.run --config configs/deeplab.yaml
python -m finland_geoai.training.run --config configs/unet_ce.yaml
```

The protected test evaluation has a committed lock. It is not a routine command: change it only for a documented dataset defect and version the split. The committed metrics and final artifact are the result of the one declared evaluation.

```bash
mlflow ui --backend-store-uri ./mlruns
make test
make serve
docker build -f docker/Dockerfile -t finland-geospatial-ai .
docker run --rm -p 8000:8000 finland-geospatial-ai
```

`POST /predict` accepts only an uploaded, at-most-5-MiB GeoTIFF with exactly four bands in blue/green/red/NIR order, 64×64 pixels, EPSG:3067 and 10 m square pixels. It returns a color PNG, a categorical GeoTIFF preserving CRS/transform, the legend, class fractions, model version and entropy summary. It never downloads a user-supplied URL.

## Evidence map

- [Experiment report](reports/experiment-report.md) — question → configuration → result → conclusion, including negative results.
- [Model card](reports/model-card.md) — intended use, geography, procedure, metrics, calibration and failure modes.
- [Dataset card](reports/dataset-card.md) — sources, licensing, AOIs, class support, splits and limitations.
- [Architecture](reports/architecture.md) — STAC/COG acquisition, alignment, training, evaluation and serving.
- [Inference contract](reports/inference.md) — accepted raster contract, outputs and safety boundaries.
- [Portfolio and market audit](reports/audit-and-market-fit.md) — which capability gap this project actually closes.
- [Test metrics](artifacts/test-metrics.json) and [experiment runs](artifacts/experiment-runs.json) — machine-readable measured evidence.

![Validation-only training curves](reports/figures/training-curves.png?v=20260828-geospatial-ai-v1)

## Repository structure

```text
src/finland_geoai/
  data/          STAC/COG build, Dataset and dataset validation
  geo/           explicit target-grid, CRS and resampling operations
  models/        compact U-Net and TinyDeepLabV3-style network
  training/      losses, seeds, DataLoader and explicit PyTorch loop
  evaluation/    metrics, calibration, error selection and figures
  inference/     CPU model service and FastAPI boundary
configs/         versioned dataset and experiment questions
artifacts/       manifest, run summaries, test lock/metrics, compact final model
reports/         evidence, model/data cards and generated figures
tests/           raster, leakage, augmentation, metrics, DL and API tests
```

Source code is MIT licensed. Sentinel-2 and WorldCover retain their source licences and required attribution; see the dataset card.
