# Finland Geospatial AI

[![CI](https://github.com/Sintagmatarches/finland-geospatial-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Sintagmatarches/finland-geospatial-ai/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)

Native-resolution semantic segmentation of Finnish aerial imagery using official data from the
National Land Survey of Finland (NLS). The project treats geospatial provenance, label uncertainty,
spatial leakage, held-out geography, calibration, and deployable inference as first-class concerns.

The imagery stays at its native **0.5 metre ground sampling distance** in **ETRS-TM35FIN
(EPSG:3067)**. No Sentinel-2 proxy, visual demo tiles, resampled screenshots, or synthetic labels
are used as evidence.

## Published v2 result

The selected pretrained SegFormer-B0 reached **0.7344 validation mIoU** and **0.6652 mIoU / 0.7742
macro Dice** on the single sealed-test evaluation. The two held-out test sheets scored 0.6447 and
0.6784 mIoU. Water was strongest at 0.9636 IoU; open-natural land was weakest at 0.3300 IoU.
The full test artifact, per-patch metrics, selection record, calibration bins, and visual cases are
committed rather than reduced to a headline score.

## Research question

How reliably can open NLS Topographic Database polygons supervise five visually meaningful classes
on 0.5 m RGB orthophotos, and how much does a pretrained SegFormer-B0 improve held-out geography
over a compact U-Net?

The class vocabulary is intentionally honest about the source data:

| ID | Class | Label source |
|---:|---|---|
| 0 | `other_land` | Residual valid terrestrial surface; includes forest and roads not represented by a precise polygon class |
| 1 | `water` | Hydrographic area polygons |
| 2 | `cultivated_land` | Cultivated-land polygons |
| 3 | `built_structures` | Building and precise built-surface polygons; coarse urban polygons are excluded |
| 4 | `open_natural` | Meadow, marsh, open woodland, peatland, rock, sand and related polygons |

Pixels within 1.5 m of mapped class transitions are ignored during the primary experiment. This
reduces false certainty caused by vector generalisation, acquisition-date mismatch, and imperfect
polygon-to-pixel alignment. A declared ablation trains the same U-Net without that exclusion.

## Spatial design

The study uses five 6 × 6 km NLS map sheets in the L324 region. A 512 m interior buffer is applied
before non-overlapping 256 × 256 patches are sampled.

| Split | Map sheets | Patches | Ground footprint per patch |
|---|---|---:|---:|
| Train | `L3244C`, `L3244H` | 384 | 128 × 128 m |
| Validation | `L3243D` | 96 | 128 × 128 m |
| Sealed test | `L3243F`, `L3243H` | 96 | 128 × 128 m |

The test sheet is never used for model selection, early stopping, normalisation, or class balancing.
Every generated patch stores its CRS, affine transform, bounds, map sheet, acquisition year, class
counts, and SHA-256 hashes in a strict versioned manifest.

## Reproduce

Python 3.12 and an [NLS API key](https://www.maanmittauslaitos.fi/en/rajapinnat/api-avaimen-ohje)
are required for source acquisition. Keep the key in the environment only.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

export NLS_API_KEY='...'
geoai-acquire --config configs/data/nls_l324.yaml --describe-only
geoai-acquire --config configs/data/nls_l324.yaml
geoai-data --config configs/data/nls_l324.yaml
geoai-validate data/manifests/nls-l324-2025-v1.json
geoai-dataset-report data/manifests/nls-l324-2025-v1.json

geoai-evaluate --manifest data/manifests/nls-l324-2025-v1.json --majority-baseline
geoai-train --config configs/models/unet.yaml --manifest data/manifests/nls-l324-2025-v1.json
geoai-train --config configs/models/unet_no_boundary_ignore.yaml --manifest data/manifests/nls-l324-2025-v1.json
geoai-train --config configs/models/segformer_b0.yaml --manifest data/manifests/nls-l324-2025-v1.json
geoai-evaluate --metadata artifacts/models/E1-unet-rgb.json --manifest data/manifests/nls-l324-2025-v1.json --split val
geoai-evaluate --metadata artifacts/models/E2-unet-no-boundary-ignore.json --manifest data/manifests/nls-l324-2025-v1.json --split val
geoai-evaluate --metadata artifacts/models/E3-segformer-b0-rgb.json --manifest data/manifests/nls-l324-2025-v1.json --split val
geoai-select artifacts/models/E1-unet-rgb.json artifacts/models/E2-unet-no-boundary-ignore.json artifacts/models/E3-segformer-b0-rgb.json

# The published result is already sealed in artifacts/evaluation/test-evaluation-lock.json.
# A clean reproduction uses a separate output directory and opens that copy once.
geoai-evaluate --metadata artifacts/models/selected-model.json \
  --manifest data/manifests/nls-l324-2025-v1.json --split test --device cpu \
  --output-dir artifacts/reproduction-evaluation
geoai-report --metrics artifacts/reproduction-evaluation/E3-segformer-b0-rgb-test-metrics.json \
  --patches artifacts/reproduction-evaluation/E3-segformer-b0-rgb-test-patches.csv \
  --manifest data/manifests/nls-l324-2025-v1.json \
  --metadata artifacts/models/selected-model.json
```

MLflow uses `sqlite:///mlflow.db` by default. Start its UI with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Inference

Inference rejects rasters that violate the trained geospatial contract: exactly three RGB bands,
0.5 m square pixels, and an EPSG:3067 horizontal CRS. Large rasters use overlap-tiled probability
blending, and the output GeoTIFF preserves CRS, transform, dimensions, and model provenance tags.

```bash
geoai-infer input.jp2 prediction.tif \
  --model artifacts/models/E3-segformer-b0-rgb.json \
  --tile-size 256 --overlap 64
```

Run the API with Docker after mounting the model directory:

```bash
docker compose up --build
curl -F 'file=@input.jp2' http://localhost:8000/predict --output prediction-package.zip
```

The API package contains `prediction.tif`, `confidence.tif`, and `summary.json` with the class
legend, pixel fractions, square-metre areas, dimensions, model version, and confidence summary.

## Repository map

```text
configs/                     versioned data and experiment declarations
src/finland_geospatial_ai/   acquisition, rasterisation, datasets, models, evaluation, inference
tests/                       synthetic geospatial contract and API tests
reports/                     architecture, dataset and model documentation
artifacts/                   published manifests, MLflow export, models and evaluation evidence
```

## Data and licences

NLS states that orthophotos use ETRS-TM35FIN, are typically 0.5 m, and are updated on a roughly
three-year cycle. The Topographic Database is a current national vector dataset available as
GeoPackage. Source data are licensed under CC BY 4.0; attribute the **National Land Survey of
Finland**, the dataset names, and delivery dates. Code in this repository is MIT licensed.

- [NLS Orthophotos product description](https://www.maanmittauslaitos.fi/en/maps-and-spatial-data/datasets-and-interfaces/product-descriptions/orthophotos)
- [NLS Topographic Database product description](https://www.maanmittauslaitos.fi/en/maps-and-spatial-data/datasets-and-interfaces/product-descriptions/topographic-database)
- [NLS OGC API Processes technical description](https://www.maanmittauslaitos.fi/paikkatiedon-tiedostopalvelu/tekninen-kuvaus)
- [NLS v2 dataset card](reports/v2/dataset-card.md)
- [NLS v2 architecture](reports/v2/architecture.md)
- [NLS v2 model card](reports/v2/model-card.md)
- [Generated experiment report](reports/v2/generated/experiment-report.md)
- [Visual error analysis](reports/v2/generated/error-analysis.md)
- [Calibration and uncertainty analysis](reports/v2/calibration.md)

## Immutable v1 track

The original Sentinel-2 / ESA WorldCover experiment remains intact under `src/finland_geoai`, its
original configs, reports, figures, checkpoints, manifest, metrics, and test lock. Its held-out Oulu
result was 0.3982 mIoU with a six-class, 10 m, four-band contract. It is historical evidence and was
not retrained or tuned during the NLS migration. The v1 and v2 values are not a direct leaderboard:
their sensors, spatial resolution, labels, classes, regions, patch sizes, and model-selection pools
differ. See the [scientific comparison](reports/v2/v1-v2-comparison.md).

## Limitations

- Topographic polygons are cartographic objects, not human pixel annotations.
- `other_land` is heterogeneous; it must not be reported as a forest class.
- Labels and imagery are temporally close, not necessarily captured on the same date.
- Evidence from four Turku-region sheets does not establish Finland-wide generalisation.
- The API is intended for bounded batch inference, not unrestricted public uploads.
