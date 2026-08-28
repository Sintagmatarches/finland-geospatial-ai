# Model card — Finland GeoAI U-Net v1

## Model and intended use

The final artifact is a compact four-band U-Net trained from scratch to demonstrate a complete semantic-segmentation workflow on real Finnish satellite imagery. Intended use is bounded technical evaluation, education and portfolio review: inspect a 640 m square Sentinel-2-like patch and receive a six-class land-cover mask plus uncertainty indicators.

It is not intended for cadastral decisions, forestry inventory, environmental enforcement, navigation, disaster response, property valuation, change detection or safety-critical use. It must not be interpreted as an authoritative land-cover map.

## Inputs and outputs

- Input: 4-band blue/green/red/NIR GeoTIFF, band order fixed, 64×64 pixels, EPSG:3067, 10 m resolution.
- Normalization: Sentinel reflectance / 10,000 and committed train-only channel statistics.
- Output: forest, shrub/grass, cropland, built-up, other natural or water per valid pixel; no-data is 255.
- Artifact: `artifacts/final-model.pt`, 1.9 MB, SHA-256 `4e103ad9683b5eb5e4831145101155e02c130826a00cf441cdbc900d19f495dc`.
- Architecture: 488,230-parameter compact U-Net; four channels; GroupNorm; trained from scratch.

## Training procedure

Training used 269 real patches from Helsinki and Lahti. Tampere's 266 patches were used for early model/checkpoint selection; Oulu's 315 patches were unopened until the selection rule was frozen. The explicit PyTorch loop performs Dataset/DataLoader batching, `train`/`eval`, `zero_grad`, forward, weighted cross-entropy, backward, AdamW update, deterministic seeds and best-checkpoint selection. Dihedral flips/90° rotations are paired between image and mask and are train-only.

Recorded environment: Python 3.12, PyTorch 2.7.1+cpu, rasterio 1.4.3 / GDAL 3.9.3, MLflow 3.3.2. Training completed on CPU; the repository does not claim GPU training.

## Selection and evaluation

E4 won the declared validation matrix with 0.4574 Tampere mIoU at epoch 11. The Oulu test set was evaluated once and locked.

| Metric | Held-out Oulu |
| --- | ---: |
| mIoU | 0.3982 |
| Macro Dice/F1 | 0.4743 |
| Pixel accuracy | 0.8056 |
| Forest IoU | 0.7598 |
| Shrub/grass IoU | 0.0909 |
| Cropland IoU | 0.0821 |
| Built-up IoU | 0.5633 |
| Other natural IoU | 0.0000 |
| Water IoU | 0.8933 |

Pixel accuracy is secondary because forest/water dominate. Per-class precision, recall, Dice, support and the full confusion matrix are in `artifacts/test-metrics.json`.

## Calibration and uncertainty

Max-softmax and normalized predictive entropy are uncertainty signals, not Bayesian uncertainty. A single temperature was fit on a deterministic validation subsample only. It did not transfer geographically: Oulu ECE worsened from 0.0406 to 0.0886 and NLL from 0.665 to 0.706. The API explicitly discloses this failure and does not call its scores calibrated probabilities.

Uncertainty still ranks some difficulty: retaining the most-confident 70% of Oulu pixels reduced error from 19.44% at full coverage to 10.25%. It is not perfectly monotonic at low coverage, so uncertainty is useful for triage, not a safety guarantee.

![Reliability before and after validation-fit temperature scaling](figures/reliability.png?v=20260828-geospatial-ai-v1)

## Known failure modes

- Rare and geographically shifted classes are weak; `other_natural` has zero test true positives.
- Shrub/grass is mostly confused with forest and built-up; cropland is confused with forest/shrub.
- Boundaries inherit 10 m mixed pixels and WorldCover label noise.
- Urban vegetation and shoreline transitions create false positives and boundary errors.
- A confident pixel can still be wrong; the reliability plot exposes this.
- Inputs from other years, seasons, sensors, resolutions, CRSs or band conventions are out of distribution.
- Oulu is one held-out geography, not proof of Finland-wide or international generalization.

## Ethical and operational considerations

Land-cover errors can affect decisions about people, property and habitats even when no personal data is used. Users must retain source imagery, uncertainty, date and provenance; validate against authoritative domain data; and never automate consequential decisions from this artifact alone. Production use would require broader seasonal/geographic sampling, independent human labels, repeated spatial folds, class redesign, monitoring, model registry controls and a documented update policy.
