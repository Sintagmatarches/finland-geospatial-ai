# Model card: NLS high-resolution v2

## Model and selection

`E3-segformer-b0-rgb` is a SegFormer-B0 with an ImageNet-pretrained `nvidia/mit-b0` encoder and a
five-class segmentation head. It was selected strictly by validation mean IoU after three declared
runs on the same dataset and spatial split.

| Experiment | Validation mIoU | Validation macro Dice | Training time |
|---|---:|---:|---:|
| E1 compact U-Net | 0.5671 | 0.6521 | 409.9 s |
| E2 U-Net, no boundary ignore | 0.6678 | 0.7750 | 979.1 s |
| E3 pretrained SegFormer-B0 | **0.7344** | **0.8305** | 414.4 s |

The selection artifact records all three MLflow run IDs, checkpoint hashes, the shared manifest hash,
and `test_metrics_consulted=false`. The selected checkpoint SHA-256 is
`06a4c34133e5c44ebdd68695f56b5cdaeacc77721b873b39b6e079a8464eaf50`.

## Single sealed-test result

The final test was evaluated once after selection and locked by a hash-bound v2 test-lock artifact.

| Metric | Result |
|---|---:|
| Mean IoU | **0.6652** |
| Macro Dice | **0.7742** |
| Pixel accuracy | 0.8024 |
| Boundary F1, 3 px / 1.5 m tolerance | 0.1777 |
| Expected calibration error | 0.0474 |
| Entropy/error Spearman correlation | 0.3104 |
| CPU throughput, 256 px patches | 1.167 MP/s |

| Class | IoU | Dice | Recall |
|---|---:|---:|---:|
| other_land | 0.7268 | 0.8418 | 0.8785 |
| water | 0.9636 | 0.9814 | 0.9723 |
| cultivated_land | 0.8312 | 0.9078 | 0.9335 |
| built_structures | 0.4743 | 0.6434 | 0.5592 |
| open_natural | 0.3300 | 0.4963 | 0.4262 |

## Intended use

The model produces a five-class categorical raster from three-band NLS RGB orthophotos at 0.5 m in
EPSG:3067. Intended uses are reproducible research, analyst-assisted mapping, and bounded offline
batch inference. It is unsuitable for cadastral, safety-critical, enforcement, navigation, or legal
land-use decisions.

## Limits

The study covers five L324 map sheets, not Finland. Labels inherit cartographic generalisation,
omissions, and 2025 imagery versus 2026 vector timing differences. `other_land` is heterogeneous and
must not be interpreted as forest. Built-structure and open-natural recall are the main measured weak
points. Pixel calibration statistics contain spatial dependence, so ECE describes this sealed sample
and is not a confidence guarantee for new regions.
