# Generated experiment report

Selected experiment: `E3-segformer-b0-rgb`.
Selection used validation evidence only; sealed-test metrics were not consulted.

## Candidate validation evidence

| Experiment | MLflow run | Validation mIoU | Validation macro Dice |
|---|---|---:|---:|
| E1-unet-rgb | `9031810c9f4b458a93671bfc97172255` | 0.5671 | 0.6521 |
| E2-unet-no-boundary-ignore | `431b47f55c5e482687361e3a6c43e204` | 0.6678 | 0.7750 |
| E3-segformer-b0-rgb | `1da67b56fa65438ca5020b17be02f78b` | 0.7344 | 0.8305 |

## Sealed-test result

- Mean IoU: **0.6652**
- Macro Dice: **0.7742**
- Pixel accuracy: 0.8024
- Validation majority-baseline mIoU (context only): 0.0782
- Boundary F1 (3 px): 0.1777
- Expected calibration error: 0.0474
- Entropy/error Spearman correlation: 0.3104

## Per-class result

| Class | IoU | Dice | Precision | Recall |
|---|---:|---:|---:|---:|
| other_land | 0.7268 | 0.8418 | 0.8080 | 0.8785 |
| water | 0.9636 | 0.9814 | 0.9908 | 0.9723 |
| cultivated_land | 0.8312 | 0.9078 | 0.8835 | 0.9335 |
| built_structures | 0.4743 | 0.6434 | 0.7575 | 0.5592 |
| open_natural | 0.3300 | 0.4963 | 0.5939 | 0.4262 |

## Geographic generalisation

| Held-out map sheet | mIoU | Mean confidence | Mean entropy |
|---|---:|---:|---:|
| L3243F | 0.6447 | 0.8253 | 0.4558 |
| L3243H | 0.6784 | 0.8641 | 0.3980 |

## Interpretation boundaries

The evidence applies to the declared L324 blocks, 2025 RGB orthophotos, the recorded Topographic Database snapshot, and the five-class mapping. It does not establish Finland-wide performance. `other_land` is a heterogeneous residual class, not forest.
