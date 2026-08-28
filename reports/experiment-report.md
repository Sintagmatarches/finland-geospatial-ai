# Experiment report

## 1. Problem and method

The task is per-pixel land-cover prediction from real four-band Earth Observation imagery. Semantic segmentation is appropriate because a patch contains multiple spatially contiguous classes; assigning one label to the whole image would destroy the boundary and area information needed for a map.

## 2. Data and spatial governance

Sentinel-2 Level-2A blue/green/red/NIR COG windows are reprojected to EPSG:3067 / 10 m and aligned to ESA WorldCover 2021 v200 with nearest-neighbour categorical resampling. Helsinki and Lahti train the models; Tampere selects models and calibration; Oulu is the single held-out test. No patch overlaps, no AOI crosses a split and automated bounds checks block leakage.

## 3. Declared experiment matrix

Every experiment used seed 42, 64×64 patches, train-only dihedral augmentation, AdamW, 12 maximum CPU epochs and best-validation-mIoU checkpointing.

| ID | Question | Configuration | Best validation mIoU | Conclusion |
| --- | --- | --- | ---: | --- |
| E0 | Do neural models beat a trivial strategy? | training-majority forest everywhere | 0.0685 | Yes; majority accuracy (0.411) hides unusable macro performance. |
| E1 | What can RGB U-Net learn? | compact U-Net, RGB, weighted CE+Dice | 0.4037 | Strong non-trivial baseline. |
| E2 | Does NIR help? | same U-Net + NIR | 0.4512 | Yes, +0.0475 absolute validation mIoU in this controlled run. |
| E3 | Does modern atrous context help? | 178,998-parameter depthwise TinyDeepLabV3, RGB+NIR | 0.3867 | No; it underperformed both U-Nets. |
| E4 | Does Dice improve the imbalanced objective? | E2 architecture, weighted CE only | **0.4574** | No; removing Dice improved by 0.0061 and selected the final checkpoint. |

These are single-seed bounded experiments, so small differences—especially E2 vs E4—should not be overinterpreted as general architectural truths.

![All selection curves use Tampere validation only](figures/training-curves.png?v=20260828-geospatial-ai-v1)

## 4. Final test result

The selection rule was highest validation mIoU across E1–E4. E4 epoch 11 was frozen, a temperature was fit using validation only, and Oulu was evaluated once. Final mIoU is 0.3982 and macro Dice is 0.4743. Strong forest/water performance coexists with failure on rare `other_natural`; this distribution-sensitive result is more informative than the 0.8056 pixel accuracy.

## 5. Error analysis

Cases are selected deterministically: lowest patch mIoU, median patch mIoU, highest patch mIoU and highest mean entropy. The repository does not cherry-pick only attractive outputs.

- Forest often absorbs shrub/grass and rare natural areas.
- Built-up predictions are materially useful but bleed into forest/vegetation boundaries.
- Cropland has only 0.58% test support and transfers poorly from southern training geography.
- Water is strongest but shoreline/mixed pixels still create forest/built confusions.
- The highest-entropy patch has mIoU 0.187, supporting the claim that entropy often surfaces difficult geography.

![Worst held-out patch by deterministic mIoU criterion](figures/error-case-worst_iou.png?v=20260828-geospatial-ai-v1)

## 6. Calibration and geographic transfer

Temperature 0.671 was fit on Tampere only. It made Oulu probabilities more overconfident and worsened ECE/NLL. This negative result suggests geographic calibration shift; a scalar fitted to one held-out city is not a universal calibration fix. The model should expose uncertainty and limitations, not relabel softmax as trustworthy probability.

Risk/coverage is more useful: full coverage error is 19.44%; at 80% it is 12.65%, and at 70% it is 10.25%. The curve is imperfect, so abstention thresholds require independent operational validation.

![Risk versus retained coverage](figures/risk-coverage.png?v=20260828-geospatial-ai-v1)

## 7. What production would require

Production use needs a geographically and seasonally broader labelled corpus, independent human validation, repeated blocked spatial folds, more support or a redesign for rare classes, scene-quality stratification, external AOI tests, explicit model registry/approval, drift monitoring, calibrated abstention evaluated per region and a retraining policy. A larger pretrained segmentation model is a sensible next experiment only after improving data coverage and labels.

## 8. Answers to the portfolio questions

- NIR helped in the controlled U-Net comparison.
- More modern context did not help at this scale.
- Dice did not help the selected objective.
- The final model fails mainly on rare/shifted vegetation and heterogeneous natural classes.
- It is not calibrated across the held-out geography; temperature scaling made it worse.
- Uncertainty identifies many difficult pixels but is not a guarantee.
- Geographic performance cannot be averaged across test AOIs because the protected test deliberately contains only Oulu; Tampere validation is reported separately and not relabelled as test.
