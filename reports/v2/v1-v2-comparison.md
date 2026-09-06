# v1 and v2 scientific comparison

The two tracks answer different questions. Their scores are preserved side by side but must not be
read as a controlled before/after model comparison.

| Property | Historical v1 | NLS high-resolution v2 |
|---|---|---|
| Imagery | Sentinel-2 L2A | NLS digital colour orthophoto |
| Native resolution | 10 m | 0.5 m |
| Bands | blue, green, red, NIR | red, green, blue |
| Labels | ESA WorldCover 2021 | NLS Topographic Database polygons |
| Classes | 6 | 5 |
| Regions | four Finnish AOIs; Oulu test | five L324 sheets; two test sheets |
| Patches | 850 × 64 px | 576 × 256 px |
| Selected model | U-Net with weighted CE | pretrained SegFormer-B0 |
| Validation mIoU | 0.4574 | 0.7344 |
| Sealed-test mIoU | 0.3982 | 0.6652 |

The 0.2670 absolute difference in test mIoU combines changes in sensor, ground sampling, region,
class vocabulary, reference source, patch scale, data volume, and model family. It does not estimate
the causal benefit of NLS imagery or SegFormer. The controlled v2 evidence is internal: SegFormer-B0
improved validation mIoU by 0.1673 over the compact U-Net on the same v2 contract, while the
boundary-ignore policy also differs in the declared E2 ablation.

The v1 lock and metrics remain unchanged. No v1 test data were reopened, no v1 hyperparameters were
retuned, and no v2 result was substituted into a v1 artifact.
