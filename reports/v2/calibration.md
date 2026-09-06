# Calibration and uncertainty analysis

The single sealed test contains 5,922,438 evaluated interior pixels. Raw maximum-softmax confidence
has an expected calibration error of 0.0474 across 15 fixed bins. Predictive entropy has a 0.3104
Spearman correlation with pixel error, so uncertainty is informative but incomplete.

Risk coverage is monotonic in the published operating points: retaining the most confident 10% of
pixels gives a 0.00013 error rate, while retaining all pixels gives 0.19762. This supports confidence
as a triage signal inside this sample; it does not define a transferable safety threshold.

Calibration differs by class. Water averages 0.9919 confidence at 0.9723 accuracy. Open-natural land
averages 0.7516 confidence at 0.4262 accuracy, and built structures average 0.8617 confidence at
0.5592 accuracy. Those gaps expose material class-specific overconfidence that the aggregate ECE
alone hides.

No temperature or threshold was fit after seeing test results. The committed reliability diagram,
risk-coverage curve, calibration bins, per-class statistics, and test lock preserve the raw selected
model result. Pixel-level samples are spatially correlated, so these numbers are descriptive for the
two held-out sheets rather than independent-sample confidence intervals.
