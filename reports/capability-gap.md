# Portfolio capability-gap audit

## Existing evidence

The owner's public work already demonstrates applied tabular machine learning, time-series
forecasting, uncertainty intervals, optimization and decision support, production APIs, Docker,
CI/CD, data engineering, lakehouse patterns, evaluation, observability, and production-oriented
GenAI/RAG systems.

## Missing evidence before this project

The portfolio did not yet contain one project where the central technical object is a trained deep
neural network over high-resolution imagery. In particular, it lacked an interview-readable PyTorch
training loop, semantic segmentation, raster/vector label construction, image-label boundary
uncertainty, and geographic generalisation measured with disjoint spatial regions.

## Positioning

This repository is intentionally narrow and complementary. It does not add another tabular model or
API-first demo. Its evidence chain is:

1. native 0.5 m official Finnish orthophotos;
2. authoritative but imperfect topographic vectors;
3. exact-grid categorical rasterisation and spatially buffered splits;
4. explicit PyTorch U-Net and pretrained SegFormer training;
5. sealed-geography, boundary, error, uncertainty and calibration analysis;
6. georeferenced CPU inference through one validated CLI/API implementation.

Portfolio claims may now use the committed measured evidence: the real source dataset, three MLflow
runs, validation-only model selection, and single sealed-test artifact all exist.
