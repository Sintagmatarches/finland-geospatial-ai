# NLS v2 inference contract

The v2 CLI and FastAPI service accept GeoTIFF or JPEG2000 rasters only when they have exactly three
bands, 0.5 m square pixels, and EPSG:3067. The predictor verifies the checkpoint SHA-256, class count,
and normalization metadata before inference. It blends overlapping 256 px tiles and writes a
single-band categorical GeoTIFF plus optional maximum-softmax confidence, preserving the input CRS,
affine transform, width, and height.

The API limits uploads to 256 MiB by default, serializes access to the model, and returns a ZIP with
`prediction.tif`, `confidence.tif`, and `summary.json`. The summary reports class pixel counts,
fractions, areas, and confidence statistics. CPU loading, health, prediction, georeferencing, and ZIP
contents are covered by unit tests and the Docker CI smoke test.

This contract is specific to the NLS v2 model. The historical v1 service remains available in the
`finland_geoai` package with its four-band, 10 m contract.
