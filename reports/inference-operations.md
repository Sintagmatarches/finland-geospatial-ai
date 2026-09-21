# Inference operations — code release 0.2.1

The model, dataset, geographic split and sealed test artifacts are unchanged.
The production predictor now blends into a row buffer and flushes rows after the
last tile that can influence them. The tile origins, padding, Hann weights,
normalization, class argmax and confidence semantics are preserved. Probability
buffer memory is O(classes × tile size × raster width), independent of image height.
This is not constant memory: extremely wide images still require more memory.

Tests compare the new implementation with the prior dense accumulation algorithm
on generated rasters, including small images, shifted final tiles, zero overlap and
nodata. They also assert that probability-buffer height never exceeds tile height.
These are serving regression tests, not another evaluation on the sealed test set.

The API retains a 256 MiB upload limit and adds a decoded-image limit of 4,194,304
pixels (`GEOAI_MAX_PIXELS`) and width limit 16,384 before model loading. Compressed
upload size alone cannot bound decoded memory. The summary still reads output arrays
within this explicit API limit to compute its exact quantile; the CLI can process
taller rasters with row buffering. Model activation memory and GDAL caches are
additional to these arrays and require container-level resource limits.

One request per process is admitted to inference/packaging; overload returns 503
with `Retry-After: 1`. It does not silently queue unlimited work. Lazy model loading
is serialized. Malformed raster inputs return 422, unavailable models 503, and
unexpected processing failures a sanitized 500. Temporary output directories are
removed on failure and after the response is sent. `/live` is dependency-free;
`/health` remains the model-readiness check. Docker uses `/live`, and CI explicitly
checks `/health`, `/model` and real multipart inference with generated smoke weights.

No hosted GPU/API deployment, production traffic, image-drift monitor or latency/RSS
benchmark is claimed. Before public hosting, add authenticated access, ingress body
limits (multipart parsing precedes the handler), driver/network isolation for GDAL,
container memory/time limits and a load/restore exercise. Checkpoint hashes validate
integrity against trusted metadata, not authenticity if both files are replaced.
Rollback the complete metadata + weights pair by immutable release digest; never
mix class maps or normalization from another model. Generated model cards and
MLflow experiment metadata already exist and should not be duplicated.

The runtime dependency refresh is verified in a clean Python 3.12 environment.
Transformers 5 renamed SegFormer state keys, so loading uses the same explicit
renaming map documented upstream, followed by strict key checking; tensors are not
reshaped or relaxed. The published checkpoint loads without network access and its
CPU logits matched the previous runtime fixture within 3.82e-6 maximum absolute
difference with identical argmax predictions. `pip-audit` reported no known
vulnerabilities in the resolved environment; local project and PyTorch CPU wheel
identities are listed as unauditable because they are not PyPI distributions. CI
also emits a CycloneDX SBOM. This is release compatibility evidence, not a rerun of
the sealed geographic evaluation.
