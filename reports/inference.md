# Inference contract and runbook

## Request

`POST /predict` accepts multipart file upload only. There is no URL field and no user-controlled local path. Payloads above 5 MiB are rejected before raster parsing.

The uploaded file must be a readable GeoTIFF with:

- exactly 4 bands in blue, green, red, NIR order;
- exactly 64×64 pixels;
- EPSG:3067;
- 10 m square pixels;
- `uint16`, `int16`, `float32` or `float64` samples;
- at least one valid pixel.

Wrong band count, dimensions, CRS, resolution, dtype, no-data-only input and malformed TIFF all return bounded 4xx errors. A missing/corrupt/incompatible artifact makes `/health` return 503.

## Response

The JSON response includes model and dataset versions, the class legend, class fractions, mean normalized predictive entropy, a calibration limitation, a base64 color PNG and a base64 categorical GeoTIFF. The GeoTIFF keeps the input CRS and affine transform and uses no-data 255.

## Run

```bash
uvicorn finland_geoai.inference.api:app --host 0.0.0.0 --port 8000
docker build -f docker/Dockerfile -t finland-geospatial-ai .
docker run --rm -p 8000:8000 finland-geospatial-ai
```

The default artifact loads with `map_location="cpu"`; CUDA is neither required nor claimed. Training data and MLflow stores are excluded from the Docker context.
