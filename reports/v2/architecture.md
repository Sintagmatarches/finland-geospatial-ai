# Architecture

The system has one reproducible path from official source files to a georeferenced prediction.

```text
NLS OGC API Processes
  ├─ 2025 RGB orthophoto JPEG2000 (0.5 m, EPSG:3067)
  └─ Topographic Database GeoPackage
             │
             ▼
strict source inspection → polygon rasterisation → boundary uncertainty mask
             │
             ▼
versioned patch manifest + spatial split GeoJSON + SHA-256 receipts
             │
             ├─ majority baseline
             ├─ compact U-Net
             ├─ boundary-ignore ablation
             └─ pretrained SegFormer-B0
             │
             ▼
sealed-sheet metrics → MLflow evidence → selected safetensors checkpoint
             │
             ▼
overlap-tiled CLI / FastAPI → categorical GeoTIFF in the input grid
```

## Design decisions

`LabelRasterizer` burns explicit source layers in declared precedence order. The residual class is
assigned first, then increasingly precise classes overwrite it. The raw categorical mask and the
transition mask are stored separately so the same pixels can support the primary experiment and a
controlled boundary-ignore ablation without regenerating labels.

`DatasetManifest` rejects extra fields and inconsistent physical resolution. `geoai-validate` then
checks artifact hashes, raster alignment, allowed class IDs, split counts, class support, duplicated
content, and any cross-split patch intersection.

Training uses deterministic seeds, paired spatial transforms, restrained RGB jitter, AdamW,
mixed precision when CUDA is present, early stopping on validation mIoU, and `safetensors` weights.
Only the validation sheet influences model selection. The test sheets are opened by the separate
evaluation command. That command writes a lock containing the selected experiment, manifest,
metadata, and evaluation hashes; the published output directory rejects a second test evaluation.

Inference validates the same CRS, band, and resolution contract before allocation. Tiles overlap by
64 pixels and blend softmax probabilities with a tapered window to reduce seams. Categorical output
is written directly to a single-band GeoTIFF with the original grid and provenance tags.
