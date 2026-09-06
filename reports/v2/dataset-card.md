# Dataset card: `nls-l324-2025-v1`

## Purpose

This dataset supports a bounded study of high-resolution semantic segmentation in south-western
Finland. It is designed for engineering evidence, not as a national benchmark.

## Sources and provenance

- NLS Orthophoto, digital colour, map-sheet download, 2025 acquisition year, JPEG2000.
- NLS Topographic Database, GeoPackage, bounded extraction covering the five sheets.
- Horizontal CRS: ETRS-TM35FIN (EPSG:3067).
- Native pixel size: 0.5 m.
- Licence: CC BY 4.0, National Land Survey of Finland.

The acquisition command writes service URL, requested sheets, timestamp, byte size, source URL, and
SHA-256 digest for every downloaded file. The processed manifest records those source digests again.

## Label construction

Five broad classes are derived from exact named Topographic Database layers in
`configs/data/nls_l324.yaml`. No visual guessing or generated masks are permitted. Polygon overlap is
resolved by the declared precedence:

`other_land < open_natural < cultivated_land < water < built_structures`

The `other_land` name is deliberate. The source data do not provide a complete dense-forest polygon
class, so the residual must not be renamed or interpreted as forest.

Rasterisation uses pixel-centre inclusion (`all_touched=False`). Invalid source pixels become 255.
Class-transition pixels within 1.5 m are recorded separately and set to 255 only when a training run
declares boundary ignoring.

## Sampling and splits

Patches are 256 × 256 pixels (128 × 128 m), aligned to source pixels, and do not overlap. Candidate
patches must contain at least 99% valid imagery. Selection is deterministic and adds high-coverage
examples of non-residual classes before seeded filling.

The published manifest contains 576 patches and 9.437184 km² of sampled ground footprint: 384 train,
96 validation, and 96 sealed test. It records the source map sheet, 2025 imagery year, affine grid,
bounds, file hashes, valid fraction, and class counts for every patch. The repository-facing
`artifacts/v2/dataset-manifest-v2.json` is the published copy of the strict v2 schema; the immutable
source-snapshot identifier inside it is `nls-l324-2025-v1`.

Train, validation and test use different 6 × 6 km map sheets with a 512 m interior buffer. The
manifest validator fails on any cross-split patch intersection.

Minimum polygon distances are 1,136 m (train-validation), 2,788.318 m (train-test), and 1,264 m
(validation-test). Validation also fails for duplicate patch IDs, duplicate image content, undeclared
map-sheet/split pairs, hash drift, invalid class IDs, or any image/mask/boundary grid mismatch.

## Known limitations

- Vector boundaries and image edges can disagree due to cartographic generalisation.
- A 2026 database snapshot may reflect edits after the 2025 image acquisition.
- Rare built footprints can be under-represented at patch scale.
- Mixed content inside `other_land` makes that class less semantically specific.
- The selected region contains coastal and agricultural geography but not Finland's full diversity.
