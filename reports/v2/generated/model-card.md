# Selected model summary

- Model: `E3-segformer-b0-rgb`
- Dataset: `nls-l324-2025-v1`
- Split: `l324-spatial-blocks-v1`
- Input: RGB, 0.5 m, EPSG:3067
- Patch: 256 px (128 m)
- Final test mIoU: 0.6652
- Final test macro Dice: 0.7742
- ECE: 0.0474

## Intended use

Analyst-assisted research and bounded offline mapping on compatible NLS-style imagery.
Not for cadastral, safety-critical, enforcement, or legal land-use decisions.

## Known limitations

Labels inherit vector generalisation and temporal mismatch. The study covers a bounded south-west Finland region. Confidence does not guarantee correctness, and `other_land` contains several visually distinct surfaces.
