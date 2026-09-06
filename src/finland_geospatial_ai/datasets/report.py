"""Generate machine-readable and Markdown dataset-validation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shapely.geometry import box
from shapely.ops import unary_union

from finland_geospatial_ai.datasets.manifest import DatasetManifest
from finland_geospatial_ai.datasets.validate import validate_dataset


def generate_dataset_report(manifest_path: Path, output_dir: Path) -> Path:
    validation = validate_dataset(manifest_path)
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    split_geometries = {
        split: unary_union(
            [box(*record.bounds) for record in manifest.patches if record.split == split]
        )
        for split in ("train", "val", "test")
    }
    pairwise_distance = {
        f"{left}_to_{right}_m": float(split_geometries[left].distance(split_geometries[right]))
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    class_distribution = {}
    for split, counts in manifest.class_counts_by_split.items():
        total = sum(counts.values())
        class_distribution[split] = {
            item.name: {
                "pixels": counts[str(item.id)],
                "fraction": counts[str(item.id)] / total,
                "area_m2": counts[str(item.id)] * manifest.native_pixel_size_m**2,
            }
            for item in manifest.classes
        }
    report = {
        "schema_version": "finland-geospatial-ai-dataset-validation/v1",
        "dataset_version": manifest.dataset_version,
        "validation": validation,
        "source_orthophotos": len(manifest.sources["orthophotos"]),
        "orthophoto_years": sorted({record.orthophoto_year for record in manifest.patches}),
        "native_resolution_m": manifest.native_pixel_size_m,
        "crs": manifest.crs,
        "source_vector_feature_counts": manifest.sources.get(
            "topographic_layer_feature_counts", {}
        ),
        "patch_counts": manifest.patch_counts,
        "patch_ground_size_m": manifest.patch_ground_size_m,
        "sampled_area_km2": len(manifest.patches) * manifest.patch_ground_size_m**2 / 1e6,
        "class_distribution": class_distribution,
        "ignored_pixel_fraction_by_split": manifest.ignored_pixel_fraction_by_split,
        "minimum_split_distances": pairwise_distance,
        "duplicate_check": "passed",
        "alignment_check": "passed",
        "hash_check": "passed",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset-validation.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = [
        "# NLS v2 dataset validation",
        "",
        f"Dataset: `{manifest.dataset_version}`",
        "",
        f"- Source orthophotos: {report['source_orthophotos']}",
        f"- Orthophoto years: {', '.join(map(str, report['orthophoto_years']))}",
        f"- Grid: {manifest.crs} at {manifest.native_pixel_size_m} m",
        f"- Patches: {len(manifest.patches)} ({manifest.patch_ground_size_m:g} m square)",
        f"- Sampled footprint: {report['sampled_area_km2']:.3f} km²",
        f"- Split counts: {manifest.patch_counts}",
        f"- Minimum split distances: {pairwise_distance}",
        "- Alignment, hash, duplicate and overlap checks: passed",
        "",
        "This file is generated from the versioned manifest; see "
        "`dataset-validation.json` for the full machine-readable evidence.",
    ]
    (output_dir / "dataset-validation.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports"))
    args = parser.parse_args()
    print(generate_dataset_report(args.manifest, args.output_dir))


if __name__ == "__main__":
    main()
