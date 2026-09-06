"""Audit official Topographic Database class support against candidate map sheets."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pyproj import CRS
from shapely.geometry import shape
from shapely.ops import unary_union

from finland_geospatial_ai.acquisition.nls import audit_orthophoto_sheets, sha256_file
from finland_geospatial_ai.labels.rasterize import LabelRasterizer


def audit_topography(
    config_path: Path, gpkg_path: Path, output_path: Path, source_url: str
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    year = int(config["source"]["orthophoto"]["acquisition_year"])
    requested = {
        sheet: split
        for split, definition in config["splits"].items()
        for sheet in definition["map_sheets"]
    }
    coverage = {
        feature["properties"]["mapSheetNumber"]: shape(feature["geometry"])
        for feature in audit_orthophoto_sheets(year, "L324")
    }
    missing = set(requested) - set(coverage)
    if missing:
        raise ValueError(f"No {year} orthophoto coverage for candidate sheets: {sorted(missing)}")
    rasterizer = LabelRasterizer(gpkg_path, config)
    populated_frames = [frame for frame in rasterizer.frames.values() if not frame.empty]
    if not populated_frames:
        raise ValueError("None of the declared label layers contains source features")
    source_crs = CRS.from_user_input(populated_frames[0].crs)
    rows = []
    for sheet, split in sorted(requested.items()):
        geometry = coverage[sheet]
        class_support = {}
        for class_name in rasterizer.precedence:
            if class_name == "other_land":
                continue
            features = [item[0] for item in rasterizer.vector_features(geometry.bounds, class_name)]
            clipped = [item.intersection(geometry) for item in features]
            clipped = [item for item in clipped if not item.is_empty]
            class_support[class_name] = {
                "features": len(clipped),
                "union_area_km2": float(unary_union(clipped).area / 1e6) if clipped else 0.0,
            }
        rows.append(
            {
                "map_sheet": sheet,
                "split": split,
                "orthophoto_year": year,
                "bounds": list(geometry.bounds),
                "class_support": class_support,
            }
        )
    artifact = {
        "schema_version": "finland-geospatial-ai-source-audit/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_url": source_url,
        "source_file": gpkg_path.name,
        "source_sha256": sha256_file(gpkg_path),
        "source_crs": source_crs.to_string(),
        "orthophoto_coverage_year": year,
        "available_layers": sorted(rasterizer.available_layers),
        "candidate_sheets": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpkg", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/data/nls_l324.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/source-audit.json"))
    parser.add_argument(
        "--source-url",
        default="https://tiedostopalvelu.maanmittauslaitos.fi/tp/julkinen/lataus/tuotteet/Maastotietojen_testituotteet_GPKG",
    )
    args = parser.parse_args()
    print(audit_topography(args.config, args.gpkg, args.output, args.source_url))


if __name__ == "__main__":
    main()
