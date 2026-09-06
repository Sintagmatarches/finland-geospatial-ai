"""CLI for source-quality NLS JPEG2000 and Topographic Database acquisition."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .nls import (
    NLSProcessesClient,
    audit_orthophoto_sheets,
    validate_process_inputs,
    write_receipt,
)


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _orthophoto_inputs(config: dict[str, Any]) -> dict[str, Any]:
    source = config["source"]["orthophoto"]
    map_sheets = sorted(
        {sheet for split in config["splits"].values() for sheet in split["map_sheets"]}
    )
    return {
        "mapSheetInput": map_sheets,
        "fileFormatInput": source["file_format"],
        "yearInput": int(source["acquisition_year"]),
        "dataSetInput": source["dataset_id"],
    }


def _topographic_inputs(config: dict[str, Any]) -> dict[str, Any]:
    source = config["source"]["topographic_database"]
    return {
        "boundingBoxInput": source["bbox"],
        "fileFormatInput": source["file_format"],
        "themeInput": source["theme"],
    }


def acquire(config_path: Path, describe_only: bool = False) -> dict[str, Any]:
    config = _load(config_path)
    source = config["source"]
    raw_dir = Path(config["paths"]["raw_dir"])
    year = int(source["orthophoto"]["acquisition_year"])
    requested_sheets = sorted(
        {sheet for split in config["splits"].values() for sheet in split["map_sheets"]}
    )
    coverage = audit_orthophoto_sheets(year, "L324")
    available = {feature["properties"]["mapSheetNumber"] for feature in coverage}
    missing = sorted(set(requested_sheets) - available)
    if missing:
        raise ValueError(f"NLS has no {year} orthophoto coverage for: {missing}")

    api_key = os.environ.get("NLS_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "NLS_API_KEY is required for source-quality downloads. Create it in the NLS "
            "My Account service and set it only in the environment; never commit it."
        )
    client = NLSProcessesClient(api_key)
    orthophoto_process = source["orthophoto"]["process_id"]
    topographic_process = source["topographic_database"]["process_id"]
    descriptions = {
        orthophoto_process: client.describe_process(orthophoto_process),
        topographic_process: client.describe_process(topographic_process),
    }
    if describe_only:
        return {"coverage": requested_sheets, "processes": descriptions}

    orthophoto_inputs = _orthophoto_inputs(config)
    topographic_inputs = _topographic_inputs(config)
    validate_process_inputs(descriptions[orthophoto_process], orthophoto_inputs)
    validate_process_inputs(descriptions[topographic_process], topographic_inputs)
    ortho_payload = client.execute(orthophoto_process, orthophoto_inputs)
    topo_payload = client.execute(topographic_process, topographic_inputs)
    ortho_receipts = client.download_results(ortho_payload, raw_dir / "orthophotos")
    topo_receipts = client.download_results(topo_payload, raw_dir / "topographic_database")
    receipt = {
        "schema_version": "nls-acquisition-receipt/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_version": config["dataset_version"],
        "official_service": client.base_url,
        "orthophoto_year": year,
        "map_sheets": requested_sheets,
        "topographic_bbox": source["topographic_database"]["bbox"],
        "files": [item.__dict__ for item in [*ortho_receipts, *topo_receipts]],
    }
    write_receipt(raw_dir / "acquisition-receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data/nls_l324.yaml"))
    parser.add_argument("--describe-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(acquire(args.config, args.describe_only), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
