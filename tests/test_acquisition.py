from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from finland_geospatial_ai.acquisition.nls import (
    NLSAcquisitionError,
    _download_links,
    _extract_zip,
    _status_link,
    validate_process_inputs,
)


class StubResponse:
    headers: dict[str, str] = {}


def test_official_result_path_is_discovered() -> None:
    payload = {
        "outputs": [
            {
                "path": "https://example.test/download/L3244C.jp2",
                "zipPath": "https://example.test/uncompressed/job-id",
            }
        ]
    }
    assert _download_links(payload) == ["https://example.test/download/L3244C.jp2"]


def test_status_self_link_is_discovered() -> None:
    payload = {
        "status": "accepted",
        "links": [{"href": "https://example.test/jobs/123", "rel": "self"}],
    }
    assert _status_link(payload, StubResponse()) == "https://example.test/jobs/123"


def test_zip_extraction_records_hash(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/topodb.gpkg", b"test")
    receipts = _extract_zip(archive_path, tmp_path / "output", "https://example.test/source.zip")
    assert Path(receipts[0].path).read_bytes() == b"test"
    assert len(receipts[0].sha256) == 64


def test_zip_slip_is_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.gpkg", b"test")
    with pytest.raises(NLSAcquisitionError, match="Unsafe path"):
        _extract_zip(archive_path, tmp_path / "output", "https://example.test/unsafe.zip")


def test_process_description_rejects_stale_input_name() -> None:
    description = {"inputs": {"boundingBoxInput": {"minOccurs": 1}}}
    with pytest.raises(NLSAcquisitionError, match="absent"):
        validate_process_inputs(description, {"bboxInput": [1, 2, 3, 4]})
    validate_process_inputs(description, {"boundingBoxInput": [1, 2, 3, 4]})
