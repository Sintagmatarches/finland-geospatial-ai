"""Small, auditable client for the official NLS OGC API Processes service."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests

DEFAULT_BASE_URL = "https://avoin-paikkatieto.maanmittauslaitos.fi/tiedostopalvelu/ogcproc/v1"
MAPSITE_API = "https://asiointi.maanmittauslaitos.fi/karttapaikka/api/spatialDataFiles"


class NLSAcquisitionError(RuntimeError):
    """Raised when NLS rejects a request or returns an incomplete job."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_or_error(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except requests.JSONDecodeError as exc:
        raise NLSAcquisitionError(
            f"NLS returned non-JSON content ({response.status_code})"
        ) from exc
    if not response.ok:
        detail = payload.get("detail") or payload.get("description") or payload
        raise NLSAcquisitionError(f"NLS request failed ({response.status_code}): {detail}")
    return payload


def _walk_links(value: Any) -> Iterable[tuple[str | None, str]]:
    if isinstance(value, dict):
        for key in ("href", "path"):
            href = value.get(key)
            if isinstance(href, str):
                yield value.get("rel") or key, href
        for child in value.values():
            yield from _walk_links(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_links(child)


def _download_links(payload: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for rel, href in _walk_links(payload):
        path = urlparse(href).path.lower()
        downloadable = path.endswith((".jp2", ".zip", ".gpkg", ".gml", ".tif", ".tiff"))
        if downloadable or rel in {"output", "result", "download", "path"}:
            candidates.append(href)
    return list(dict.fromkeys(candidates))


def _status_link(payload: dict[str, Any], response: requests.Response) -> str | None:
    location = response.headers.get("Location")
    if location:
        return location
    for rel, href in _walk_links(payload):
        if rel in {"status", "monitor", "self"}:
            return href
    return None


def validate_process_inputs(description: dict[str, Any], inputs: dict[str, Any]) -> None:
    """Check an execution payload against the service's self-described input IDs."""
    described = description.get("inputs", {})
    if isinstance(described, dict):
        definitions = described
    elif isinstance(described, list):
        definitions = {item["id"]: item for item in described if isinstance(item, dict)}
    else:
        raise NLSAcquisitionError("NLS process description has an unsupported inputs schema")
    declared = set(definitions)
    unknown = set(inputs) - declared
    if unknown:
        raise NLSAcquisitionError(
            f"Execution uses inputs absent from NLS description: {sorted(unknown)}"
        )
    required = {
        name
        for name, definition in definitions.items()
        if isinstance(definition, dict)
        and (definition.get("minOccurs", 0) not in {0, "0"} or definition.get("required") is True)
    }
    missing = required - set(inputs)
    if missing:
        raise NLSAcquisitionError(f"Execution omits required NLS inputs: {sorted(missing)}")


@dataclass(frozen=True)
class DownloadReceipt:
    path: str
    sha256: str
    bytes: int
    source_url: str


def _response_filename(response: requests.Response, url: str, fallback: str) -> str:
    disposition = response.headers.get("Content-Disposition", "")
    match = re.search(r"filename\*?=(?:UTF-8''|\")?([^\";]+)", disposition, re.IGNORECASE)
    if match:
        return unquote(match.group(1).strip())
    return unquote(Path(urlparse(url).path).name) or fallback


def _extract_zip(path: Path, output_dir: Path, source_url: str) -> list[DownloadReceipt]:
    receipts: list[DownloadReceipt] = []
    root = output_dir.resolve()
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            destination = (output_dir / member.filename).resolve()
            if root not in destination.parents:
                raise NLSAcquisitionError(f"Unsafe path in NLS ZIP archive: {member.filename}")
            if destination.exists():
                raise NLSAcquisitionError(f"NLS archive would overwrite: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            receipts.append(
                DownloadReceipt(
                    path=str(destination),
                    sha256=sha256_file(destination),
                    bytes=destination.stat().st_size,
                    source_url=f"{source_url}#{member.filename}",
                )
            )
    return receipts


class NLSProcessesClient:
    """Authenticated client using an NLS API key as HTTP Basic username."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: int = 120,
    ) -> None:
        if not api_key.strip():
            raise ValueError("NLS_API_KEY is empty")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.auth = (api_key, "")
        self.session.headers.update(
            {"Accept": "application/json", "User-Agent": "finland-geospatial-ai/0.2"}
        )

    def list_processes(self) -> dict[str, Any]:
        return _json_or_error(
            self.session.get(f"{self.base_url}/processes", timeout=self.timeout_seconds)
        )

    def describe_process(self, process_id: str) -> dict[str, Any]:
        return _json_or_error(
            self.session.get(
                f"{self.base_url}/processes/{process_id}", timeout=self.timeout_seconds
            )
        )

    def execute(
        self,
        process_id: str,
        inputs: dict[str, Any],
        poll_seconds: float = 3.0,
        maximum_wait_seconds: int = 1800,
    ) -> dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/processes/{process_id}/execution",
            json={"id": process_id, "inputs": inputs},
            headers={"Prefer": "respond-async"},
            timeout=self.timeout_seconds,
        )
        payload = _json_or_error(response)
        if _download_links(payload):
            return payload
        status_url = _status_link(payload, response)
        if not status_url:
            raise NLSAcquisitionError("NLS execution returned neither results nor a status URL")
        deadline = time.monotonic() + maximum_wait_seconds
        while time.monotonic() < deadline:
            status_response = self.session.get(status_url, timeout=self.timeout_seconds)
            status_payload = _json_or_error(status_response)
            status = str(status_payload.get("status", "")).lower()
            if status in {"successful", "succeeded", "success", "finished"}:
                if _download_links(status_payload):
                    return status_payload
                result_url = next(
                    (
                        href
                        for rel, href in _walk_links(status_payload)
                        if rel == "results" or urlparse(href).path.rstrip("/").endswith("/results")
                    ),
                    None,
                )
                if result_url:
                    return _json_or_error(
                        self.session.get(result_url, timeout=self.timeout_seconds)
                    )
                raise NLSAcquisitionError("Successful NLS job has no downloadable result")
            if status in {"failed", "dismissed", "cancelled", "canceled"}:
                raise NLSAcquisitionError(f"NLS job ended with status={status!r}")
            time.sleep(poll_seconds)
        raise NLSAcquisitionError("Timed out while waiting for NLS data preparation")

    def download_results(self, payload: dict[str, Any], output_dir: Path) -> list[DownloadReceipt]:
        links = _download_links(payload)
        if not links:
            raise NLSAcquisitionError("NLS response contains no downloadable files")
        output_dir.mkdir(parents=True, exist_ok=True)
        receipts: list[DownloadReceipt] = []
        for index, url in enumerate(links, start=1):
            response = self.session.get(url, stream=True, timeout=self.timeout_seconds)
            response.raise_for_status()
            filename = _response_filename(response, url, f"nls-output-{index}")
            path = output_dir / filename
            with path.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            receipts.append(
                DownloadReceipt(
                    path=str(path),
                    sha256=sha256_file(path),
                    bytes=path.stat().st_size,
                    source_url=url,
                )
            )
            if zipfile.is_zipfile(path):
                receipts.extend(_extract_zip(path, output_dir, url))
        return receipts


def audit_orthophoto_sheets(year: int, prefix: str) -> list[dict[str, Any]]:
    """Read public MapSite coverage metadata without downloading imagery."""
    response = requests.get(f"{MAPSITE_API}/sheets/{year}", timeout=120)
    response.raise_for_status()
    features = response.json().get("features", [])
    return [
        feature
        for feature in features
        if str(feature.get("properties", {}).get("mapSheetNumber", "")).startswith(prefix)
    ]


def write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
