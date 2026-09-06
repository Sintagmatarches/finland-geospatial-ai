"""FastAPI boundary: uploads only, bounded payloads, no URL or filesystem input."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile

from finland_geoai.inference.service import InferenceInputError, SegmentationService

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def create_app(model_path: Path | str | None = None) -> FastAPI:
    app = FastAPI(
        title="Finland Geospatial AI",
        version="0.1.0",
        description="CPU semantic segmentation for validated 4-band EPSG:3067 GeoTIFF patches.",
    )
    configured_model_path = model_path or os.environ.get("MODEL_PATH")
    resolved_model_path = Path(configured_model_path or "artifacts/final-model.pt")

    @lru_cache(maxsize=1)
    def service() -> SegmentationService:
        return SegmentationService(resolved_model_path)

    @app.get("/health")
    def health() -> dict[str, str]:
        try:
            loaded = service()
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"status": "ok", "device": "cpu", "model_version": loaded.model_version}

    @app.post("/predict")
    async def predict(file: Annotated[UploadFile, File()]) -> dict[str, object]:
        if file.content_type not in {"image/tiff", "image/geotiff", "application/octet-stream"}:
            raise HTTPException(status_code=415, detail="Expected a GeoTIFF upload")
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Upload exceeds 5 MiB limit")
        try:
            return service().predict(payload)
        except InferenceInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
