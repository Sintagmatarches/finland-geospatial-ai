"""Versioned dataset-manifest schema."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClassDefinition(BaseModel):
    id: int = Field(ge=0, le=254)
    name: str
    color: tuple[int, int, int]
    source_layers: list[str]
    rule: str


class PatchRecord(BaseModel):
    id: str
    split: Literal["train", "val", "test"]
    source_map_sheet: str
    source_orthophoto: str
    orthophoto_year: int
    image_path: str
    mask_path: str
    boundary_path: str
    bounds: tuple[float, float, float, float]
    transform: tuple[float, float, float, float, float, float]
    width: int
    height: int
    valid_fraction: float = Field(ge=0, le=1)
    class_counts: dict[str, int]
    image_sha256: str
    mask_sha256: str
    boundary_sha256: str


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["finland-geospatial-ai-dataset/v2"]
    dataset_version: str
    split_version: str
    class_map_version: str
    generated_at: str
    generation_commit: str
    config_sha256: str
    crs: Literal["EPSG:3067"]
    aoi: dict[str, Any]
    native_pixel_size_m: float
    bands: list[str]
    patch_size_px: int
    patch_ground_size_m: float
    ignore_index: int
    boundary_ignore_width_m: float
    sources: dict[str, Any]
    classes: list[ClassDefinition]
    precedence: list[str]
    normalization: dict[str, Any]
    patch_counts: dict[str, int]
    class_counts_by_split: dict[str, dict[str, int]]
    ignored_pixel_fraction_by_split: dict[str, float]
    patches: list[PatchRecord]

    @model_validator(mode="after")
    def validate_vocabulary(self) -> DatasetManifest:
        ids = [item.id for item in self.classes]
        names = [item.name for item in self.classes]
        if len(ids) != len(set(ids)) or len(names) != len(set(names)):
            raise ValueError("Class IDs and names must be unique")
        if set(names) != set(self.precedence):
            raise ValueError("Precedence and class vocabulary differ")
        if self.patch_ground_size_m != self.patch_size_px * self.native_pixel_size_m:
            raise ValueError("Patch physical extent is inconsistent with native resolution")
        return self
