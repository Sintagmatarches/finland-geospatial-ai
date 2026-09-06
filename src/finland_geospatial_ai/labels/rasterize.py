"""Rasterize NLS polygons directly onto an orthophoto window grid."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pyogrio
from affine import Affine
from pyproj import CRS
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from finland_geospatial_ai.geospatial.raster import TARGET_CRS, bounds_from_transform


class LabelSourceError(ValueError):
    """Raised for missing, invalid or incompatible Topographic Database layers."""


def boundary_pixels(mask: np.ndarray, width_pixels: int, ignore_index: int = 255) -> np.ndarray:
    """Return a dilated class-transition mask, excluding pre-existing ignore pixels."""
    if mask.ndim != 2:
        raise ValueError("Categorical mask must be two-dimensional")
    if width_pixels < 0:
        raise ValueError("Boundary width cannot be negative")
    valid = mask != ignore_index
    edges = np.zeros_like(valid)
    edges[:, 1:] |= valid[:, 1:] & valid[:, :-1] & (mask[:, 1:] != mask[:, :-1])
    edges[1:, :] |= valid[1:, :] & valid[:-1, :] & (mask[1:, :] != mask[:-1, :])
    if width_pixels == 0:
        return edges
    return binary_dilation(edges, iterations=width_pixels)


def _available_layers(path: Path) -> set[str]:
    try:
        return {name for name, _geometry_type in pyogrio.list_layers(path)}
    except Exception as exc:
        raise LabelSourceError(f"Unreadable Topographic Database: {path}") from exc


def _read_layer(path: Path, layer: str, bbox: tuple[float, ...]) -> gpd.GeoDataFrame:
    frame = pyogrio.read_dataframe(path, layer=layer, bbox=bbox)
    if frame.empty:
        return frame
    if frame.crs is None:
        raise LabelSourceError(f"Layer {layer!r} has no CRS")
    horizontal = CRS.from_user_input(frame.crs).to_2d()
    if not horizontal.equals(TARGET_CRS):
        frame = frame.to_crs(TARGET_CRS)
    elif not CRS.from_user_input(frame.crs).equals(TARGET_CRS):
        # Transform the compound EPSG:3903 representation to its EPSG:3067 horizontal CRS.
        frame = frame.to_crs(TARGET_CRS)
    invalid = ~frame.geometry.is_valid
    if invalid.any():
        frame.loc[invalid, "geometry"] = frame.loc[invalid, "geometry"].make_valid()
    frame = frame[~frame.geometry.is_empty & frame.geometry.notna()]
    return frame


class LabelRasterizer:
    """Deterministic class mapping with an explicit overlap precedence."""

    def __init__(self, gpkg_path: Path | str, config: dict[str, Any]) -> None:
        self.path = Path(gpkg_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Topographic Database not found: {self.path}")
        self.classes = {item["name"]: item for item in config["classes"]}
        self.precedence = list(config["precedence"])
        if set(self.precedence) != set(self.classes):
            raise LabelSourceError("Class precedence must list every class exactly once")
        self.ignore_index = int(config["patch"]["ignore_index"])
        self.available_layers = _available_layers(self.path)
        declared = {
            layer for item in self.classes.values() for layer in item.get("source_layers", [])
        }
        missing = sorted(declared - self.available_layers)
        if missing:
            raise LabelSourceError(f"Topographic Database is missing declared layers: {missing}")
        study_bbox = tuple(config["source"]["topographic_database"]["bbox"])
        self.frames = {
            layer: _read_layer(self.path, layer, study_bbox) for layer in sorted(declared)
        }

    def vector_features(
        self, bounds: tuple[float, ...], class_name: str
    ) -> Iterable[tuple[BaseGeometry, int]]:
        class_config = self.classes[class_name]
        class_id = int(class_config["id"])
        clip_geometry = box(*bounds)
        for layer in class_config.get("source_layers", []):
            frame = self.frames[layer]
            if frame.empty:
                continue
            candidates = frame.iloc[list(frame.sindex.query(clip_geometry, predicate="intersects"))]
            clipped = candidates.geometry.intersection(clip_geometry)
            for geometry in clipped:
                if not geometry.is_empty:
                    yield geometry, class_id

    def rasterize_window(
        self,
        transform: Affine,
        height: int,
        width: int,
        valid_pixels: np.ndarray | None = None,
        boundary_width_pixels: int = 0,
        use_boundary_ignore: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if valid_pixels is not None and valid_pixels.shape != (height, width):
            raise ValueError("valid_pixels does not match the target grid")
        mask = np.full((height, width), self.classes["other_land"]["id"], dtype=np.uint8)
        bounds = bounds_from_transform(transform, width, height)
        for class_name in self.precedence:
            if class_name == "other_land":
                continue
            shapes = list(self.vector_features(bounds, class_name))
            if shapes:
                burned = rasterize(
                    shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype="uint8",
                    all_touched=False,
                )
                class_id = int(self.classes[class_name]["id"])
                mask[burned == class_id] = class_id
        boundary = boundary_pixels(mask, boundary_width_pixels, self.ignore_index)
        if valid_pixels is not None:
            mask[~valid_pixels] = self.ignore_index
            boundary &= valid_pixels
        if use_boundary_ignore:
            mask[boundary] = self.ignore_index
        declared_ids = {int(item["id"]) for item in self.classes.values()} | {self.ignore_index}
        unexpected = set(np.unique(mask).tolist()) - declared_ids
        if unexpected:
            raise LabelSourceError(f"Rasterization produced undeclared class IDs: {unexpected}")
        return mask, boundary
