"""Raster alignment primitives with resampling choices visible at call sites."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds


@dataclass(frozen=True)
class TargetGrid:
    crs: str
    transform: Affine
    width: int
    height: int
    bounds: tuple[float, float, float, float]
    resolution: float


def grid_from_wgs84_bbox(
    bbox: tuple[float, float, float, float], target_crs: str, resolution: float
) -> TargetGrid:
    """Transform a WGS84 box and snap its outer edges to a metric pixel grid."""
    left, bottom, right, top = transform_bounds("EPSG:4326", target_crs, *bbox, densify_pts=21)
    left = math.floor(left / resolution) * resolution
    bottom = math.floor(bottom / resolution) * resolution
    right = math.ceil(right / resolution) * resolution
    top = math.ceil(top / resolution) * resolution
    width = int(round((right - left) / resolution))
    height = int(round((top - bottom) / resolution))
    return TargetGrid(
        crs=target_crs,
        transform=from_origin(left, top, resolution, resolution),
        width=width,
        height=height,
        bounds=(left, bottom, right, top),
        resolution=resolution,
    )


def read_aligned(url: str, grid: TargetGrid, resampling: Resampling) -> np.ndarray:
    """Read a remote COG onto ``grid`` using the explicitly supplied algorithm."""
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MULTIRANGE="YES",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
    ):
        with rasterio.open(url) as source:
            with WarpedVRT(
                source,
                crs=grid.crs,
                transform=grid.transform,
                width=grid.width,
                height=grid.height,
                resampling=resampling,
            ) as vrt:
                return vrt.read(1)


def patch_bounds(transform: Affine, row: int, col: int, size: int) -> list[float]:
    left, top = transform * (col, row)
    right, bottom = transform * (col + size, row + size)
    return [float(left), float(bottom), float(right), float(top)]
