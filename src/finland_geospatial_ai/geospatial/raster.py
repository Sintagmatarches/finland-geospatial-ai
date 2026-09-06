"""Strict raster validation and exact image-grid alignment helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import rasterio
from affine import Affine
from pyproj import CRS

TARGET_CRS = CRS.from_epsg(3067)


class RasterContractError(ValueError):
    """Raised when a raster cannot satisfy the model's geospatial contract."""


@dataclass(frozen=True)
class RasterMetadata:
    path: str
    driver: str
    width: int
    height: int
    bands: int
    dtypes: tuple[str, ...]
    crs: str
    horizontal_epsg: int | None
    transform: tuple[float, float, float, float, float, float]
    bounds: tuple[float, float, float, float]
    resolution_m: tuple[float, float]
    nodata: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def horizontal_crs(crs: CRS | rasterio.crs.CRS | str) -> CRS:
    return CRS.from_user_input(crs).to_2d()


def is_tm35fin(crs: CRS | rasterio.crs.CRS | str) -> bool:
    return horizontal_crs(crs).equals(TARGET_CRS)


def inspect_raster(
    path: Path | str,
    *,
    expected_bands: int | None = None,
    expected_dtype: str | None = None,
    expected_resolution_m: float = 0.5,
    require_tm35fin: bool = True,
) -> RasterMetadata:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Raster not found: {path}")
    try:
        with rasterio.open(path) as source:
            if source.crs is None:
                raise RasterContractError(f"Raster has no CRS: {path}")
            if require_tm35fin and not is_tm35fin(source.crs):
                raise RasterContractError(
                    f"Expected ETRS-TM35FIN horizontal CRS (EPSG:3067), got {source.crs}"
                )
            if expected_bands is not None and source.count != expected_bands:
                raise RasterContractError(
                    f"Expected {expected_bands} bands, got {source.count}: {path}"
                )
            xres, yres = abs(float(source.transform.a)), abs(float(source.transform.e))
            if abs(xres - expected_resolution_m) > 1e-6 or abs(yres - expected_resolution_m) > 1e-6:
                raise RasterContractError(
                    f"Expected {expected_resolution_m} m square pixels, got {xres} x {yres} m"
                )
            if source.transform.b != 0 or source.transform.d != 0:
                raise RasterContractError("Rotated/skewed rasters are not supported")
            if len(set(source.dtypes)) != 1:
                raise RasterContractError("All imagery bands must have one dtype")
            if expected_dtype is not None and source.dtypes[0] != expected_dtype:
                raise RasterContractError(
                    f"Expected {expected_dtype} imagery, got {source.dtypes[0]}: {path}"
                )
            bounds = source.bounds
            return RasterMetadata(
                path=str(path),
                driver=source.driver,
                width=source.width,
                height=source.height,
                bands=source.count,
                dtypes=tuple(source.dtypes),
                crs=source.crs.to_string(),
                horizontal_epsg=horizontal_crs(source.crs).to_epsg(),
                transform=tuple(source.transform)[:6],
                bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                resolution_m=(xres, yres),
                nodata=source.nodata,
            )
    except rasterio.errors.RasterioError as exc:
        raise RasterContractError(f"Unreadable raster: {path}") from exc


def assert_aligned(
    image_shape: tuple[int, int],
    image_transform: Affine,
    image_crs: CRS | rasterio.crs.CRS | str,
    mask_shape: tuple[int, int],
    mask_transform: Affine,
    mask_crs: CRS | rasterio.crs.CRS | str,
) -> None:
    if image_shape != mask_shape:
        raise RasterContractError(f"Image/mask dimensions differ: {image_shape} != {mask_shape}")
    if image_transform != mask_transform:
        raise RasterContractError("Image/mask affine transforms differ")
    if not horizontal_crs(image_crs).equals(horizontal_crs(mask_crs)):
        raise RasterContractError("Image/mask horizontal CRSs differ")


def bounds_from_transform(transform: Affine, width: int, height: int) -> tuple[float, ...]:
    bounds = rasterio.transform.array_bounds(height, width, transform)
    return tuple(float(value) for value in bounds)
