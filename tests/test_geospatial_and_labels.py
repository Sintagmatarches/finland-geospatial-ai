from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from finland_geospatial_ai.geospatial.raster import RasterContractError, assert_aligned
from finland_geospatial_ai.labels.rasterize import boundary_pixels


def test_alignment_accepts_identical_tm35fin_grids() -> None:
    transform = Affine(0.5, 0, 1000, 0, -0.5, 7000000)
    assert_aligned((32, 32), transform, "EPSG:3067", (32, 32), transform, "EPSG:3067")


def test_alignment_rejects_shifted_mask() -> None:
    with pytest.raises(RasterContractError, match="transforms differ"):
        assert_aligned(
            (32, 32),
            Affine.identity(),
            "EPSG:3067",
            (32, 32),
            Affine.translation(1, 0),
            "EPSG:3067",
        )


def test_boundary_pixels_dilate_class_transition() -> None:
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[:, 5:] = 1
    boundary = boundary_pixels(mask, width_pixels=1)
    assert boundary[:, 3:7].any()
    assert not boundary[:, 0].any()
