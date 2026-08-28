from __future__ import annotations

from finland_geoai.geo.raster import grid_from_wgs84_bbox


def test_target_grid_is_snapped_to_ten_metres() -> None:
    grid = grid_from_wgs84_bbox((24.87, 60.17, 25.0, 60.26), "EPSG:3067", 10)
    assert all(value % 10 == 0 for value in grid.bounds)
    assert grid.transform.a == 10
    assert grid.transform.e == -10
