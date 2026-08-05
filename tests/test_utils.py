import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from yg_eo_soilnet.utils import LogTransformer, assign_grid_ids, rpd_score, rpiq_score


def test_log_transformer_round_trip() -> None:
    transformer = LogTransformer()
    values = np.array([0.0, 1.5, 10.0])

    transformed = transformer.transform(values)
    restored = transformer.inverse_transform(transformed)

    assert np.allclose(restored, values)


def test_assign_grid_ids_returns_grid_and_gdf() -> None:
    frame = pd.DataFrame(
        {
            "lat": [0.0, 0.01, 0.02],
            "lon": [0.0, 0.0, 0.0],
            "value": [1, 2, 3],
        }
    )

    grid_ids, grid_gdf = assign_grid_ids(frame, cell_size_m=1000)

    assert len(grid_ids) == len(frame)
    assert grid_gdf.crs.to_epsg() == 4326
    assert isinstance(grid_gdf, GeoDataFrame)
    assert grid_gdf["Grid_ID"].dtype.kind in {"i", "u"}


def test_assign_grid_ids_rejects_invalid_cell_size() -> None:
    frame = pd.DataFrame({"lat": [0.0], "lon": [0.0]})

    for cell_size_m in (None, 0, 50):
        try:
            assign_grid_ids(frame, cell_size_m=cell_size_m)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for invalid cell size")


def test_rpd_and_rpiq_scores_match_manual_calculation() -> None:
    predictions = np.array([1.0, 2.0, 3.0, 4.0])
    targets = np.array([1.0, 2.0, 2.0, 5.0])

    expected_rpd = np.std(targets, ddof=1) / np.sqrt(np.mean((targets - predictions) ** 2))
    expected_rpiq = (np.percentile(targets, 75) - np.percentile(targets, 25)) / np.sqrt(
        np.mean((targets - predictions) ** 2)
    )

    assert np.isclose(rpd_score(predictions, targets), expected_rpd)
    assert np.isclose(rpiq_score(predictions, targets), expected_rpiq)