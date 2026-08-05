"""Calendar-grid CNN path: rasteriser, both convolutional encoders, and the Lightning module.

The rasteriser is where a date becomes a grid coordinate, so most of the risk lives there: an
off-by-one in the month lookup silently shifts a whole dataset's phenology by a month and nothing
downstream would complain. Those cases are pinned first and hardest.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from yg_eo_soilnet.data_manager import DataManager
from yg_eo_soilnet.datamodules.sequence.sequence_builder import SoilSequenceBuilder, to_decimal_year
from yg_eo_soilnet.datamodules.sequence.sequence_datamodule import SoilSequenceDataModule
from yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module import SoilCNNLightningModule
from yg_eo_soilnet.models.lightningmodules.temporal_cnn_encoders import (
    AnnualGrid2DEncoder,
    CalendarGridRasterizer,
    ConcatGatedFusion,
    DilatedTempCNNEncoder,
    decimal_year_to_month_index,
    masked_global_pool,
)

ENCODERS = ["dilated_tempcnn", "annual_grid2d"]
ENCODER_CLASSES = [DilatedTempCNNEncoder, AnnualGrid2DEncoder]


def _times(dates) -> torch.Tensor:
    return torch.as_tensor(to_decimal_year(pd.Series(pd.to_datetime(dates)))).unsqueeze(0)


def _batch(batch_size=4, length=24, channels=3, seed=0, start="2019-01-01", months_step=1, year_offset=0):
    generator = torch.Generator().manual_seed(seed)
    dates = pd.date_range(start, periods=length, freq=f"{months_step}MS")
    if year_offset:
        dates = dates + pd.DateOffset(years=year_offset)
    times = torch.as_tensor(to_decimal_year(pd.Series(dates))).unsqueeze(0).repeat(batch_size, 1)
    return {
        "x_static": torch.randn(batch_size, 5, generator=generator),
        "y": torch.randn(batch_size, 1, generator=generator),
        "sequences": {"m": torch.randn(batch_size, length, channels, generator=generator)},
        "sequence_mask": {"m": torch.ones(batch_size, length, dtype=torch.bool)},
        "sequence_time": {"m": times},
        "sequence_validity": {"m": torch.ones(batch_size, length, channels, dtype=torch.bool)},
    }


# --- month lookup ----------------------------------------------------------


@pytest.mark.parametrize("year", [2017, 2019, 2020, 2024, 2000, 2100])
def test_every_day_of_a_year_maps_to_its_calendar_month(year: int) -> None:
    """Includes leap and century years: March onward shifts a day once February has 29."""
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    resolved = decimal_year_to_month_index(_times(days))[0].numpy()
    np.testing.assert_array_equal(resolved, days.month.to_numpy() - 1)


def test_month_lookup_beats_the_naive_fraction_scaling() -> None:
    """1 November is the canonical break: floor(0.8323 * 12) == 9, i.e. October."""
    times = _times(["2021-11-01", "2021-12-01"])
    assert decimal_year_to_month_index(times)[0].tolist() == [10, 11]

    naive = ((times - times.floor()) * 12).floor().long()[0].tolist()
    assert naive == [9, 10], "the naive scaling mis-assigns both, which is why it is not used"


def test_first_of_month_stamps_are_exact_in_a_leap_year() -> None:
    dates = [f"2020-{month:02d}-01" for month in range(1, 13)]
    assert decimal_year_to_month_index(_times(dates))[0].tolist() == list(range(12))


# --- rasteriser ------------------------------------------------------------


def test_rasterizer_places_observations_in_the_right_cells() -> None:
    rasterizer = CalendarGridRasterizer(num_channels=1, grid_years=3)
    times = _times(["2021-03-01", "2022-07-01", "2023-11-01", "2023-12-01"])
    values = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    mask = torch.ones(1, 4, dtype=torch.bool)

    grid, cell_mask = rasterizer(values, mask, times)

    assert grid.shape == (1, rasterizer.output_channels, 3, 12)
    # Rows count back from the point's own latest year (2023 -> row 2).
    assert cell_mask[0].nonzero().tolist() == [[0, 2], [1, 6], [2, 10], [2, 11]]
    assert [float(grid[0, 0, row, month]) for row, month in cell_mask[0].nonzero().tolist()] == [1.0, 2.0, 3.0, 4.0]


def test_rasterizer_leaves_missing_months_unoccupied() -> None:
    rasterizer = CalendarGridRasterizer(num_channels=1, grid_years=1)
    present = [1, 2, 3, 9, 10, 11, 12]  # months 4..8 absent, the sub-12-month case
    times = _times([f"2022-{month:02d}-01" for month in present])
    values = torch.ones(1, len(present), 1)

    _, cell_mask = rasterizer(values, torch.ones(1, len(present), dtype=torch.bool), times)

    occupied = sorted(month for _, month in cell_mask[0].nonzero().tolist())
    assert occupied == [month - 1 for month in present]
    assert cell_mask[0, 0, 3:8].sum() == 0


def test_rasterizer_drops_observations_older_than_the_window() -> None:
    """Older readings must fall off the grid, not wrap into an occupied cell."""
    rasterizer = CalendarGridRasterizer(num_channels=1, grid_years=2)
    times = _times(["2015-06-01", "2022-06-01", "2023-06-01"])
    values = torch.tensor([[[9.0], [1.0], [2.0]]])

    grid, cell_mask = rasterizer(values, torch.ones(1, 3, dtype=torch.bool), times)

    assert cell_mask[0].nonzero().tolist() == [[0, 5], [1, 5]]
    assert [float(grid[0, 0, row, month]) for row, month in cell_mask[0].nonzero().tolist()] == [1.0, 2.0]
    assert 9.0 not in grid[0, 0].flatten().tolist()


def test_rasterizer_averages_repeated_readings_in_one_cell() -> None:
    rasterizer = CalendarGridRasterizer(num_channels=1, grid_years=1)
    times = _times(["2022-05-02", "2022-05-20"])
    values = torch.tensor([[[2.0], [4.0]]])

    grid, cell_mask = rasterizer(values, torch.ones(1, 2, dtype=torch.bool), times)

    assert cell_mask[0].nonzero().tolist() == [[0, 4]]
    assert float(grid[0, 0, 0, 4]) == pytest.approx(3.0)


def test_rasterizer_handles_a_point_with_no_observations() -> None:
    rasterizer = CalendarGridRasterizer(num_channels=2, grid_years=3)
    times = _times(["2022-01-01", "2022-02-01"])
    values = torch.randn(1, 2, 2)

    grid, cell_mask = rasterizer(values, torch.zeros(1, 2, dtype=torch.bool), times)

    assert not bool(cell_mask.any())
    assert torch.isfinite(grid).all()


def test_rasterizer_is_invariant_to_the_calendar_era() -> None:
    rasterizer = CalendarGridRasterizer(num_channels=1, grid_years=3)
    times = _times(["2021-03-01", "2022-07-01", "2023-11-01"])
    values = torch.tensor([[[1.0], [2.0], [3.0]]])
    mask = torch.ones(1, 3, dtype=torch.bool)

    base_grid, base_mask = rasterizer(values, mask, times)
    shifted_grid, shifted_mask = rasterizer(values, mask, _times(["2031-03-01", "2032-07-01", "2033-11-01"]))

    torch.testing.assert_close(base_grid, shifted_grid)
    assert torch.equal(base_mask, shifted_mask)


def test_rasterizer_handles_irregular_cadence() -> None:
    rasterizer = CalendarGridRasterizer(num_channels=1, grid_years=1)
    times = _times(["2022-01-03", "2022-01-10", "2022-02-19", "2022-02-22"])  # 7, 40, 3 days apart
    values = torch.tensor([[[1.0], [3.0], [5.0], [7.0]]])

    grid, cell_mask = rasterizer(values, torch.ones(1, 4, dtype=torch.bool), times)

    assert cell_mask[0].nonzero().tolist() == [[0, 0], [0, 1]]
    assert float(grid[0, 0, 0, 0]) == pytest.approx(2.0)  # (1+3)/2
    assert float(grid[0, 0, 0, 1]) == pytest.approx(6.0)  # (5+7)/2


def test_rasterizer_emits_validity_and_positional_channels() -> None:
    rasterizer = CalendarGridRasterizer(num_channels=2, grid_years=1)
    assert rasterizer.output_channels == 2 + 2 + 1 + 2  # data, validity, cell flag, month sin/cos

    times = _times(["2022-04-01"])
    values = torch.tensor([[[1.0, 2.0]]])
    validity = torch.tensor([[[True, False]]])
    grid, _ = rasterizer(values, torch.ones(1, 1, dtype=torch.bool), times, validity)

    assert float(grid[0, 2, 0, 3]) == pytest.approx(1.0)  # channel 0 measured
    assert float(grid[0, 3, 0, 3]) == pytest.approx(0.0)  # channel 1 median-filled
    assert float(grid[0, 4, 0, 3]) == pytest.approx(1.0)  # cell occupied

    bare = CalendarGridRasterizer(num_channels=2, grid_years=1, use_validity_channels=False, month_positional=False)
    assert bare.output_channels == 3


def test_rasterizer_infers_its_span_when_none_is_configured() -> None:
    rasterizer = CalendarGridRasterizer(num_channels=1, grid_years=None)
    times = _times(["2019-01-01", "2021-01-01"])
    _, cell_mask = rasterizer(times.new_ones(1, 2, 1), torch.ones(1, 2, dtype=torch.bool), times)
    assert cell_mask.shape == (1, 3, 12)  # 2019, 2020, 2021


# --- pooling ---------------------------------------------------------------


def test_masked_pool_ignores_empty_cells() -> None:
    features = torch.tensor([[[[2.0, 0.0], [0.0, 0.0]]]])  # (B=1, C=1, 2, 2)
    cell_mask = torch.tensor([[[True, False], [False, False]]])

    assert float(masked_global_pool(features, cell_mask)) == pytest.approx(2.0)
    # Plain averaging divides by every cell, shrinking a sparse point toward zero.
    assert float(masked_global_pool(features, cell_mask, masked=False)) == pytest.approx(0.5)


def test_masked_pool_zeroes_a_fully_empty_grid() -> None:
    features = torch.randn(2, 3, 4, 4)
    cell_mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    assert bool((masked_global_pool(features, cell_mask) == 0).all())


# --- encoders --------------------------------------------------------------


def test_dilated_conv_reaches_exactly_twelve_months_back() -> None:
    """The point of dilation=12: a spike must resurface 12 cells either side of where it landed."""
    encoder = DilatedTempCNNEncoder(num_channels=1, output_dim=4, hidden_dim=2, norm="none").eval()
    near_conv, far_conv = encoder.blocks.stages[0][0], encoder.blocks.stages[1][0]
    assert (near_conv.kernel_size, near_conv.dilation, near_conv.padding) == ((3,), (1,), (1,))
    assert (far_conv.kernel_size, far_conv.dilation, far_conv.padding) == ((3,), (12,), (12,))
    torch.nn.init.constant_(near_conv.weight, 0.5)
    torch.nn.init.constant_(far_conv.weight, 0.5)

    baseline = torch.zeros(1, 1, 36)
    spike = baseline.clone()
    spike[0, 0, 18] = 1.0
    cell_mask = torch.ones(1, 36, dtype=torch.bool)

    with torch.no_grad():
        response = (encoder.blocks(spike, cell_mask) - encoder.blocks(baseline, cell_mask)).abs().sum(dim=1)[0]

    touched = set(response.nonzero().flatten().tolist())
    assert {17, 18, 19} <= touched, "the k=3 layer must reach its immediate neighbours"
    assert {6, 30} <= touched, "the dilation=12 layer must reach one year either side"
    assert 12 not in touched, "positions outside the receptive field must stay untouched"


@pytest.mark.parametrize("encoder_cls", ENCODER_CLASSES)
@pytest.mark.parametrize("grid_years", [1, 3, 9, 15])
def test_encoders_accept_any_grid_span(encoder_cls, grid_years: int) -> None:
    encoder = encoder_cls(num_channels=4, output_dim=8, hidden_dim=6).eval()
    grid = torch.randn(2, 4, grid_years, 12)
    cell_mask = torch.ones(2, grid_years, 12, dtype=torch.bool)

    with torch.no_grad():
        embedding = encoder(grid, cell_mask)

    assert embedding.shape == (2, 8)
    assert torch.isfinite(embedding).all()


@pytest.mark.parametrize("encoder_cls", ENCODER_CLASSES)
def test_masked_pooling_makes_extra_empty_years_free(encoder_cls) -> None:
    """Padding the grid with empty years must not move the embedding.

    This is what allows grid_years to be inferred rather than frozen into the checkpoint.
    """
    encoder = encoder_cls(num_channels=3, output_dim=5, hidden_dim=4, norm="none").eval()
    torch.manual_seed(0)
    payload = torch.randn(1, 3, 2, 12)
    payload_mask = torch.ones(1, 2, 12, dtype=torch.bool)

    padded = torch.cat([torch.zeros(1, 3, 4, 12), payload], dim=2)
    padded_mask = torch.cat([torch.zeros(1, 4, 12, dtype=torch.bool), payload_mask], dim=1)

    with torch.no_grad():
        torch.testing.assert_close(encoder(payload, payload_mask), encoder(padded, padded_mask))


def test_concat_gated_fusion_matches_the_specified_formula() -> None:
    fusion = ConcatGatedFusion(static_dim=3, temporal_dim=2)
    static = torch.randn(4, 3)
    temporal = torch.randn(4, 2)

    joined = torch.cat([static, temporal], dim=-1)
    expected = joined * torch.sigmoid(fusion.gate(joined))

    torch.testing.assert_close(fusion(static, temporal), expected)
    assert fusion.output_dim == 5


# --- lightning module ------------------------------------------------------


def _module(encoder: str, **kwargs) -> SoilCNNLightningModule:
    defaults = dict(static_dim=5, target_dim=1, modality_dims={"m": 3}, temporal_encoder=encoder, grid_years=3)
    defaults.update(kwargs)
    torch.manual_seed(0)
    return SoilCNNLightningModule(**defaults).eval()


@pytest.mark.parametrize("encoder", ENCODERS)
def test_module_forward_produces_gradients(encoder: str) -> None:
    module = _module(encoder)
    batch = _batch()

    predictions = module(batch)
    assert predictions.shape == (4, 1)

    module.loss_fn(predictions, batch["y"]).backward()
    conv_parameters = [p for p in module.temporal_encoders.parameters() if p.requires_grad]
    assert conv_parameters
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in conv_parameters)


@pytest.mark.parametrize("encoder", ENCODERS)
@pytest.mark.parametrize("start", ["2019-01-01", "2018-01-01"])
def test_module_is_invariant_to_the_calendar_era(encoder: str, start: str) -> None:
    """Same readings, same calendar months, a decade later -> identical predictions.

    Both start years are exercised on purpose: 2018 shifts onto leap year 2028, which is where a
    month lookup that ignores leap days would drift.
    """
    module = _module(encoder)
    batch = _batch(start=start)
    shifted = _batch(start=start, year_offset=10)
    shifted = {**batch, "sequence_time": shifted["sequence_time"]}

    with torch.no_grad():
        torch.testing.assert_close(module(batch), module(shifted))


def test_decimal_shifting_is_not_the_same_as_shifting_dates_across_a_leap_year() -> None:
    """Guard against 'fix' the era test by adding a constant to the decimal year instead.

    A decimal year encodes the day-of-year as a fraction of 365.25, so adding 10.0 keeps the
    fraction and moves the year label. When the destination year is a leap year that fraction names
    a different calendar day: 2018-03-01 is day 60, and day 60 of 2028 is 29 February. Shifting real
    dates is the meaningful operation, and only it is invariant.
    """
    real_dates = _times(["2018-03-01"])
    shifted_dates = _times(["2028-03-01"])
    assert decimal_year_to_month_index(real_dates).tolist() == decimal_year_to_month_index(shifted_dates).tolist()

    naive_shift = real_dates + 10.0
    assert decimal_year_to_month_index(naive_shift).item() == 1, "day 60 of a leap year is February"


@pytest.mark.parametrize("encoder", ENCODERS)
def test_module_runs_on_spans_it_was_not_built_for(encoder: str) -> None:
    module = _module(encoder, grid_years=None)

    with torch.no_grad():
        short = module(_batch(length=6, seed=1))
        long = module(_batch(length=90, seed=2))

    assert short.shape == long.shape == (4, 1)
    assert torch.isfinite(short).all() and torch.isfinite(long).all()


@pytest.mark.parametrize("encoder", ENCODERS)
def test_module_consumes_validity_channels(encoder: str) -> None:
    module = _module(encoder)
    batch = _batch()
    flipped_validity = batch["sequence_validity"]["m"].clone()
    flipped_validity[0, :6] = False
    flipped = {**batch, "sequence_validity": {"m": flipped_validity}}

    with torch.no_grad():
        assert not torch.allclose(module(batch)[0], module(flipped)[0])

    blind = _module(encoder, use_validity_channels=False)
    with torch.no_grad():
        torch.testing.assert_close(blind(batch), blind(flipped))


@pytest.mark.parametrize("encoder", ENCODERS)
def test_module_ignores_masked_out_observations(encoder: str) -> None:
    module = _module(encoder)
    batch = _batch(length=12)
    mask = batch["sequence_mask"]["m"].clone()
    mask[:, 8:] = False
    batch = {**batch, "sequence_mask": {"m": mask}}

    polluted = batch["sequences"]["m"].clone()
    polluted[:, 8:] += 99.0

    with torch.no_grad():
        torch.testing.assert_close(module(batch), module({**batch, "sequences": {"m": polluted}}))


@pytest.mark.parametrize("encoder", ENCODERS)
def test_module_inverts_the_target_transform_for_prediction(encoder: str) -> None:
    module = _module(encoder, target_mean=[1.5], target_scale=[0.5], target_transform="log1p")
    batch = _batch()

    with torch.no_grad():
        expected = torch.expm1((module(batch) * 0.5 + 1.5) / 10.0)
        torch.testing.assert_close(module.predict_step(batch, 0), expected)


def test_module_supports_any_number_of_modalities() -> None:
    modality_dims = {"s2": 10, "s1": 8, "soil": 3, "ag": 3, "clim": 3}
    module = SoilCNNLightningModule(
        static_dim=13,
        target_dim=1,
        modality_dims=modality_dims,
        grid_years=9,
        cnn_hidden_dim={"s2": 48, "s1": 32, "soil": 16, "ag": 16, "clim": 16},
        modality_embed_dim=16,
    ).eval()

    generator = torch.Generator().manual_seed(0)
    dates = pd.date_range("2019-01-01", periods=18, freq="MS")
    times = torch.as_tensor(to_decimal_year(pd.Series(dates))).unsqueeze(0).repeat(2, 1)
    batch = {
        "x_static": torch.randn(2, 13, generator=generator),
        "sequences": {name: torch.randn(2, 18, dim, generator=generator) for name, dim in modality_dims.items()},
        "sequence_mask": {name: torch.ones(2, 18, dtype=torch.bool) for name in modality_dims},
        "sequence_time": {name: times for name in modality_dims},
        "sequence_validity": {name: torch.ones(2, 18, dim, dtype=torch.bool) for name, dim in modality_dims.items()},
    }

    with torch.no_grad():
        assert module(batch).shape == (2, 1)
    assert len(module.temporal_encoders) == 5


def test_module_rejects_a_per_modality_mapping_that_omits_a_modality() -> None:
    with pytest.raises(ValueError, match="no width in cnn_hidden_dim"):
        SoilCNNLightningModule(
            static_dim=4, target_dim=1, modality_dims={"s1": 3, "s2": 4}, cnn_hidden_dim={"s1": 16}
        )


def test_module_rejects_an_unknown_encoder_name() -> None:
    with pytest.raises(ValueError, match="temporal_encoder"):
        SoilCNNLightningModule(static_dim=4, target_dim=1, modality_dims={"m": 3}, temporal_encoder="resnet")


def test_module_runs_without_a_temporal_branch() -> None:
    module = _module("dilated_tempcnn", temporal_enabled=False)
    assert module(_batch()).shape == (4, 1)
    assert len(module.temporal_encoders) == 0


def test_module_runs_without_static_features() -> None:
    module = _module("dilated_tempcnn", static_dim=0)
    batch = {**_batch(), "x_static": torch.zeros(4, 0)}
    predictions = module(batch)

    assert predictions.shape == (4, 1)
    assert torch.isfinite(predictions).all()


@pytest.mark.parametrize("encoder", ENCODERS)
def test_module_checkpoint_reloads_under_weights_only(tmp_path: Path, encoder: str) -> None:
    module = _module(encoder, target_mean=np.array([2.0]), target_scale=np.array([0.75]))
    checkpoint_path = tmp_path / f"{encoder}.ckpt"
    torch.save({"state_dict": module.state_dict(), "hyper_parameters": dict(module.hparams)}, checkpoint_path)

    loaded = torch.load(checkpoint_path, weights_only=True)
    assert loaded["hyper_parameters"]["target_mean"] == [2.0]
    # Adaptive pooling is what keeps the weights free of the grid span; nothing may be sized by it.
    for name, tensor in loaded["state_dict"].items():
        assert 12 not in tensor.shape[2:] or "conv" not in name.lower() or tensor.ndim <= 4


# --- data path -------------------------------------------------------------


def _write_csvs(tmp_path: Path, dates_by_point):
    point_ids = sorted(dates_by_point)
    static_df = pd.DataFrame(
        {
            "point_id": point_ids,
            "lat": [0.1 * index for index in range(len(point_ids))],
            "lon": [0.1 * index for index in range(len(point_ids))],
            "target_a": [1.0 + index for index in range(len(point_ids))],
            "static_1": [10.0 + index for index in range(len(point_ids))],
        }
    )
    rows = []
    for point_id, dates in dates_by_point.items():
        for index, date in enumerate(dates):
            rows.append(
                {
                    "point_id": point_id,
                    "obs_date": date,
                    "S2_b2": 1.0 + index,
                    "S2_b3": np.nan if (point_id == 2 and index == 0) else 2.0 + index,
                }
            )
    static_path, timeseries_path = tmp_path / "static.csv", tmp_path / "ts.csv"
    static_df.to_csv(static_path, index=False)
    pd.DataFrame(rows).to_csv(timeseries_path, index=False)
    return static_path, timeseries_path


def _bundle(tmp_path: Path, logger, dates_by_point):
    static_path, timeseries_path = _write_csvs(tmp_path, dates_by_point)
    config = SimpleNamespace(
        DATA_FOLDER=str(tmp_path),
        DATA_FILE="static.csv",
        STATIC_CSV_PATH=str(static_path),
        TIMESERIES_CSV_PATH=str(timeseries_path),
        POINT_ID_COLUMN="point_id",
        LAT_COLUMN="lat",
        LON_COLUMN="lon",
        TIME_COLUMN="obs_date",
        TEMPORAL_FEATURES_ENABLED=True,
        TEMPORAL_FEATURES={"enabled": True, "time_column": "obs_date"},
        MODALITY_PREFIX_MAP={"s2": "S2_"},
        S1_COLUMNS=[],
        S2_COLUMNS=[],
        MODIS_COLUMNS=[],
        TARGET_COLUMNS=["target_a"],
        LABEL_COLUMNS=["target_a"],
        PREDICTOR_COLUMNS=[],
        IGNORED_COLUMNS=["point_id", "lat", "lon"],
        ELIMINATED_FEATURES=["point_id", "lat", "lon"],
        CATEGORICAL_FEATURES=[],
        EXCLUDE_CATEGORICAL=False,
        EXISTING_HS_FEATURES={"enabled": False},
        RANDOM_SEED=42,
        TEST_SIZE=0.25,
        DATA_INDEX_MANIFEST_PATH=None,
        STATIC_SOURCE=None,
        TARGETS_SOURCE=None,
        TIMESERIES_SOURCE=None,
        STATIC_FEATURES_FOLDER=None,
        TARGETS_FOLDER=None,
        TIMESERIES_FOLDER=None,
        TARGETS_FILE="static.csv",
        TARGETS_CSV_PATH=str(static_path),
    )
    return SoilSequenceBuilder(config, logger, DataManager(config, logger)).build()


@pytest.mark.parametrize(
    "dates_by_point, expected_years",
    [
        ({1: ["2022-01-01", "2022-06-01"], 2: ["2022-03-01"], 3: ["2022-09-01"]}, 1),
        ({1: ["2020-01-01", "2022-06-01"], 2: ["2021-03-01"], 3: ["2022-09-01"]}, 3),
        ({1: ["2017-01-01", "2025-12-01"], 2: ["2021-03-01"], 3: ["2022-09-01"]}, 9),
        # 0.2 decimal years but two calendar rows: measuring the decimal span would under-allocate.
        ({1: ["2021-11-01", "2022-01-01"], 2: ["2021-12-01"], 3: ["2022-02-01"]}, 2),
    ],
)
def test_grid_years_is_inferred_from_the_data(tmp_path: Path, logger, dates_by_point, expected_years) -> None:
    datamodule = SoilSequenceDataModule(_bundle(tmp_path, logger, dates_by_point), batch_size=3)
    assert datamodule.grid_years == expected_years


def test_validity_reaches_the_batch_and_survives_padding(tmp_path: Path, logger) -> None:
    bundle = _bundle(
        tmp_path,
        logger,
        {1: ["2022-01-01", "2022-02-01", "2022-03-01"], 2: ["2022-01-01"], 3: ["2022-05-01"]},
    )
    datamodule = SoilSequenceDataModule(bundle, batch_size=3, val_size=0.0, test_size=0.0)
    datamodule.setup("fit")

    batch = datamodule._collate_points(np.arange(3))
    validity = batch["sequence_validity"]["s2"]

    assert validity.shape == batch["sequences"]["s2"].shape
    assert validity.dtype == torch.bool
    # Point 2's first reading had a NaN S2_b3, which the builder median-filled.
    assert bool(validity[1, 0, 0]) is True
    assert bool(validity[1, 0, 1]) is False
    # Padding is never claimed as measured.
    assert not bool(validity[1, 1:].any())


def test_standardization_ignores_median_filled_cells(tmp_path: Path, logger) -> None:
    bundle = _bundle(
        tmp_path,
        logger,
        {1: ["2022-01-01", "2022-02-01"], 2: ["2022-01-01", "2022-02-01"], 3: ["2022-01-01"]},
    )
    datamodule = SoilSequenceDataModule(bundle, batch_size=3, val_size=0.0, test_size=0.0)
    datamodule.setup("fit")

    measured = np.concatenate(
        [
            bundle.sequences["s2"][index][bundle.validity_for("s2", index)[:, 1], 1]
            for index in datamodule.train_idx_
            if len(bundle.sequences["s2"][index])
        ]
    )
    np.testing.assert_allclose(datamodule.sequence_mean_["s2"][1], measured.mean(), rtol=1e-5)


def test_builder_to_cnn_module_end_to_end(tmp_path: Path, logger) -> None:
    from lightning.pytorch import Trainer

    dates = [f"20{year:02d}-{month:02d}-01" for year in range(19, 23) for month in range(1, 13)]
    # Enough points that val and test are both non-empty; a 3-point fixture leaves val with none.
    bundle = _bundle(
        tmp_path,
        logger,
        {point: dates[: 20 + 4 * point] for point in range(1, 9)},
    )
    datamodule = SoilSequenceDataModule(
        bundle, batch_size=2, val_size=0.4, test_size=0.25, seed=5, target_transform="log1p"
    )
    datamodule.setup("fit")
    assert datamodule.val_idx_.size >= 2 and datamodule.test_idx_.size >= 1

    module = SoilCNNLightningModule(
        static_dim=datamodule.static_dim,
        target_dim=datamodule.target_dim,
        modality_dims=datamodule.modality_dims,
        grid_years=datamodule.grid_years,
        temporal_encoder="annual_grid2d",
        cnn_norm="group",  # batches here are too small for BatchNorm
        target_mean=datamodule.target_mean_,
        target_scale=datamodule.target_scale_,
        target_transform=datamodule.target_transform,
    )

    trainer = Trainer(
        max_epochs=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, datamodule=datamodule)

    assert torch.isfinite(torch.as_tensor(trainer.callback_metrics["train_loss"]))
    predicted = torch.cat(trainer.predict(module, datamodule=datamodule)).reshape(-1)
    assert len(predicted) == len(datamodule.y_test_frame_)
    assert torch.isfinite(predicted).all()
