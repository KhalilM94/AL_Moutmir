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


LABEL_NAMES = ["lab_a", "lab_b", "lab_c"]


def _batch(
    batch_size=4, length=24, channels=3, seed=0, start="2019-01-01", months_step=1, year_offset=0, labels=3
):
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
        # Appended last so the generator draws above keep their values and every existing
        # expectation in this file still holds.
        "x_labels": torch.randn(batch_size, labels, generator=generator),
        "x_label_validity": torch.ones(batch_size, labels, dtype=torch.bool),
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
    encoder = DilatedTempCNNEncoder(num_channels=1, output_dim=4, hidden_dims=[2], norm="none").eval()
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
    encoder = encoder_cls(num_channels=4, output_dim=8, hidden_dims=[6]).eval()
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
    encoder = encoder_cls(num_channels=3, output_dim=5, hidden_dims=[4], norm="none").eval()
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
        cnn_hidden_dims={"s2": [48], "s1": [32], "soil": [16], "ag": [16], "clim": [16]},
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
    with pytest.raises(ValueError, match="no entry in cnn_hidden_dims"):
        SoilCNNLightningModule(
            static_dim=4, target_dim=1, modality_dims={"s1": 3, "s2": 4}, cnn_hidden_dims={"s1": [16]}
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


# --- auxiliary lab columns ---------------------------------------------------


def _auxiliary_module(**kwargs) -> SoilCNNLightningModule:
    defaults = dict(
        target_names=["target_a"],
        auxiliary_available_names=LABEL_NAMES,
        auxiliary_label_columns=["lab_a", "lab_c"],
    )
    defaults.update(kwargs)
    return _module("dilated_tempcnn", **defaults)


def test_a_selected_lab_column_reaches_the_head_and_an_unselected_one_does_not() -> None:
    """The whole point of the feature, and the guard that selection is by position not by luck."""
    module = _auxiliary_module()
    batch = _batch()
    baseline = module(batch)

    for column, expected_change in (("lab_a", True), ("lab_b", False), ("lab_c", True)):
        perturbed = {**batch, "x_labels": batch["x_labels"].clone()}
        perturbed["x_labels"][:, LABEL_NAMES.index(column)] += 5.0
        changed = not torch.allclose(module(perturbed), baseline, atol=1e-6)
        assert changed is expected_change, f"{column} should{'' if expected_change else ' not'} affect the head"


def test_validity_flags_reach_the_head_too() -> None:
    module = _auxiliary_module()
    batch = _batch()

    flipped = {**batch, "x_label_validity": batch["x_label_validity"].clone()}
    flipped["x_label_validity"][:, LABEL_NAMES.index("lab_a")] = False

    assert not torch.allclose(module(flipped), module(batch), atol=1e-6)


def test_validity_channels_can_be_switched_off() -> None:
    module = _auxiliary_module(auxiliary_validity_channels=False)
    batch = _batch()

    flipped = {**batch, "x_label_validity": torch.zeros_like(batch["x_label_validity"])}
    assert torch.allclose(module(flipped), module(batch), atol=1e-6)
    # Two selected columns, no flags -> the raw branch is 2 wide rather than 4.
    assert module.auxiliary_output_dim == 2


def test_selecting_a_column_that_is_also_a_target_is_refused() -> None:
    """A filter would be worse than an error: the run would train a model that reads its answer."""
    with pytest.raises(ValueError, match="may not name a column being fitted"):
        _auxiliary_module(
            target_names=["target_a", "lab_c"],
            auxiliary_available_names=[*LABEL_NAMES, "target_a"],
            auxiliary_label_columns=["lab_a", "lab_c"],
        )


def test_selecting_an_unknown_column_names_what_is_available() -> None:
    with pytest.raises(ValueError, match=r"does not carry.*lab_zzz"):
        _auxiliary_module(auxiliary_label_columns=["lab_a", "lab_zzz"])


def test_selecting_a_duplicate_column_is_refused() -> None:
    with pytest.raises(ValueError, match="duplicate column"):
        _auxiliary_module(auxiliary_label_columns=["lab_a", "lab_a"])


def test_auxiliary_columns_without_target_names_are_refused() -> None:
    """Skipping the leakage check silently is exactly the failure this feature could cause."""
    with pytest.raises(ValueError, match="requires target_names"):
        _auxiliary_module(target_names=[])


def test_raw_concat_widens_the_head_by_the_column_count() -> None:
    module = _auxiliary_module(auxiliary_hidden_dims=[])
    head_input = next(layer for layer in module.output_head.modules() if isinstance(layer, torch.nn.Linear))

    # 2 selected columns + 2 validity flags, appended straight onto the fused vector.
    assert module.auxiliary_output_dim == 4
    assert isinstance(module.auxiliary_encoder, torch.nn.Identity)
    assert head_input.in_features == module.fusion.output_dim + 4


def test_an_encoder_width_decouples_the_head_from_the_column_count() -> None:
    module = _auxiliary_module(auxiliary_hidden_dims=[16, 8])
    head_input = next(layer for layer in module.output_head.modules() if isinstance(layer, torch.nn.Linear))

    assert module.auxiliary_output_dim == 8
    assert head_input.in_features == module.fusion.output_dim + 8


def test_no_auxiliary_columns_leaves_the_architecture_untouched() -> None:
    """The default must be byte-for-byte the model that existed before this feature."""
    plain = _module("dilated_tempcnn")
    head_input = next(layer for layer in plain.output_head.modules() if isinstance(layer, torch.nn.Linear))

    assert plain.has_auxiliary_labels is False
    assert plain.auxiliary_output_dim == 0
    assert head_input.in_features == plain.fusion.output_dim
    # No target_names needed, and a batch without the lab keys still runs.
    batch = {key: value for key, value in _batch().items() if not key.startswith("x_label")}
    assert torch.isfinite(plain(batch)).all()


def test_a_batch_that_no_longer_matches_the_label_roster_is_refused() -> None:
    """Positions were resolved at build time; a narrower batch would read a different measurement."""
    module = _auxiliary_module()
    batch = _batch(labels=2)

    with pytest.raises(ValueError, match="no longer matches the checkpoint's label roster"):
        module(batch)


def test_auxiliary_selection_survives_a_weights_only_checkpoint_round_trip(tmp_path: Path) -> None:
    module = _auxiliary_module(auxiliary_hidden_dims=[16])
    checkpoint_path = tmp_path / "auxiliary.ckpt"
    torch.save({"state_dict": module.state_dict(), "hyper_parameters": dict(module.hparams)}, checkpoint_path)

    loaded = torch.load(checkpoint_path, weights_only=True)
    assert loaded["hyper_parameters"]["auxiliary_label_columns"] == ["lab_a", "lab_c"]
    assert loaded["hyper_parameters"]["auxiliary_available_names"] == LABEL_NAMES
    # The resolved positions travel with the weights rather than being re-derived on load.
    assert loaded["state_dict"]["auxiliary_index"].tolist() == [0, 2]


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


def _write_csvs(tmp_path: Path, dates_by_point, split: bool = False):
    """Write the fixture as one joint file, or as separate static and targets files.

    `split` is not decoration: the targets join used to carry only the ACTIVE targets, so the two
    layouts disagreed about which lab columns a model could select from the very same declaration.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    point_ids = sorted(dates_by_point)
    static_df = pd.DataFrame(
        {
            "point_id": point_ids,
            "lat": [0.1 * index for index in range(len(point_ids))],
            "lon": [0.1 * index for index in range(len(point_ids))],
            "target_a": [1.0 + index for index in range(len(point_ids))],
            "static_1": [10.0 + index for index in range(len(point_ids))],
            # Measured lab values: named in LABEL_COLUMNS so they are never features, but carried
            # on the bundle so a model may opt into them. lab_sparse is deliberately incomplete.
            "lab_dense": [100.0 + 10.0 * index for index in range(len(point_ids))],
            "lab_sparse": [
                np.nan if index % 2 else 5.0 + index for index in range(len(point_ids))
            ],
        }
    )
    targets_df = static_df
    if split:
        # The lab values live with the targets, which is where a real targets file keeps them.
        lab_columns = ["point_id", "target_a", "lab_dense", "lab_sparse"]
        targets_df = static_df[lab_columns]
        static_df = static_df.drop(columns=[column for column in lab_columns if column != "point_id"])

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
    targets_path = tmp_path / "targets.csv" if split else static_path
    static_df.to_csv(static_path, index=False)
    if split:
        targets_df.to_csv(targets_path, index=False)
    pd.DataFrame(rows).to_csv(timeseries_path, index=False)
    return static_path, timeseries_path, targets_path


def _bundle(tmp_path: Path, logger, dates_by_point, carry_labels: bool = True, split: bool = False):
    static_path, timeseries_path, targets_path = _write_csvs(tmp_path, dates_by_point, split=split)
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
        LABEL_COLUMNS=["target_a", "lab_dense", "lab_sparse"],
        CARRY_LABEL_COLUMNS=carry_labels,
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
        TARGETS_FILE=targets_path.name,
        TARGETS_CSV_PATH=str(targets_path),
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


def test_label_columns_travel_on_the_bundle_without_becoming_features(tmp_path: Path, logger) -> None:
    bundle = _bundle(
        tmp_path,
        logger,
        {1: ["2022-01-01", "2022-02-01"], 2: ["2022-01-01"], 3: ["2022-03-01"]},
    )

    # Every LABEL_COLUMNS entry the frame carries, target included - selection happens in the model.
    assert bundle.label_feature_names == ["target_a", "lab_dense", "lab_sparse"]
    assert bundle.label_dim == 3
    # ...and none of them leaked into the predictors.
    assert bundle.static_feature_names == ["static_1"]
    # NaN is preserved at build time: the fill value is a train-split median and the split does not
    # exist yet.
    assert bundle.label_missing_fraction("lab_sparse") > 0
    assert bundle.label_missing_fraction("lab_dense") == 0
    assert not np.isfinite(bundle.label_features[:, 2]).all()


def test_a_split_targets_file_offers_the_same_lab_columns_as_a_joint_one(tmp_path: Path, logger) -> None:
    """The regression: the join used to carry only the ACTIVE targets, so the same LABEL_COLUMNS
    declaration meant 3 selectable columns on a joint file and 1 on split files."""
    dates = {1: ["2022-01-01", "2022-02-01"], 2: ["2022-01-01"], 3: ["2022-03-01"]}
    joint = _bundle(tmp_path / "joint", logger, dates)
    split = _bundle(tmp_path / "split", logger, dates, split=True)

    assert split.label_feature_names == joint.label_feature_names == ["target_a", "lab_dense", "lab_sparse"]
    np.testing.assert_array_equal(
        np.isfinite(split.label_features), np.isfinite(joint.label_features)
    )
    # The join must not have promoted anything: features come from filter_schema either way.
    assert split.static_feature_names == joint.static_feature_names == ["static_1"]


@pytest.mark.parametrize("split", [False, True])
def test_the_carry_flag_off_leaves_no_lab_columns_on_the_bundle(tmp_path: Path, logger, split: bool) -> None:
    bundle = _bundle(
        tmp_path,
        logger,
        {1: ["2022-01-01", "2022-02-01"], 2: ["2022-01-01"], 3: ["2022-03-01"]},
        carry_labels=False,
        split=split,
    )

    assert bundle.label_feature_names == []
    assert bundle.label_dim == 0
    # Everything else is untouched, so a run that never opted in is unaffected.
    assert bundle.static_feature_names == ["static_1"]
    assert bundle.target_names == ["target_a"]


def test_the_carry_flag_off_produces_batches_without_lab_values(tmp_path: Path, logger) -> None:
    bundle = _bundle(
        tmp_path,
        logger,
        {1: ["2022-01-01", "2022-02-01"], 2: ["2022-01-01"], 3: ["2022-03-01"]},
        carry_labels=False,
    )
    datamodule = SoilSequenceDataModule(bundle, batch_size=3, val_size=0.0, test_size=0.0)
    datamodule.setup("fit")

    batch = datamodule._collate_points(np.arange(3))
    assert datamodule.label_feature_names == []
    assert batch["x_labels"].shape == (3, 0)
    assert datamodule.label_median_ is None


def test_selecting_a_column_with_nothing_carried_names_the_flag(tmp_path: Path, logger) -> None:
    """The message the failing run should have shown: the config was right, the flag was off."""
    bundle = _bundle(
        tmp_path,
        logger,
        {1: ["2022-01-01", "2022-02-01"], 2: ["2022-01-01"], 3: ["2022-03-01"]},
        carry_labels=False,
    )
    datamodule = SoilSequenceDataModule(bundle, batch_size=3, val_size=0.0, test_size=0.0)
    datamodule.setup("fit")

    with pytest.raises(ValueError, match="CARRY_LABEL_COLUMNS"):
        SoilCNNLightningModule(
            static_dim=datamodule.static_dim,
            target_dim=datamodule.target_dim,
            target_names=datamodule.target_names,
            modality_dims=datamodule.modality_dims,
            auxiliary_available_names=datamodule.label_feature_names,
            auxiliary_label_columns=["lab_dense"],
        )


def test_lab_values_reach_the_batch_standardized_with_validity(tmp_path: Path, logger) -> None:
    bundle = _bundle(
        tmp_path,
        logger,
        {1: ["2022-01-01", "2022-02-01"], 2: ["2022-01-01"], 3: ["2022-03-01"]},
    )
    datamodule = SoilSequenceDataModule(bundle, batch_size=3, val_size=0.0, test_size=0.0)
    datamodule.setup("fit")

    assert datamodule.label_feature_names == ["target_a", "lab_dense", "lab_sparse"]
    batch = datamodule._collate_points(np.arange(3))

    assert batch["x_labels"].shape == (3, 3)
    assert batch["x_label_validity"].dtype == torch.bool
    # Filling happens, but never silently: the flag is what tells a fill from a measurement.
    sparse_valid = batch["x_label_validity"][:, 2]
    assert not bool(sparse_valid.all())
    assert torch.isfinite(batch["x_labels"]).all()


def test_lab_fill_and_scaling_come_from_the_train_split_only(tmp_path: Path, logger) -> None:
    """A median fitted over val/test would leak their distribution into every filled cell."""
    bundle = _bundle(
        tmp_path,
        logger,
        {point: ["2022-01-01", "2022-02-01"] for point in range(1, 9)},
    )
    datamodule = SoilSequenceDataModule(bundle, batch_size=2, val_size=0.25, test_size=0.25, seed=5)
    datamodule.setup("fit")

    column = bundle.label_feature_names.index("lab_sparse")
    train_values = bundle.label_features[datamodule.train_idx_, column]
    measured = train_values[np.isfinite(train_values)]

    assert measured.size and measured.size < train_values.size
    np.testing.assert_allclose(datamodule.label_median_[column], np.median(measured), rtol=1e-5)


def test_a_lab_column_with_no_measured_train_value_stays_inert(tmp_path: Path, logger) -> None:
    bundle = _bundle(
        tmp_path,
        logger,
        {1: ["2022-01-01", "2022-02-01"], 2: ["2022-01-01"], 3: ["2022-03-01"]},
    )
    column = bundle.label_feature_names.index("lab_sparse")
    bundle.label_features[:, column] = np.nan

    datamodule = SoilSequenceDataModule(bundle, batch_size=3, val_size=0.0, test_size=0.0)
    datamodule.setup("fit")
    batch = datamodule._collate_points(np.arange(3))

    # Nothing to fill from, so it becomes a constant the model can only ignore - not a NaN, and not
    # a fabricated centre.
    assert datamodule.label_median_[column] == 0.0
    assert torch.isfinite(batch["x_labels"]).all()
    assert not bool(batch["x_label_validity"][:, column].any())


def test_builder_to_cnn_module_with_auxiliary_labels_end_to_end(tmp_path: Path, logger) -> None:
    from lightning.pytorch import Trainer

    dates = [f"20{year:02d}-{month:02d}-01" for year in range(19, 23) for month in range(1, 13)]
    bundle = _bundle(tmp_path, logger, {point: dates[: 20 + 4 * point] for point in range(1, 9)})
    datamodule = SoilSequenceDataModule(bundle, batch_size=2, val_size=0.4, test_size=0.25, seed=5)
    datamodule.setup("fit")

    module = SoilCNNLightningModule(
        static_dim=datamodule.static_dim,
        target_dim=datamodule.target_dim,
        target_names=datamodule.target_names,
        modality_dims=datamodule.modality_dims,
        grid_years=datamodule.grid_years,
        temporal_encoder="annual_grid2d",
        cnn_norm="group",
        auxiliary_available_names=datamodule.label_feature_names,
        auxiliary_label_columns=["lab_dense", "lab_sparse"],
        auxiliary_hidden_dims=[8],
        target_mean=datamodule.target_mean_,
        target_scale=datamodule.target_scale_,
    )
    assert module.auxiliary_index.tolist() == [1, 2]

    trainer = Trainer(
        max_epochs=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, datamodule=datamodule)

    # The sparse column is partly train-median filled, so this also pins that the fill path does not
    # produce a NaN loss.
    assert torch.isfinite(torch.as_tensor(trainer.callback_metrics["train_loss"]))
    predicted = torch.cat(trainer.predict(module, datamodule=datamodule)).reshape(-1)
    assert torch.isfinite(predicted).all()


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


# --- per-block conv widths ---------------------------------------------------


@pytest.mark.parametrize("encoder_cls", [DilatedTempCNNEncoder, AnnualGrid2DEncoder])
def test_a_repeated_width_reproduces_the_old_num_blocks_stack(encoder_cls) -> None:
    """`hidden_dims=[64, 64]` must be exactly what `hidden_dim=64, num_blocks=2` built."""
    encoder = encoder_cls(num_channels=7, output_dim=8, hidden_dims=[64, 64])
    convs = [m for m in encoder.blocks.modules() if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d))]

    # Two conv stages per block, and the first of each block reads the previous block's width.
    assert [(c.in_channels, c.out_channels) for c in convs] == [
        (7, 64), (64, 64),      # block 1: raw channels in, then the dilated/inter-annual pass
        (64, 64), (64, 64),     # block 2
    ]
    assert encoder.projection.in_features == 64


@pytest.mark.parametrize("encoder_cls", [DilatedTempCNNEncoder, AnnualGrid2DEncoder])
def test_the_conv_stack_can_widen_across_blocks(encoder_cls) -> None:
    """The shape a single shared width could not express, and the conventional one for convs."""
    encoder = encoder_cls(num_channels=7, output_dim=8, hidden_dims=[32, 64, 128])
    convs = [m for m in encoder.blocks.modules() if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d))]

    assert [(c.in_channels, c.out_channels) for c in convs] == [
        (7, 32), (32, 32),
        (32, 64), (64, 64),
        (64, 128), (128, 128),
    ]
    assert encoder.projection.in_features == 128


@pytest.mark.parametrize("encoder_cls", [DilatedTempCNNEncoder, AnnualGrid2DEncoder])
def test_a_widening_stack_still_runs_end_to_end(encoder_cls) -> None:
    encoder = encoder_cls(num_channels=4, output_dim=8, hidden_dims=[8, 16], norm="none").eval()
    grid = torch.randn(2, 4, 3, 12)
    cell_mask = torch.ones(2, 3, 12, dtype=torch.bool)

    assert encoder(grid, cell_mask).shape == (2, 8)


@pytest.mark.parametrize("encoder_cls", [DilatedTempCNNEncoder, AnnualGrid2DEncoder])
def test_an_empty_conv_width_list_is_refused(encoder_cls) -> None:
    """Zero blocks would pool the raw rasterized channels: a different model, not a smaller one."""
    with pytest.raises(ValueError, match="at least one hidden width"):
        encoder_cls(num_channels=4, output_dim=8, hidden_dims=[])


def test_a_scalar_cnn_width_still_means_one_block() -> None:
    """`cnn_hidden_dims: 64` in a config is the same as `[64]`, so old spellings keep working."""
    module = _module("dilated_tempcnn", cnn_hidden_dims=64)

    encoder = module.temporal_encoders["m"]
    assert encoder.hidden_dims == [64]


def test_a_per_modality_mapping_may_carry_a_width_list_each() -> None:
    module = _module(
        "dilated_tempcnn",
        modality_dims={"s2": 3, "s1": 2},
        cnn_hidden_dims={"s2": [16, 32], "s1": [8]},
    )

    assert module.temporal_encoders["s2"].hidden_dims == [16, 32]
    assert module.temporal_encoders["s1"].hidden_dims == [8]
