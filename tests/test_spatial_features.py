"""The two switchable spatial-awareness features: harmonic coordinates and the context group.

Both are opt-in, and the property that matters most is what happens when they are OFF: the module
must have the parameter count and the state_dict key set it had before either existed, so a
checkpoint trained without them still loads. That is pinned first and hardest.

The second theme is the train bounding box. It is a fitted statistic like any scaler here, and it
has a failure mode the scalers do not: a serving request can be a SINGLE point, whose own bounding
box is degenerate, so a re-fit would normalize every served point to the centre of itself and every
prediction would silently be made at the middle of the study area.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from yg_eo_soilnet.data_manager import DataManager
from yg_eo_soilnet.datamodules.sequence.sequence_builder import SoilSequenceBuilder
from yg_eo_soilnet.datamodules.sequence.sequence_datamodule import SoilSequenceDataModule
from yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module import SoilCNNLightningModule
from yg_eo_soilnet.models.lightningmodules.spatial_encoders import HarmonicPositionEncoder
from yg_eo_soilnet.models.lightningmodules.temporal_cnn_encoders import ConcatGatedFusion

MODEL_ARGS = dict(
    static_dim=4,
    target_dim=1,
    modality_dims={"s2": 3},
    grid_years=2,
    cnn_hidden_dims=[8],
    modality_embed_dim=6,
    static_hidden_dims=[8],
    head_hidden_dims=[8],
    target_names=["target_a"],
)


def _batch(rows: int = 5, *, coords: bool = True, length: int = 4) -> dict:
    generator = torch.Generator().manual_seed(0)
    batch = {
        "x_static": torch.randn(rows, 4, generator=generator),
        "x_categorical": torch.zeros(rows, 0, dtype=torch.long),
        "sequences": {"s2": torch.randn(rows, length, 3, generator=generator)},
        "sequence_mask": {"s2": torch.ones(rows, length, dtype=torch.bool)},
        "sequence_time": {
            "s2": torch.tensor([[2020.1, 2020.5, 2021.2, 2021.8]] * rows, dtype=torch.float64)
        },
        "sequence_validity": {"s2": torch.ones(rows, length, 3, dtype=torch.bool)},
    }
    if coords:
        batch["x_coords"] = torch.rand(rows, 2, generator=generator) * 2 - 1
    return batch


# --- backward compatibility ------------------------------------------------


def test_coordinates_off_adds_no_parameters_and_no_state_dict_keys() -> None:
    """The load-bearing one: off must be indistinguishable from the module before the branch.

    Not merely "the same count" - the same KEYS, because a checkpoint is restored by key. A branch
    that registered even an unused buffer would make every existing soil_cnn checkpoint fail to
    load strictly.
    """
    off = SoilCNNLightningModule(**MODEL_ARGS)
    explicit_zero = SoilCNNLightningModule(**MODEL_ARGS, coord_dim=0)

    assert set(off.state_dict()) == set(explicit_zero.state_dict())
    assert sum(p.numel() for p in off.parameters()) == sum(
        p.numel() for p in explicit_zero.parameters()
    )
    assert not [key for key in off.state_dict() if "coord" in key]
    assert off.coordinate_output_dim == 0
    assert off.has_coordinates is False


def test_a_checkpoint_trained_without_coordinates_loads_strictly() -> None:
    """The consequence of the test above, stated as the thing users actually do."""
    trained_before = SoilCNNLightningModule(**MODEL_ARGS)
    rebuilt = SoilCNNLightningModule(**MODEL_ARGS)
    rebuilt.load_state_dict(trained_before.state_dict(), strict=True)


def test_two_argument_fusion_still_builds_the_module_it_always_did() -> None:
    """coordinate_dim=0 must not perturb the gate: it is the whole backward-compatibility story."""
    two = ConcatGatedFusion(static_dim=8, temporal_dim=6)
    three = ConcatGatedFusion(static_dim=8, temporal_dim=6, coordinate_dim=0)

    assert two.output_dim == three.output_dim == 14
    assert two.gate.weight.shape == three.gate.weight.shape
    assert set(two.state_dict()) == set(three.state_dict())


def test_the_fusion_still_accepts_two_positional_branches() -> None:
    fusion = ConcatGatedFusion(static_dim=3, temporal_dim=2)
    joined = torch.cat([torch.ones(2, 3), torch.zeros(2, 2)], dim=-1)

    expected = joined * torch.sigmoid(fusion.gate(joined))
    torch.testing.assert_close(fusion(torch.ones(2, 3), torch.zeros(2, 2)), expected)


def test_coordinates_widen_the_fusion_by_exactly_the_branch_output() -> None:
    off = SoilCNNLightningModule(**MODEL_ARGS)
    on = SoilCNNLightningModule(**MODEL_ARGS, coord_dim=2, harmonic_num_frequencies=6)

    assert on.coordinate_encoder.output_dim == 4 * 6 + 2
    assert on.fusion.output_dim == off.fusion.output_dim + on.coordinate_encoder.output_dim


# --- the encoder -----------------------------------------------------------


@pytest.mark.parametrize("num_frequencies", [1, 3, 6])
@pytest.mark.parametrize("include_input", [True, False])
def test_output_dim_is_the_documented_arithmetic(num_frequencies: int, include_input: bool) -> None:
    encoder = HarmonicPositionEncoder(
        num_frequencies=num_frequencies, include_input=include_input
    )
    expected = 2 * (2 * num_frequencies + (1 if include_input else 0))

    assert encoder.output_dim == expected
    assert encoder(torch.zeros(3, 2)).shape == (3, expected)


def test_channel_layout_accounts_for_every_channel_exactly_once() -> None:
    """Same contract CalendarGridRasterizer.channel_layout carries, and for the same reason: an
    explainer that mislabels a channel produces a plausible plot rather than an error."""
    encoder = HarmonicPositionEncoder(num_frequencies=4)
    layout = encoder.channel_layout()

    claimed = layout["input"] + layout["sin"] + layout["cos"]
    assert sorted(claimed) == list(range(encoder.embedding_dim))


def test_the_layout_names_the_channel_forward_actually_writes() -> None:
    """Reading the layout must find sines where it says sines are.

    Pinned by construction rather than by inspection: at coords=0 every sine is 0 and every cosine
    is 1, so a swapped pair is visible without depending on any particular frequency.
    """
    encoder = HarmonicPositionEncoder(num_frequencies=3, include_input=True, hidden_dims=[])
    embedded = encoder(torch.zeros(1, 2))[0]
    layout = encoder.channel_layout()

    assert torch.allclose(embedded[layout["sin"]], torch.zeros(len(layout["sin"])))
    assert torch.allclose(embedded[layout["cos"]], torch.ones(len(layout["cos"])))


def test_distinct_positions_get_distinct_embeddings() -> None:
    """The point of the branch. Note this is the OPPOSITE of the temporal contract, which requires
    invariance to shifting every date - there, absolute epoch is noise; here, absolute position is
    the signal."""
    encoder = HarmonicPositionEncoder(num_frequencies=6)
    here, there = torch.tensor([[-0.5, 0.25]]), torch.tensor([[0.5, -0.25]])

    assert not torch.allclose(encoder(here), encoder(there))


def test_a_point_outside_the_training_extent_is_encoded_not_rejected() -> None:
    """Normalization is not clipped, so an out-of-box point arrives outside [-1, 1]. Sine and
    cosine are periodic, so it stays bounded and finite instead of blowing up."""
    encoder = HarmonicPositionEncoder(num_frequencies=6, include_input=False)
    embedded = encoder(torch.tensor([[4.0, -7.5]]))

    assert torch.isfinite(embedded).all()
    assert embedded.abs().max() <= 1.0 + 1e-6


def test_higher_frequencies_separate_points_a_low_one_cannot() -> None:
    """Why the bank is geometric rather than one scale: two nearby points are indistinguishable to
    the lowest frequency and separated by the highest."""
    near, other = torch.tensor([[0.0, 0.0]]), torch.tensor([[0.02, 0.0]])
    coarse = HarmonicPositionEncoder(num_frequencies=1, include_input=False)
    fine = HarmonicPositionEncoder(num_frequencies=8, include_input=False)

    assert (coarse(near) - coarse(other)).abs().max() < (fine(near) - fine(other)).abs().max()


def test_hidden_dims_decouple_the_branch_width_from_the_frequency_count() -> None:
    wide = HarmonicPositionEncoder(num_frequencies=10, hidden_dims=[])
    projected = HarmonicPositionEncoder(num_frequencies=10, hidden_dims=[7])

    assert wide.output_dim == 42
    assert projected.output_dim == 7


@pytest.mark.parametrize("num_frequencies", [0, -1])
def test_an_empty_frequency_bank_is_refused(num_frequencies: int) -> None:
    with pytest.raises(ValueError, match="num_frequencies must be positive"):
        HarmonicPositionEncoder(num_frequencies=num_frequencies)


def test_a_batch_of_the_wrong_width_is_refused() -> None:
    """The alternative is an axis swap: at the wrong width longitude would be read out of the
    latitude column and the result would still be well-shaped."""
    with pytest.raises(ValueError, match="but this encoder was built for"):
        HarmonicPositionEncoder(num_coordinates=2)(torch.zeros(3, 3))


# --- the module ------------------------------------------------------------


def test_position_changes_the_prediction() -> None:
    model = SoilCNNLightningModule(**MODEL_ARGS, coord_dim=2).eval()
    batch = _batch()
    moved = dict(batch, x_coords=batch["x_coords"] + 0.4)

    with torch.no_grad():
        assert not torch.allclose(model(batch), model(moved))


def test_gradients_reach_the_coordinates() -> None:
    model = SoilCNNLightningModule(**MODEL_ARGS, coord_dim=2)
    batch = _batch()
    coords = batch["x_coords"].clone().requires_grad_(True)

    model(dict(batch, x_coords=coords)).sum().backward()
    assert coords.grad is not None and float(coords.grad.abs().sum()) > 0


def test_a_batch_without_coordinates_is_refused_by_name() -> None:
    model = SoilCNNLightningModule(**MODEL_ARGS, coord_dim=2)

    with pytest.raises(KeyError, match="x_coords"):
        model(_batch(coords=False))


def test_a_coordinate_free_model_ignores_coordinates_in_the_batch() -> None:
    """The datamodule always emits x_coords, zero-width when off. A model built without the branch
    must not care either way, or a shared datamodule could not serve both."""
    model = SoilCNNLightningModule(**MODEL_ARGS).eval()

    with torch.no_grad():
        torch.testing.assert_close(model(_batch(coords=True)), model(_batch(coords=False)))


def test_the_attribution_seam_reproduces_forward_exactly() -> None:
    """forward_from_parts(explanation_parts(batch)) == forward(batch), with coordinates on.

    Exact, not approximate: any drift attributes importance to a model that is not the one being
    scored. The coordinate part is the raw normalized columns, so the harmonic expansion is
    recomputed inside forward_from_parts and the two paths must agree bit for bit.
    """
    model = SoilCNNLightningModule(**MODEL_ARGS, coord_dim=2).eval()
    batch = _batch()

    with torch.no_grad():
        parts, _groups = model.explanation_parts(batch)
        torch.testing.assert_close(model.forward_from_parts(parts), model(batch), rtol=0, atol=0)


def test_coordinates_are_attributed_as_two_named_rows_not_as_harmonics() -> None:
    """Two rows meaning `lat` and `lon`, not 26 rows that individually mean nothing."""
    model = SoilCNNLightningModule(**MODEL_ARGS, coord_dim=2)
    model.coord_names = ["lat", "lon"]
    parts, groups = model.explanation_parts(_batch())

    spatial = [group for group in groups if group["kind"] == "spatial"]
    assert [group["name"] for group in spatial] == ["lat", "lon"]
    assert parts[spatial[0]["part"]].shape[-1] == 2


def test_every_group_stays_inside_its_own_part() -> None:
    model = SoilCNNLightningModule(**MODEL_ARGS, coord_dim=2)
    parts, groups = model.explanation_parts(_batch())

    for group in groups:
        assert max(group["columns"]) < parts[group["part"]].shape[-1]


def test_the_module_survives_a_weights_only_checkpoint_round_trip(tmp_path: Path) -> None:
    """The recurring trap in this codebase: anything non-builtin in hyper_parameters makes the
    checkpoint unloadable under torch.load's weights_only=True default."""
    import lightning.pytorch as pl

    model = SoilCNNLightningModule(
        **MODEL_ARGS, coord_dim=2, harmonic_num_frequencies=4, harmonic_hidden_dims=[5]
    )
    path = tmp_path / "model.ckpt"
    pl.Trainer(logger=False, enable_checkpointing=False).strategy.connect(model)
    torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams)}, path)

    loaded = torch.load(path, weights_only=True)
    assert loaded["hyper_parameters"]["harmonic_hidden_dims"] == [5]
    assert loaded["hyper_parameters"]["coord_dim"] == 2


# --- the data path ---------------------------------------------------------


def _write_csvs(tmp_path: Path, *, coords=None, context_value=1.0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    point_ids = [1, 2, 3, 4, 5, 6]
    latitudes = [31.0, 32.0, 33.0, 34.0, 35.0, 36.0] if coords is None else coords[0]
    longitudes = [-8.0, -7.0, -6.0, -5.0, -4.0, -3.0] if coords is None else coords[1]
    static = pd.DataFrame(
        {
            "point_id": point_ids,
            "lat": latitudes,
            "lon": longitudes,
            "target_a": [1.0 + index for index in range(len(point_ids))],
            "static_1": [10.0 + index for index in range(len(point_ids))],
            "patch_variance_b04": [context_value + index for index in range(len(point_ids))],
        }
    )
    rows = [
        {"point_id": point_id, "obs_date": date, "S2_b2": 1.0 + index, "S2_b3": 2.0 + index}
        for point_id in point_ids
        for index, date in enumerate(["2021-03-01", "2021-09-01", "2022-03-01"])
    ]
    static_path, timeseries_path = tmp_path / "static.csv", tmp_path / "ts.csv"
    static.to_csv(static_path, index=False)
    pd.DataFrame(rows).to_csv(timeseries_path, index=False)
    return static_path, timeseries_path


def _config(tmp_path: Path, static_path, timeseries_path, **overrides):
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
        CARRY_LABEL_COLUMNS=False,
        USE_HARMONIC_COORDS=False,
        CONTEXT_FEATURES=[],
        USE_CONTEXT_FEATURES=True,
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
        MAX_MISSING_COLUMN_RATIO=0.9,
        ALLOW_SPARSE_COLUMNS=[],
        FAIL_ON_SPARSE_COLUMNS=False,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _bundle(tmp_path: Path, logger, *, csv_kwargs=None, **overrides):
    static_path, timeseries_path = _write_csvs(tmp_path, **(csv_kwargs or {}))
    config = _config(tmp_path, static_path, timeseries_path, **overrides)
    return SoilSequenceBuilder(config, logger, DataManager(config, logger)).build()


def test_the_flag_off_leaves_no_coordinates_on_the_bundle(tmp_path: Path, logger) -> None:
    bundle = _bundle(tmp_path, logger)

    assert bundle.coord_dim == 0
    assert bundle.coord_names == []
    assert np.asarray(bundle.coords).shape[1] == 0


def test_the_flag_on_carries_coordinates_without_making_them_features(
    tmp_path: Path, logger
) -> None:
    """The whole point: they travel, and they are still not predictors. lat/lon must not appear
    among the continuous covariates that reach TabularStaticEncoder."""
    bundle = _bundle(tmp_path, logger, USE_HARMONIC_COORDS=True)

    assert bundle.coord_dim == 2
    assert bundle.coord_names == ["lat", "lon"]
    assert "lat" not in bundle.static_feature_names
    assert "lon" not in bundle.static_feature_names


def test_a_point_with_no_coordinate_is_dropped_only_when_the_flag_is_on(
    tmp_path: Path, logger
) -> None:
    """There is no honest fill for a coordinate, so the row goes - but only when it is being used.
    With the flag off the same point must survive, or turning the feature off would not restore the
    dataset it was meant to restore."""
    csv_kwargs = {
        "coords": ([31.0, np.nan, 33.0, 34.0, 35.0, 36.0], [-8.0, -7.0, -6.0, -5.0, -4.0, -3.0])
    }

    assert _bundle(tmp_path / "off", logger, csv_kwargs=csv_kwargs).num_points == 6
    assert (
        _bundle(tmp_path / "on", logger, csv_kwargs=csv_kwargs, USE_HARMONIC_COORDS=True).num_points
        == 5
    )


def test_the_row_rule_and_the_split_population_agree(tmp_path: Path, logger) -> None:
    """usable_point_ids feeds the unified splitter, and build() produces what actually trains. If
    they disagreed the splitter would assign points the builder then deletes, silently shrinking
    the run's population under population_policy=intersect."""
    csv_kwargs = {
        "coords": ([31.0, np.nan, 33.0, 34.0, 35.0, 36.0], [-8.0, -7.0, -6.0, -5.0, -4.0, -3.0])
    }
    static_path, timeseries_path = _write_csvs(tmp_path, **csv_kwargs)
    config = _config(tmp_path, static_path, timeseries_path, USE_HARMONIC_COORDS=True)
    builder = SoilSequenceBuilder(config, logger, DataManager(config, logger))

    assert list(builder.usable_point_ids()) == list(builder.build().point_ids)


def test_a_missing_coordinate_column_names_where_to_look(tmp_path: Path, logger) -> None:
    static_path, timeseries_path = _write_csvs(tmp_path)
    pd.read_csv(static_path).drop(columns=["lat"]).to_csv(static_path, index=False)
    config = _config(tmp_path, static_path, timeseries_path, USE_HARMONIC_COORDS=True)

    with pytest.raises(KeyError, match="USE_HARMONIC_COORDS is on"):
        SoilSequenceBuilder(config, logger, DataManager(config, logger)).build()


# --- normalization and serving ---------------------------------------------


def test_coordinates_are_normalized_onto_the_train_bounding_box(tmp_path: Path, logger) -> None:
    datamodule = SoilSequenceDataModule(
        _bundle(tmp_path, logger, USE_HARMONIC_COORDS=True), batch_size=6, val_size=0.0, test_size=0.0
    )
    datamodule.setup("fit")
    coords = datamodule.collate(datamodule.train_idx_)["x_coords"].numpy()

    assert datamodule.coord_dim == 2
    assert pytest.approx(coords.min(axis=0).tolist()) == [-1.0, -1.0]
    assert pytest.approx(coords.max(axis=0).tolist()) == [1.0, 1.0]


def test_the_box_is_fitted_on_train_alone_so_a_test_point_may_fall_outside(
    tmp_path: Path, logger
) -> None:
    """Not clipped: collapsing everything beyond the edge onto the boundary would make a distant
    point indistinguishable from one just outside."""
    datamodule = SoilSequenceDataModule(
        _bundle(tmp_path, logger, USE_HARMONIC_COORDS=True), batch_size=6, val_size=0.0, test_size=0.34
    )
    datamodule.setup("fit")

    train_max = datamodule.collate(datamodule.train_idx_)["x_coords"].abs().max()
    everything = datamodule.collate(np.arange(datamodule.sequence_bundle.num_points))["x_coords"]

    assert float(train_max) <= 1.0 + 1e-6
    assert float(everything.abs().max()) > 1.0
    assert torch.isfinite(everything).all()


def test_a_degenerate_axis_becomes_a_constant_rather_than_an_infinity(
    tmp_path: Path, logger
) -> None:
    """Every training point on one meridian carries no east-west information at all. Zero is the
    honest answer; dividing by the zero span would poison every downstream channel."""
    bundle = _bundle(
        tmp_path,
        logger,
        csv_kwargs={"coords": ([31.0, 32.0, 33.0, 34.0, 35.0, 36.0], [-8.0] * 6)},
        USE_HARMONIC_COORDS=True,
    )
    datamodule = SoilSequenceDataModule(bundle, batch_size=6, val_size=0.0, test_size=0.0)
    datamodule.setup("fit")
    coords = datamodule.collate(datamodule.train_idx_)["x_coords"].numpy()

    assert np.isfinite(coords).all()
    assert np.allclose(coords[:, 1], 0.0)


def test_a_single_point_request_is_placed_on_the_training_box_not_on_itself(
    tmp_path: Path, logger
) -> None:
    """The failure the stored bounding box exists to prevent.

    A serving batch can be one point. Re-fitting a box on it gives a degenerate span, so that point
    would normalize to the centre of itself - every served point silently predicted at the middle
    of the study area, with no error anywhere.
    """
    bundle = _bundle(tmp_path, logger, USE_HARMONIC_COORDS=True)
    trained = SoilSequenceDataModule(bundle, batch_size=6, val_size=0.0, test_size=0.0)
    trained.setup("fit")
    expected = trained.collate(np.array([4]))["x_coords"]

    served = SoilSequenceDataModule(bundle, batch_size=1)
    served.apply_preprocessing_state(trained.preprocessing_state())

    torch.testing.assert_close(served.collate(np.array([4]))["x_coords"], expected)
    assert not torch.allclose(expected, torch.zeros_like(expected))


def test_the_bounding_box_travels_in_the_preprocessing_state(tmp_path: Path, logger) -> None:
    datamodule = SoilSequenceDataModule(
        _bundle(tmp_path, logger, USE_HARMONIC_COORDS=True), batch_size=6, val_size=0.0, test_size=0.0
    )
    datamodule.setup("fit")
    state = datamodule.preprocessing_state()

    assert state["coord_names"] == ["lat", "lon"]
    assert state["coord_min"] == [31.0, -8.0]
    assert state["coord_max"] == [36.0, -3.0]
    assert all(isinstance(value, float) for value in state["coord_min"] + state["coord_max"])


def test_the_batch_always_carries_a_coordinate_key(tmp_path: Path, logger) -> None:
    """Zero-width when off, exactly as x_categorical is, so nothing downstream needs a None
    branch and one datamodule can serve a model with the branch and one without."""
    datamodule = SoilSequenceDataModule(_bundle(tmp_path, logger), batch_size=6)
    datamodule.setup("fit")
    batch = datamodule.collate(np.arange(6))

    assert batch["x_coords"].shape == (6, 0)


# --- the spatial-context group ---------------------------------------------

CONTEXT = ["patch_variance_b04"]


def test_an_empty_group_changes_nothing(tmp_path: Path, logger) -> None:
    """The default. A column not named anywhere behaves as the ordinary covariate it always was."""
    bundle = _bundle(tmp_path, logger)

    assert bundle.context_feature_names == []
    assert "patch_variance_b04" in bundle.static_feature_names


def test_the_group_on_keeps_the_columns_as_ordinary_continuous_features(
    tmp_path: Path, logger
) -> None:
    """Named, but not moved. They must still reach TabularStaticEncoder inside x_static, with the
    width and the standardization they would have had unnamed."""
    plain = _bundle(tmp_path / "plain", logger)
    grouped = _bundle(tmp_path / "grouped", logger, CONTEXT_FEATURES=CONTEXT)

    assert grouped.static_feature_names == plain.static_feature_names
    assert grouped.context_feature_names == CONTEXT
    np.testing.assert_array_equal(grouped.static_features, plain.static_features)


def test_the_group_off_removes_the_columns(tmp_path: Path, logger) -> None:
    """The ablation. Naming a column makes it switchable, which necessarily means the switch can
    take away a column that would otherwise be a feature."""
    bundle = _bundle(
        tmp_path, logger, CONTEXT_FEATURES=CONTEXT, USE_CONTEXT_FEATURES=False
    )

    assert "patch_variance_b04" not in bundle.static_feature_names
    assert bundle.context_feature_names == []


def test_group_membership_follows_the_frame_not_the_config_order(tmp_path: Path, logger) -> None:
    """The names index positions in static_features, so declaration order must not leak in."""
    bundle = _bundle(
        tmp_path, logger, CONTEXT_FEATURES=["patch_variance_b04", "static_1"]
    )
    positions = [bundle.static_feature_names.index(name) for name in bundle.context_feature_names]

    assert positions == sorted(positions)


def test_an_unknown_context_column_is_refused(tmp_path: Path, logger) -> None:
    with pytest.raises(KeyError, match="CONTEXT_FEATURES names column"):
        _bundle(tmp_path, logger, CONTEXT_FEATURES=["no_such_column"])


def test_a_context_column_something_else_drops_is_refused_distinctly(
    tmp_path: Path, logger
) -> None:
    """Present but filtered out elsewhere. A different message, because the fix is elsewhere too."""
    with pytest.raises(ValueError, match="present but not continuous features"):
        _bundle(
            tmp_path,
            logger,
            CONTEXT_FEATURES=["patch_variance_b04"],
            IGNORED_COLUMNS=["point_id", "lat", "lon", "patch_variance_b04"],
            ELIMINATED_FEATURES=["point_id", "lat", "lon", "patch_variance_b04"],
        )


def test_the_group_reaches_the_model_as_its_own_attribution_block(tmp_path: Path, logger) -> None:
    """It rides attach_preprocessing_state, the channel static_feature_names already uses, so it
    lands in the checkpoint without becoming a hyperparameter."""
    datamodule = SoilSequenceDataModule(
        _bundle(tmp_path, logger, CONTEXT_FEATURES=CONTEXT), batch_size=6
    )
    datamodule.setup("fit")
    model = SoilCNNLightningModule(
        **{
            **MODEL_ARGS,
            "static_dim": datamodule.static_dim,
            "modality_dims": datamodule.modality_dims,
        },
        coord_dim=0,
    )
    model.attach_preprocessing_state(datamodule.preprocessing_state())

    batch = datamodule.collate(np.arange(6))
    _parts, groups = model.explanation_parts(batch)
    kinds = {group["name"]: group["kind"] for group in groups}

    assert model.context_feature_names == CONTEXT
    assert kinds["patch_variance_b04"] == "context"
    assert kinds["static_1"] == "static"


def test_the_context_group_adds_no_parameters(tmp_path: Path, logger) -> None:
    """It is config plumbing and a name list. Nothing about the network changes."""
    model = SoilCNNLightningModule(**MODEL_ARGS)
    before = sum(parameter.numel() for parameter in model.parameters())
    model.attach_preprocessing_state({"context_feature_names": CONTEXT})

    assert sum(parameter.numel() for parameter in model.parameters()) == before


# --- end to end ------------------------------------------------------------


def test_both_features_on_train_together(tmp_path: Path, logger) -> None:
    """Coordinates and the context group are independent switches; the run must survive both."""
    bundle = _bundle(
        tmp_path, logger, USE_HARMONIC_COORDS=True, CONTEXT_FEATURES=CONTEXT
    )
    datamodule = SoilSequenceDataModule(bundle, batch_size=3, target_transform="log1p")
    datamodule.setup("fit")
    model = SoilCNNLightningModule(
        **{
            **MODEL_ARGS,
            "static_dim": datamodule.static_dim,
            "modality_dims": datamodule.modality_dims,
        },
        coord_dim=datamodule.coord_dim,
    )
    model.attach_preprocessing_state(datamodule.preprocessing_state())

    batch = datamodule.collate(np.arange(6))
    # forward + loss_fn rather than training_step: self.log needs a Trainer attached, and this test
    # is about the branches reaching the optimizer, not about Lightning's logging.
    loss = model.loss_fn(model(batch), batch["y"])
    loss.backward()

    assert torch.isfinite(loss)
    assert model.has_coordinates
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        for parameter in model.fusion.parameters()
    )
