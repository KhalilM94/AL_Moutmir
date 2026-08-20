"""Builder -> datamodule -> module tests for the graph-free sequence path.

The properties that matter most here are the ones that let a trained checkpoint outlive the window
it was trained on: era invariance, length agnosticism and cadence agnosticism. Each has a dedicated
test below.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from yg_eo_soilnet.data_manager import DataManager
from yg_eo_soilnet.datamodules.categorical import CategoricalEncoder
from yg_eo_soilnet.datamodules.sequence.sequence_bundle import SoilSequenceBundle
from yg_eo_soilnet.datamodules.sequence.sequence_builder import SoilSequenceBuilder, to_decimal_year
from yg_eo_soilnet.datamodules.sequence.sequence_datamodule import SoilSequenceDataModule
from yg_eo_soilnet.models.lightningmodules.soil_sequence_lightning_module import SoilSequenceLightningModule
from yg_eo_soilnet.models.lightningmodules.soil_tabular_lightning_module import SoilTabularLightningModule
from yg_eo_soilnet.models.lightningmodules.temporal_encoders import (
    TemporalTransformerEncoder,
    TimeAwareLSTMEncoder,
    compact_observations,
    sequence_time_features,
)

ENCODERS = ["time_transformer", "time_lstm"]


# --- fixtures --------------------------------------------------------------


def _write_csvs(tmp_path: Path, *, year_offset: int = 0, dates_by_point=None, with_categoricals=False):
    """Static + time-series CSVs where each point deliberately has a different observation count."""
    static_df = pd.DataFrame(
        {
            "point_id": [1, 2, 3, 4, 5, 6],
            "lat": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "lon": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "target_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "static_1": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "static_2": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
        }
    )
    if with_categoricals:
        # A unique texture per point, so whatever the split, some categories exist only outside the
        # train split - which is the case a fitted vocabulary has to handle. Point 6 is blank: under
        # the old float encoding that NaN deleted the whole point.
        static_df["texture"] = ["lo", "cl", "salo", "sicllo", "sacllo", ""]
        static_df["landform"] = ["valley", "peak", "valley", "upper", "valley", "valley"]

    if dates_by_point is None:
        # Ragged on purpose: 6, 4 and 2 observations, with gaps and non-zero starts.
        dates_by_point = {
            1: ["2020-01-15", "2020-02-15", "2020-05-15", "2020-06-15", "2021-01-15", "2021-07-15"],
            2: ["2020-03-15", "2020-04-15", "2021-02-15", "2021-11-15"],
            3: ["2020-01-15", "2022-12-15"],
            4: ["2020-02-15", "2020-08-15", "2021-03-15"],
            5: ["2020-04-15", "2020-09-15", "2021-05-15", "2021-12-15", "2022-02-15"],
            6: ["2020-06-15", "2021-06-15"],
        }

    rows = []
    for point_id, dates in dates_by_point.items():
        for index, date in enumerate(dates):
            stamp = pd.Timestamp(date) + pd.DateOffset(years=year_offset)
            rows.append(
                {
                    "point_id": point_id,
                    "obs_date": stamp.strftime("%Y-%m-%d"),
                    "S1_vv": 0.1 * point_id + 0.01 * index,
                    "S2_b2": 1.0 + 0.1 * index,
                    "S2_b3": 2.0 - 0.05 * index,
                }
            )
    timeseries_df = pd.DataFrame(rows)

    static_path = tmp_path / "static.csv"
    timeseries_path = tmp_path / "timeseries.csv"
    static_df.to_csv(static_path, index=False)
    timeseries_df.to_csv(timeseries_path, index=False)
    return static_path, timeseries_path


def _config(
    tmp_path: Path, static_path: Path, timeseries_path: Path, categorical_features=()
) -> SimpleNamespace:
    return SimpleNamespace(
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
        MODALITY_PREFIX_MAP={"s1": "S1_", "s2": "S2_"},
        S1_COLUMNS=[],
        S2_COLUMNS=[],
        MODIS_COLUMNS=[],
        TARGET_COLUMNS=["target_a"],
        LABEL_COLUMNS=["target_a"],
        PREDICTOR_COLUMNS=[],
        IGNORED_COLUMNS=["point_id", "lat", "lon"],
        ELIMINATED_FEATURES=["point_id", "lat", "lon"],
        CATEGORICAL_FEATURES=list(categorical_features),
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


def _build_bundle(tmp_path: Path, logger, *, categorical_features=(), **kwargs) -> SoilSequenceBundle:
    static_path, timeseries_path = _write_csvs(tmp_path, **kwargs)
    config = _config(tmp_path, static_path, timeseries_path, categorical_features)
    data_manager = DataManager(config, logger)
    return SoilSequenceBuilder(config, logger, data_manager).build()


def _categorical_bundle(tmp_path: Path, logger) -> SoilSequenceBundle:
    return _build_bundle(
        tmp_path, logger, with_categoricals=True, categorical_features=["texture", "landform"]
    )


def _synthetic_batch(batch_size=4, length=10, channels=3, seed=0, start_year=2018.0):
    generator = torch.Generator().manual_seed(seed)
    times = torch.sort(
        start_year + torch.rand(batch_size, length, generator=generator, dtype=torch.float64) * 5.0, dim=1
    ).values
    values = torch.randn(batch_size, length, channels, generator=generator)
    mask = torch.ones(batch_size, length, dtype=torch.bool)
    return {
        "x_static": torch.randn(batch_size, 4, generator=generator),
        "y": torch.randn(batch_size, 1, generator=generator),
        "sequences": {"m": values},
        "sequence_mask": {"m": mask},
        "sequence_time": {"m": times},
    }


# --- builder ---------------------------------------------------------------


def test_builder_produces_ragged_date_stamped_sequences(tmp_path: Path, logger) -> None:
    bundle = _build_bundle(tmp_path, logger)

    assert bundle.num_points == 6
    assert set(bundle.sequences) == {"s1", "s2"}
    assert bundle.modality_dims == {"s1": 1, "s2": 2}
    assert bundle.temporal_enabled

    counts = bundle.observation_counts("s2")
    # The whole point of the representation: lengths genuinely differ per point.
    assert counts.tolist() == [6, 4, 2, 3, 5, 2]
    assert len(set(counts.tolist())) > 1

    for modality in ("s1", "s2"):
        for values, times in zip(bundle.sequences[modality], bundle.sequence_times[modality]):
            assert len(values) == len(times)
            assert np.all(np.diff(times) > 0), "timestamps must be strictly ascending"


def test_decimal_year_conversion_is_continuous_and_leap_aware() -> None:
    dates = pd.Series(pd.to_datetime(["2020-01-01", "2020-07-01", "2021-01-01", "2019-12-31"]))
    decimal = to_decimal_year(dates)

    assert decimal.dtype == np.float64  # float32 would blur sub-monthly spacing near year 2020
    assert decimal[0] == pytest.approx(2020.0)
    assert decimal[1] == pytest.approx(2020.0 + 182 / 365.25, abs=1e-6)
    assert decimal[2] == pytest.approx(2021.0)
    assert decimal[3] < decimal[2]


def test_builder_handles_irregular_cadence(tmp_path: Path, logger) -> None:
    """Nothing may assume a monthly step: daily, yearly and ragged spacing must all build."""
    bundle = _build_bundle(
        tmp_path,
        logger,
        dates_by_point={
            1: ["2020-01-01", "2020-01-08", "2020-02-17", "2020-02-20"],  # 7, 40, 3 days
            2: ["2019-03-01", "2022-09-14"],
            3: ["2020-05-05"],
            4: ["2020-01-01", "2020-01-02", "2020-01-03"],
            5: ["2018-11-11", "2021-04-02"],
            6: ["2020-12-31", "2021-01-01"],
        },
    )
    bundle.validate()
    assert bundle.observation_counts("s2").tolist() == [4, 2, 1, 3, 2, 2]


def test_builder_zero_fills_points_absent_from_the_timeseries(tmp_path: Path, logger) -> None:
    dates = {1: ["2020-01-15", "2020-02-15"], 2: ["2020-03-15"]}
    bundle = _build_bundle(tmp_path, logger, dates_by_point=dates)

    counts = bundle.observation_counts("s2")
    assert counts.tolist() == [2, 1, 0, 0, 0, 0]
    assert bundle.sequences["s2"][2].shape == (0, 2)
    bundle.validate()


def test_bundle_validate_names_the_offending_point_and_column(tmp_path: Path, logger) -> None:
    bundle = _build_bundle(tmp_path, logger)
    bundle.sequences["s2"][1][0, 1] = np.nan

    with pytest.raises(ValueError, match=r"Non-finite value in sequences\['s2'\] at point 2.*S2_b3"):
        bundle.validate()


def test_bundle_validate_rejects_unsorted_timestamps(tmp_path: Path, logger) -> None:
    bundle = _build_bundle(tmp_path, logger)
    bundle.sequence_times["s2"][0] = bundle.sequence_times["s2"][0][::-1].copy()

    with pytest.raises(ValueError, match="strictly ascending"):
        bundle.validate()


# --- datamodule ------------------------------------------------------------


def test_datamodule_pads_per_batch_and_mask_sum_is_the_true_length(tmp_path: Path, logger) -> None:
    bundle = _build_bundle(tmp_path, logger)
    datamodule = SoilSequenceDataModule(bundle, batch_size=6, val_size=0.0, test_size=0.0, seed=7)
    datamodule.setup("fit")

    batch = datamodule._collate_points(np.arange(6))
    values = batch["sequences"]["s2"]
    mask = batch["sequence_mask"]["s2"]

    assert values.shape == (6, 6, 2)  # padded to the longest series in THIS batch, which is 6
    assert mask.dtype == torch.bool
    # Every unmasked token is a real observation, so the count is a genuine length.
    assert mask.sum(dim=1).tolist() == bundle.observation_counts("s2").tolist()
    # Padding is right-padding: the mask is a dense prefix.
    assert torch.equal(mask, mask.sort(dim=1, descending=True).values)


def test_datamodule_batch_length_follows_the_batch_not_a_global_axis(tmp_path: Path, logger) -> None:
    bundle = _build_bundle(tmp_path, logger)
    datamodule = SoilSequenceDataModule(bundle, batch_size=2, val_size=0.0, test_size=0.0)
    datamodule.setup("fit")

    # Points 0 and 2 have 6 and 2 observations; points 2 and 5 have 2 and 2.
    long_batch = datamodule._collate_points(np.array([0, 2]))
    short_batch = datamodule._collate_points(np.array([2, 5]))
    assert long_batch["sequences"]["s2"].shape[1] == 6
    assert short_batch["sequences"]["s2"].shape[1] == 2


def test_datamodule_exposes_the_factory_contract_without_graph_or_step_attributes(
    tmp_path: Path, logger
) -> None:
    bundle = _build_bundle(tmp_path, logger)
    datamodule = SoilSequenceDataModule(bundle, batch_size=2, target_transform="log1p")
    datamodule.setup("fit")

    assert datamodule.static_dim == 2
    assert datamodule.target_dim == 1
    assert datamodule.modality_dims == {"s1": 1, "s2": 2}
    assert datamodule.target_mean_ is not None and datamodule.target_scale_ is not None
    # A length-agnostic, graph-free model must never be handed these by the factory.
    assert not hasattr(datamodule, "temporal_steps")
    assert not hasattr(datamodule, "edge_attr_dim")


def test_datamodule_fits_standardization_on_the_train_split_only(tmp_path: Path, logger) -> None:
    bundle = _build_bundle(tmp_path, logger)
    datamodule = SoilSequenceDataModule(bundle, batch_size=2, val_size=0.25, test_size=0.25, seed=3)
    datamodule.setup("fit")

    train_static = bundle.static_features[datamodule.train_idx_]
    np.testing.assert_allclose(datamodule.static_mean_, train_static.mean(axis=0), rtol=1e-5)

    observed = np.concatenate([bundle.sequences["s2"][i] for i in datamodule.train_idx_ if len(bundle.sequences["s2"][i])])
    np.testing.assert_allclose(datamodule.sequence_mean_["s2"], observed.mean(axis=0), rtol=1e-5)


def test_datamodule_predict_order_matches_the_evaluation_frame(tmp_path: Path, logger) -> None:
    bundle = _build_bundle(tmp_path, logger)
    datamodule = SoilSequenceDataModule(bundle, batch_size=2, val_size=0.25, test_size=0.34, seed=11)
    datamodule.setup("fit")

    predicted_ids = [pid for batch in datamodule.predict_dataloader() for pid in batch["point_ids"]]
    expected_ids = [bundle.point_ids[index] for index in datamodule.test_idx_]
    assert predicted_ids == expected_ids
    assert len(datamodule.y_test_frame_) == len(expected_ids)


# --- time features and encoders -------------------------------------------


def test_time_features_are_invariant_to_the_calendar_era() -> None:
    times = torch.tensor([[2018.0, 2018.25, 2019.5, 2021.0]], dtype=torch.float64)
    mask = torch.ones(1, 4, dtype=torch.bool)

    base = sequence_time_features(times, mask)
    shifted = sequence_time_features(times + 10.0, mask)
    torch.testing.assert_close(base, shifted)


def test_time_features_encode_seasonality_and_gaps() -> None:
    # Same month in different years must land on the same point of the seasonal circle.
    times = torch.tensor([[2018.0, 2019.0, 2019.5]], dtype=torch.float64)
    mask = torch.ones(1, 3, dtype=torch.bool)
    features = sequence_time_features(times, mask)

    torch.testing.assert_close(features[0, 0, :2], features[0, 1, :2])
    assert not torch.allclose(features[0, 0, :2], features[0, 2, :2])
    # Relative age is measured from this row's own latest observation, so it ends at 0.
    assert features[0, -1, 2].item() == pytest.approx(0.0)
    assert features[0, 0, 2].item() < 0.0
    # A 6-month gap must read larger than a 0-month one.
    assert features[0, 2, 3].item() > 0.0


def test_time_features_zero_a_row_with_no_observations() -> None:
    times = torch.tensor([[2018.0, 2019.0]], dtype=torch.float64)
    mask = torch.zeros(1, 2, dtype=torch.bool)
    features = sequence_time_features(times, mask)

    assert torch.isfinite(features).all()
    assert bool((features == 0).all())


def test_compact_observations_moves_real_tokens_to_the_front() -> None:
    values = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    times = torch.tensor([[2018.0, 2019.0, 2020.0, 2021.0]], dtype=torch.float64)
    mask = torch.tensor([[False, True, False, True]])

    compacted_values, compacted_mask, compacted_times = compact_observations(values, mask, times)

    assert compacted_mask[0].tolist() == [True, True, False, False]
    assert compacted_values[0, :2, 0].tolist() == [2.0, 4.0]
    assert compacted_times[0, :2].tolist() == [2019.0, 2021.0]


def test_transformer_rejects_a_head_count_that_does_not_divide_d_model() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TemporalTransformerEncoder(input_dim=3, output_dim=8, d_model=10, nhead=4)


@pytest.mark.parametrize("encoder_cls", [TemporalTransformerEncoder, TimeAwareLSTMEncoder])
def test_encoders_run_at_any_length_with_one_instance(encoder_cls) -> None:
    """No parameter may be sized by sequence length, or a future series of a different shape fails."""
    encoder = encoder_cls(input_dim=3, output_dim=8).eval()

    for length in (1, 4, 37):
        batch = _synthetic_batch(length=length, channels=3)
        with torch.no_grad():
            embedding = encoder(
                batch["sequences"]["m"], batch["sequence_mask"]["m"], batch["sequence_time"]["m"]
            )
        assert embedding.shape == (4, 8)
        assert torch.isfinite(embedding).all()


@pytest.mark.parametrize("encoder_cls", [TemporalTransformerEncoder, TimeAwareLSTMEncoder])
def test_encoders_zero_a_point_with_no_observations(encoder_cls) -> None:
    encoder = encoder_cls(input_dim=3, output_dim=8).eval()
    batch = _synthetic_batch(length=6, channels=3)
    mask = batch["sequence_mask"]["m"].clone()
    mask[0] = False

    with torch.no_grad():
        embedding = encoder(batch["sequences"]["m"], mask, batch["sequence_time"]["m"])

    assert torch.isfinite(embedding).all()
    assert bool((embedding[0] == 0).all())
    assert not bool((embedding[1] == 0).all())


# --- categorical covariates and entity embeddings --------------------------


def test_bundle_keeps_categoricals_out_of_the_continuous_block(tmp_path: Path, logger) -> None:
    """The categorical columns must not reach the scaler; a category code is not a magnitude."""
    bundle = _categorical_bundle(tmp_path, logger)

    assert bundle.static_feature_names == ["static_1", "static_2"]
    assert bundle.static_features.shape == (6, 2)
    assert bundle.categorical_feature_names == ["texture", "landform"]
    assert bundle.static_categoricals.shape == (6, 2)
    # Raw labels, not codes: encoding needs a vocabulary and the split does not exist yet.
    assert bundle.static_categoricals[0].tolist() == ["lo", "valley"]


def test_a_missing_category_no_longer_deletes_the_point(tmp_path: Path, logger) -> None:
    """Point 6 has a blank texture. Under the old float encoding its NaN dropped the whole row."""
    bundle = _categorical_bundle(tmp_path, logger)

    assert bundle.num_points == 6
    assert 6 in bundle.point_ids


def test_vocabulary_is_fitted_on_the_train_split_only(tmp_path: Path, logger) -> None:
    """The leak this replaces: pd.factorize saw every split, so held-out categories got real codes."""
    bundle = _categorical_bundle(tmp_path, logger)
    datamodule = SoilSequenceDataModule(bundle, batch_size=2, val_size=0.25, test_size=0.25, seed=3)
    datamodule.setup("fit")

    raw = np.asarray(bundle.static_categoricals, dtype=object)
    train_textures = {
        label for label in raw[datamodule.train_idx_, 0].tolist() if label not in (None, "")
    }
    assert datamodule.categorical_vocabularies[0] == sorted(train_textures)
    assert datamodule.categorical_cardinalities[0] == len(train_textures) + 1

    # Every point outside the train split carries a texture the vocabulary never saw, so it must
    # land on the reserved index rather than borrowing another category's row.
    holdout = np.concatenate([datamodule.val_idx_, datamodule.test_idx_])
    assert (datamodule.categorical_codes_[holdout, 0] == 0).all()
    assert (datamodule.categorical_codes_[datamodule.train_idx_, 0] > 0).all()


def test_datamodule_exports_the_categorical_contract(tmp_path: Path, logger) -> None:
    bundle = _categorical_bundle(tmp_path, logger)
    datamodule = SoilSequenceDataModule(bundle, batch_size=2, val_size=0.25, test_size=0.25, seed=3)
    datamodule.setup("fit")

    # static_dim counts the CONTINUOUS covariates only - this is what the factory injects.
    assert datamodule.static_dim == 2
    assert datamodule.categorical_feature_names == ["texture", "landform"]
    assert len(datamodule.categorical_cardinalities) == 2
    assert all(
        len(vocabulary) + 1 == cardinality
        for vocabulary, cardinality in zip(
            datamodule.categorical_vocabularies, datamodule.categorical_cardinalities
        )
    )


def test_batch_carries_categorical_indices_in_range(tmp_path: Path, logger) -> None:
    bundle = _categorical_bundle(tmp_path, logger)
    datamodule = SoilSequenceDataModule(bundle, batch_size=6, val_size=0.0, test_size=0.0, seed=7)
    datamodule.setup("fit")

    batch = next(iter(datamodule.train_dataloader()))

    assert batch["x_categorical"].dtype == torch.int64
    assert batch["x_categorical"].shape == (6, 2)
    assert batch["x_static"].shape == (6, 2)
    for column, cardinality in enumerate(datamodule.categorical_cardinalities):
        assert int(batch["x_categorical"][:, column].min()) >= 0
        assert int(batch["x_categorical"][:, column].max()) < cardinality


def test_batch_has_a_well_shaped_categorical_key_with_no_categoricals(tmp_path: Path, logger) -> None:
    """No None branch downstream: the key is always present, just empty."""
    datamodule = SoilSequenceDataModule(_build_bundle(tmp_path, logger), batch_size=6, val_size=0.0, test_size=0.0)
    datamodule.setup("fit")

    batch = next(iter(datamodule.train_dataloader()))

    assert batch["x_categorical"].shape == (6, 0)
    assert batch["x_categorical"].dtype == torch.int64


@pytest.mark.parametrize("encoder", ENCODERS)
def test_end_to_end_training_step_with_embeddings(tmp_path: Path, logger, encoder: str) -> None:
    bundle = _categorical_bundle(tmp_path, logger)
    datamodule = SoilSequenceDataModule(bundle, batch_size=6, val_size=0.0, test_size=0.0, seed=7)
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))

    torch.manual_seed(0)
    module = SoilSequenceLightningModule(
        static_dim=datamodule.static_dim,
        target_dim=datamodule.target_dim,
        categorical_cardinalities=datamodule.categorical_cardinalities,
        categorical_vocabularies=datamodule.categorical_vocabularies,
        categorical_feature_names=datamodule.categorical_feature_names,
        modality_dims=datamodule.modality_dims,
        temporal_encoder=encoder,
    )

    predictions = module(batch)
    assert predictions.shape == (6, 1)

    module.loss_fn(predictions, batch["y"]).backward()
    tables = module.static_encoder.embeddings.embeddings
    assert len(tables) == 2
    assert all(table.weight.grad is not None for table in tables)
    assert any(table.weight.grad.abs().sum() > 0 for table in tables)


def test_embedding_widths_follow_the_heuristic(tmp_path: Path, logger) -> None:
    bundle = _categorical_bundle(tmp_path, logger)
    datamodule = SoilSequenceDataModule(bundle, batch_size=6, val_size=0.0, test_size=0.0, seed=7)
    datamodule.setup("fit")

    module = SoilSequenceLightningModule(
        static_dim=datamodule.static_dim,
        target_dim=1,
        categorical_cardinalities=datamodule.categorical_cardinalities,
        modality_dims=datamodule.modality_dims,
    )

    expected = [min(50, (c + 1) // 2) for c in datamodule.categorical_cardinalities]
    assert module.static_encoder.embedding_dims == expected
    assert module.static_encoder.input_dim == datamodule.static_dim + sum(expected)


def test_checkpoint_carries_the_vocabulary_under_weights_only(tmp_path: Path, logger) -> None:
    """The portability fix: the label->index mapping travels with the weights.

    Without it a checkpoint re-derives its mapping from whatever frame it is handed, so the same
    category means a different embedding row and the model silently predicts nonsense.
    """
    bundle = _categorical_bundle(tmp_path, logger)
    datamodule = SoilSequenceDataModule(bundle, batch_size=6, val_size=0.0, test_size=0.0, seed=7)
    datamodule.setup("fit")

    module = SoilSequenceLightningModule(
        static_dim=datamodule.static_dim,
        target_dim=1,
        categorical_cardinalities=datamodule.categorical_cardinalities,
        categorical_vocabularies=datamodule.categorical_vocabularies,
        categorical_feature_names=datamodule.categorical_feature_names,
        modality_dims=datamodule.modality_dims,
    )
    path = tmp_path / "module.ckpt"
    torch.save({"hyper_parameters": dict(module.hparams), "state_dict": module.state_dict()}, path)

    # weights_only=True is the PyTorch >= 2.6 default; anything but plain builtins breaks it.
    loaded = torch.load(path, weights_only=True)
    hparams = loaded["hyper_parameters"]

    assert hparams["categorical_vocabularies"] == datamodule.categorical_vocabularies
    assert hparams["categorical_feature_names"] == ["texture", "landform"]
    # And the mapping can be rebuilt from it without touching the training data.
    restored = CategoricalEncoder.from_vocabularies(
        hparams["categorical_feature_names"], hparams["categorical_vocabularies"]
    )
    assert restored.cardinalities == datamodule.categorical_cardinalities
    assert restored.transform(bundle.static_categoricals).tolist() == datamodule.categorical_codes_.tolist()


def test_an_undeclared_non_numeric_column_fails_loudly(tmp_path: Path, logger) -> None:
    """Silently factorizing it into a magnitude is the behaviour this replaces."""
    with pytest.raises(ValueError, match="texture"):
        _build_bundle(tmp_path, logger, with_categoricals=True, categorical_features=["landform"])


def test_a_declared_column_absent_from_the_data_fails_loudly(tmp_path: Path, logger) -> None:
    with pytest.raises(KeyError, match="SU_WRB1_PH"):
        _build_bundle(tmp_path, logger, categorical_features=["SU_WRB1_PH"])


def test_tabular_module_runs_on_the_same_batch(tmp_path: Path, logger) -> None:
    """The static-only baseline shares the datamodule and ignores every sequence key."""
    bundle = _categorical_bundle(tmp_path, logger)
    datamodule = SoilSequenceDataModule(bundle, batch_size=6, val_size=0.0, test_size=0.0, seed=7)
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))

    torch.manual_seed(0)
    module = SoilTabularLightningModule(
        static_dim=datamodule.static_dim,
        target_dim=datamodule.target_dim,
        categorical_cardinalities=datamodule.categorical_cardinalities,
        categorical_feature_names=datamodule.categorical_feature_names,
    )

    predictions = module(batch)
    assert predictions.shape == (6, 1)

    module.loss_fn(predictions, batch["y"]).backward()
    assert module.static_encoder.embeddings.embeddings[0].weight.grad is not None


# --- lightning module ------------------------------------------------------


def _module(encoder: str, **kwargs) -> SoilSequenceLightningModule:
    defaults = dict(static_dim=4, target_dim=1, modality_dims={"m": 3}, temporal_encoder=encoder)
    defaults.update(kwargs)
    torch.manual_seed(0)
    return SoilSequenceLightningModule(**defaults).eval()


@pytest.mark.parametrize("encoder", ENCODERS)
def test_module_forward_produces_gradients(encoder: str) -> None:
    module = _module(encoder)
    batch = _synthetic_batch()

    predictions = module(batch)
    assert predictions.shape == (4, 1)

    module.loss_fn(predictions, batch["y"]).backward()
    encoder_parameters = [p for p in module.temporal_encoders.parameters() if p.requires_grad]
    assert encoder_parameters, "the dynamic branch should have trainable parameters"
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder_parameters)


@pytest.mark.parametrize("encoder", ENCODERS)
def test_module_predictions_are_invariant_to_the_calendar_era(encoder: str) -> None:
    """A model trained on 2017-2025 must behave identically on the same series shifted ten years."""
    module = _module(encoder)
    batch = _synthetic_batch()
    shifted = {**batch, "sequence_time": {"m": batch["sequence_time"]["m"] + 10.0}}

    with torch.no_grad():
        torch.testing.assert_close(module(batch), module(shifted))


@pytest.mark.parametrize("encoder", ENCODERS)
def test_module_runs_on_sequence_lengths_it_was_not_built_for(encoder: str) -> None:
    module = _module(encoder)

    with torch.no_grad():
        short = module(_synthetic_batch(length=2, seed=1))
        long = module(_synthetic_batch(length=64, seed=2))

    assert short.shape == long.shape == (4, 1)
    assert torch.isfinite(short).all() and torch.isfinite(long).all()


@pytest.mark.parametrize("encoder", ENCODERS)
def test_module_reads_the_whole_series_not_just_a_prefix(encoder: str) -> None:
    """Regression guard: perturbing the LAST observation must move the prediction.

    A consumer that mistook the observation count for a sequence length would truncate here and
    show no sensitivity at all - the failure that previously reduced the temporal branch to noise.
    """
    module = _module(encoder)
    batch = _synthetic_batch(length=12)
    mask = batch["sequence_mask"]["m"].clone()
    mask[:, 4:] = False  # only the first four tokens are real
    mask[0, 9] = True  # ...except one straggler far past that count for row 0
    batch = {**batch, "sequence_mask": {"m": mask}}

    perturbed_values = batch["sequences"]["m"].clone()
    perturbed_values[0, 9] += 5.0
    perturbed = {**batch, "sequences": {"m": perturbed_values}}

    with torch.no_grad():
        difference = (module(batch)[0] - module(perturbed)[0]).abs().item()
    assert difference > 1e-6


@pytest.mark.parametrize("encoder", ENCODERS)
def test_module_ignores_masked_out_observations(encoder: str) -> None:
    module = _module(encoder)
    batch = _synthetic_batch(length=8)
    mask = batch["sequence_mask"]["m"].clone()
    mask[:, 5:] = False
    batch = {**batch, "sequence_mask": {"m": mask}}

    polluted_values = batch["sequences"]["m"].clone()
    polluted_values[:, 5:] += 99.0
    polluted = {**batch, "sequences": {"m": polluted_values}}

    with torch.no_grad():
        torch.testing.assert_close(module(batch), module(polluted))


@pytest.mark.parametrize("encoder", ENCODERS)
def test_module_inverts_the_target_transform_for_prediction(encoder: str) -> None:
    module = _module(encoder, target_mean=[1.5], target_scale=[0.5], target_transform="log1p")
    batch = _synthetic_batch()

    with torch.no_grad():
        standardized = module(batch)
        predicted = module.predict_step(batch, 0)

    expected = torch.expm1((standardized * 0.5 + 1.5) / 10.0)
    torch.testing.assert_close(predicted, expected)


def test_module_runs_without_a_temporal_branch() -> None:
    module = _module("time_transformer", temporal_enabled=False)
    batch = _synthetic_batch()

    predictions = module(batch)
    assert predictions.shape == (4, 1)
    assert len(module.temporal_encoders) == 0


def test_module_runs_without_static_features() -> None:
    module = _module("time_transformer", static_dim=0)
    batch = {**_synthetic_batch(), "x_static": torch.zeros(4, 0)}

    predictions = module(batch)
    assert predictions.shape == (4, 1)
    assert torch.isfinite(predictions).all()


def test_module_rejects_an_unknown_encoder_name() -> None:
    with pytest.raises(ValueError, match="temporal_encoder"):
        SoilSequenceLightningModule(static_dim=4, target_dim=1, modality_dims={"m": 3}, temporal_encoder="gru")


def test_module_rejects_a_per_modality_mapping_that_omits_a_modality() -> None:
    with pytest.raises(ValueError, match="no width in modality_embed_dim"):
        SoilSequenceLightningModule(
            static_dim=4,
            target_dim=1,
            modality_dims={"s1": 3, "s2": 4},
            modality_embed_dim={"s1": 16},
        )


@pytest.mark.parametrize("encoder", ENCODERS)
def test_module_checkpoint_reloads_under_weights_only(tmp_path: Path, encoder: str) -> None:
    module = _module(encoder, target_mean=np.array([2.0]), target_scale=np.array([0.75]))
    checkpoint_path = tmp_path / "model.ckpt"
    torch.save(
        {"state_dict": module.state_dict(), "hyper_parameters": dict(module.hparams)}, checkpoint_path
    )

    # weights_only=True is the PyTorch >= 2.6 default; a numpy array in hyper_parameters breaks it.
    loaded = torch.load(checkpoint_path, weights_only=True)
    assert loaded["hyper_parameters"]["target_mean"] == [2.0]

    # Nothing in the weights may be sized by sequence length or tied to a date range.
    for name, tensor in loaded["state_dict"].items():
        assert "time_values" not in name and "temporal_steps" not in name
        assert tensor.numel() < 100_000, name


# --- end to end ------------------------------------------------------------


@pytest.mark.parametrize("encoder", ENCODERS)
def test_builder_to_module_end_to_end(tmp_path: Path, logger, encoder: str) -> None:
    """A real Trainer.fit: covers the steps, the epoch metrics and the predict path together."""
    from lightning.pytorch import Trainer

    bundle = _build_bundle(tmp_path, logger)
    # val needs >= 2 points or R2 is undefined and correctly goes unlogged.
    datamodule = SoilSequenceDataModule(
        bundle, batch_size=3, val_size=0.5, test_size=0.34, seed=5, target_transform="log1p"
    )
    datamodule.setup("fit")
    assert datamodule.val_idx_.size >= 2

    module = SoilSequenceLightningModule(
        static_dim=datamodule.static_dim,
        target_dim=datamodule.target_dim,
        modality_dims=datamodule.modality_dims,
        temporal_encoder=encoder,
        target_mean=datamodule.target_mean_,
        target_scale=datamodule.target_scale_,
        target_transform=datamodule.target_transform,
        fusion_norm_type="none",  # batches here are too small for BatchNorm1d
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
    # The two health numbers this architecture exists to move.
    assert "val_r2" in trainer.callback_metrics
    assert "val_pred_std_ratio" in trainer.callback_metrics

    predictions = trainer.predict(module, datamodule=datamodule)
    predicted = torch.cat(predictions).reshape(-1)
    assert len(predicted) == len(datamodule.y_test_frame_)
    assert torch.isfinite(predicted).all()


def test_end_to_end_predictions_survive_a_ten_year_shift(tmp_path: Path, logger) -> None:
    """The transfer guarantee, end to end: same readings, dates a decade later, same predictions."""
    base_dir = tmp_path / "base"
    shifted_dir = tmp_path / "shifted"
    base_dir.mkdir()
    shifted_dir.mkdir()
    base_bundle = _build_bundle(base_dir, logger)
    shifted_bundle = _build_bundle(shifted_dir, logger, year_offset=10)

    module = SoilSequenceLightningModule(
        static_dim=2,
        target_dim=1,
        modality_dims={"s1": 1, "s2": 2},
        temporal_encoder="time_transformer",
        fusion_norm_type="none",
    ).eval()

    predictions = []
    for bundle in (base_bundle, shifted_bundle):
        datamodule = SoilSequenceDataModule(bundle, batch_size=6, val_size=0.0, test_size=0.0)
        datamodule.setup("fit")
        with torch.no_grad():
            predictions.append(module(datamodule._collate_points(np.arange(6))))

    # Not exact: a decade shifts which years are leap years, moving each date a fraction of a day
    # along the seasonal circle. The tolerance is that drift, not a modelling approximation.
    torch.testing.assert_close(predictions[0], predictions[1], atol=1e-3, rtol=1e-3)


# --- the shared split plan -------------------------------------------------


def _plan_over(point_ids, *, test_ids, val_ids=()):
    """A hand-built SplitPlan, so the assertion names the exact points rather than a ratio."""
    from yg_eo_soilnet.datamodules.splitting import SplitPlan

    assignments = pd.Series("train", index=pd.Index(list(point_ids), name="point_id"), dtype=object)
    assignments.loc[list(val_ids)] = "val"
    assignments.loc[list(test_ids)] = "test"
    return SplitPlan(
        assignments=assignments,
        strategy="random",
        test_size=0.2,
        val_size=0.2,
        seed=42,
        population_policy="intersect",
    )


def test_a_split_plan_decides_the_holdout_instead_of_val_size_and_test_size(tmp_path, logger):
    """The datamodule stops splitting for itself once the run hands it the shared plan.

    That is the whole fix: this family and the sklearn family used to carve independent holdouts
    from the same data, so a test point here was usually a training point there.
    """
    bundle = _build_bundle(tmp_path, logger)
    point_ids = list(bundle.point_ids)
    plan = _plan_over(point_ids, test_ids=point_ids[:2], val_ids=point_ids[2:3])

    datamodule = SoilSequenceDataModule(
        bundle, batch_size=2, val_size=0.5, test_size=0.5, seed=7, split_plan=plan
    )
    datamodule.setup("fit")

    assert {point_ids[i] for i in datamodule.test_idx_} == set(point_ids[:2])
    assert {point_ids[i] for i in datamodule.val_idx_} == set(point_ids[2:3])
    # val_size=0.5 / test_size=0.5 would have given 3 test points; the plan wins.
    assert datamodule.train_idx_.size == len(point_ids) - 3


def test_points_the_plan_does_not_cover_are_used_by_no_split(tmp_path, logger):
    """Expected under population_policy=intersect, where other families dropped those points."""
    bundle = _build_bundle(tmp_path, logger)
    point_ids = list(bundle.point_ids)
    plan = _plan_over(point_ids[:-1], test_ids=point_ids[:1])

    datamodule = SoilSequenceDataModule(bundle, batch_size=2, split_plan=plan)
    datamodule.setup("fit")

    covered = datamodule.train_idx_.size + datamodule.val_idx_.size + datamodule.test_idx_.size
    assert covered == len(point_ids) - 1


def test_a_plan_that_leaves_no_training_points_raises_rather_than_training_on_nothing(tmp_path, logger):
    bundle = _build_bundle(tmp_path, logger)
    point_ids = list(bundle.point_ids)
    plan = _plan_over(point_ids, test_ids=point_ids)

    datamodule = SoilSequenceDataModule(bundle, batch_size=2, split_plan=plan)

    with pytest.raises(ValueError, match="no training points"):
        datamodule.setup("fit")


def test_without_a_plan_the_legacy_ratio_carve_still_applies(tmp_path, logger):
    """Standalone and serving use construct a datamodule with no plan; that must keep working."""
    bundle = _build_bundle(tmp_path, logger)

    datamodule = SoilSequenceDataModule(bundle, batch_size=2, val_size=0.25, test_size=0.25, seed=3)
    datamodule.setup("fit")

    assert datamodule.test_idx_.size > 0
    assert datamodule.train_idx_.size > 0
