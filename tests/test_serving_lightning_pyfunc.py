"""The pyfunc contract for the sequence models.

Why a pyfunc at all: MLflow 3 defaults `mlflow.pytorch.log_model` to `serialization_format="pt2"`,
which traces `model.forward` from an example input. These models consume a dict batch of ragged,
date-stamped sequences, so nothing can trace them - the run failed with "If serialization_format is
set to 'pt2', then input_example is required" and left the logged model in status FAILED. A pyfunc
sidesteps tracing and, unlike a bare checkpoint, carries the preprocessing needed to consume raw
data.

The load-bearing test is the round trip: build an example, predict through the wrapper, and get the
same numbers as SoilSequencePredictor on the same points.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from yg_eo_soilnet.datamodules.sequence.sequence_bundle import SoilSequenceBundle
from yg_eo_soilnet.datamodules.sequence.sequence_datamodule import SoilSequenceDataModule
from yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module import SoilCNNLightningModule
from yg_eo_soilnet.serving import SoilSequencePredictor
from yg_eo_soilnet.serving.lightning_pyfunc import (
    SoilSequencePyfunc,
    build_input_example,
    bundle_from_frame,
    frame_from_bundle,
    time_column,
    values_column,
)

STATIC = ["clay_pct", "ph"]
BANDS = ["S2_B02", "S2_B08"]
N_POINTS = 16
# The full lab roster the bundle carries when CARRY_LABEL_COLUMNS is on. It includes the target,
# which is exactly why the serving contract must never ask for all of it.
LAB_ROSTER = ["organic_matter_pct", "ph_lab", "clay_lab", "sand_lab"]


@pytest.fixture
def trained():
    torch.manual_seed(0)
    generator = np.random.default_rng(0)

    bundle = SoilSequenceBundle(
        point_ids=[f"p{index}" for index in range(N_POINTS)],
        static_features=generator.normal(20, 5, (N_POINTS, len(STATIC))).astype(np.float32),
        static_feature_names=list(STATIC),
        static_categoricals=np.asarray(
            [[generator.choice(["sandy", "loam"])] for _ in range(N_POINTS)], dtype=object
        ),
        categorical_feature_names=["texture"],
        targets=generator.normal(3, 1, (N_POINTS, 1)).astype(np.float32),
        target_names=["organic_matter_pct"],
        sequences={
            "s2": [
                generator.normal(0.2, 0.05, (int(generator.integers(3, 7)), len(BANDS))).astype(np.float32)
                for _ in range(N_POINTS)
            ]
        },
        sequence_times={},
        modality_columns={"s2": list(BANDS)},
        temporal_enabled=True,
    )
    bundle.sequence_times = {
        "s2": [
            2020.0 + np.sort(generator.random(values.shape[0])) * 2.0
            for values in bundle.sequences["s2"]
        ]
    }

    datamodule = SoilSequenceDataModule(
        sequence_bundle=bundle, batch_size=8, val_size=0.25, test_size=0.25, seed=42
    )
    datamodule.setup("fit")

    model = SoilCNNLightningModule(
        static_dim=datamodule.static_dim,
        target_dim=datamodule.target_dim,
        target_names=datamodule.target_names,
        categorical_cardinalities=datamodule.categorical_cardinalities,
        categorical_vocabularies=datamodule.categorical_vocabularies,
        categorical_feature_names=datamodule.categorical_feature_names,
        modality_dims=datamodule.modality_dims,
        temporal_enabled=True,
        grid_years=datamodule.grid_years,
        static_hidden_dims=[6],
        head_hidden_dims=[6],
        cnn_hidden_dims=[4],
        modality_embed_dim=4,
        dropout=0.0,
        target_mean=datamodule.target_mean_,
        target_scale=datamodule.target_scale_,
    )
    model.attach_preprocessing_state(datamodule.preprocessing_state())
    model.eval()
    return model, bundle


def test_input_example_carries_the_documented_columns(trained) -> None:
    model, bundle = trained
    example = build_input_example(model, bundle, n_rows=3)

    assert len(example) == 3
    for name in ["point_id", *STATIC, "texture", time_column("s2"), values_column("s2")]:
        assert name in example.columns, name

    # Ragged sequences travel as nested lists: one time per observation, one value per band.
    times = example[time_column("s2")].iloc[0]
    values = example[values_column("s2")].iloc[0]
    assert len(times) == len(values)
    assert all(len(row) == len(BANDS) for row in values)


def test_the_wrapper_matches_the_predictor_on_the_same_points(trained) -> None:
    """The round trip that proves the contract reconstructs the model's real inputs."""
    model, bundle = trained

    example = build_input_example(model, bundle, n_rows=N_POINTS)
    through_pyfunc = SoilSequencePyfunc(model).predict(None, example).to_numpy(dtype=float)
    through_predictor = SoilSequencePredictor(model).predict(bundle)

    assert np.allclose(through_pyfunc, through_predictor, atol=1e-5)


def test_predictions_are_labelled_with_the_target_names(trained) -> None:
    model, bundle = trained
    example = build_input_example(model, bundle, n_rows=2)

    assert list(SoilSequencePyfunc(model).predict(None, example).columns) == ["organic_matter_pct"]


def test_a_frame_round_trips_through_the_bundle(trained) -> None:
    model, bundle = trained
    state = model.get_preprocessing_state()

    rebuilt = bundle_from_frame(frame_from_bundle(bundle, state), state)

    assert rebuilt.num_points == bundle.num_points
    assert rebuilt.static_feature_names == bundle.static_feature_names
    assert rebuilt.modality_columns == bundle.modality_columns
    assert np.allclose(rebuilt.static_features, bundle.static_features, atol=1e-5)
    for original, restored in zip(bundle.sequences["s2"], rebuilt.sequences["s2"]):
        assert restored.shape == original.shape
        assert np.allclose(restored, original, atol=1e-5)


def test_a_missing_static_column_is_refused_by_name(trained) -> None:
    model, bundle = trained
    state = model.get_preprocessing_state()
    example = frame_from_bundle(bundle, state).drop(columns=["ph"])

    with pytest.raises(KeyError, match="ph"):
        bundle_from_frame(example, state)


def test_a_missing_modality_column_is_refused_with_the_band_order(trained) -> None:
    model, bundle = trained
    state = model.get_preprocessing_state()
    example = frame_from_bundle(bundle, state).drop(columns=[values_column("s2")])

    with pytest.raises(KeyError, match="s2__values"):
        bundle_from_frame(example, state)


def test_misaligned_times_and_values_are_refused(trained) -> None:
    model, bundle = trained
    state = model.get_preprocessing_state()
    example = frame_from_bundle(bundle, state)
    example.at[0, time_column("s2")] = list(example[time_column("s2")].iloc[0])[:-1]

    with pytest.raises(ValueError, match="line up"):
        bundle_from_frame(example, state)


def test_a_point_with_no_observations_is_accepted(trained) -> None:
    """A real request can carry a point that has never been imaged; it must not crash the batch."""
    model, bundle = trained
    state = model.get_preprocessing_state()
    example = frame_from_bundle(bundle, state, n_rows=4)
    example.at[0, time_column("s2")] = []
    example.at[0, values_column("s2")] = []

    predictions = SoilSequencePyfunc(model).predict(None, example)

    assert len(predictions) == 4
    assert np.isfinite(predictions.to_numpy(dtype=float)).all()


def test_signature_inference_produces_array_types(trained) -> None:
    """MLflow has to describe the nested columns, or a served request cannot be validated."""
    from mlflow.models import infer_signature

    model, bundle = trained
    example = build_input_example(model, bundle, n_rows=3)
    predictions = SoilSequencePyfunc(model).predict(None, example)

    rendered = str(infer_signature(example, predictions).inputs)

    assert "Array(double)" in rendered
    assert "Array(Array(double))" in rendered


def test_a_model_without_preprocessing_state_cannot_build_an_example() -> None:
    model = SoilCNNLightningModule(static_dim=2, target_dim=1, temporal_enabled=False)

    with pytest.raises(ValueError, match="no preprocessing state"):
        build_input_example(model, SoilSequenceBundle(), n_rows=1)


def test_frame_from_bundle_honours_the_row_cap(trained) -> None:
    model, bundle = trained
    assert len(frame_from_bundle(bundle, model.get_preprocessing_state(), n_rows=2)) == 2
    # Asking for more rows than exist yields what there is, rather than raising.
    assert len(frame_from_bundle(bundle, model.get_preprocessing_state(), n_rows=999)) == N_POINTS


def test_extra_columns_in_a_request_are_ignored(trained) -> None:
    """A caller sending a wider frame must not reshape the model's inputs."""
    model, bundle = trained
    state = model.get_preprocessing_state()
    example = frame_from_bundle(bundle, state, n_rows=3)
    example["an_unrelated_column"] = 1.0

    rebuilt = bundle_from_frame(example, state)

    assert rebuilt.static_feature_names == STATIC
    assert rebuilt.static_features.shape[1] == len(STATIC)


def test_predict_accepts_a_plain_dict_of_columns(trained) -> None:
    """MLflow hands scoring payloads over as records; a DataFrame constructor must cover it."""
    model, bundle = trained
    example = build_input_example(model, bundle, n_rows=2)

    from_records = SoilSequencePyfunc(model).predict(None, pd.DataFrame(example.to_dict("list")))

    assert np.allclose(
        from_records.to_numpy(dtype=float),
        SoilSequencePyfunc(model).predict(None, example).to_numpy(dtype=float),
    )


# --- models that USE auxiliary lab values ------------------------------------
# The shape every test above misses: with auxiliary_label_columns=[], _select_auxiliary returns
# before the roster-width check, so the path that broke in production is never executed here.

AUXILIARY = ["ph_lab", "clay_lab"]          # a strict subset of LAB_ROSTER, like soil_cnn-58e689


@pytest.fixture
def trained_with_auxiliary():
    """A model that reads a subset of the lab roster, as the tuned config does."""
    torch.manual_seed(0)
    generator = np.random.default_rng(0)

    bundle = SoilSequenceBundle(
        point_ids=[f"p{index}" for index in range(N_POINTS)],
        static_features=generator.normal(20, 5, (N_POINTS, len(STATIC))).astype(np.float32),
        static_feature_names=list(STATIC),
        static_categoricals=np.empty((N_POINTS, 0), dtype=object),
        categorical_feature_names=[],
        targets=generator.normal(3, 1, (N_POINTS, 1)).astype(np.float32),
        target_names=["organic_matter_pct"],
        label_features=generator.normal(5, 1, (N_POINTS, len(LAB_ROSTER))).astype(np.float32),
        label_feature_names=list(LAB_ROSTER),
        sequences={
            "s2": [
                generator.normal(0.2, 0.05, (5, len(BANDS))).astype(np.float32)
                for _ in range(N_POINTS)
            ]
        },
        sequence_times={"s2": [2020.0 + np.sort(generator.random(5)) * 2.0 for _ in range(N_POINTS)]},
        modality_columns={"s2": list(BANDS)},
        temporal_enabled=True,
    )

    datamodule = SoilSequenceDataModule(
        sequence_bundle=bundle, batch_size=8, val_size=0.25, test_size=0.25, seed=42
    )
    datamodule.setup("fit")

    model = SoilCNNLightningModule(
        static_dim=datamodule.static_dim,
        target_dim=datamodule.target_dim,
        target_names=datamodule.target_names,
        modality_dims=datamodule.modality_dims,
        temporal_enabled=True,
        grid_years=datamodule.grid_years,
        auxiliary_label_columns=AUXILIARY,
        auxiliary_available_names=list(LAB_ROSTER),
        static_hidden_dims=[6],
        head_hidden_dims=[6],
        cnn_hidden_dims=[4],
        modality_embed_dim=4,
        dropout=0.0,
        target_mean=datamodule.target_mean_,
        target_scale=datamodule.target_scale_,
    )
    model.attach_preprocessing_state(datamodule.preprocessing_state())
    model.eval()
    return model, bundle


def test_the_example_asks_only_for_the_columns_the_model_reads(trained_with_auxiliary) -> None:
    model, bundle = trained_with_auxiliary
    columns = set(build_input_example(model, bundle, n_rows=3).columns)

    assert set(AUXILIARY) <= columns
    # The roster columns it does not read stay out of the contract, and so does the target.
    assert "sand_lab" not in columns
    assert "organic_matter_pct" not in columns


def test_the_rebuilt_bundle_keeps_the_full_roster_width(trained_with_auxiliary) -> None:
    """THE regression. The model index_selects roster POSITIONS, so a narrower block is unusable.

    Before the fix this came back (3, 0), and the model rejected the batch with
    'Batch carries 0 lab column(s) ... against 6' - which is what stopped every model being logged
    and therefore registered.
    """
    model, bundle = trained_with_auxiliary
    state = model.get_preprocessing_state()

    rebuilt = bundle_from_frame(build_input_example(model, bundle, n_rows=3), state)

    assert np.asarray(rebuilt.label_features).shape == (3, len(LAB_ROSTER))


def test_supplied_values_land_at_their_roster_positions(trained_with_auxiliary) -> None:
    """Position, not order of appearance - a shifted column would feed the model the wrong
    measurement without raising anything."""
    model, bundle = trained_with_auxiliary
    state = model.get_preprocessing_state()
    example = build_input_example(model, bundle, n_rows=3)
    example["clay_lab"] = [111.0, 222.0, 333.0]

    rebuilt = bundle_from_frame(example, state)
    column = np.asarray(rebuilt.label_features)[:, LAB_ROSTER.index("clay_lab")]

    assert np.allclose(column, [111.0, 222.0, 333.0])
    # A column the caller did not supply is absent, not zero: NaN is the bundle's own convention,
    # and the datamodule median-fills it and flags it as unmeasured.
    assert np.isnan(np.asarray(rebuilt.label_features)[:, LAB_ROSTER.index("sand_lab")]).all()


def test_a_model_with_auxiliary_columns_still_predicts(trained_with_auxiliary) -> None:
    model, bundle = trained_with_auxiliary
    example = build_input_example(model, bundle, n_rows=N_POINTS)

    through_pyfunc = SoilSequencePyfunc(model).predict(None, example).to_numpy(dtype=float)
    through_predictor = SoilSequencePredictor(model).predict(bundle)

    assert np.allclose(through_pyfunc, through_predictor, atol=1e-5)
