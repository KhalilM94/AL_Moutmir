"""Reloading a trained deep-learning model and predicting with it.

The gap this closes: the checkpoint always carried the weights and the TARGET inverse-transform
(both buffers), but the INPUT standardization and the categorical vocabulary were fitted on the
datamodule and thrown away with it. A restored model could therefore only be fed data some
datamodule had already scaled - which is to say, it could not be deployed the way a pickled sklearn
Pipeline can.

The load-bearing test here is the round-trip one: save, reload, predict from a raw bundle, and get
back what trainer.predict produced on the same points.
"""

import numpy as np
import pytest
import torch

from yg_eo_soilnet.datamodules.sequence.sequence_bundle import SoilSequenceBundle
from yg_eo_soilnet.datamodules.sequence.sequence_datamodule import SoilSequenceDataModule
from yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module import SoilCNNLightningModule
from yg_eo_soilnet.serving import SoilSequencePredictor

STATIC_NAMES = ["clay_pct", "ph"]
S2_BANDS = ["S2_B02", "S2_B08"]
N_POINTS = 24


def _bundle(seed: int = 0) -> SoilSequenceBundle:
    generator = np.random.default_rng(seed)
    return SoilSequenceBundle(
        point_ids=[f"p{index}" for index in range(N_POINTS)],
        static_features=generator.normal(20, 5, (N_POINTS, len(STATIC_NAMES))).astype(np.float32),
        static_feature_names=list(STATIC_NAMES),
        static_categoricals=np.asarray(
            [[generator.choice(["sandy", "loam"])] for _ in range(N_POINTS)], dtype=object
        ),
        categorical_feature_names=["texture"],
        targets=generator.normal(3, 1, (N_POINTS, 1)).astype(np.float32),
        target_names=["organic_matter_pct"],
        sequences={
            "s2": [
                generator.normal(0.2, 0.05, (generator.integers(3, 8), len(S2_BANDS))).astype(np.float32)
                for _ in range(N_POINTS)
            ]
        },
        sequence_times={},
        modality_columns={"s2": list(S2_BANDS)},
        temporal_enabled=True,
    )


def _with_times(bundle: SoilSequenceBundle, seed: int = 0) -> SoilSequenceBundle:
    generator = np.random.default_rng(seed + 100)
    bundle.sequence_times = {
        "s2": [
            2020.0 + np.sort(generator.random(values.shape[0])) * 2.0
            for values in bundle.sequences["s2"]
        ]
    }
    return bundle


@pytest.fixture
def trained() -> tuple[SoilCNNLightningModule, SoilSequenceDataModule, SoilSequenceBundle]:
    torch.manual_seed(0)
    bundle = _with_times(_bundle())

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
    return model, datamodule, bundle


def test_preprocessing_state_is_plain_builtins(trained) -> None:
    """A numpy array in hyper_parameters makes the checkpoint unloadable under weights_only=True."""
    _model, datamodule, _bundle_ = trained
    state = datamodule.preprocessing_state()

    def assert_plain(value, path="state"):
        if isinstance(value, dict):
            for key, item in value.items():
                assert isinstance(key, str), f"{path} has a non-string key {key!r}"
                assert_plain(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                assert_plain(item, f"{path}[{index}]")
        else:
            assert isinstance(value, (str, int, float, bool)), f"{path} is {type(value).__name__}"

    assert_plain(state)


def test_preprocessing_state_carries_the_input_scalers(trained) -> None:
    _model, datamodule, _bundle_ = trained
    state = datamodule.preprocessing_state()

    assert len(state["static_mean"]) == len(STATIC_NAMES)
    assert len(state["static_scale"]) == len(STATIC_NAMES)
    assert state["static_feature_names"] == STATIC_NAMES
    assert state["modality_column_names"] == {"s2": S2_BANDS}
    assert len(state["sequence_mean"]["s2"]) == len(S2_BANDS)
    assert state["categorical_vocabularies"] and state["categorical_feature_names"] == ["texture"]


def test_a_reloaded_checkpoint_predicts_the_same_values(trained, tmp_path) -> None:
    """The round trip: save, reload with weights_only semantics, predict from a RAW bundle."""
    model, _datamodule, bundle = trained

    before = SoilSequencePredictor(model).predict(bundle)

    checkpoint = {"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams)}
    model.on_save_checkpoint(checkpoint)

    checkpoint_path = tmp_path / "model.ckpt"
    torch.save(checkpoint, checkpoint_path)
    # weights_only=True is torch's default from 2.6 on, so anything the checkpoint carries has to
    # survive it. This is why preprocessing_state is plain builtins.
    payload = torch.load(checkpoint_path, weights_only=True)

    assert "preprocessing_state" in payload

    restored = SoilCNNLightningModule(**payload["hyper_parameters"])
    restored.load_state_dict(payload["state_dict"])
    restored.on_load_checkpoint(payload)
    restored.eval()

    after = SoilSequencePredictor(restored).predict(bundle)

    assert np.allclose(before, after)


def test_predictions_come_back_in_original_target_units(trained) -> None:
    """predict_step inverts the standardization; forward alone would return standardized values."""
    model, _datamodule, bundle = trained

    predictions = SoilSequencePredictor(model).predict(bundle)

    datamodule = SoilSequenceDataModule(sequence_bundle=bundle, batch_size=8)
    datamodule.apply_preprocessing_state(model.get_preprocessing_state())
    with torch.no_grad():
        standardized = model(datamodule.collate(np.arange(bundle.num_points))).numpy()

    assert predictions.shape == (bundle.num_points, 1)
    assert not np.allclose(predictions, standardized)


def test_scalers_are_not_refitted_on_the_incoming_points(trained) -> None:
    """A serving batch is not a training split.

    Predicting for a subset must give each point the same answer it gets in the full batch. If the
    scaler were refitted per request, a point's prediction would depend on which other points
    happened to arrive with it.
    """
    model, _datamodule, bundle = trained
    predictor = SoilSequencePredictor(model)

    full = predictor.predict(bundle)

    subset = SoilSequenceBundle.from_mapping(
        {
            **{field: getattr(bundle, field) for field in bundle.keys()},
            "point_ids": bundle.point_ids[:4],
            "static_features": bundle.static_features[:4],
            "static_categoricals": bundle.static_categoricals[:4],
            "targets": bundle.targets[:4],
            "sequences": {"s2": bundle.sequences["s2"][:4]},
            "sequence_times": {"s2": bundle.sequence_times["s2"][:4]},
        }
    )

    assert np.allclose(predictor.predict(subset), full[:4], atol=1e-5)


def test_a_single_point_is_scored_consistently(trained) -> None:
    model, _datamodule, bundle = trained
    predictor = SoilSequencePredictor(model)
    full = predictor.predict(bundle)

    single = SoilSequenceBundle.from_mapping(
        {
            **{field: getattr(bundle, field) for field in bundle.keys()},
            "point_ids": bundle.point_ids[:1],
            "static_features": bundle.static_features[:1],
            "static_categoricals": bundle.static_categoricals[:1],
            "targets": bundle.targets[:1],
            "sequences": {"s2": bundle.sequences["s2"][:1]},
            "sequence_times": {"s2": bundle.sequence_times["s2"][:1]},
        }
    )

    assert np.allclose(predictor.predict(single), full[:1], atol=1e-5)


def test_an_unseen_category_lands_on_the_reserved_index_rather_than_shifting_the_others(
    trained,
) -> None:
    model, _datamodule, bundle = trained
    predictor = SoilSequencePredictor(model)

    unseen = SoilSequenceBundle.from_mapping(
        {**{field: getattr(bundle, field) for field in bundle.keys()}}
    )
    unseen.static_categoricals = np.asarray([["volcanic"]] * bundle.num_points, dtype=object)

    predictions = predictor.predict(unseen)

    assert predictions.shape == (bundle.num_points, 1)
    assert np.isfinite(predictions).all()


def test_predict_frame_labels_columns_with_the_target_names(trained) -> None:
    model, _datamodule, bundle = trained

    frame = SoilSequencePredictor(model).predict_frame(bundle)

    assert list(frame.columns) == ["organic_matter_pct"]
    assert list(frame.index) == list(bundle.point_ids)


def test_a_checkpoint_without_the_state_is_refused_with_an_actionable_message() -> None:
    model = SoilCNNLightningModule(static_dim=2, target_dim=1, temporal_enabled=False)

    with pytest.raises(ValueError, match="no preprocessing state"):
        SoilSequencePredictor(model)


def test_an_empty_bundle_predicts_nothing_rather_than_raising(trained) -> None:
    model, _datamodule, _bundle_ = trained

    predictions = SoilSequencePredictor(model).predict(SoilSequenceBundle())

    assert predictions.shape[0] == 0
