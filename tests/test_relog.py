"""Recovering a model from its checkpoint, without retraining.

A run can train perfectly and still fail to package - weights on disk, metrics logged, no model
saved and nothing registered. That happened for real, and retraining 70 epochs to recover a model
that already exists is the wrong trade.

What makes recovery possible is that a checkpoint written by this project is self-describing: it
carries the hyper-parameters and the fitted preprocessing_state alongside the weights. The tests
here pin that property, because the moment a checkpoint stops being self-contained this command
silently starts needing the dataset back.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from yg_eo_soilnet.datamodules.sequence.sequence_bundle import SoilSequenceBundle
from yg_eo_soilnet.datamodules.sequence.sequence_datamodule import SoilSequenceDataModule
from yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module import SoilCNNLightningModule
from yg_eo_soilnet.serving.lightning_pyfunc import bundle_from_frame, example_from_state

STATIC = ["clay_pct", "ph", "elevation"]
BANDS = {"s2": ["S2_B02", "S2_B08"], "clim": ["CLIM_precip"]}
LAB_ROSTER = ["organic_matter_g_kg", "total_silt_pct", "c_e_c_meq_100g", "c_n_ratio"]
AUXILIARY = ["total_silt_pct", "c_n_ratio"]
N_POINTS = 12


def _bundle() -> SoilSequenceBundle:
    generator = np.random.default_rng(0)
    return SoilSequenceBundle(
        point_ids=[f"p{index}" for index in range(N_POINTS)],
        static_features=generator.normal(20, 5, (N_POINTS, len(STATIC))).astype(np.float32),
        static_feature_names=list(STATIC),
        static_categoricals=np.asarray(
            [[generator.choice(["sandy", "loam"])] for _ in range(N_POINTS)], dtype=object
        ),
        categorical_feature_names=["texture"],
        targets=generator.normal(3, 1, (N_POINTS, 1)).astype(np.float32),
        target_names=["organic_matter_g_kg"],
        label_features=generator.normal(5, 1, (N_POINTS, len(LAB_ROSTER))).astype(np.float32),
        label_feature_names=list(LAB_ROSTER),
        sequences={
            name: [generator.normal(0.2, 0.05, (5, len(columns))).astype(np.float32) for _ in range(N_POINTS)]
            for name, columns in BANDS.items()
        },
        sequence_times={
            name: [2020.0 + np.sort(generator.random(5)) * 2.0 for _ in range(N_POINTS)] for name in BANDS
        },
        modality_columns={name: list(columns) for name, columns in BANDS.items()},
        temporal_enabled=True,
    )


@pytest.fixture
def trained(tmp_path):
    """A trained model, its checkpoint on disk, and nothing else - as after a failed packaging."""
    torch.manual_seed(0)
    bundle = _bundle()
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
        auxiliary_label_columns=AUXILIARY,
        auxiliary_available_names=datamodule.label_feature_names,
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

    import lightning.pytorch as pl

    checkpoint = {
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "pytorch-lightning_version": pl.__version__,
    }
    model.on_save_checkpoint(checkpoint)
    path = tmp_path / "epoch=70-step=3763.ckpt"
    torch.save(checkpoint, path)
    return model, bundle, path


# --- the property the whole command rests on --------------------------------


def test_a_checkpoint_is_self_describing(trained) -> None:
    """No dataset required: the checkpoint alone must carry everything needed to rebuild."""
    _model, _bundle, path = trained

    payload = torch.load(path, weights_only=True)
    state = payload["preprocessing_state"]

    assert state["static_feature_names"] == STATIC
    assert state["label_feature_names"] == LAB_ROSTER
    assert set(state["modality_column_names"]) == set(BANDS)
    assert state["categorical_vocabularies"]
    assert payload["hyper_parameters"]["auxiliary_label_columns"] == AUXILIARY


def test_a_model_rebuilt_from_the_checkpoint_alone_predicts(trained) -> None:
    _model, _bundle, path = trained

    restored = SoilCNNLightningModule.load_from_checkpoint(path, map_location="cpu")
    restored.eval()
    state = restored.get_preprocessing_state()

    example = example_from_state(state, auxiliary_columns=AUXILIARY, n_rows=3)
    predictions = restored.predict_step(
        _collate(bundle_from_frame(example, state), state), 0
    )

    assert predictions.shape[0] == 3
    assert torch.isfinite(predictions).all()


def _collate(rebuilt, state):
    datamodule = SoilSequenceDataModule(sequence_bundle=rebuilt, batch_size=8)
    datamodule.apply_preprocessing_state(state)
    return datamodule.collate(np.arange(rebuilt.num_points))


# --- example_from_state ------------------------------------------------------


def test_the_example_satisfies_the_reader(trained) -> None:
    _model, _bundle, path = trained
    state = torch.load(path, weights_only=True)["preprocessing_state"]

    rebuilt = bundle_from_frame(example_from_state(state, auxiliary_columns=AUXILIARY), state)

    assert rebuilt.num_points == 3
    # Roster width, not the count of supplied columns - the model index_selects positions.
    assert np.asarray(rebuilt.label_features).shape == (3, len(LAB_ROSTER))
    for name, columns in BANDS.items():
        assert rebuilt.sequences[name][0].shape[1] == len(columns)


def test_supplied_statics_override_the_stored_means(trained) -> None:
    _model, _bundle, path = trained
    state = torch.load(path, weights_only=True)["preprocessing_state"]

    example = example_from_state(
        state,
        static_frame=pd.DataFrame({"clay_pct": [77.0, 78.0, 79.0]}),
        auxiliary_columns=AUXILIARY,
    )

    assert list(example["clay_pct"]) == [77.0, 78.0, 79.0]
    # A column the frame does not carry falls back, so a partial CSV degrades rather than fails.
    assert np.isclose(example["ph"].iloc[0], state["static_mean"][STATIC.index("ph")])


def test_the_example_works_with_no_static_frame_at_all(trained) -> None:
    """The case that matters on a machine that no longer has the data or the run's artifacts."""
    _model, _bundle, path = trained
    state = torch.load(path, weights_only=True)["preprocessing_state"]

    example = example_from_state(state, static_frame=None, auxiliary_columns=AUXILIARY)

    assert set(STATIC) <= set(example.columns)
    assert np.isfinite(example[STATIC].to_numpy(dtype=float)).all()


def test_the_example_asks_only_for_the_auxiliary_columns_the_model_reads(trained) -> None:
    _model, _bundle, path = trained
    state = torch.load(path, weights_only=True)["preprocessing_state"]

    columns = set(example_from_state(state, auxiliary_columns=AUXILIARY).columns)

    assert set(AUXILIARY) <= columns
    assert "c_e_c_meq_100g" not in columns          # in the roster, not read by this model
    assert "organic_matter_g_kg" not in columns     # the target is never an input


def test_categoricals_use_a_real_vocabulary_entry(trained) -> None:
    """An unknown label would land on the reserved OOV index and exercise a different embedding
    row than production does."""
    _model, _bundle, path = trained
    state = torch.load(path, weights_only=True)["preprocessing_state"]

    example = example_from_state(state, auxiliary_columns=AUXILIARY)

    assert example["texture"].iloc[0] in state["categorical_vocabularies"][0]


# --- the command -------------------------------------------------------------


def _relog(tmp_path, checkpoint, *, register=True):
    """Run the relog entry point against a temp store holding one finished, unpackaged run."""
    import mlflow

    from relog import relog as run_relog
    from yg_eo_soilnet.tracking import configure_tracking

    configure_tracking(
        type("_C", (), {
            "MLFLOW_TRACKING_URI": (tmp_path / "mlruns").as_uri(),
            "MLFLOW_EXPERIMENT_NAME": "Relog",
        })()
    )
    with mlflow.start_run() as run:
        mlflow.set_tags({"model_name": "soil_cnn", "target": "organic_matter_g_kg"})
        mlflow.log_metric("rmse_test", 7.66)
        run_id = run.info.run_id

    args = type("_A", (), {
        "checkpoint": str(checkpoint),
        "run_id": run_id,
        "model_class": "yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module.SoilCNNLightningModule",
        "config_path": "configs/main_config.yml",
        "no_register": not register,
        "rows": 3,
    })()
    return run_relog(args), run_id


def test_relog_logs_registers_and_promotes(trained, tmp_path) -> None:
    """The end-to-end recovery: checkpoint plus run id in, registered champion out."""
    import mlflow

    _model, _bundle, checkpoint = trained
    summary, run_id = _relog(tmp_path, checkpoint)

    assert summary["registered_model_version"] is not None
    assert summary["champion"]["promoted"] is True
    assert summary["rmse_test"] == 7.66

    client = mlflow.MlflowClient()
    name = "organic_matter_g_kg_soil_cnn"
    version = client.get_model_version_by_alias(name, "champion")
    # The model belongs to the run that produced it, which is what lets promotion read its score.
    assert version.run_id == run_id


def test_the_recovered_model_actually_serves(trained, tmp_path) -> None:
    import mlflow

    _model, _bundle, checkpoint = trained
    _summary, _run_id = _relog(tmp_path, checkpoint)

    loaded = mlflow.pyfunc.load_model("models:/organic_matter_g_kg_soil_cnn@champion")
    state = torch.load(checkpoint, weights_only=True)["preprocessing_state"]
    predictions = loaded.predict(example_from_state(state, auxiliary_columns=AUXILIARY))

    assert len(predictions) == 3
    assert np.isfinite(np.asarray(predictions, dtype=float)).all()


def test_no_register_logs_without_registering(trained, tmp_path) -> None:
    _model, _bundle, checkpoint = trained

    summary, _run_id = _relog(tmp_path, checkpoint, register=False)

    assert summary["registered_model_version"] is None
    assert summary["champion"]["reason"] == "registration was disabled"


def test_a_checkpoint_without_preprocessing_state_is_refused(tmp_path) -> None:
    """Such a model cannot standardize raw input, so logging it would produce something unservable."""
    import lightning.pytorch as pl

    model = SoilCNNLightningModule(static_dim=2, target_dim=1, temporal_enabled=False)
    path = tmp_path / "bare.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": pl.__version__,
        },
        path,
    )

    with pytest.raises(SystemExit, match="no preprocessing state"):
        _relog(tmp_path, path)
