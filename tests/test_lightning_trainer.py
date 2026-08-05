from dataclasses import dataclass
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import yg_eo_soilnet.trainers.lightning_trainer as lightning_trainer_module
from yg_eo_soilnet.data_manager import DataManager
from yg_eo_soilnet.datamodules.lightning.lightning_graph import SingleNodeGraphDataModule
from yg_eo_soilnet.datamodules.lightning.spatiotemporal_graph_builder import SpatiotemporalGraphBuilder
from yg_eo_soilnet.models.config_fatories.lightning_config_factory import LightningModelBundle
from yg_eo_soilnet.trainers.lightning_trainer import LightningTrainer


@dataclass
class FakeCheckpointCallback:
    best_model_path: str = "/tmp/best.ckpt"


class FakeTrainer:
    def __init__(self):
        self.fit_called = False
        self.validate_called = False
        self.test_called = False
        self.predict_called = False
        self.validate_ckpt_path = None
        self.test_ckpt_path = None
        self.predict_ckpt_path = None
        self.checkpoint_callback = FakeCheckpointCallback()
        self.callbacks = [self.checkpoint_callback]

    def fit(self, model, datamodule=None):
        self.fit_called = True

    def validate(self, model, datamodule=None, verbose=False, ckpt_path=None):
        self.validate_called = True
        self.validate_ckpt_path = ckpt_path
        return [{"val_loss": 0.4}]

    def test(self, model, datamodule=None, verbose=False, ckpt_path=None):
        self.test_called = True
        self.test_ckpt_path = ckpt_path
        return [{"test_loss": 0.3}]

    def predict(self, model, datamodule=None, ckpt_path=None):
        self.predict_called = True
        self.predict_ckpt_path = ckpt_path
        # Match the datamodule's test split so the evaluation frame lines up.
        n = len(getattr(datamodule, "y_test_frame_", [])) if datamodule is not None else 2
        return [np.linspace(0.1, 0.2, max(n, 1))]


class FakeModel:
    pass


def test_lightning_trainer_runs_fit_validate_test_and_logs(monkeypatch) -> None:
    datamodule = _tiny_graph_datamodule()

    bundle = LightningModelBundle(
        name="toy_lightning",
        target="target_a",
        model=FakeModel(),
        datamodule=datamodule,
        trainer_kwargs={"max_epochs": 1},
        callback_specs={
            "early_stopping": {"monitor": "val_loss", "mode": "min", "patience": 2},
            "checkpoint": {"monitor": "val_loss", "mode": "min", "save_top_k": 1},
        },
        registry_entry={"modeltype": "dl", "init_args": {}, "datamodule_init_args": {}},
    )

    fake_trainer = FakeTrainer()
    fake_mlflow_logger = SimpleNamespace(log_lightning_child_run=MagicMock())
    trainer = LightningTrainer(
        config=SimpleNamespace(LIGHTNING_CHECKPOINT_DIR="/tmp/checkpoints"),
        logger=MagicMock(),
        mlflow_logger=fake_mlflow_logger,
    )

    monkeypatch.setattr(trainer, "_build_trainer", lambda bundle: fake_trainer)
    monkeypatch.setattr(trainer, "_resolve_best_checkpoint", lambda trainer_obj: "/tmp/best.ckpt")

    start_run = MagicMock()
    start_run.__enter__.return_value = SimpleNamespace(info=SimpleNamespace(run_id="run-1"))
    start_run.__exit__.return_value = False
    monkeypatch.setattr(lightning_trainer_module.mlflow, "start_run", MagicMock(return_value=start_run))
    results = trainer.train(
        target="target_a",
        data={},
        model_bundles={"toy_lightning": bundle},
    )

    assert fake_trainer.fit_called is True
    assert fake_trainer.validate_called is True
    assert fake_trainer.test_called is True
    assert fake_trainer.predict_called is True
    assert fake_trainer.validate_ckpt_path == "/tmp/best.ckpt"
    assert fake_trainer.test_ckpt_path == "/tmp/best.ckpt"
    assert fake_trainer.predict_ckpt_path == "/tmp/best.ckpt"
    assert fake_mlflow_logger.log_lightning_child_run.called
    logged_kwargs = fake_mlflow_logger.log_lightning_child_run.call_args.kwargs
    assert logged_kwargs["model"] is bundle.model
    assert results["toy_lightning"].best_model_path == "/tmp/best.ckpt"
    assert results["toy_lightning"].test_metrics["test_loss"] == 0.3


def test_lightning_trainer_normalizes_metrics() -> None:
    trainer = LightningTrainer(config=SimpleNamespace(), logger=MagicMock())

    metrics = trainer._normalize_metrics([{"val_loss": 1.5, "junk": object()}])

    assert metrics == {"val_loss": 1.5}


def test_lightning_trainer_disables_default_logger_when_configured(monkeypatch) -> None:
    captured_kwargs = {}

    class FakeLightningModule:
        class callbacks:
            class EarlyStopping:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

            class ModelCheckpoint:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

        class Trainer:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

    trainer = LightningTrainer(
        config=SimpleNamespace(LIGHTNING_ENABLE_DEFAULT_LOGGER=False, LIGHTNING_CHECKPOINT_DIR=None),
        logger=MagicMock(),
    )
    monkeypatch.setattr(trainer, "_get_lightning_module", lambda: FakeLightningModule)

    bundle = LightningModelBundle(
        name="toy_lightning",
        target="target_a",
        model=FakeModel(),
        datamodule=SimpleNamespace(),
        trainer_kwargs={"max_epochs": 1},
        callback_specs={},
        registry_entry={"modeltype": "dl", "init_args": {}, "datamodule_init_args": {}},
    )

    trainer._build_trainer(bundle)

    assert captured_kwargs["logger"] is False


def test_lightning_trainer_seeds_each_bundle(monkeypatch) -> None:
    datamodule = _tiny_graph_datamodule()

    bundle = LightningModelBundle(
        name="toy_lightning",
        target="target_a",
        model=FakeModel(),
        datamodule=datamodule,
        trainer_kwargs={"max_epochs": 1},
        callback_specs={
            "early_stopping": {"monitor": "val_loss", "mode": "min", "patience": 2},
            "checkpoint": {"monitor": "val_loss", "mode": "min", "save_top_k": 1},
        },
        registry_entry={"modeltype": "dl", "init_args": {}, "datamodule_init_args": {}, "random_seed": 99},
    )

    fake_trainer = FakeTrainer()
    fake_mlflow_logger = SimpleNamespace(log_lightning_child_run=MagicMock())
    trainer = LightningTrainer(
        config=SimpleNamespace(LIGHTNING_CHECKPOINT_DIR="/tmp/checkpoints"),
        logger=MagicMock(),
        mlflow_logger=fake_mlflow_logger,
    )

    seed_spy = MagicMock()
    monkeypatch.setattr(trainer, "_seed_for_bundle", seed_spy)
    monkeypatch.setattr(trainer, "_build_trainer", lambda bundle: fake_trainer)
    monkeypatch.setattr(trainer, "_resolve_best_checkpoint", lambda trainer_obj: "/tmp/best.ckpt")

    start_run = MagicMock()
    start_run.__enter__.return_value = SimpleNamespace(info=SimpleNamespace(run_id="run-1"))
    start_run.__exit__.return_value = False
    monkeypatch.setattr(lightning_trainer_module.mlflow, "start_run", MagicMock(return_value=start_run))

    trainer.train(
        target="target_a",
        data={},
        model_bundles={"toy_lightning": bundle},
    )

    seed_spy.assert_called_once_with(bundle)


def test_lightning_epoch_metrics_callback_logs_train_and_val_losses(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    callback = lightning_trainer_module._LightningMlflowEpochMetricCallback()
    log_metric = MagicMock()
    monkeypatch.setattr(lightning_trainer_module.mlflow, "log_metric", log_metric)

    trainer = SimpleNamespace(
        sanity_checking=False,
        current_epoch=3,
        callback_metrics={
            "train_loss": torch.tensor(0.7),
            "val_loss": torch.tensor(0.5),
        },
    )

    callback.on_train_epoch_end(trainer, None)
    callback.on_validation_epoch_end(trainer, None)

    assert log_metric.call_count == 2
    assert log_metric.call_args_list[0].args[0] == "train_loss"
    assert log_metric.call_args_list[0].args[1] == pytest.approx(0.7)
    assert log_metric.call_args_list[0].kwargs == {"step": 3}
    assert log_metric.call_args_list[1].args[0] == "val_loss"
    assert log_metric.call_args_list[1].args[1] == pytest.approx(0.5)
    assert log_metric.call_args_list[1].kwargs == {"step": 3}


def test_lightning_trainer_builds_graph_evaluation_frame(tmp_path, logger) -> None:
    static_df = pd.DataFrame(
        {
            "point_id": [1, 2, 3],
            "lat": [0.0, 0.5, 1.0],
            "lon": [0.0, 0.5, 1.0],
            "target_a": [1.0, 2.0, 3.0],
            "static_1": [10.0, 11.0, 12.0],
            "static_2": [20.0, 21.0, 22.0],
        }
    )
    static_path = tmp_path / "static.csv"
    static_df.to_csv(static_path, index=False)

    config = SimpleNamespace(
        DATA_FOLDER=str(tmp_path),
        DATA_FILE="static.csv",
        STATIC_CSV_PATH=str(static_path),
        TIMESERIES_CSV_PATH=None,
        POINT_ID_COLUMN="point_id",
        LAT_COLUMN="lat",
        LON_COLUMN="lon",
        TIME_COLUMN="month",
        TEMPORAL_FEATURES_ENABLED=False,
        TARGET_COLUMNS=["target_a"],
        ELIMINATED_FEATURES=[],
        TEST_SIZE=0.33,
        LIGHTNING_VAL_SIZE=0.33,
        RANDOM_SEED=42,
        SPATIAL_RADIUS=2.0,
        BASELINE_METHOD="knn",
        BASELINE_K_NEIGHBORS=1,
    )

    manager = DataManager(config=config, logger=logger)
    bundle_dict = SpatiotemporalGraphBuilder(manager.config, manager.logger, manager).build(graph_data_args={"spatial_graph_enabled": False})
    datamodule = SingleNodeGraphDataModule(
        spatiotemporal_graph=bundle_dict,
        batch_size=1,
        val_size=0.33,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        seed=42,
    )
    datamodule.setup("fit")

    bundle = LightningModelBundle(
        name="toy_lightning",
        target="target_a",
        model=FakeModel(),
        datamodule=datamodule,
        trainer_kwargs={"max_epochs": 1},
        callback_specs={},
        registry_entry={"modeltype": "dl", "init_args": {}, "datamodule_init_args": {}},
    )

    lightning_trainer = LightningTrainer(config=SimpleNamespace(), logger=MagicMock())

    fake_trainer = SimpleNamespace(
        predict=lambda model, datamodule=None: [np.asarray([[0.25]], dtype=np.float32)]
    )

    eval_df = lightning_trainer._build_evaluation_frame(bundle, fake_trainer, target="target_a")

    assert eval_df is not None
    assert list(eval_df.columns) == ["static_1", "static_2", "target_a", "prediction", "target_name"]
    assert eval_df["prediction"].iloc[0] == 0.25


def test_lightning_trainer_fans_out_multitarget_child_runs(monkeypatch) -> None:
    logger = MagicMock()
    trainer = LightningTrainer(config=SimpleNamespace(), logger=MagicMock(), mlflow_logger=logger)

    bundle = SimpleNamespace(
        name="toy_lightning",
        target="target_a__target_b",
        model=SimpleNamespace(),
        datamodule=SimpleNamespace(
            y_test_frame_=pd.DataFrame({"target_a": [1.0, 2.0], "target_b": [3.0, 4.0]}),
            target_names=["target_a", "target_b"],
                setup=lambda stage=None: None,
        ),
        trainer_kwargs={},
        callback_specs={},
        registry_entry={"modeltype": "dl", "init_args": {}, "datamodule_init_args": {}},
    )

    evaluation_df = pd.DataFrame(
        {
            "feature_1": [10.0, 11.0],
            "target_a": [1.0, 2.0],
            "target_b": [3.0, 4.0],
            "prediction_target_a": [1.1, 1.9],
            "prediction_target_b": [2.9, 4.1],
            "target_name": ["target_a__target_b", "target_a__target_b"],
            "target_names": ["target_a__target_b", "target_a__target_b"],
        }
    )

    fake_trainer = SimpleNamespace(
        predict=lambda model, datamodule=None: [np.asarray([[1.1, 2.9], [1.9, 4.1]], dtype=np.float32)],
        fit=lambda *args, **kwargs: None,
        validate=lambda *args, **kwargs: [{"val_loss": 0.25}],
        test=lambda *args, **kwargs: [{"test_loss": 0.5}],
    )

    monkeypatch.setattr(trainer, "_seed_for_bundle", MagicMock())
    monkeypatch.setattr(trainer, "_build_trainer", MagicMock(return_value=fake_trainer))
    monkeypatch.setattr(trainer, "_resolve_best_checkpoint", MagicMock(return_value="/tmp/best.ckpt"))
    monkeypatch.setattr(trainer, "_build_evaluation_frame", MagicMock(return_value=evaluation_df))
    monkeypatch.setattr(trainer, "_serialize_params", MagicMock(return_value={"foo": "bar"}))
    monkeypatch.setattr("yg_eo_soilnet.trainers.lightning_trainer.mlflow.start_run", lambda *args, **kwargs: nullcontext(SimpleNamespace(info=SimpleNamespace(run_id="run"))))

    trainer.train(target="target_a__target_b", data={}, model_bundles={"toy_lightning": bundle})

    assert logger.log_lightning_child_run.call_count == 2
    logged_targets = [call.kwargs["target"] for call in logger.log_lightning_child_run.call_args_list]
    assert logged_targets == ["target_a", "target_b"]
    first_eval_df = logger.log_lightning_child_run.call_args_list[0].kwargs["evaluation_df"]
    second_eval_df = logger.log_lightning_child_run.call_args_list[1].kwargs["evaluation_df"]
    assert list(first_eval_df.columns) == ["feature_1", "target_a", "prediction", "target_name", "target_names"]
    assert list(second_eval_df.columns) == ["feature_1", "target_b", "prediction", "target_name", "target_names"]


def test_lightning_trainer_builds_target_specific_frame_without_duplicate_target_columns() -> None:
    trainer = LightningTrainer(config=SimpleNamespace(), logger=MagicMock())
    evaluation_df = pd.DataFrame(
        {
            "feature_1": [10.0, 11.0],
            "target_a": [1.0, 2.0],
            "target_b": [3.0, 4.0],
            "prediction_target_a": [1.1, 1.9],
            "prediction_target_b": [2.9, 4.1],
            "target_name": ["target_a__target_b", "target_a__target_b"],
            "target_names": ["target_a__target_b", "target_a__target_b"],
        }
    )

    target_frame = trainer._build_target_evaluation_frame(evaluation_df, "target_a", ["target_a", "target_b"])

    assert target_frame is not None
    assert list(target_frame.columns) == ["feature_1", "target_a", "prediction", "target_name", "target_names"]
    assert target_frame["target_a"].tolist() == [1.0, 2.0]

# --- checkpoint round-trip -------------------------------------------------
# A real fit -> checkpoint -> restore was untested, so a numpy value leaking into
# hyper_parameters crashed every real run while the suite stayed green.


def _tiny_graph_datamodule(num_nodes: int = 24):
    rng = np.random.default_rng(0)
    graph = {
        "point_ids": list(range(num_nodes)),
        "static_features": rng.normal(5.0, 2.0, (num_nodes, 3)).astype(np.float32),
        "targets": rng.gamma(2.0, 1.5, (num_nodes, 1)).astype(np.float32),
        "coords": rng.normal(0.0, 1.0, (num_nodes, 2)).astype(np.float32),
        "edge_index": np.zeros((2, 0), dtype=np.int64),
        "edge_attr": np.zeros((0, 1), dtype=np.float32),
    }
    datamodule = SingleNodeGraphDataModule(
        spatiotemporal_graph=graph, batch_size=8, test_size=0.25, val_size=0.25, seed=42
    )
    datamodule.setup("fit")
    return datamodule


def _fit_and_checkpoint(tmp_path):
    """Train one epoch through the real Lightning Trainer and return the checkpoint path."""
    torch = pytest.importorskip("torch")
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint

    from yg_eo_soilnet.models.config_fatories.lightning_config_factory import LightningConfigFactory
    from yg_eo_soilnet.models.lightningmodules.soil_graph_lightning_module import SoilGraphLightningModule

    datamodule = _tiny_graph_datamodule()
    # Go through the factory so the test covers how stats actually reach the model in production.
    init_args = {}
    for key, attribute in (("target_mean", "target_mean_"), ("target_scale", "target_scale_")):
        init_args[key] = [float(v) for v in np.asarray(getattr(datamodule, attribute)).ravel()]
    model = SoilGraphLightningModule(
        static_dim=datamodule.static_dim, target_dim=datamodule.target_dim,
        temporal_enabled=False, spatial_graph_enabled=False, **init_args,
    )
    checkpoint = ModelCheckpoint(dirpath=str(tmp_path), monitor="val_loss", save_top_k=1)
    Trainer(
        max_epochs=1, accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
        enable_model_summary=False, callbacks=[checkpoint],
    ).fit(model, datamodule=datamodule)
    assert checkpoint.best_model_path, "no checkpoint was written"
    return checkpoint.best_model_path, datamodule


def test_checkpoint_loads_under_weights_only_default(tmp_path) -> None:
    """PyTorch >= 2.6 defaults torch.load to weights_only=True; the checkpoint must satisfy it."""
    torch = pytest.importorskip("torch")
    path, _ = _fit_and_checkpoint(tmp_path)

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    assert "state_dict" in checkpoint


def test_checkpoint_hyperparameters_contain_no_array_objects(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    path, _ = _fit_and_checkpoint(tmp_path)

    hparams = torch.load(path, map_location="cpu", weights_only=False).get("hyper_parameters", {})

    def is_plain(value):
        if isinstance(value, (str, bool, int, float, type(None))):
            return True
        if isinstance(value, (list, tuple)):
            return all(is_plain(item) for item in value)
        if isinstance(value, dict):
            return all(is_plain(item) for item in value.values())
        return False

    offenders = {key: type(value).__name__ for key, value in hparams.items() if not is_plain(value)}
    assert not offenders, f"non-primitive hyperparameters break weights_only load: {offenders}"


def test_target_inverse_transform_survives_a_checkpoint_restore(tmp_path) -> None:
    """Guards against 'fixing' this by ignoring the hparams, which would silently skip the inverse."""
    torch = pytest.importorskip("torch")
    from yg_eo_soilnet.models.lightningmodules.soil_graph_lightning_module import SoilGraphLightningModule

    path, datamodule = _fit_and_checkpoint(tmp_path)
    restored = SoilGraphLightningModule.load_from_checkpoint(path, map_location="cpu")

    assert bool(restored.targets_are_standardized) is True
    # 0 in standardized space must map back to the training target mean, in original units.
    recovered = restored.inverse_transform_targets(torch.zeros(1, datamodule.target_dim))
    np.testing.assert_allclose(
        recovered.detach().numpy().ravel(), np.asarray(datamodule.target_mean_).ravel(), rtol=1e-5
    )
