from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

import yg_eo_soilnet.trainers.lightning_trainer as lightning_trainer_module
from yg_eo_soilnet.datamodules.lightning_datamodule import LightningTabularDataModule
from yg_eo_soilnet.models.lightning_config_factory import LightningModelBundle
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
        self.checkpoint_callback = FakeCheckpointCallback()
        self.callbacks = [self.checkpoint_callback]

    def fit(self, model, datamodule=None):
        self.fit_called = True

    def validate(self, model, datamodule=None, verbose=False):
        self.validate_called = True
        return [{"val_loss": 0.4}]

    def test(self, model, datamodule=None, verbose=False):
        self.test_called = True
        return [{"test_loss": 0.3}]

    def predict(self, model, datamodule=None):
        self.predict_called = True
        return [pd.Series([0.1, 0.2]).to_numpy()]


class FakeModel:
    pass


def test_lightning_trainer_runs_fit_validate_test_and_logs(monkeypatch) -> None:
    X_train = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0], "cat": ["a", "b", "a", "b"]})
    y_train = pd.DataFrame({"target_a": [10.0, 11.0, 12.0, 13.0]})
    X_test = pd.DataFrame({"feature": [5.0, 6.0], "cat": ["a", "b"]})
    y_test = pd.DataFrame({"target_a": [14.0, 15.0]})

    datamodule = LightningTabularDataModule(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        batch_size=2,
        val_size=0.25,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        seed=7,
        target_columns=["target_a"],
    )
    datamodule.setup("fit")

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
    trainer = LightningTrainer(
        config=SimpleNamespace(LIGHTNING_CHECKPOINT_DIR="/tmp/checkpoints"),
        logger=MagicMock(),
    )

    monkeypatch.setattr(trainer, "_build_trainer", lambda bundle: fake_trainer)
    monkeypatch.setattr(trainer, "_resolve_best_checkpoint", lambda trainer_obj: "/tmp/best.ckpt")

    start_run = MagicMock()
    start_run.__enter__.return_value = SimpleNamespace(info=SimpleNamespace(run_id="run-1"))
    start_run.__exit__.return_value = False
    monkeypatch.setattr(lightning_trainer_module.mlflow, "start_run", MagicMock(return_value=start_run))
    monkeypatch.setattr(lightning_trainer_module.mlflow, "set_tags", MagicMock())
    monkeypatch.setattr(lightning_trainer_module.mlflow, "log_params", MagicMock())
    monkeypatch.setattr(lightning_trainer_module.mlflow, "log_metric", MagicMock())
    monkeypatch.setattr(lightning_trainer_module.mlflow, "log_artifact", MagicMock())

    results = trainer.train(
        target="target_a",
        data={"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test},
        model_bundles={"toy_lightning": bundle},
    )

    assert fake_trainer.fit_called is True
    assert fake_trainer.validate_called is True
    assert fake_trainer.test_called is True
    assert fake_trainer.predict_called is True
    assert results["toy_lightning"].best_model_path == "/tmp/best.ckpt"
    assert results["toy_lightning"].test_metrics["test_loss"] == 0.3
    assert lightning_trainer_module.mlflow.log_artifact.called


def test_lightning_trainer_normalizes_metrics() -> None:
    trainer = LightningTrainer(config=SimpleNamespace(), logger=MagicMock())

    metrics = trainer._normalize_metrics([{"val_loss": 1.5, "junk": object()}])

    assert metrics == {"val_loss": 1.5}