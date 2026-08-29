"""The Lightning ensemble path: member runs, aggregation, calibration on val, and the off-switch."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import yg_eo_soilnet.trainers.lightning_trainer as lightning_trainer_module
from yg_eo_soilnet.models.config_fatories.lightning_config_factory import LightningModelBundle
from yg_eo_soilnet.trainers.lightning_trainer import LightningTrainer

N_TEST = 12
N_VAL = 40


@dataclass
class FakeCheckpointCallback:
    best_model_path: str = "/tmp/best.ckpt"


class FakeDataModule:
    """The frames and loaders the ensemble path reads, and nothing else."""

    def __init__(self, target_names=("target_a",), seed: int = 0):
        rng = np.random.default_rng(seed)
        self.target_names = list(target_names)
        self.X_test_frame_ = pd.DataFrame({"f": rng.normal(size=N_TEST)})
        self.y_test_frame_ = pd.DataFrame(
            {name: rng.normal(size=N_TEST) for name in self.target_names}
        )
        self.X_val_frame_ = pd.DataFrame({"f": rng.normal(size=N_VAL)})
        self.y_val_frame_ = pd.DataFrame(
            {name: rng.normal(size=N_VAL) for name in self.target_names}
        )

    def setup(self, stage=None):
        return None

    def val_dataloader(self):
        return "val-loader"


class FakeTrainer:
    """Returns a different constant per member, so the ensemble spread is predictable."""

    def __init__(self, offset: float, n_targets: int = 1):
        self.offset = offset
        self.n_targets = n_targets
        self.checkpoint_callback = FakeCheckpointCallback()
        self.callbacks = [self.checkpoint_callback]
        self.predicted_splits = []

    def fit(self, model, datamodule=None):
        return None

    def validate(self, model, datamodule=None, verbose=False, ckpt_path=None):
        return [{"val_loss": 0.4 + self.offset}]

    def test(self, model, datamodule=None, verbose=False, ckpt_path=None):
        return [{"test_loss": 0.3 + self.offset}]

    def predict(self, model, datamodule=None, dataloaders=None, ckpt_path=None):
        if dataloaders is not None:
            self.predicted_splits.append("val")
            rows = N_VAL
        else:
            self.predicted_splits.append("test")
            rows = N_TEST
        return [np.full((rows, self.n_targets), self.offset, dtype=np.float32)]


def _config(**overrides) -> SimpleNamespace:
    base = dict(
        RANDOM_SEED=42,
        UNCERTAINTY_ENABLED=True,
        UNCERTAINTY_N_MEMBERS=3,
        UNCERTAINTY_SEED_STRIDE=1000,
        UNCERTAINTY_ALPHA=0.05,
        UNCERTAINTY_INTERVAL_METHOD="conformal",
        UNCERTAINTY_INTERVAL_K=1.0,
        # Config keeps both names in step; the stub does too, so a test that sets one does not
        # silently exercise the other's default.
        UNCERTAINTY_CALIBRATION_METHOD="split_conformal",
        UNCERTAINTY_CALIBRATION_SOURCE="val",
        UNCERTAINTY_MODELS=[],
        UNCERTAINTY_SKIP_MODELS=[],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _bundle(datamodule) -> LightningModelBundle:
    return LightningModelBundle(
        name="soil_tabular",
        target="target_a",
        model=SimpleNamespace(),
        datamodule=datamodule,
        trainer_kwargs={"max_epochs": 1},
        callback_specs={},
        registry_entry={"modeltype": "dl", "init_args": {}, "datamodule_init_args": {}},
    )


def _run(monkeypatch, config, target_names=("target_a",)):
    """Train an ensemble with fake Lightning trainers; return (logged kwargs, run names)."""
    datamodule = FakeDataModule(target_names)
    bundle = _bundle(datamodule)

    built = []

    def bundle_builder(seed):
        built.append(seed)
        return {"soil_tabular": _bundle(datamodule)}

    fake_trainers = [
        FakeTrainer(offset=float(index + 1), n_targets=len(target_names)) for index in range(10)
    ]
    handed_out = []

    # _log_checkpoint is part of the real ChildRunLogger's interface and _fit_member calls it for
    # every member, so the double carries it too rather than being guarded against in production.
    fake_mlflow_logger = SimpleNamespace(
        log_lightning_child_run=MagicMock(), _log_checkpoint=MagicMock()
    )
    trainer = LightningTrainer(
        config=config, logger=MagicMock(), mlflow_logger=fake_mlflow_logger
    )

    def build_trainer(_bundle):
        fake = fake_trainers[len(handed_out)]
        handed_out.append(fake)
        return fake

    monkeypatch.setattr(trainer, "_build_trainer", build_trainer)

    run_names = []

    def fake_start_child_run(run_name, tags=None):
        run_names.append((run_name, dict(tags or {})))
        from contextlib import nullcontext

        return nullcontext()

    monkeypatch.setattr(lightning_trainer_module, "start_child_run", fake_start_child_run)
    monkeypatch.setattr(lightning_trainer_module.mlflow, "log_params", MagicMock())

    trainer.train(
        target="__".join(target_names),
        data={},
        model_bundles={"soil_tabular": bundle},
        bundle_builder=bundle_builder,
    )

    return (fake_mlflow_logger.log_lightning_child_run.call_args, run_names, built,
            handed_out, fake_mlflow_logger)


# --- the run tree ----------------------------------------------------------


def test_each_member_gets_its_own_child_run_tagged_as_a_member(monkeypatch):
    _call, run_names, _built, _trainers, _lg = _run(monkeypatch, _config())

    members = [(name, tags) for name, tags in run_names if tags.get("run_kind") == "ensemble_member"]
    assert len(members) == 3
    assert [name for name, _ in members] == [
        "target_a_soil_tabular_member0",
        "target_a_soil_tabular_member1",
        "target_a_soil_tabular_member2",
    ]
    # And one model run wrapping them.
    assert ("target_a_soil_tabular", {}) in run_names


def test_the_factory_is_asked_for_a_fresh_bundle_at_each_strided_seed(monkeypatch):
    _call, _run_names, built, _trainers, _lg = _run(monkeypatch, _config())
    # Rebuilt per member, because weights are constructed by the factory at seeding time.
    assert built == [42, 1042, 2042]


def test_uncertainty_disabled_takes_the_single_fit_path(monkeypatch):
    _call, run_names, built, _trainers, _lg = _run(
        monkeypatch, _config(UNCERTAINTY_ENABLED=False)
    )
    assert built == []
    assert not any(tags.get("run_kind") == "ensemble_member" for _name, tags in run_names)


def test_a_skipped_entry_takes_the_single_fit_path(monkeypatch):
    _call, run_names, built, _trainers, _lg = _run(
        monkeypatch, _config(UNCERTAINTY_SKIP_MODELS=["soil_tabular"])
    )
    assert built == []
    assert not any(tags.get("run_kind") == "ensemble_member" for _name, tags in run_names)


# --- aggregation -----------------------------------------------------------


def test_the_logged_frame_carries_the_ensemble_mean_not_one_members_predictions(monkeypatch):
    call, _run_names, _built, _trainers, _lg = _run(monkeypatch, _config())
    frame = call.kwargs["evaluation_df"]

    # Members predict the constants 1, 2 and 3, so the ensemble mean is 2 and the population
    # standard deviation is sqrt(2/3).
    assert np.allclose(frame["prediction"], 2.0)
    assert np.allclose(frame["prediction_epistemic_std"], np.sqrt(2.0 / 3.0))


def test_a_point_predicting_ensemble_reports_no_aleatoric_component(monkeypatch):
    call, _run_names, _built, _trainers, _lg = _run(monkeypatch, _config())
    frame = call.kwargs["evaluation_df"]
    assert np.allclose(frame["prediction_aleatoric_std"], 0.0)
    assert np.allclose(frame["prediction_std"], frame["prediction_epistemic_std"])


def test_member_metrics_are_averaged_so_they_describe_the_ensemble(monkeypatch):
    call, _run_names, _built, _trainers, _lg = _run(monkeypatch, _config())
    # Offsets 1, 2, 3 on a base of 0.3 -> mean 2.3.
    assert call.kwargs["test_metrics"]["test_loss"] == pytest.approx(2.3)
    assert call.kwargs["validation_metrics"]["val_loss"] == pytest.approx(2.4)


# --- calibration -----------------------------------------------------------


def test_calibration_uses_the_val_split_and_reaches_the_logger(monkeypatch):
    call, _run_names, _built, trainers, _lg = _run(monkeypatch, _config())

    calibrators = call.kwargs["calibrators"]
    assert set(calibrators) == {"target_a"}
    assert calibrators["target_a"].n_calib == N_VAL

    # Both splits are predicted for every member: test for the eval frame, val to calibrate.
    for fake in trainers[:3]:
        assert fake.predicted_splits.count("val") == 1
        assert fake.predicted_splits.count("test") >= 1


def test_the_interval_columns_are_written_from_the_calibrator(monkeypatch):
    call, _run_names, _built, _trainers, _lg = _run(monkeypatch, _config())
    frame = call.kwargs["evaluation_df"]

    assert "prediction_lower" in frame.columns
    assert "prediction_upper" in frame.columns
    assert (frame["prediction_lower"] < frame["prediction"]).all()
    assert (frame["prediction_upper"] > frame["prediction"]).all()


def test_calibration_off_still_reports_sigma_but_writes_no_interval(monkeypatch):
    call, _run_names, _built, _trainers, _lg = _run(
        monkeypatch,
        _config(UNCERTAINTY_INTERVAL_METHOD="none", UNCERTAINTY_CALIBRATION_METHOD="none"),
    )
    frame = call.kwargs["evaluation_df"]

    assert call.kwargs["calibrators"] == {}
    assert "prediction_std" in frame.columns
    assert "prediction_lower" not in frame.columns


def test_a_joint_group_suffixes_every_uncertainty_column(monkeypatch):
    call, _run_names, _built, _trainers, _lg = _run(
        monkeypatch, _config(), target_names=("target_a", "target_b")
    )
    frame = call.kwargs["evaluation_df"]

    for target_name in ("target_a", "target_b"):
        assert f"prediction_{target_name}" in frame.columns
        assert f"prediction_std_{target_name}" in frame.columns
        assert f"prediction_lower_{target_name}" in frame.columns
    assert "prediction_std" not in frame.columns


def test_a_missing_validation_split_skips_calibration_rather_than_pairing_wrong_rows(monkeypatch):
    """A calibrator fitted on misaligned rows would be silently, confidently wrong."""
    datamodule = FakeDataModule()
    datamodule.y_val_frame_ = pd.DataFrame({"target_a": []})

    trainer = LightningTrainer(
        config=_config(), logger=MagicMock(), mlflow_logger=SimpleNamespace()
    )
    members = [
        {"calibration_predictions": np.zeros((N_VAL, 1))},
    ]
    assert trainer._calibrate_ensemble(members, datamodule, ["target_a"]) == {}
    assert trainer.logger.warning.called


def test_every_member_logs_its_own_checkpoint(monkeypatch):
    """Without this the run keeps one checkpoint for the whole ensemble - the reference member's -
    and the other n-1 exist only as local files that nothing records, which is what made an
    ensemble unrecoverable from MLflow alone."""
    _call, _run_names, _built, _trainers, logger = _run(monkeypatch, _config())

    assert logger._log_checkpoint.call_count == 3          # one per member
    assert all(call.args[0] == "/tmp/best.ckpt" for call in logger._log_checkpoint.call_args_list)
