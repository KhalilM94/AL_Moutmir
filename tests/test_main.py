from types import SimpleNamespace
from unittest.mock import MagicMock
from pathlib import Path

import pandas as pd
import pytest

import main as main_module


def test_parse_args_supports_cli_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["main.py", "--config-path", "custom.yml"])

    args = main_module.parse_args()

    assert args.config_path == "custom.yml"


def _fake_plan():
    """A stand-in SplitPlan: main() only reads describe() and counts() off it."""
    return SimpleNamespace(
        describe=MagicMock(return_value={"split_strategy": "random"}),
        counts=MagicMock(return_value={"train": 2, "val": 1, "test": 1}),
    )


def test_main_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_trainer = SimpleNamespace(
        config=SimpleNamespace(config_path="config.yml", registry_path="registry.yml"),
        scikit_datamodule=SimpleNamespace(
            load_frame=MagicMock(return_value="raw"),
            preprocess=MagicMock(return_value="processed"),
            split=MagicMock(return_value={"X_train": "x"}),
        ),
        # main() decides the split once, before either family sees the data, and logs its
        # provenance - so a stub trainer has to offer the provider.
        split_plan_provider=SimpleNamespace(plan=MagicMock(return_value=_fake_plan())),
        logger=SimpleNamespace(info=MagicMock(), error=MagicMock()),
        logger_wrapper=SimpleNamespace(log_file=None),
        train_models=MagicMock(),
    )
    fake_parent_logger = SimpleNamespace(log_parent_summary=MagicMock())

    monkeypatch.setattr(main_module.mlflow, "enable_system_metrics_logging", MagicMock())
    monkeypatch.setattr(main_module.mlflow, "set_experiment", MagicMock())
    monkeypatch.setattr(main_module.mlflow, "create_experiment", MagicMock())
    monkeypatch.setattr(main_module.mlflow, "active_run", MagicMock(return_value=None))
    monkeypatch.setattr(main_module.mlflow, "end_run", MagicMock())
    monkeypatch.setattr(main_module.mlflow, "log_param", MagicMock())
    # log_params too, not just log_param: an unstubbed one auto-starts a REAL run against the
    # default tracking store and never ends it, which surfaces three test files later as
    # "Run with UUID ... is already active".
    monkeypatch.setattr(main_module.mlflow, "log_params", MagicMock())
    monkeypatch.setattr(main_module.mlflow, "log_artifact", MagicMock())
    monkeypatch.setattr(main_module.datetime, "datetime", SimpleNamespace(now=lambda: SimpleNamespace(strftime=lambda fmt: "20260703_120000")))
    monkeypatch.setattr(main_module, "SoilModelTraining", MagicMock(return_value=fake_trainer))
    monkeypatch.setattr(main_module, "ParentRunLogger", MagicMock(return_value=fake_parent_logger))

    class FakeRun:
        info = SimpleNamespace(run_id="run-123")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(main_module.mlflow, "start_run", MagicMock(return_value=FakeRun()))

    monkeypatch.setattr("sys.argv", ["main.py", "--config-path", "config.yml"])

    main_module.main()

    fake_trainer.scikit_datamodule.load_frame.assert_called_once()
    fake_trainer.scikit_datamodule.preprocess.assert_called_once_with("raw")
    # The shared plan is handed to the sklearn family rather than each family splitting for itself.
    fake_trainer.scikit_datamodule.split.assert_called_once_with(
        "processed", fake_trainer.split_plan_provider.plan.return_value
    )
    fake_trainer.train_models.assert_called_once()
    fake_parent_logger.log_parent_summary.assert_called_once_with("run-123", fake_trainer)


def test_main_logs_and_reraises_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    failing_trainer = SimpleNamespace(
        config=SimpleNamespace(config_path="config.yml", registry_path="registry.yml"),
        scikit_datamodule=SimpleNamespace(
            load_frame=MagicMock(side_effect=RuntimeError("load failed")),
            preprocess=MagicMock(),
            split=MagicMock(),
        ),
        # main() decides the split once, before either family sees the data, and logs its
        # provenance - so a stub trainer has to offer the provider.
        split_plan_provider=SimpleNamespace(plan=MagicMock(return_value=_fake_plan())),
        logger=SimpleNamespace(info=MagicMock(), error=MagicMock()),
        logger_wrapper=SimpleNamespace(log_file=None),
        train_models=MagicMock(),
    )

    monkeypatch.setattr(main_module.mlflow, "enable_system_metrics_logging", MagicMock())
    monkeypatch.setattr(main_module.mlflow, "set_experiment", MagicMock())
    monkeypatch.setattr(main_module.mlflow, "create_experiment", MagicMock())
    monkeypatch.setattr(main_module.mlflow, "active_run", MagicMock(return_value=None))
    monkeypatch.setattr(main_module.mlflow, "end_run", MagicMock())
    monkeypatch.setattr(main_module.mlflow, "log_param", MagicMock())
    # log_params too, not just log_param: an unstubbed one auto-starts a REAL run against the
    # default tracking store and never ends it, which surfaces three test files later as
    # "Run with UUID ... is already active".
    monkeypatch.setattr(main_module.mlflow, "log_params", MagicMock())
    monkeypatch.setattr(main_module.mlflow, "log_artifact", MagicMock())
    monkeypatch.setattr(main_module.datetime, "datetime", SimpleNamespace(now=lambda: SimpleNamespace(strftime=lambda fmt: "20260703_120000")))
    monkeypatch.setattr(main_module, "SoilModelTraining", MagicMock(return_value=failing_trainer))
    monkeypatch.setattr(main_module, "ParentRunLogger", MagicMock(return_value=SimpleNamespace(log_parent_summary=MagicMock())))

    class FakeRun:
        info = SimpleNamespace(run_id="run-123")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(main_module.mlflow, "start_run", MagicMock(return_value=FakeRun()))
    monkeypatch.setattr("sys.argv", ["main.py"])

    with pytest.raises(RuntimeError, match="load failed"):
        main_module.main()

    failing_trainer.logger.error.assert_called_once()


def test_main_skips_artifact_upload_when_logger_file_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_trainer = SimpleNamespace(
        config=SimpleNamespace(config_path="config.yml", registry_path="registry.yml"),
        scikit_datamodule=SimpleNamespace(
            load_frame=MagicMock(return_value="raw"),
            preprocess=MagicMock(return_value="processed"),
            split=MagicMock(return_value={"X_train": "x"}),
        ),
        # main() decides the split once, before either family sees the data, and logs its
        # provenance - so a stub trainer has to offer the provider.
        split_plan_provider=SimpleNamespace(plan=MagicMock(return_value=_fake_plan())),
        logger=SimpleNamespace(info=MagicMock(), error=MagicMock()),
        logger_wrapper=SimpleNamespace(log_file=None),
        train_models=MagicMock(),
    )
    fake_parent_logger = SimpleNamespace(log_parent_summary=MagicMock())

    monkeypatch.setattr(main_module.mlflow, "enable_system_metrics_logging", MagicMock())
    monkeypatch.setattr(main_module.mlflow, "set_experiment", MagicMock())
    monkeypatch.setattr(main_module.mlflow, "create_experiment", MagicMock())
    monkeypatch.setattr(main_module.mlflow, "active_run", MagicMock(return_value=None))
    monkeypatch.setattr(main_module.mlflow, "end_run", MagicMock())
    monkeypatch.setattr(main_module.mlflow, "log_param", MagicMock())
    # log_params too, not just log_param: an unstubbed one auto-starts a REAL run against the
    # default tracking store and never ends it, which surfaces three test files later as
    # "Run with UUID ... is already active".
    monkeypatch.setattr(main_module.mlflow, "log_params", MagicMock())
    log_artifact = MagicMock()
    monkeypatch.setattr(main_module.mlflow, "log_artifact", log_artifact)
    monkeypatch.setattr(main_module.datetime, "datetime", SimpleNamespace(now=lambda: SimpleNamespace(strftime=lambda fmt: "20260703_120000")))
    monkeypatch.setattr(main_module, "SoilModelTraining", MagicMock(return_value=fake_trainer))
    monkeypatch.setattr(main_module, "ParentRunLogger", MagicMock(return_value=fake_parent_logger))

    class FakeRun:
        info = SimpleNamespace(run_id="run-123")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(main_module.mlflow, "start_run", MagicMock(return_value=FakeRun()))
    monkeypatch.setattr("sys.argv", ["main.py", "--config-path", "config.yml"])

    main_module.main()

    log_artifact.assert_not_called()


def test_main_exports_mlflow_experiment_when_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    fake_trainer = SimpleNamespace(
        config=SimpleNamespace(
            config_path="config.yml",
            registry_path="registry.yml",
            MLFLOW_EXPERIMENT_EXPORT_ENABLED=True,
            MLFLOW_EXPERIMENT_EXPORT_PATH=str(tmp_path / "exports"),
        ),
        scikit_datamodule=SimpleNamespace(
            load_frame=MagicMock(return_value="raw"),
            preprocess=MagicMock(return_value="processed"),
            split=MagicMock(return_value={"X_train": "x"}),
        ),
        # main() decides the split once, before either family sees the data, and logs its
        # provenance - so a stub trainer has to offer the provider.
        split_plan_provider=SimpleNamespace(plan=MagicMock(return_value=_fake_plan())),
        logger=SimpleNamespace(info=MagicMock(), error=MagicMock()),
        logger_wrapper=SimpleNamespace(log_file=None),
        train_models=MagicMock(),
    )
    fake_parent_logger = SimpleNamespace(log_parent_summary=MagicMock())
    experiment_id = "12345"
    run_id = "67890"
    experiment_name = "Soil Model Training Experiment"
    run_name = "Run_20260703_120000"
    tracking_root = tmp_path / "mlruns"
    run_dir = tracking_root / experiment_id / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "meta.yaml").write_text(f"name: {run_name}\n")
    (run_dir / "dummy.txt").write_text("content\n")

    monkeypatch.setattr(main_module.mlflow, "enable_system_metrics_logging", MagicMock())
    monkeypatch.setattr(main_module.mlflow, "set_experiment", MagicMock())
    monkeypatch.setattr(main_module.mlflow, "create_experiment", MagicMock())
    monkeypatch.setattr(main_module.mlflow, "active_run", MagicMock(return_value=None))
    monkeypatch.setattr(main_module.mlflow, "end_run", MagicMock())
    monkeypatch.setattr(main_module.mlflow, "log_param", MagicMock())
    # log_params too, not just log_param: an unstubbed one auto-starts a REAL run against the
    # default tracking store and never ends it, which surfaces three test files later as
    # "Run with UUID ... is already active".
    monkeypatch.setattr(main_module.mlflow, "log_params", MagicMock())
    monkeypatch.setattr(main_module.mlflow, "log_artifact", MagicMock())
    monkeypatch.setattr(main_module.mlflow, "get_tracking_uri", MagicMock(return_value=f"file://{tracking_root}"))
    monkeypatch.setattr(main_module.mlflow, "get_experiment", MagicMock(return_value=SimpleNamespace(name=experiment_name)))
    monkeypatch.setattr(main_module.datetime, "datetime", SimpleNamespace(now=lambda: SimpleNamespace(strftime=lambda fmt: "20260703_120000")))
    monkeypatch.setattr(main_module, "SoilModelTraining", MagicMock(return_value=fake_trainer))
    monkeypatch.setattr(main_module, "ParentRunLogger", MagicMock(return_value=fake_parent_logger))

    copied = {}

    def fake_copytree(src, dst):
        copied["src"] = src
        copied["dst"] = dst
        return dst

    monkeypatch.setattr(main_module.shutil, "copytree", fake_copytree)
    monkeypatch.setattr(main_module.shutil, "rmtree", MagicMock())

    class FakeRun:
        info = SimpleNamespace(run_id=run_id, experiment_id=experiment_id)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(main_module.mlflow, "start_run", MagicMock(return_value=FakeRun()))
    monkeypatch.setattr("sys.argv", ["main.py", "--config-path", "config.yml"])

    main_module.main()

    assert copied["src"] == run_dir
    assert copied["dst"] == tmp_path / "exports" / "Soil_Model_Training_Experiment" / run_name


def test_train_models_dispatches_sklearn_and_lightning(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_sklearn_trainer = SimpleNamespace(train=MagicMock())
    fake_lightning_trainer = SimpleNamespace(train=MagicMock())

    trainer = main_module.SoilModelTraining.__new__(main_module.SoilModelTraining)
    trainer.config = SimpleNamespace(
        TARGET_COLUMNS=["target_a"],
        COLUMNS_TO_TRANSFORM=[],
        ENABLE_CLUSTERING=False,
        SPLIT_STRATEGY="kfold",
        RANDOM_SEED=42,
    )
    trainer.logger = MagicMock()
    trainer.sklearn_logger = MagicMock()
    trainer.model_configs = SimpleNamespace(
        build_model_configs=MagicMock(return_value={"sklearn_model": {"model": object(), "params": {}, "modeltype": "ml"}})
    )
    trainer.lightning_model_configs = SimpleNamespace(
        build_lightning_configs=MagicMock(return_value={"lightning_model": object()}),
        covers_all_targets_in_one_run=MagicMock(return_value=False),
    )
    trainer.lightning_trainer = fake_lightning_trainer

    monkeypatch.setattr(main_module, "ModelTrainer", MagicMock(return_value=fake_sklearn_trainer))

    data = {
        "X_train": pd.DataFrame({"feature": [1.0, 2.0]}),
        "y_train": pd.DataFrame({"target_a": [3.0, 4.0]}),
        "X_test": pd.DataFrame({"feature": [5.0]}),
        "y_test": pd.DataFrame({"target_a": [6.0]}),
    }

    trainer.train_models(data)

    fake_sklearn_trainer.train.assert_called_once()
    fake_lightning_trainer.train.assert_called_once()

def _multi_target_trainer(monkeypatch: pytest.MonkeyPatch, *, covers_all_targets: bool):
    fake_sklearn_trainer = SimpleNamespace(train=MagicMock())
    fake_lightning_trainer = SimpleNamespace(train=MagicMock())

    trainer = main_module.SoilModelTraining.__new__(main_module.SoilModelTraining)
    trainer.config = SimpleNamespace(
        TARGET_COLUMNS=["target_a", "target_b"],
        COLUMNS_TO_TRANSFORM=[],
        ENABLE_CLUSTERING=False,
        SPLIT_STRATEGY="kfold",
        RANDOM_SEED=42,
    )
    trainer.logger = MagicMock()
    trainer.sklearn_logger = MagicMock()
    trainer.model_configs = SimpleNamespace(
        build_model_configs=MagicMock(return_value={"sklearn_model": {"model": object(), "params": {}, "modeltype": "ml"}})
    )
    trainer.lightning_model_configs = SimpleNamespace(
        build_lightning_configs=MagicMock(return_value={"lightning_model": object()}),
        covers_all_targets_in_one_run=MagicMock(return_value=covers_all_targets),
    )
    trainer.lightning_trainer = fake_lightning_trainer

    monkeypatch.setattr(main_module, "ModelTrainer", MagicMock(return_value=fake_sklearn_trainer))

    data = {
        "X_train": pd.DataFrame({"feature": [1.0, 2.0]}),
        "y_train": pd.DataFrame({"target_a": [3.0, 4.0], "target_b": [5.0, 6.0]}),
        "X_test": pd.DataFrame({"feature": [5.0]}),
        "y_test": pd.DataFrame({"target_a": [6.0], "target_b": [7.0]}),
    }
    trainer.train_models(data)
    return trainer, fake_sklearn_trainer, fake_lightning_trainer


def test_multi_target_graph_runs_lightning_once_on_a_combined_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """A graph spans every point at once, so it cannot be trained per target."""
    trainer, sklearn_trainer, lightning_trainer = _multi_target_trainer(monkeypatch, covers_all_targets=True)

    trainer.lightning_model_configs.covers_all_targets_in_one_run.assert_called_once()
    assert sklearn_trainer.train.call_count == 2  # sklearn stays per-target
    lightning_trainer.train.assert_called_once()
    assert lightning_trainer.train.call_args.kwargs["target"] == "target_a__target_b"


def test_multi_target_without_graph_runs_lightning_per_target(monkeypatch: pytest.MonkeyPatch) -> None:
    _, sklearn_trainer, lightning_trainer = _multi_target_trainer(monkeypatch, covers_all_targets=False)

    assert sklearn_trainer.train.call_count == 2
    assert lightning_trainer.train.call_count == 2
    assert [call.kwargs["target"] for call in lightning_trainer.train.call_args_list] == ["target_a", "target_b"]
