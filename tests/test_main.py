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
        MULTI_TARGET_MODE="joint",
        COLUMNS_TO_TRANSFORM=[],
        ENABLE_CLUSTERING=False,
        SPLIT_STRATEGY="kfold",
        RANDOM_SEED=42,
        MODEL_REGISTRY={"sklearn_model": {"enabled": True}},
        LIGHTNING_MODEL_REGISTRY={"lightning_model": {"enabled": True}},
    )
    trainer.logger = MagicMock()
    trainer.sklearn_logger = MagicMock()
    trainer.model_configs = SimpleNamespace(
        build_model_configs=MagicMock(return_value={"sklearn_model": {"model": object(), "params": {}, "modeltype": "ml"}})
    )
    trainer.lightning_model_configs = SimpleNamespace(
        build_lightning_configs=MagicMock(return_value={"lightning_model": object()}),
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

def _multi_target_trainer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str,
    sklearn_spec: dict | None = None,
    lightning_spec: dict | None = None,
):
    fake_sklearn_trainer = SimpleNamespace(train=MagicMock())
    fake_lightning_trainer = SimpleNamespace(train=MagicMock())

    trainer = main_module.SoilModelTraining.__new__(main_module.SoilModelTraining)
    trainer.config = SimpleNamespace(
        TARGET_COLUMNS=["target_a", "target_b"],
        MULTI_TARGET_MODE=mode,
        COLUMNS_TO_TRANSFORM=[],
        ENABLE_CLUSTERING=False,
        SPLIT_STRATEGY="kfold",
        RANDOM_SEED=42,
        MODEL_REGISTRY={"sklearn_model": {"enabled": True, **(sklearn_spec or {})}},
        LIGHTNING_MODEL_REGISTRY={"lightning_model": {"enabled": True, **(lightning_spec or {})}},
    )
    trainer.logger = MagicMock()
    trainer.sklearn_logger = MagicMock()
    trainer.model_configs = SimpleNamespace(
        build_model_configs=MagicMock(return_value={"sklearn_model": {"model": object(), "params": {}, "modeltype": "ml"}})
    )
    trainer.lightning_model_configs = SimpleNamespace(
        build_lightning_configs=MagicMock(return_value={"lightning_model": object()}),
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


def test_joint_mode_fits_one_lightning_model_over_every_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """MULTI_TARGET_MODE: joint gives Lightning one run with a target_dim-wide head."""
    _, _, lightning_trainer = _multi_target_trainer(monkeypatch, mode="joint")

    lightning_trainer.train.assert_called_once()
    assert lightning_trainer.train.call_args.kwargs["target"] == "target_a__target_b"


def test_per_target_mode_fits_one_lightning_model_each(monkeypatch: pytest.MonkeyPatch) -> None:
    """The switch the family never had: Lightning was joint whatever the config said."""
    _, _, lightning_trainer = _multi_target_trainer(monkeypatch, mode="per_target")

    assert lightning_trainer.train.call_count == 2
    assert [call.kwargs["target"] for call in lightning_trainer.train.call_args_list] == ["target_a", "target_b"]


def test_sklearn_joins_targets_only_when_the_estimator_declares_it_can(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Most scikit estimators cannot fit a 2-D y, so an undeclared entry stays per-target."""
    _, undeclared, _ = _multi_target_trainer(monkeypatch, mode="joint")
    assert undeclared.train.call_count == 2
    assert [call.kwargs["target"] for call in undeclared.train.call_args_list] == ["target_a", "target_b"]

    _, native, _ = _multi_target_trainer(
        monkeypatch, mode="joint", sklearn_spec={"multi_target": "native"}
    )
    native.train.assert_called_once()
    assert native.train.call_args.kwargs["target"] == "target_a__target_b"
    assert native.train.call_args.kwargs["targets"] == ["target_a", "target_b"]


def test_a_registry_entry_overrides_the_global_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, lightning_trainer = _multi_target_trainer(
        monkeypatch, mode="joint", lightning_spec={"multi_target": "per_target"}
    )

    assert lightning_trainer.train.call_count == 2
    assert [call.kwargs["target"] for call in lightning_trainer.train.call_args_list] == ["target_a", "target_b"]


def test_sklearn_entries_that_agree_on_a_grouping_are_trained_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FAIL_IF_ALL_MODELS_FAIL_FOR_TARGET is a check ACROSS the models tried for a target.

    Splitting the entries into one call each would turn "every model failed" into "any model
    failed", so entries resolving to the same groups share a call.
    """
    fake_sklearn_trainer = SimpleNamespace(train=MagicMock())

    trainer = main_module.SoilModelTraining.__new__(main_module.SoilModelTraining)
    trainer.config = SimpleNamespace(
        TARGET_COLUMNS=["target_a", "target_b"],
        MULTI_TARGET_MODE="joint",
        COLUMNS_TO_TRANSFORM=[],
        ENABLE_CLUSTERING=False,
        SPLIT_STRATEGY="kfold",
        RANDOM_SEED=42,
        MODEL_REGISTRY={
            "Ridge": {"enabled": True, "multi_target": "native"},
            "PLSRegression": {"enabled": True, "multi_target": "native"},
            # Cannot take a 2-D y, so it falls back to one model per target on its own.
            "GradientBoosting": {"enabled": True},
        },
        LIGHTNING_MODEL_REGISTRY={},
    )
    trainer.logger = MagicMock()
    trainer.sklearn_logger = MagicMock()
    trainer.model_configs = SimpleNamespace(
        build_model_configs=MagicMock(
            return_value={name: {"model": object(), "params": {}, "modeltype": "ml"} for name in
                          ("Ridge", "PLSRegression", "GradientBoosting")}
        )
    )
    trainer.lightning_model_configs = SimpleNamespace(build_lightning_configs=MagicMock(return_value={}))
    trainer.lightning_trainer = SimpleNamespace(train=MagicMock())
    monkeypatch.setattr(main_module, "ModelTrainer", MagicMock(return_value=fake_sklearn_trainer))

    trainer.train_models(
        {
            "X_train": pd.DataFrame({"feature": [1.0, 2.0]}),
            "y_train": pd.DataFrame({"target_a": [3.0, 4.0], "target_b": [5.0, 6.0]}),
            "X_test": pd.DataFrame({"feature": [5.0]}),
            "y_test": pd.DataFrame({"target_a": [6.0], "target_b": [7.0]}),
        }
    )

    calls = {call.kwargs["target"]: sorted(call.kwargs["model_pipelines"]) for call in
             fake_sklearn_trainer.train.call_args_list}
    assert calls == {
        # The two native estimators share the joint group, in ONE call.
        "target_a__target_b": ["PLSRegression", "Ridge"],
        # The one that cannot fit a 2-D y gets its own per-target calls.
        "target_a": ["GradientBoosting"],
        "target_b": ["GradientBoosting"],
    }


@pytest.mark.parametrize(
    "mode,expected_sklearn",
    [
        ("joint", "target_a__target_b"),
        ("per_target", "target_a | target_b"),
    ],
)
def test_train_models_records_what_it_decided_to_fit(
    monkeypatch: pytest.MonkeyPatch, mode, expected_sklearn
) -> None:
    """The run logs how it split the data but used to say nothing about what it planned to fit.

    That gap is why an OOM-killed multi-target run was mistaken for a bug in the grouping: the
    mode and the resolved groups were only recoverable by reading child run names afterwards.
    """
    logged: dict = {}
    monkeypatch.setattr(main_module.mlflow, "log_params", lambda params: logged.update(params))

    trainer = main_module.SoilModelTraining.__new__(main_module.SoilModelTraining)
    trainer.config = SimpleNamespace(
        TARGET_COLUMNS=["target_a", "target_b"],
        MULTI_TARGET_MODE=mode,
        COLUMNS_TO_TRANSFORM=[],
        ENABLE_CLUSTERING=False,
        SPLIT_STRATEGY="kfold",
        RANDOM_SEED=42,
        MODEL_REGISTRY={"Ridge": {"enabled": True, "multi_target": "native"}},
        LIGHTNING_MODEL_REGISTRY={"soil_cnn": {"enabled": True}},
    )
    trainer.logger = MagicMock()
    trainer.sklearn_logger = MagicMock()
    trainer.model_configs = SimpleNamespace(
        build_model_configs=MagicMock(
            return_value={"Ridge": {"model": object(), "params": {}, "modeltype": "ml"}}
        )
    )
    trainer.lightning_model_configs = SimpleNamespace(build_lightning_configs=MagicMock(return_value={}))
    trainer.lightning_trainer = SimpleNamespace(train=MagicMock())
    monkeypatch.setattr(main_module, "ModelTrainer", MagicMock(return_value=SimpleNamespace(train=MagicMock())))

    trainer.train_models(
        {
            "X_train": pd.DataFrame({"feature": [1.0, 2.0]}),
            "y_train": pd.DataFrame({"target_a": [3.0, 4.0], "target_b": [5.0, 6.0]}),
            "X_test": pd.DataFrame({"feature": [5.0]}),
            "y_test": pd.DataFrame({"target_a": [6.0], "target_b": [7.0]}),
        }
    )

    assert logged["MULTI_TARGET_MODE"] == mode
    assert logged["TARGET_COLUMNS"] == "target_a,target_b"
    assert logged["sklearn_target_groups"] == expected_sklearn
    assert logged["lightning_target_groups"] == expected_sklearn


def test_the_target_plan_shows_a_fallback_as_a_mode_group_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """joint mode beside per-target groups IS the fallback, and is otherwise invisible afterwards."""
    logged: dict = {}
    monkeypatch.setattr(main_module.mlflow, "log_params", lambda params: logged.update(params))

    trainer = main_module.SoilModelTraining.__new__(main_module.SoilModelTraining)
    trainer.config = SimpleNamespace(
        TARGET_COLUMNS=["target_a", "target_b"],
        MULTI_TARGET_MODE="joint",
        COLUMNS_TO_TRANSFORM=[],
        ENABLE_CLUSTERING=False,
        SPLIT_STRATEGY="kfold",
        RANDOM_SEED=42,
        # No multi_target declaration, so it cannot take a 2-D y.
        MODEL_REGISTRY={"TabICL": {"enabled": True}},
        LIGHTNING_MODEL_REGISTRY={},
    )
    trainer.logger = MagicMock()
    trainer.sklearn_logger = MagicMock()
    trainer.model_configs = SimpleNamespace(
        build_model_configs=MagicMock(
            return_value={"TabICL": {"model": object(), "params": {}, "modeltype": "ml"}}
        )
    )
    trainer.lightning_model_configs = SimpleNamespace(build_lightning_configs=MagicMock(return_value={}))
    trainer.lightning_trainer = SimpleNamespace(train=MagicMock())
    monkeypatch.setattr(main_module, "ModelTrainer", MagicMock(return_value=SimpleNamespace(train=MagicMock())))

    trainer.train_models(
        {
            "X_train": pd.DataFrame({"feature": [1.0, 2.0]}),
            "y_train": pd.DataFrame({"target_a": [3.0, 4.0], "target_b": [5.0, 6.0]}),
            "X_test": pd.DataFrame({"feature": [5.0]}),
            "y_test": pd.DataFrame({"target_a": [6.0], "target_b": [7.0]}),
        }
    )

    assert logged["MULTI_TARGET_MODE"] == "joint"
    assert logged["sklearn_target_groups"] == "target_a | target_b"
