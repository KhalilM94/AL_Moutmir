from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import main as main_module


def test_parse_args_supports_cli_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["main.py", "--config-path", "custom.yml", "--registry-path", "registry.yml"])

    args = main_module.parse_args()

    assert args.config_path == "custom.yml"
    assert args.registry_path == "registry.yml"


def test_main_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_trainer = SimpleNamespace(
        config=SimpleNamespace(config_path="config.yml", registry_path="registry.yml"),
        data_manager=SimpleNamespace(
            load_data=MagicMock(return_value="raw"),
            preprocess_data=MagicMock(return_value="processed"),
            split_data=MagicMock(return_value={"X_train": "x"}),
        ),
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

    monkeypatch.setattr("sys.argv", ["main.py", "--config-path", "config.yml", "--registry-path", "registry.yml"])

    main_module.main()

    fake_trainer.data_manager.load_data.assert_called_once()
    fake_trainer.data_manager.preprocess_data.assert_called_once_with("raw")
    fake_trainer.data_manager.split_data.assert_called_once_with("processed")
    fake_trainer.train_models.assert_called_once()
    fake_parent_logger.log_parent_summary.assert_called_once_with("run-123", fake_trainer)


def test_main_logs_and_reraises_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    failing_trainer = SimpleNamespace(
        config=SimpleNamespace(config_path="config.yml", registry_path="registry.yml"),
        data_manager=SimpleNamespace(
            load_data=MagicMock(side_effect=RuntimeError("load failed")),
            preprocess_data=MagicMock(),
            split_data=MagicMock(),
        ),
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