from types import SimpleNamespace
from unittest.mock import MagicMock
from pathlib import Path

import mlflow.pyfunc
import mlflow.pytorch
import pandas as pd

import yg_eo_soilnet.logger.mlflow_loggers as mlflow_loggers_module
from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger, ParentRunLogger


def _checkpoint(tmp_path) -> Path:
    """A checkpoint file that actually exists.

    The logger copies the best checkpoint to a stable `best.ckpt` name rather than uploading the
    path it was handed, so these tests need a real file on disk.
    """
    path = tmp_path / "epoch=7-step=42.ckpt"
    path.write_bytes(b"fake-checkpoint")
    return path


def test_log_lightning_child_run_logs_metrics_artifacts_and_tags(monkeypatch, tmp_path) -> None:
    logger = ChildRunLogger()

    set_tags = MagicMock()
    log_params = MagicMock()
    log_metric = MagicMock()
    log_artifact = MagicMock()
    log_model = MagicMock()

    monkeypatch.setattr(mlflow_loggers_module.mlflow, "set_tags", set_tags)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_params", log_params)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_metric", log_metric)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_artifact", log_artifact)
    monkeypatch.setattr(mlflow.pyfunc, "log_model", log_model)
    monkeypatch.setattr(mlflow.pytorch, "save_model", MagicMock())
    monkeypatch.setattr(logger, "_log_plots", MagicMock())

    fake_model = SimpleNamespace()

    logger.log_lightning_child_run(
        config=SimpleNamespace(
            LIGHTNING_BATCH_SIZE=8,
            LIGHTNING_VAL_SIZE=0.2,
            LIGHTNING_MAX_EPOCHS=10,
            LIGHTNING_ACCELERATOR="cpu",
            LIGHTNING_DEVICES=1,
            LIGHTNING_PRECISION="32-true",
        ),
        target="target_a",
        model_name="toy_lightning",
        evaluation_df=pd.DataFrame({
            "target_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "prediction": [1.3, 1.8, 3.4, 3.6, 5.5, 5.7],
        }),
        validation_metrics={"val_loss": 0.5},
        test_metrics={"test_loss": 0.4},
        best_model_path=str(_checkpoint(tmp_path)),
        extra_params={"foo": "bar"},
        plot_functions={},
        model=fake_model,
    )

    # Two calls now: the run identity, then the checkpoint's original filename.
    assert set_tags.call_count == 2
    log_params.assert_any_call({
        "LIGHTNING_BATCH_SIZE": 8,
        "LIGHTNING_VAL_SIZE": 0.2,
        "LIGHTNING_MAX_EPOCHS": 10,
        "LIGHTNING_ACCELERATOR": "cpu",
        "LIGHTNING_DEVICES": 1,
        "LIGHTNING_PRECISION": "32-true",
    })
    log_metric.assert_any_call("val_loss", 0.5)
    log_metric.assert_any_call("test_loss", 0.4)
    # Logged under a STABLE leaf name, not Lightning's epoch=NN-step=MMM, so two runs of the
    # same model produce the same artifact path and MLflow can compare them.
    checkpoint_calls = [
        call for call in log_artifact.call_args_list
        if call.kwargs.get("artifact_path") == "checkpoints"
    ]
    assert len(checkpoint_calls) == 1
    assert Path(checkpoint_calls[0].args[0]).name == "best.ckpt"
    # The original name survives as a tag rather than in the path.
    assert any(
        c.args and c.args[0].get("checkpoint_filename") == "epoch=7-step=42.ckpt"
        for c in set_tags.call_args_list
    )
    assert any(call.kwargs.get("artifact_path") == "eval_results" for call in log_artifact.call_args_list)
    assert log_model.called


def test_log_lightning_child_run_logs_split_summary_metrics(monkeypatch, tmp_path) -> None:
    logger = ChildRunLogger()

    set_tags = MagicMock()
    log_params = MagicMock()
    log_metric = MagicMock()
    log_artifact = MagicMock()
    log_model = MagicMock()

    monkeypatch.setattr(mlflow_loggers_module.mlflow, "set_tags", set_tags)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_params", log_params)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_metric", log_metric)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_artifact", log_artifact)
    monkeypatch.setattr(mlflow.pyfunc, "log_model", log_model)
    monkeypatch.setattr(mlflow.pytorch, "save_model", MagicMock())
    monkeypatch.setattr(logger, "_log_plots", MagicMock())

    bundle = SimpleNamespace(
        datamodule=SimpleNamespace(
            y_train_frame_=pd.DataFrame({"target_a": [1.0, 2.0, 3.0]}),
            y_val_frame_=pd.DataFrame({"target_a": [4.0, 5.0]}),
            y_test_frame_=pd.DataFrame({"target_a": [6.0, 7.0]}),
        ),
        trainer_kwargs={},
        registry_entry={"modeltype": "dl"},
    )

    logger.log_lightning_child_run(
        config=SimpleNamespace(
            LIGHTNING_BATCH_SIZE=8,
            LIGHTNING_VAL_SIZE=0.2,
            LIGHTNING_MAX_EPOCHS=10,
            LIGHTNING_ACCELERATOR="cpu",
            LIGHTNING_DEVICES=1,
            LIGHTNING_PRECISION="32-true",
        ),
        target="target_a",
        model_name="toy_lightning",
        evaluation_df=pd.DataFrame({
            "target_a": [6.0, 7.0],
            "prediction": [5.5, 7.5],
        }),
        validation_metrics={"val_loss": 0.5},
        test_metrics={"test_loss": 0.4},
        best_model_path=str(_checkpoint(tmp_path)),
        extra_params={"foo": "bar"},
        plot_functions={},
        bundle=bundle,
        model=SimpleNamespace(),
    )

    assert any(call.args[0] == "train_mean" for call in log_metric.call_args_list)
    assert any(call.args[0] == "test_mean" for call in log_metric.call_args_list)
    assert any(call.args[0] == "eval_mae" for call in log_metric.call_args_list)
    assert any(call.kwargs.get("artifact_path") == "eval_results" for call in log_artifact.call_args_list)


def test_log_lightning_child_run_logs_architecture_dimensions(monkeypatch, tmp_path) -> None:
    logger = ChildRunLogger()

    set_tags = MagicMock()
    log_params = MagicMock()
    log_metric = MagicMock()
    log_artifact = MagicMock()
    log_model = MagicMock()

    monkeypatch.setattr(mlflow_loggers_module.mlflow, "set_tags", set_tags)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_params", log_params)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_metric", log_metric)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_artifact", log_artifact)
    monkeypatch.setattr(mlflow.pyfunc, "log_model", log_model)
    monkeypatch.setattr(mlflow.pytorch, "save_model", MagicMock())
    monkeypatch.setattr(logger, "_log_plots", MagicMock())

    model = SimpleNamespace(
        static_dim=12,
        target_dim=1,
        hidden_dim=64,
        temporal_hidden_dim=64,
        edge_attr_dim=4,
        learning_rate=0.0015,
        temporal_enabled=True,
        temporal_steps=8,
        temporal_lstm_hidden_dim=32,
        temporal_lstm_num_layers=2,
        temporal_lstm_dropout=0.0,
        temporal_lstm_bidirectional=False,
        temporal_pooling="last",
        spatial_graph_enabled=True,
        graph_blocks=[object(), object(), object(), object()],
        static_encoder=SimpleNamespace(in_features=12, out_features=64),
        output_head=SimpleNamespace(in_features=128, out_features=1),
        temporal_encoders={"modis": SimpleNamespace(input_size=5, hidden_size=32, num_layers=2, bidirectional=False)},
        modality_dims={"modis": 5},
    )

    bundle = SimpleNamespace(
        datamodule=SimpleNamespace(
            static_dim=12,
            target_dim=1,
            temporal_steps=8,
            edge_attr_dim=4,
            feature_dim=76,
            modality_dims={"modis": 5},
        ),
        trainer_kwargs={"max_epochs": 100, "accelerator": "cuda", "devices": 1},
        registry_entry={"modeltype": "dl"},
    )

    logger.log_lightning_child_run(
        config=SimpleNamespace(
            LIGHTNING_BATCH_SIZE=8,
            LIGHTNING_VAL_SIZE=0.2,
            LIGHTNING_MAX_EPOCHS=10,
            LIGHTNING_ACCELERATOR="cpu",
            LIGHTNING_DEVICES=1,
            LIGHTNING_PRECISION="32-true",
        ),
        target="target_a",
        model_name="toy_lightning",
        evaluation_df=pd.DataFrame({
            "target_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "prediction": [1.3, 1.8, 3.4, 3.6, 5.5, 5.7],
        }),
        validation_metrics={"val_loss": 0.5},
        test_metrics={"test_loss": 0.4},
        best_model_path=str(_checkpoint(tmp_path)),
        extra_params={"foo": "bar"},
        plot_functions={},
        bundle=bundle,
        model=model,
    )

    architecture_calls = [call for call in log_params.call_args_list if any(key.startswith("architecture.") for key in call.args[0])]
    assert architecture_calls, "Expected architecture dimensions to be logged"
    logged_architecture = architecture_calls[0].args[0]
    assert logged_architecture["architecture.num_graph_layers"] == 4
    assert logged_architecture["architecture.output_head_in_features"] == 128
    assert logged_architecture["architecture.temporal_encoder.modis.hidden_size"] == 32


def test_log_lightning_child_run_skips_pred_obs_for_multi_output(monkeypatch) -> None:
    logger = ChildRunLogger()

    log_artifact = MagicMock()
    log_model = MagicMock()

    monkeypatch.setattr(mlflow_loggers_module.mlflow, "set_tags", MagicMock())
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_params", MagicMock())
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_metric", MagicMock())
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_artifact", log_artifact)
    monkeypatch.setattr(mlflow.pyfunc, "log_model", log_model)
    monkeypatch.setattr(mlflow.pytorch, "save_model", MagicMock())

    logger.log_lightning_child_run(
        config=SimpleNamespace(),
        target="target_a",
        model_name="toy_lightning",
        evaluation_df=pd.DataFrame(
            {
                "target_a": [1.0, 2.0],
                "prediction_0": [0.9, 2.1],
                "prediction_1": [1.1, 1.9],
            }
        ),
        model=SimpleNamespace(),
    )

    eval_plot_calls = [
        call for call in log_artifact.call_args_list if call.kwargs.get("artifact_path") == "plots"
    ]
    assert eval_plot_calls == []


def test_collect_eval_dfs_uses_optional_fallback_path(monkeypatch, tmp_path) -> None:
    logger = ParentRunLogger()

    csv_path = tmp_path / "eval_results_target_a_toy_lightning.csv"
    pd.DataFrame({"target_a": [1.0], "prediction": [1.2]}).to_csv(csv_path, index=False)

    class FakeRun:
        def __init__(self, run_id: str, target: str, model_name: str):
            self.info = SimpleNamespace(run_id=run_id)
            self.data = SimpleNamespace(tags={"target": target, "model_name": model_name})

    class FakeRunInfo:
        def __init__(self, experiment_id: str):
            self.info = SimpleNamespace(experiment_id=experiment_id)

    class FakeMlflowClient:
        def get_run(self, parent_run_id):
            return FakeRunInfo("experiment-1")

        def search_runs(self, experiment_ids, filter_string):
            return [FakeRun("child-run-1", "target_a", "toy_lightning")]

    def fake_download_artifacts(*, run_id, artifact_path):
        if artifact_path == "eval_results/eval_results_target_a_toy_lightning.csv":
            raise FileNotFoundError("missing nested path")
        if artifact_path == "eval_results_target_a_toy_lightning.csv":
            return str(csv_path)
        raise AssertionError(f"Unexpected artifact path: {artifact_path}")

    monkeypatch.setattr(mlflow_loggers_module.mlflow.tracking, "MlflowClient", FakeMlflowClient)
    monkeypatch.setattr(mlflow_loggers_module.mlflow.artifacts, "download_artifacts", fake_download_artifacts)

    eval_dfs = logger._collect_eval_dfs("parent-run-1")

    assert len(eval_dfs) == 1
    assert eval_dfs[0]["target_name"].iloc[0] == "target_a"
    assert eval_dfs[0]["model_name"].iloc[0] == "toy_lightning"


def test_collect_eval_dfs_warns_when_optional_eval_csv_missing(monkeypatch, capsys) -> None:
    logger = ParentRunLogger()

    class FakeRun:
        def __init__(self, run_id: str, target: str, model_name: str):
            self.info = SimpleNamespace(run_id=run_id)
            self.data = SimpleNamespace(tags={"target": target, "model_name": model_name})

    class FakeRunInfo:
        def __init__(self, experiment_id: str):
            self.info = SimpleNamespace(experiment_id=experiment_id)

    class FakeMlflowClient:
        def get_run(self, parent_run_id):
            return FakeRunInfo("experiment-1")

        def search_runs(self, experiment_ids, filter_string):
            return [FakeRun("child-run-1", "target_a", "toy_lightning")]

    def fake_download_artifacts(*, run_id, artifact_path):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(mlflow_loggers_module.mlflow.tracking, "MlflowClient", FakeMlflowClient)
    monkeypatch.setattr(mlflow_loggers_module.mlflow.artifacts, "download_artifacts", fake_download_artifacts)

    eval_dfs = logger._collect_eval_dfs("parent-run-1")
    captured = capsys.readouterr()

    assert eval_dfs == []
    assert "Optional eval CSV missing" in captured.out


def test_collect_eval_dfs_ignores_unlabeled_runs(monkeypatch) -> None:
    logger = ParentRunLogger()

    class FakeRun:
        def __init__(self, run_id: str, target: str | None, model_name: str | None):
            self.info = SimpleNamespace(run_id=run_id)
            tags = {}
            if target is not None:
                tags["target"] = target
            if model_name is not None:
                tags["model_name"] = model_name
            self.data = SimpleNamespace(tags=tags)

    class FakeRunInfo:
        def __init__(self, experiment_id: str):
            self.info = SimpleNamespace(experiment_id=experiment_id)

    class FakeMlflowClient:
        def get_run(self, parent_run_id):
            return FakeRunInfo("experiment-1")

        def search_runs(self, experiment_ids, filter_string):
            return [
                FakeRun("wrapper-run", None, None),
                FakeRun("child-run-1", "target_a", "toy_lightning"),
            ]

    monkeypatch.setattr(mlflow_loggers_module.mlflow.tracking, "MlflowClient", FakeMlflowClient)
    monkeypatch.setattr(
        mlflow_loggers_module.mlflow.artifacts,
        "download_artifacts",
        lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )

    eval_dfs = logger._collect_eval_dfs("parent-run-1")

    assert eval_dfs == []


def test_log_lightning_pred_obs_artifact_separates_targets_by_directory(monkeypatch, tmp_path) -> None:
    """Two targets in one run must not overwrite each other - but they separate by DIRECTORY now.

    They used to separate by filename (`..._target_a_soil_graph.png`), which kept them apart within
    a run at the cost of giving every run a different artifact path, so MLflow's compare view found
    nothing in common. Nesting under `plots/<target>/` keeps both properties.
    """
    logger = ChildRunLogger()

    logged: list[tuple[str, str]] = []

    def fake_create_pred_obs_plot(eval_df, builtin_metrics, artifacts_dir):
        artifact = Path(artifacts_dir) / "obs_pred_and_residual_plot.png"
        artifact.write_text("plot", encoding="utf-8")
        return {"obs_pred_and_residual_plot": str(artifact)}

    def fake_log_artifact(path, artifact_path=None):
        logged.append((path, artifact_path))

    monkeypatch.setattr(mlflow_loggers_module, "create_pred_obs_plot", fake_create_pred_obs_plot)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_artifact", fake_log_artifact)

    eval_a = pd.DataFrame({"target_a": [1.0, 2.0], "prediction": [1.1, 1.9]})
    eval_b = pd.DataFrame({"target_b": [3.0, 4.0], "prediction": [3.1, 3.9]})

    assert logger._log_lightning_pred_obs_artifact(
        eval_a, target="target_a", model_name="soil_graph", artifact_path="plots/target_a"
    )
    assert logger._log_lightning_pred_obs_artifact(
        eval_b, target="target_b", model_name="soil_graph", artifact_path="plots/target_b"
    )

    assert len(logged) == 2
    # Same stable leaf, different directory: distinct destinations, comparable across runs.
    assert [Path(path).name for path, _ in logged] == ["pred_obs.png", "pred_obs.png"]
    assert [artifact_path for _, artifact_path in logged] == ["plots/target_a", "plots/target_b"]


def test_single_target_pred_obs_lands_on_the_flat_comparable_path(monkeypatch) -> None:
    logger = ChildRunLogger()
    logged: list[tuple[str, str]] = []

    def fake_create_pred_obs_plot(eval_df, builtin_metrics, artifacts_dir):
        artifact = Path(artifacts_dir) / "obs_pred_and_residual_plot.png"
        artifact.write_text("plot", encoding="utf-8")
        return {"obs_pred_and_residual_plot": str(artifact)}

    monkeypatch.setattr(mlflow_loggers_module, "create_pred_obs_plot", fake_create_pred_obs_plot)
    monkeypatch.setattr(
        mlflow_loggers_module.mlflow,
        "log_artifact",
        lambda path, artifact_path=None: logged.append((path, artifact_path)),
    )

    frame = pd.DataFrame({"target_a": [1.0, 2.0], "prediction": [1.1, 1.9]})
    assert logger._log_lightning_pred_obs_artifact(frame, target="target_a", model_name="soil_cnn")

    assert logged == [(logged[0][0], "plots")]
    assert Path(logged[0][0]).name == "pred_obs.png"


def test_log_lightning_child_run_logs_pred_obs_for_single_target_frame(monkeypatch) -> None:
    logger = ChildRunLogger()

    set_tags = MagicMock()
    log_params = MagicMock()
    log_metric = MagicMock()
    log_artifact = MagicMock()
    log_model = MagicMock()

    monkeypatch.setattr(mlflow_loggers_module.mlflow, "set_tags", set_tags)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_params", log_params)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_metric", log_metric)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_artifact", log_artifact)
    monkeypatch.setattr(mlflow.pyfunc, "log_model", log_model)
    monkeypatch.setattr(mlflow.pytorch, "save_model", MagicMock())

    logger.log_lightning_child_run(
        config=SimpleNamespace(),
        target="target_a",
        model_name="toy_lightning",
        evaluation_df=pd.DataFrame(
            {
                "feature_1": [1.0, 2.0],
                "target_a": [1.0, 2.0],
                "prediction": [1.1, 1.9],
            }
        ),
        validation_metrics={"val_loss": 0.5},
        test_metrics={"test_loss": 0.4},
        model=SimpleNamespace(),
    )

    assert any(call.kwargs.get("artifact_path") == "plots" for call in log_artifact.call_args_list)

# --- architecture introspection over real modules ---------------------------
# The fakes above use SimpleNamespace(in_features=...), which hides the case that actually
# matters: an nn.Sequential head, whose dims used to log as None.


def test_architecture_params_report_dims_for_a_sequential_head() -> None:
    import pytest

    torch = pytest.importorskip("torch")
    nn = torch.nn

    model = SimpleNamespace(
        static_encoder=nn.Sequential(nn.Linear(11, 24), nn.LayerNorm(24), nn.ReLU()),
        output_head=nn.Sequential(nn.Linear(152, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)),
    )

    params = ChildRunLogger()._collect_lightning_architecture_params(model)

    assert params["architecture.static_encoder_in_features"] == 11
    assert params["architecture.static_encoder_out_features"] == 24
    # first Linear in, last Linear out - the whole block, not just one layer
    assert params["architecture.output_head_in_features"] == 152
    assert params["architecture.output_head_out_features"] == 1


def test_architecture_params_still_handle_a_plain_linear_head() -> None:
    import pytest

    torch = pytest.importorskip("torch")

    model = SimpleNamespace(output_head=torch.nn.Linear(152, 1))

    params = ChildRunLogger()._collect_lightning_architecture_params(model)

    assert params["architecture.output_head_in_features"] == 152
    assert params["architecture.output_head_out_features"] == 1
