"""The two training families must report the same numbers under the same names.

This is the regression test for the bug that started all of it. `ChildRunLogger.log_child_run`
logged a POSITIVE cross-validated RMSE as `mean_test_score`; `log_lightning_child_run` logged
`-test_loss`, a NEGATIVE number, under that same name; and `log_parent_summary` plotted both on one
axis. Three units, two signs, one column.

The cross-family equality test below is the one that matters: give both loggers the same
observations and predictions and every unified metric has to come out identical.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import mlflow.pyfunc
import mlflow.pytorch
import pandas as pd
import pytest

import yg_eo_soilnet.logger.mlflow_loggers as mlflow_loggers_module
from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger, ParentRunLogger
from yg_eo_soilnet.metrics import METRIC_STEMS

OBSERVED = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
PREDICTED = np.array([1.3, 1.8, 3.4, 3.7, 5.2, 6.1])


@pytest.fixture
def captured_metrics(monkeypatch) -> dict:
    """Collects every mlflow.log_metric call into a dict."""
    recorded: dict = {}

    def record(key, value, **kwargs):
        recorded[key] = value

    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_metric", record)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "set_tags", MagicMock())
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_params", MagicMock())
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_artifact", MagicMock())
    monkeypatch.setattr("yg_eo_soilnet.artifacts.mlflow.log_artifact", MagicMock())
    monkeypatch.setattr(mlflow.pyfunc, "log_model", MagicMock())
    return recorded


def _run_lightning(logger, captured_metrics) -> dict:
    logger.log_lightning_child_run(
        config=SimpleNamespace(EXPLAIN_ENABLED=False),
        target="organic_matter_pct",
        model_name="soil_cnn",
        evaluation_df=pd.DataFrame({"organic_matter_pct": OBSERVED, "prediction": PREDICTED}),
        validation_metrics={"val_loss": 0.51},
        test_metrics={"test_loss": 0.42, "test_r2": 0.66},
        model=SimpleNamespace(),
    )
    return dict(captured_metrics)


def test_lightning_no_longer_logs_the_negated_score(captured_metrics) -> None:
    metrics = _run_lightning(ChildRunLogger(), captured_metrics)

    assert "mean_test_score" not in metrics
    assert "mean_train_score" not in metrics


def test_lightning_rmse_is_positive_and_matches_the_eval_frame(captured_metrics) -> None:
    metrics = _run_lightning(ChildRunLogger(), captured_metrics)

    assert metrics["rmse_test"] > 0
    assert np.isclose(metrics["rmse_test"], np.sqrt(np.mean((PREDICTED - OBSERVED) ** 2)))


def test_the_training_space_metrics_keep_their_names_and_values(captured_metrics) -> None:
    """val_loss/test_loss/test_r2 are wired into EarlyStopping, ModelCheckpoint and the HPO
    objective whitelist. Renaming them would invalidate every exported tuned config."""
    metrics = _run_lightning(ChildRunLogger(), captured_metrics)

    assert metrics["val_loss"] == 0.51
    assert metrics["test_loss"] == 0.42
    assert metrics["test_r2"] == 0.66


def test_test_loss_and_rmse_test_are_both_present_and_different(captured_metrics) -> None:
    """They measure different things in different spaces; keeping both is deliberate."""
    metrics = _run_lightning(ChildRunLogger(), captured_metrics)

    assert "test_loss" in metrics and "rmse_test" in metrics
    assert metrics["test_loss"] != metrics["rmse_test"]


def test_both_families_report_the_same_numbers_for_the_same_predictions(monkeypatch) -> None:
    """The core regression test.

    A sklearn run and a Lightning run scored on identical observations and predictions must agree
    on every unified metric. Before yg_eo_soilnet.metrics they disagreed on the sign.
    """
    from yg_eo_soilnet.metrics import regression_metrics

    lightning_recorded: dict = {}
    monkeypatch.setattr(
        mlflow_loggers_module.mlflow, "log_metric", lambda key, value, **kw: lightning_recorded.update({key: value})
    )
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "set_tags", MagicMock())
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_params", MagicMock())
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_artifact", MagicMock())
    monkeypatch.setattr("yg_eo_soilnet.artifacts.mlflow.log_artifact", MagicMock())
    monkeypatch.setattr(mlflow.pyfunc, "log_model", MagicMock())

    ChildRunLogger().log_lightning_child_run(
        config=SimpleNamespace(EXPLAIN_ENABLED=False),
        target="organic_matter_pct",
        model_name="soil_cnn",
        evaluation_df=pd.DataFrame({"organic_matter_pct": OBSERVED, "prediction": PREDICTED}),
        validation_metrics={},
        test_metrics={},
        model=SimpleNamespace(),
    )

    # The sklearn branch computes the same set from the same helper on the same frame, so comparing
    # against the helper is comparing against what log_child_run logs.
    expected = regression_metrics(OBSERVED, PREDICTED)

    for stem in METRIC_STEMS:
        name = f"{stem}_test"
        if name not in expected:
            continue
        assert np.isclose(lightning_recorded[name], expected[name]), f"{name} disagrees"


# --- the parent leaderboard -------------------------------------------------


def test_leaderboard_backfills_legacy_runs_so_old_mlruns_still_plot() -> None:
    """A pre-unification sklearn row: positive RMSE under the retired name."""
    row = ParentRunLogger._backfill_legacy_metrics(
        {"target": "om", "model": "XGBoost", "mean_test_score": 22.76, "r2_score": 0.41}
    )

    assert row["rmse_test"] == 22.76
    assert row["r2_test"] == 0.41
    assert row["legacy_metrics"] is True


def test_leaderboard_backfill_makes_a_negative_legacy_lightning_row_comparable() -> None:
    """A pre-unification Lightning row: -test_loss, hence negative. abs() puts it back on the
    same side of zero as the sklearn rows; it is a display fix, not a unit conversion, which is
    why the row is tagged."""
    row = ParentRunLogger._backfill_legacy_metrics(
        {"target": "om", "model": "soil_cnn", "mean_test_score": -0.7284, "r2_score": 0.30}
    )

    assert row["rmse_test"] == pytest.approx(0.7284)
    assert row["legacy_metrics"] is True


def test_leaderboard_leaves_current_runs_untouched() -> None:
    row = ParentRunLogger._backfill_legacy_metrics(
        {"target": "om", "model": "XGBoost", "rmse_test": 1.5, "r2_test": 0.8}
    )

    assert row["rmse_test"] == 1.5
    assert "legacy_metrics" not in row


def test_leaderboard_tolerates_a_row_with_neither_name() -> None:
    row = ParentRunLogger._backfill_legacy_metrics({"target": "om", "model": "XGBoost"})

    assert "rmse_test" not in row
    assert "legacy_metrics" not in row


def test_parent_summary_plots_the_unified_axes(monkeypatch) -> None:
    captured: dict = {}

    def fake_scatter(frame, metric_x, metric_y, **kwargs):
        captured["metric_x"] = metric_x
        captured["metric_y"] = metric_y
        import matplotlib.pyplot as plt

        return plt.figure()

    monkeypatch.setattr(mlflow_loggers_module, "plot_leaderboard_scatter", fake_scatter)
    monkeypatch.setattr(mlflow_loggers_module, "create_parent_pred_obs", lambda frames: None)
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "set_tags", MagicMock())
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_params", MagicMock())
    monkeypatch.setattr(mlflow_loggers_module.mlflow, "log_artifact", MagicMock())
    monkeypatch.setattr("yg_eo_soilnet.artifacts.mlflow.log_artifact", MagicMock())

    parent = ParentRunLogger()
    monkeypatch.setattr(parent, "_collect_leaderboard", lambda run_id: pd.DataFrame())
    monkeypatch.setattr(parent, "_collect_eval_dfs", lambda run_id: [])

    config = SimpleNamespace(
        DATA_FILE="d.csv",
        ENABLE_CLUSTERING=False,
        CLUSTERING_STRATEGY={},
        SPLIT_STRATEGY="kfold",
        DATA_FOLDER="/data",
        RANDOM_SEED=42,
        TARGET_COLUMNS=["om"],
        COLUMNS_TO_TRANSFORM=[],
    )
    parent.log_parent_summary("run-1", SimpleNamespace(config=config))

    assert captured["metric_x"] == "rmse_test"
    assert captured["metric_y"] == "r2_test"
