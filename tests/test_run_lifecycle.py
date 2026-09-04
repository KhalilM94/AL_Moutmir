"""Runs that outlive their process, and inference paid for twice.

A multi-target run once produced a model for only its first target and looked like a bug in the
target grouping. It was not: the kernel's OOM killer had taken the process partway through. The
evidence was hard to reach because the run recorded nothing about what it had planned to fit, and
what it left behind was still marked RUNNING - which reads as "in progress", or as a model that was
successfully made.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import yg_eo_soilnet.logger.mlflow_loggers as loggers_module
from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger
from yg_eo_soilnet.tracking import (
    HOST_NAME_TAG,
    HOST_PID_TAG,
    close_stale_runs,
    run_owner_tags,
    start_child_run,
)


# --- ownership tags --------------------------------------------------------


def test_a_run_records_which_process_wrote_it() -> None:
    tags = run_owner_tags()
    assert tags[HOST_PID_TAG] == str(os.getpid())
    assert tags[HOST_NAME_TAG]


# --- the stale sweep -------------------------------------------------------


def _dead_pid() -> int:
    """A pid that certainly does not exist: fork a child and reap it."""
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never returns
        os._exit(0)
    os.waitpid(pid, 0)
    return pid


def _fake_client(runs, terminated):
    return SimpleNamespace(
        get_experiment_by_name=lambda name: SimpleNamespace(experiment_id="1"),
        search_runs=lambda experiment_ids, filter_string, max_results: runs,
        set_terminated=lambda run_id, status: terminated.append((run_id, status)),
    )


def _run(run_id, tags):
    return SimpleNamespace(info=SimpleNamespace(run_id=run_id), data=SimpleNamespace(tags=tags))


def test_a_run_abandoned_by_a_dead_process_is_marked_killed(monkeypatch) -> None:
    import socket

    host = socket.gethostname()
    terminated: list = []
    runs = [_run("abandoned", {HOST_NAME_TAG: host, HOST_PID_TAG: str(_dead_pid())})]
    monkeypatch.setattr(
        loggers_module.mlflow.tracking, "MlflowClient", lambda: _fake_client(runs, terminated)
    )

    assert close_stale_runs("exp") == ["abandoned"]
    assert terminated == [("abandoned", "KILLED")]


def test_the_sweep_never_touches_a_run_that_could_still_be_writing(monkeypatch) -> None:
    """The failure this guards against is worse than the one it fixes.

    Sweeping on "status is RUNNING" alone would let one training process terminate a second one
    running beside it. Only a run from THIS host whose pid is genuinely gone is closed.
    """
    import socket

    host = socket.gethostname()
    terminated: list = []
    runs = [
        # Alive: this very process.
        _run("mine", {HOST_NAME_TAG: host, HOST_PID_TAG: str(os.getpid())}),
        # Another machine's run, whose pid means nothing here.
        _run("elsewhere", {HOST_NAME_TAG: "some-other-box", HOST_PID_TAG: str(_dead_pid())}),
        # Written before runs carried ownership tags: left alone rather than guessed at.
        _run("untagged", {}),
    ]
    monkeypatch.setattr(
        loggers_module.mlflow.tracking, "MlflowClient", lambda: _fake_client(runs, terminated)
    )

    assert close_stale_runs("exp") == []
    assert terminated == []


def test_a_sweep_that_cannot_reach_the_store_does_not_stop_the_run(monkeypatch) -> None:
    def explode():
        raise RuntimeError("tracking server is down")

    monkeypatch.setattr(loggers_module.mlflow.tracking, "MlflowClient", explode)
    logger = MagicMock()

    assert close_stale_runs("exp", logger=logger) == []
    logger.warning.assert_called_once()


# --- nesting ---------------------------------------------------------------


def test_a_child_run_nests_under_a_parent_but_stands_alone_without_one() -> None:
    """`nested=True` is an ERROR when nothing is active.

    Hardcoding it tied the trainers to being called from inside main.py's parent run, but they are
    also driven directly - by tests, and by anyone fitting a single model.
    """
    import mlflow

    client = mlflow.tracking.MlflowClient()

    with start_child_run("standalone") as run:
        assert mlflow.active_run().info.run_id == run.info.run_id
        standalone_id = run.info.run_id
    # Read back from the store: the object start_run returns is a snapshot taken before the tags
    # were written.
    standalone_tags = client.get_run(standalone_id).data.tags
    assert standalone_tags[HOST_PID_TAG] == str(os.getpid())
    assert "mlflow.parentRunId" not in standalone_tags

    with mlflow.start_run(run_name="parent") as parent:
        with start_child_run("child") as child:
            child_id = child.info.run_id
    assert client.get_run(child_id).data.tags["mlflow.parentRunId"] == parent.info.run_id


def test_a_child_that_fails_to_tag_does_not_strand_its_parent(monkeypatch) -> None:
    """start_run pushes onto the active-run stack; the caller's `with` is what pops it.

    If set_tags raises in between, the ActiveRun never reaches a `with` and nothing pops it. That
    is worse than one lost child: mlflow.end_run() pops the TOP of the stack rather than a named
    run, so the parent's own `with` would close the orphan and leave the PARENT at RUNNING forever
    - which is exactly the "parent run never ends" symptom, reachable with no OOM involved.
    """
    import mlflow

    with mlflow.start_run(run_name="parent") as parent:
        monkeypatch.setattr(
            mlflow, "set_tags", MagicMock(side_effect=RuntimeError("tracking store down"))
        )
        with pytest.raises(RuntimeError):
            start_child_run("doomed")
        monkeypatch.undo()

        # The parent, not the orphan, is what is active again.
        assert mlflow.active_run().info.run_id == parent.info.run_id

    assert mlflow.active_run() is None
    assert mlflow.tracking.MlflowClient().get_run(parent.info.run_id).info.status == "FINISHED"


# --- inference paid for once -----------------------------------------------


class _CountingEstimator:
    """Counts prediction passes, by row, the way an in-context model would charge for them."""

    def __init__(self):
        self.rows_predicted = 0
        self.calls = 0

    def predict(self, X):
        self.calls += 1
        self.rows_predicted += len(X)
        return np.arange(len(X), dtype=float)


def test_mlflow_evaluate_scores_the_predictions_already_computed(monkeypatch) -> None:
    """Static-dataset evaluation: no model, so no reload and no second inference pass."""
    captured = {}
    monkeypatch.setattr(
        loggers_module.mlflow.models, "evaluate", lambda **kwargs: captured.update(kwargs)
    )

    frame = pd.DataFrame({"target_a": [1.0, 2.0, 3.0], "prediction": [1.1, 1.9, 3.2]})
    ChildRunLogger()._evaluate_sklearn_target(frame, "target_a")

    assert captured["predictions"] == "prediction"
    assert captured["targets"] == "target_a"
    # The absence is the point: passing a model URI made MLflow reload the model and predict the
    # test set all over again, duplicating what the evaluation frame had just done.
    assert "model" not in captured
    assert captured["data"] is frame


def test_evaluate_is_skipped_when_the_frame_has_nothing_to_score(monkeypatch) -> None:
    called = MagicMock()
    monkeypatch.setattr(loggers_module.mlflow.models, "evaluate", called)

    logger = ChildRunLogger()
    logger._evaluate_sklearn_target(None, "target_a")
    logger._evaluate_sklearn_target(pd.DataFrame({"prediction": [1.0]}), "target_a")

    called.assert_not_called()


@pytest.mark.parametrize("enabled", [True, False])
def test_the_train_fit_diagnostic_is_switchable(enabled, monkeypatch) -> None:
    """r2_train_fit costs a full pass over the TRAINING split for one number.

    Free for a tree, minutes per target for an in-context model - which is the difference between
    a run that finishes and one the OOM killer gets to first.
    """
    # Serialising a locally-defined estimator is not what this test is about, and skops refuses it.
    monkeypatch.setattr(
        loggers_module.mlflow.sklearn,
        "log_model",
        lambda **kwargs: SimpleNamespace(model_uri="models:/toy/1", registered_model_version=None),
    )
    monkeypatch.setattr(loggers_module, "infer_signature", lambda *args, **kwargs: None)

    estimator = _CountingEstimator()
    X_train = pd.DataFrame({"feature": np.arange(50, dtype=float)})
    X_test = pd.DataFrame({"feature": np.arange(10, dtype=float)})
    y_train = pd.Series(np.arange(50, dtype=float), name="target_a")
    y_test = pd.Series(np.arange(10, dtype=float), name="target_a")

    logger = ChildRunLogger()
    for name in ("_log_cv_results", "_log_table_artifact", "_log_metric_dict", "_promote_champion",
                 "_log_plots", "_log_shap_slice", "_write_split_summary", "_write_json_artifact",
                 "_evaluate_sklearn_target"):
        setattr(logger, name, MagicMock())
    # The explanation is built once, on the model run, and sliced per target. (None, {}) is "nothing
    # to explain"; a bare MagicMock would fail the tuple unpack at the call site.
    logger._build_shap_results = MagicMock(return_value=(None, {}))
    logged: dict = {}
    logger._log_metric_dict = lambda metrics: logged.update(metrics)

    import mlflow

    with mlflow.start_run(run_name="probe"):
        logger.log_child_run(
            config=SimpleNamespace(
                ENABLE_CLUSTERING=False,
                CLUSTERING_STRATEGY={},
                MLFLOW_REGISTER_MODELS=False,
                EXPLAIN_ENABLED=False,
                LOG_TRAIN_FIT_METRIC=enabled,
            ),
            search=SimpleNamespace(best_params_={}, best_index_=0),
            cv_results=pd.DataFrame({"params": [{}], "mean_test_score": [-1.0]}),
            best_model=estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            target="target_a",
            targets=["target_a"],
            param_names=[],
            model_name="Toy",
            plot_functions={},
        )

    assert ("r2_train_fit" in logged) is enabled
    # 10 test rows either way; the 50 training rows are what the switch buys back.
    assert estimator.rows_predicted == (60 if enabled else 10)


def test_the_train_fit_diagnostic_can_be_declined_per_model(monkeypatch) -> None:
    """One expensive estimator should not force the diagnostic off for the cheap ones.

    The cost is a property of the model, not of a row budget, so it can only be declined by name -
    the same reasoning EXPLAIN_SKIP_MODELS is built on.
    """
    monkeypatch.setattr(
        loggers_module.mlflow.sklearn,
        "log_model",
        lambda **kwargs: SimpleNamespace(model_uri="models:/toy/1", registered_model_version=None),
    )
    monkeypatch.setattr(loggers_module, "infer_signature", lambda *args, **kwargs: None)

    estimator = _CountingEstimator()
    logger = ChildRunLogger()
    for name in ("_log_cv_results", "_log_table_artifact", "_promote_champion", "_log_plots",
                 "_log_shap_slice", "_write_split_summary", "_write_json_artifact",
                 "_evaluate_sklearn_target"):
        setattr(logger, name, MagicMock())
    logger._build_shap_results = MagicMock(return_value=(None, {}))
    logged: dict = {}
    logger._log_metric_dict = lambda metrics: logged.update(metrics)

    import mlflow

    with mlflow.start_run(run_name="probe"):
        logger.log_child_run(
            config=SimpleNamespace(
                ENABLE_CLUSTERING=False,
                CLUSTERING_STRATEGY={},
                MLFLOW_REGISTER_MODELS=False,
                EXPLAIN_ENABLED=False,
                LOG_TRAIN_FIT_METRIC=True,
                LOG_TRAIN_FIT_METRIC_SKIP_MODELS=["Toy"],
            ),
            search=SimpleNamespace(best_params_={}, best_index_=0),
            cv_results=pd.DataFrame({"params": [{}], "mean_test_score": [-1.0]}),
            best_model=estimator,
            X_train=pd.DataFrame({"feature": np.arange(50, dtype=float)}),
            y_train=pd.Series(np.arange(50, dtype=float), name="target_a"),
            X_test=pd.DataFrame({"feature": np.arange(10, dtype=float)}),
            y_test=pd.Series(np.arange(10, dtype=float), name="target_a"),
            target="target_a",
            targets=["target_a"],
            param_names=[],
            model_name="Toy",
            plot_functions={},
        )

    assert "r2_train_fit" not in logged
    # The 50 training rows were never asked for; only the one test pass happened.
    assert estimator.rows_predicted == 10


# --- registration follows the metrics --------------------------------------


def test_a_model_enters_the_registry_only_after_its_metrics_exist(monkeypatch) -> None:
    """Registering at log_model time meant a run killed partway through still minted a version.

    Four such versions of organic_matter_g_kg_TabICL accumulated that way, each READY, each backed
    by a run stuck at RUNNING with no rmse_test - and promote_if_better would have refused every
    one of them. Registration is now a separate step that the metrics happen before.
    """
    monkeypatch.setattr(
        loggers_module.mlflow.sklearn,
        "log_model",
        lambda **kwargs: SimpleNamespace(model_uri="models:/toy/1", registered_model_version=None),
    )
    monkeypatch.setattr(loggers_module, "infer_signature", lambda *args, **kwargs: None)

    order: list[str] = []
    monkeypatch.setattr(
        loggers_module.mlflow,
        "register_model",
        lambda uri, name: order.append("register") or SimpleNamespace(version=3),
    )

    logger = ChildRunLogger()
    for name in ("_log_cv_results", "_log_table_artifact", "_log_plots", "_log_shap_slice",
                 "_write_split_summary", "_write_json_artifact", "_evaluate_sklearn_target"):
        setattr(logger, name, MagicMock())
    logger._build_shap_results = MagicMock(return_value=(None, {}))
    logger._log_metric_dict = lambda metrics: order.append("metrics")
    promote = MagicMock(return_value={})
    logger._promote_champion = promote

    import mlflow

    with mlflow.start_run(run_name="probe"):
        logger.log_child_run(
            config=SimpleNamespace(
                ENABLE_CLUSTERING=False,
                CLUSTERING_STRATEGY={},
                MLFLOW_REGISTER_MODELS=True,
                EXPLAIN_ENABLED=False,
                LOG_TRAIN_FIT_METRIC=False,
            ),
            search=SimpleNamespace(best_params_={}, best_index_=0),
            cv_results=pd.DataFrame({"params": [{}], "mean_test_score": [-1.0]}),
            best_model=_CountingEstimator(),
            X_train=pd.DataFrame({"feature": np.arange(50, dtype=float)}),
            y_train=pd.Series(np.arange(50, dtype=float), name="target_a"),
            X_test=pd.DataFrame({"feature": np.arange(10, dtype=float)}),
            y_test=pd.Series(np.arange(10, dtype=float), name="target_a"),
            target="target_a",
            targets=["target_a"],
            param_names=[],
            model_name="Toy",
            plot_functions={},
        )

    assert order.index("metrics") < order.index("register")
    # The version reaches promotion, which is the only consumer that decides on it.
    assert promote.call_args.args[-1] == 3


def test_a_registry_outage_does_not_discard_a_finished_run(monkeypatch) -> None:
    """By this point the model is logged and servable; only its registry entry is missing.

    Throwing away a completed fit over that is the worse trade, so it is recorded on the run and
    escalated only under FAIL_ON_MODEL_ERROR.
    """
    monkeypatch.setattr(
        loggers_module.mlflow,
        "register_model",
        MagicMock(side_effect=RuntimeError("registry unreachable")),
    )
    logger = ChildRunLogger()
    config = SimpleNamespace(MLFLOW_REGISTER_MODELS=True, FAIL_ON_MODEL_ERROR=False)
    model_info = SimpleNamespace(model_uri="models:/toy/1")

    import mlflow

    with mlflow.start_run(run_name="probe") as run:
        assert logger._register_sklearn_model(config, model_info, "target_a_Toy") is None

    tags = mlflow.tracking.MlflowClient().get_run(run.info.run_id).data.tags
    assert tags["model_registered"] == "false"
    assert "registry unreachable" in tags["model_registration_error"]

    config.FAIL_ON_MODEL_ERROR = True
    with mlflow.start_run(run_name="strict"):
        with pytest.raises(RuntimeError):
            logger._register_sklearn_model(config, model_info, "target_a_Toy")


# --- immutable params ------------------------------------------------------
# MLflow params cannot change value once written. The parent run is written to twice - once at the
# start of training and once in the summary at the end - so a key written in both places with two
# different renderings does not fail fast. It fails AFTER every model has been fitted, logged and
# registered, and takes the summary, the leaderboard and the run's FINISHED status with it.


def _params_of(run_id):
    import mlflow

    return mlflow.tracking.MlflowClient().get_run(run_id).data.params


def test_log_params_once_writes_new_keys_and_tolerates_an_identical_relog() -> None:
    import mlflow
    from yg_eo_soilnet.tracking import log_params_once

    with mlflow.start_run() as run:
        log_params_once({"a": "1", "b": 2})
        log_params_once({"a": "1", "c": "3"})  # 'a' unchanged: allowed, and must not raise
        params = _params_of(run.info.run_id)

    assert params == {"a": "1", "b": "2", "c": "3"}


def test_log_params_once_keeps_the_first_value_and_warns_on_a_clash() -> None:
    import mlflow
    from yg_eo_soilnet.tracking import log_params_once

    logger = MagicMock()
    with mlflow.start_run() as run:
        log_params_once({"TARGET_COLUMNS": "clay_pct,sand_pct"})
        # The exact shape of the bug: same key, different rendering of the same information.
        log_params_once({"TARGET_COLUMNS": ["clay_pct", "sand_pct"], "other": "kept"}, logger=logger)
        params = _params_of(run.info.run_id)

    assert params["TARGET_COLUMNS"] == "clay_pct,sand_pct"
    # The rest of the batch still lands - the file store applies params one at a time and would
    # otherwise have written some and dropped others.
    assert params["other"] == "kept"
    assert "TARGET_COLUMNS" in logger.warning.call_args[0][0]


def test_the_target_plan_and_the_parent_summary_can_share_one_run() -> None:
    """The regression itself, and the test whose absence let it ship.

    `_log_target_plan` writes TARGET_COLUMNS at the start of training and `log_parent_summary`
    used to write it again at the end in a different format. Nothing exercised both against one
    run, so a green suite said nothing about it.
    """
    import mlflow
    import main as main_module

    trainer = main_module.SoilModelTraining.__new__(main_module.SoilModelTraining)
    trainer.config = SimpleNamespace(
        TARGET_COLUMNS=["clay_pct", "sand_pct", "total_silt_pct"], MULTI_TARGET_MODE="per_target"
    )
    trainer.logger = MagicMock()

    with mlflow.start_run() as run:
        trainer._log_target_plan({("clay_pct",): {}, ("sand_pct",): {}}, {})
        # The payload log_parent_summary writes at the end of a run.
        loggers_module.log_params_once({
            "RANDOM_SEED": 42,
            "COLUMNS_TO_TRANSFORM": [],
            "SPLIT_TEST_SIZE": 0.2,
        })
        params = _params_of(run.info.run_id)

    assert params["TARGET_COLUMNS"] == "clay_pct,sand_pct,total_silt_pct"
    assert params["MULTI_TARGET_MODE"] == "per_target"
    assert params["sklearn_target_groups"] == "clay_pct | sand_pct"
    assert params["RANDOM_SEED"] == "42"


def test_the_parent_summary_no_longer_writes_target_columns(monkeypatch) -> None:
    """One writer, and it is the early one. Pins the invariant rather than the symptom."""
    from yg_eo_soilnet.logger.mlflow_loggers import ParentRunLogger

    payloads: list[dict] = []
    monkeypatch.setattr(loggers_module, "log_params_once", lambda params, **kw: payloads.append(params))
    monkeypatch.setattr(loggers_module.mlflow, "set_tags", MagicMock())
    parent = ParentRunLogger()
    monkeypatch.setattr(parent, "_collect_leaderboard", lambda run_id: pd.DataFrame())

    config = SimpleNamespace(
        DATA_FOLDER="d", DATA_FILE="f.csv", RANDOM_SEED=42,
        TARGET_COLUMNS=["clay_pct"], COLUMNS_TO_TRANSFORM=[],
        ENABLE_CLUSTERING=False, CLUSTERING_STRATEGY={}, SPLIT_STRATEGY="kfold",
    )
    try:
        parent.log_parent_summary("run-1", SimpleNamespace(config=config))
    except Exception:
        # Whatever the leaderboard half does is not this test's business; the params are.
        pass

    assert payloads, "log_parent_summary should log params through log_params_once"
    assert all("TARGET_COLUMNS" not in payload for payload in payloads)
