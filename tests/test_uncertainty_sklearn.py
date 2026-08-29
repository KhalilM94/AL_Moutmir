"""The sklearn ensemble end to end: real estimators, the real trainer, the real logger."""

from types import SimpleNamespace

import mlflow
import numpy as np
import pandas as pd
import pytest

from yg_eo_soilnet.trainers import ModelTrainer
from yg_eo_soilnet.uncertainty import uncertainty_enabled_for
from yg_eo_soilnet.uncertainty.predictors import EnsembleRegressor


def _config(**overrides) -> SimpleNamespace:
    base = dict(
        CATEGORICAL_FEATURES=[],
        CLUSTERING_STRATEGY={"enabled": False, "params": {}},
        ENABLE_CLUSTERING=False,
        MIN_FEATURE_COUNT=2,
        LOG_TRAIN_FIT_METRIC=False,
        MLFLOW_REGISTER_MODELS=False,
        EXPLAIN_ENABLED=False,
        FAIL_ON_MODEL_ERROR=True,
        UNCERTAINTY_ENABLED=True,
        UNCERTAINTY_N_MEMBERS=4,
        UNCERTAINTY_SEED_STRIDE=1000,
        UNCERTAINTY_BOOTSTRAP="auto",
        UNCERTAINTY_ALPHA=0.05,
        UNCERTAINTY_CALIBRATION_METHOD="split_conformal",
        UNCERTAINTY_CALIBRATION_SOURCE="val",
        UNCERTAINTY_MODELS=[],
        UNCERTAINTY_SKIP_MODELS=[],
        POINT_ID_COLUMN="uuid",
        EXPORT_POINT_PREDICTIONS=False,
        EXPORT_POINT_PREDICTIONS_MODELS=[],
        EXPORT_POINT_PREDICTIONS_SKIP_MODELS=[],
        EXPORT_POINT_PREDICTIONS_FAIL_ON_ERROR=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _split(n_rows: int = 200, targets=("target_a",), seed: int = 0) -> dict:
    """A split dict shaped like SklearnDataSplitter's, including the audit-only val keys."""
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {"f1": rng.normal(size=n_rows), "f2": rng.normal(size=n_rows)}
    )
    signal = 3.0 * features["f1"] - 2.0 * features["f2"]
    y = pd.DataFrame(
        {name: signal + rng.normal(scale=1.0, size=n_rows) for name in targets}
    )

    train = slice(0, 120)
    val = slice(120, 160)
    test = slice(160, n_rows)
    return {
        # X_train is the FIT POOL: train u val, matching the real splitter's contract.
        "X_train": features.iloc[0:160].reset_index(drop=True),
        "y_train": y.iloc[0:160].reset_index(drop=True),
        "X_train_only": features.iloc[train].reset_index(drop=True),
        "y_train_only": y.iloc[train].reset_index(drop=True),
        "X_val": features.iloc[val].reset_index(drop=True),
        "y_val": y.iloc[val].reset_index(drop=True),
        "X_test": features.iloc[test].reset_index(drop=True),
        "y_test": y.iloc[test].reset_index(drop=True),
        # The whole featurized population plus its ids, the way SklearnDataSplitter returns them:
        # `point_ids` is a Series indexed like X, which is what lets the export reindex rather than
        # pair by position.
        "X_all": features,
        "point_ids": pd.Series(
            [f"p{index}" for index in range(n_rows)], index=features.index, name="point_id"
        ),
    }


def _trainer(config) -> ModelTrainer:
    import logging

    logger = logging.getLogger("uncertainty-sklearn-test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return ModelTrainer(config=config, seed=42, n_splits=3, logger=logger)


def _pipelines(name="Ridge"):
    from sklearn.linear_model import Ridge

    return {name: {"model": Ridge(), "params": {"model__alpha": [1.0]}, "modeltype": "ml"}}


def _run_and_collect(config, data, pipelines, target="target_a", targets=None):
    """Train inside a parent run and return (child runs by name, the MlflowClient)."""
    client = mlflow.tracking.MlflowClient()
    with mlflow.start_run(run_name="parent") as parent:
        _trainer(config).train(
            target=target, data=data, model_pipelines=pipelines, targets=targets
        )
        parent_id = parent.info.run_id

    experiment_id = client.get_run(parent_id).info.experiment_id

    def descendants(run_id):
        children = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{run_id}'",
        )
        collected = list(children)
        for child in children:
            collected.extend(descendants(child.info.run_id))
        return collected

    return parent_id, descendants(parent_id), client


# --- the switch ------------------------------------------------------------


def test_the_master_switch_turns_the_whole_thing_off():
    assert uncertainty_enabled_for(_config(UNCERTAINTY_ENABLED=False), "Ridge") is False


def test_a_skipped_model_gets_no_ensemble():
    config = _config(UNCERTAINTY_SKIP_MODELS=["TabICL"])
    assert uncertainty_enabled_for(config, "TabICL") is False
    assert uncertainty_enabled_for(config, "Ridge") is True


def test_naming_a_model_explicitly_beats_the_skip_list():
    config = _config(UNCERTAINTY_MODELS=["TabICL"], UNCERTAINTY_SKIP_MODELS=["TabICL"])
    assert uncertainty_enabled_for(config, "TabICL") is True
    # An allowlist is exclusive: everything not named is off.
    assert uncertainty_enabled_for(config, "Ridge") is False


# --- the fit pool ----------------------------------------------------------


def test_val_calibration_moves_the_fit_pool_off_the_val_rows():
    data = _split()
    fit_pool, calibration = _trainer(_config())._resolve_fit_pool(data)
    assert len(fit_pool["X"]) == 120
    assert calibration is not None
    assert len(calibration["X"]) == 40


def test_uncertainty_disabled_keeps_the_full_train_plus_val_fit_pool():
    data = _split()
    fit_pool, calibration = _trainer(_config(UNCERTAINTY_ENABLED=False))._resolve_fit_pool(data)
    assert len(fit_pool["X"]) == 160
    assert calibration is None


def test_cv_oof_calibration_keeps_every_row_in_the_fit_pool():
    data = _split()
    fit_pool, calibration = _trainer(
        _config(UNCERTAINTY_CALIBRATION_SOURCE="cv_oof")
    )._resolve_fit_pool(data)
    assert len(fit_pool["X"]) == 160
    assert calibration is None


def test_a_split_without_val_rows_falls_back_and_says_so():
    data = _split()
    data["X_val"] = data["X_val"].iloc[0:0]
    trainer = _trainer(_config())
    fit_pool, calibration = trainer._resolve_fit_pool(data)
    assert len(fit_pool["X"]) == 160
    assert calibration is None


# --- training --------------------------------------------------------------


def test_every_member_gets_its_own_tagged_child_run():
    _parent, runs, _client = _run_and_collect(_config(), _split(), _pipelines())

    members = [run for run in runs if run.data.tags.get("run_kind") == "ensemble_member"]
    assert len(members) == 4
    assert {run.data.tags["ensemble_member"] for run in members} == {"0", "1", "2", "3"}
    # Seeds are strided, not consecutive.
    assert {int(run.data.tags["ensemble_seed"]) for run in members} == {42, 1042, 2042, 3042}


def test_the_eval_frame_carries_the_estimate_and_its_uncertainty(tmp_path):
    _parent, runs, client = _run_and_collect(_config(), _split(), _pipelines())

    model_run = next(
        run for run in runs
        if run.data.tags.get("model_name") == "Ridge"
        and run.data.tags.get("run_kind") != "ensemble_member"
    )
    local = client.download_artifacts(
        model_run.info.run_id, "eval_results/eval_results.csv", str(tmp_path)
    )
    frame = pd.read_csv(local)

    for column in (
        "prediction",
        "prediction_std",
        "prediction_epistemic_std",
        "prediction_aleatoric_std",
        "prediction_lower",
        "prediction_upper",
    ):
        assert column in frame.columns

    # A sklearn ensemble has no aleatoric component: a point predictor cannot report one.
    assert (frame["prediction_aleatoric_std"] == 0.0).all()
    assert (frame["prediction_std"] > 0.0).all()
    assert (frame["prediction_lower"] < frame["prediction"]).all()
    assert (frame["prediction_upper"] > frame["prediction"]).all()


def test_a_deterministic_estimator_still_produces_a_non_degenerate_ensemble():
    # The bug the bootstrap exists to prevent: Ridge ignores its seed, so without resampling every
    # member is identical, sigma is exactly 0, and the run claims perfect confidence.
    _parent, runs, _client = _run_and_collect(_config(), _split(), _pipelines())
    model_run = next(
        run for run in runs
        if run.data.tags.get("model_name") == "Ridge"
        and run.data.tags.get("run_kind") != "ensemble_member"
    )
    assert model_run.data.metrics["mean_sigma_test"] > 0.0


def test_the_interval_covers_close_to_the_nominal_level():
    _parent, runs, _client = _run_and_collect(_config(), _split(400), _pipelines())
    model_run = next(
        run for run in runs
        if run.data.tags.get("model_name") == "Ridge"
        and run.data.tags.get("run_kind") != "ensemble_member"
    )
    # Conformal on 40 calibration rows is noisy, hence the wide band; the point is that the
    # interval is somewhere near its claim rather than covering 20% or 100%.
    assert model_run.data.metrics["picp_test"] == pytest.approx(0.95, abs=0.12)


def test_the_uncertainty_metrics_are_logged_beside_the_point_metrics():
    _parent, runs, _client = _run_and_collect(_config(), _split(), _pipelines())
    model_run = next(
        run for run in runs
        if run.data.tags.get("model_name") == "Ridge"
        and run.data.tags.get("run_kind") != "ensemble_member"
    )
    for metric in ("rmse_test", "r2_test", "picp_test", "mpiw_test", "interval_score_test"):
        assert metric in model_run.data.metrics


def test_the_fit_pool_is_recorded_so_a_smaller_rmse_is_not_read_as_a_regression():
    _parent, runs, _client = _run_and_collect(_config(), _split(), _pipelines())
    model_run = next(
        run for run in runs
        if run.data.tags.get("model_name") == "Ridge"
        and run.data.tags.get("run_kind") != "ensemble_member"
    )
    assert model_run.data.params["uncertainty_fit_pool"] == "train_only"
    assert model_run.data.params["uncertainty_n_train_rows"] == "120"
    assert model_run.data.params["uncertainty_n_members"] == "4"
    assert "conformal_q" in model_run.data.params


def test_uncertainty_disabled_writes_no_extra_columns(tmp_path):
    _parent, runs, client = _run_and_collect(
        _config(UNCERTAINTY_ENABLED=False), _split(), _pipelines()
    )
    model_run = next(run for run in runs if run.data.tags.get("model_name") == "Ridge")
    local = client.download_artifacts(
        model_run.info.run_id, "eval_results/eval_results.csv", str(tmp_path)
    )
    frame = pd.read_csv(local)

    assert "prediction" in frame.columns
    assert "prediction_std" not in frame.columns
    assert not any(run.data.tags.get("run_kind") == "ensemble_member" for run in runs)


def test_a_joint_group_suffixes_the_uncertainty_columns_per_target(tmp_path):
    data = _split(targets=("target_a", "target_b"))
    _parent, runs, client = _run_and_collect(
        _config(), data, _pipelines(), target="target_a__target_b",
        targets=["target_a", "target_b"],
    )

    model_run = next(
        run for run in runs
        if run.data.tags.get("target") == "target_a__target_b"
        and run.data.tags.get("run_kind") != "ensemble_member"
    )
    local = client.download_artifacts(
        model_run.info.run_id, "eval_results/eval_results.csv", str(tmp_path)
    )
    frame = pd.read_csv(local)

    for target_name in ("target_a", "target_b"):
        assert f"prediction_{target_name}" in frame.columns
        assert f"prediction_std_{target_name}" in frame.columns
        assert f"prediction_lower_{target_name}" in frame.columns
    assert "prediction_std" not in frame.columns


# --- the leaderboard -------------------------------------------------------


def test_members_do_not_replace_their_model_on_the_leaderboard():
    from yg_eo_soilnet.logger.mlflow_loggers import ParentRunLogger

    parent_id, _runs, _client = _run_and_collect(_config(), _split(), _pipelines())
    leaderboard = ParentRunLogger()._collect_leaderboard(parent_id)

    # Exactly one row: the model. Without the run_kind filter this is four member rows and the
    # model itself vanishes, because the collector prefers a run's grandchildren to the run.
    assert len(leaderboard) == 1
    assert leaderboard.iloc[0]["model"] == "Ridge"
    assert leaderboard.iloc[0]["target"] == "target_a"
    assert "rmse_test" in leaderboard.columns


def test_the_parent_eval_frames_exclude_the_members():
    from yg_eo_soilnet.logger.mlflow_loggers import ParentRunLogger

    parent_id, _runs, _client = _run_and_collect(_config(), _split(), _pipelines())
    frames = ParentRunLogger()._collect_eval_dfs(parent_id)
    assert len(frames) == 1


# --- the fitted ensemble ---------------------------------------------------


def test_predict_returns_the_mean_in_the_shape_a_single_member_returns():
    from sklearn.linear_model import Ridge

    X = pd.DataFrame({"f": np.arange(20.0)})
    y = pd.Series(np.arange(20.0) * 2.0)
    members = [Ridge().fit(X, y) for _ in range(3)]

    ensemble = EnsembleRegressor(members, ["target_a"])
    predictions = ensemble.predict(X)

    # 1-D, exactly like one member: the MLflow signature, the evaluator and champion promotion all
    # depend on this staying true.
    assert predictions.ndim == 1
    assert predictions.shape == (20,)
    assert np.allclose(predictions, members[0].predict(X))


def test_predict_frame_returns_the_wide_output_with_named_columns():
    from sklearn.linear_model import Ridge

    from yg_eo_soilnet.uncertainty.conformal import ConformalCalibrator

    X = pd.DataFrame({"f": np.arange(20.0)})
    y = pd.Series(np.arange(20.0) * 2.0)
    ensemble = EnsembleRegressor(
        [Ridge().fit(X, y) for _ in range(3)],
        ["target_a"],
        calibrators={"target_a": ConformalCalibrator(q=1.5, alpha=0.05, n_calib=100)},
    )

    frame = ensemble.predict_frame(X)
    assert list(frame.columns) == [
        "target_a_pred",
        "target_a_std",
        "target_a_epistemic_std",
        "target_a_aleatoric_std",
        "target_a_lower",
        "target_a_upper",
    ]


def test_an_ensemble_needs_at_least_one_member():
    with pytest.raises(ValueError, match="at least one fitted member"):
        EnsembleRegressor([], ["target_a"])


# --- the per-point prediction export ---------------------------------------


def _export_config(**overrides):
    return _config(EXPORT_POINT_PREDICTIONS=True, **overrides)


def _download(client, run, path, tmp_path):
    return pd.read_csv(client.download_artifacts(run.info.run_id, path, str(tmp_path)))


def _export_parent(parent_id, config):
    """Run the parent-level combine step, which main.py does inside the parent run.

    Reopening the run matters: log_table writes to whichever run is ACTIVE, so calling this outside
    one would file the parent's summary against nothing.
    """
    from yg_eo_soilnet.logger.mlflow_loggers import ParentRunLogger

    with mlflow.start_run(run_id=parent_id):
        ParentRunLogger()._log_point_prediction_export(parent_id, config)


def test_the_export_covers_every_point_not_just_the_test_split(tmp_path):
    parent, runs, client = _run_and_collect(_export_config(), _split(200), _pipelines())
    model_run = next(
        run for run in runs
        if run.data.tags.get("model_name") == "Ridge"
        and run.data.tags.get("run_kind") != "ensemble_member"
    )

    child = _download(client, model_run, "predictions/point_predictions.csv", tmp_path)
    # 200 points in the population; the test split is only 40 of them.
    assert len(child) == 200
    assert list(child.columns) == ["uuid", "target_a"]
    assert child["uuid"].iloc[0] == "p0"


def test_the_exported_test_rows_match_the_eval_frame_value_for_value(tmp_path):
    """The check that the ids are RIGHT rather than merely present.

    The export predicts the whole population in one pass; eval_results.csv predicts the test split
    in another. Where they overlap they must agree exactly, or the two passes are not describing
    the same points.
    """
    parent, runs, client = _run_and_collect(_export_config(), _split(200), _pipelines())
    model_run = next(
        run for run in runs
        if run.data.tags.get("model_name") == "Ridge"
        and run.data.tags.get("run_kind") != "ensemble_member"
    )

    child = _download(client, model_run, "predictions/point_predictions.csv", tmp_path / "a")
    evaluation = _download(client, model_run, "eval_results/eval_results.csv", tmp_path / "b")

    # Test rows are the last 40 points, ids p160..p199, in order.
    exported = child.set_index("uuid").loc[[f"p{i}" for i in range(160, 200)], "target_a"]
    assert np.allclose(exported.to_numpy(), evaluation["prediction"].to_numpy())


def test_the_parent_writes_both_the_wide_and_the_long_file(tmp_path):
    config = _export_config()
    parent_id, _runs, client = _run_and_collect(config, _split(), _pipelines())
    _export_parent(parent_id, config)
    parent_run = client.get_run(parent_id)

    wide = _download(client, parent_run, "predictions/point_predictions_wide.csv", tmp_path / "w")
    long_frame = _download(client, parent_run, "predictions/point_predictions_long.csv", tmp_path / "l")

    assert list(wide.columns) == ["uuid", "target_a__Ridge"]
    assert wide["uuid"].is_unique
    assert list(long_frame.columns) == ["uuid", "target", "model", "prediction"]
    assert set(long_frame["model"]) == {"Ridge"}


def test_a_joint_group_gets_one_wide_column_per_target(tmp_path):
    data = _split(targets=("target_a", "target_b"))
    config = _export_config()
    parent_id, _runs, client = _run_and_collect(
        config, data, _pipelines(), target="target_a__target_b",
        targets=["target_a", "target_b"],
    )
    _export_parent(parent_id, config)
    wide = _download(
        client, client.get_run(parent_id), "predictions/point_predictions_wide.csv", tmp_path
    )
    assert set(wide.columns) == {"uuid", "target_a__Ridge", "target_b__Ridge"}


def test_the_ensemble_exports_its_mean_and_no_uncertainty_columns(tmp_path):
    config = _export_config(UNCERTAINTY_ENABLED=True, UNCERTAINTY_N_MEMBERS=3)
    parent_id, _runs, client = _run_and_collect(config, _split(200), _pipelines())
    _export_parent(parent_id, config)

    wide = _download(
        client, client.get_run(parent_id), "predictions/point_predictions_wide.csv", tmp_path
    )
    assert list(wide.columns) == ["uuid", "target_a__Ridge"]
    assert not [c for c in wide.columns if c.endswith(("_std", "_lower", "_upper"))]
    # One row per point, not one per ensemble member.
    assert len(wide) == 200


def test_the_export_is_absent_when_the_switch_is_off():
    config = _config()
    parent_id, runs, client = _run_and_collect(config, _split(), _pipelines())
    _export_parent(parent_id, config)
    model_run = next(run for run in runs if run.data.tags.get("model_name") == "Ridge")

    assert [a.path for a in client.list_artifacts(model_run.info.run_id, "predictions")] == []
    assert [a.path for a in client.list_artifacts(parent_id, "predictions")] == []


def test_a_skipped_model_exports_nothing():
    config = _export_config(EXPORT_POINT_PREDICTIONS_SKIP_MODELS=["Ridge"])
    parent_id, runs, client = _run_and_collect(config, _split(), _pipelines())
    model_run = next(run for run in runs if run.data.tags.get("model_name") == "Ridge")
    assert [a.path for a in client.list_artifacts(model_run.info.run_id, "predictions")] == []
