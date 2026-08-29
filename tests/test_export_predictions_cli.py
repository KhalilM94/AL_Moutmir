"""Backfilling the per-point export onto a finished run.

What makes the backfill possible is that a finished run kept its models: a sklearn child records a
loadable model URI, and a Lightning child's checkpoint is self-describing. The features, though, are
NOT kept - they are rebuilt from the configured source - so these tests concentrate on the two
things that can go wrong because of that: the rebuilt data silently not matching what the run
trained on, and the rebuild writing over the run's own record of its split.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import export_predictions as ep


def _args(**overrides) -> SimpleNamespace:
    base = dict(
        parent_run_id="parent",
        config_path="configs/main_config.yml",
        models=None,
        skip_models=None,
        allow_population_drift=False,
        dry_run=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _logger():
    import logging

    logger = logging.getLogger("backfill-test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


# --- arguments -------------------------------------------------------------


def test_the_parent_run_id_is_required():
    with pytest.raises(SystemExit):
        ep.parse_args([])


def test_parse_args_defaults_match_the_other_clis():
    args = ep.parse_args(["--parent-run-id", "abc"])
    assert args.parent_run_id == "abc"
    assert args.config_path == "configs/main_config.yml"
    assert args.allow_population_drift is False
    assert args.dry_run is False


# --- the export config proxy ----------------------------------------------


def test_running_the_command_is_the_opt_in():
    """The config switch governs training runs; typing the command is the opt-in here."""
    config = SimpleNamespace(
        EXPORT_POINT_PREDICTIONS=False,
        EXPORT_POINT_PREDICTIONS_MODELS=[],
        EXPORT_POINT_PREDICTIONS_SKIP_MODELS=["TabICL"],
    )
    proxy = ep._export_config(config, _args())
    assert proxy.EXPORT_POINT_PREDICTIONS is True
    # ...and the original is not mutated.
    assert config.EXPORT_POINT_PREDICTIONS is False


def test_a_child_failure_is_raised_rather_than_swallowed():
    # During training a failed export is recorded and the run kept, because the model is the
    # deliverable. Here the export IS the deliverable, so it has to surface.
    proxy = ep._export_config(SimpleNamespace(), _args())
    assert proxy.EXPORT_POINT_PREDICTIONS_FAIL_ON_ERROR is True


def test_the_cli_model_filters_override_the_config_lists():
    config = SimpleNamespace(
        EXPORT_POINT_PREDICTIONS_MODELS=[], EXPORT_POINT_PREDICTIONS_SKIP_MODELS=["TabICL"]
    )
    proxy = ep._export_config(config, _args(models="Ridge,XGBoost", skip_models="PLSRegression"))
    assert proxy.EXPORT_POINT_PREDICTIONS_MODELS == ["Ridge", "XGBoost"]
    assert proxy.EXPORT_POINT_PREDICTIONS_SKIP_MODELS == ["PLSRegression"]


def test_the_config_skip_list_survives_when_the_cli_names_none():
    config = SimpleNamespace(
        EXPORT_POINT_PREDICTIONS_MODELS=[], EXPORT_POINT_PREDICTIONS_SKIP_MODELS=["TabICL"]
    )
    proxy = ep._export_config(config, _args())
    assert proxy.EXPORT_POINT_PREDICTIONS_SKIP_MODELS == ["TabICL"]


# --- the drift guard -------------------------------------------------------


def _recorded(tmp_path, columns, point_ids):
    """Stand in for the run's logged data_splits artifacts."""
    test_path = tmp_path / "X_test.parquet"
    split_path = tmp_path / "split_assignments.parquet"
    pd.DataFrame({"point_id": point_ids[:1], **{c: [0.0] for c in columns}}).to_parquet(test_path)
    pd.DataFrame({"point_id": point_ids, "split": ["train"] * len(point_ids)}).to_parquet(split_path)
    return str(test_path), str(split_path)


def _patch_downloads(monkeypatch, test_path, split_path):
    def fake(run_id, artifact_path):
        if artifact_path.endswith("X_test.parquet"):
            return test_path
        if artifact_path.endswith("split_assignments.parquet"):
            return split_path
        return None

    monkeypatch.setattr(ep, "_download", fake)


def test_matching_data_passes_the_check(monkeypatch, tmp_path):
    ids = ["a", "b", "c"]
    _patch_downloads(monkeypatch, *_recorded(tmp_path, ["f1", "f2"], ids))

    report = ep.check_for_drift(
        "parent",
        pd.DataFrame({"f1": [0.0] * 3, "f2": [0.0] * 3}),
        pd.Series(ids),
        allow_population_drift=False,
        logger=_logger(),
    )
    assert report["checked"] is True
    assert report["n_points_added"] == 0 and report["n_points_removed"] == 0


def test_a_changed_feature_column_aborts_and_names_it(monkeypatch, tmp_path):
    """The models cannot legitimately predict on a different column set, so this never relaxes."""
    ids = ["a", "b"]
    _patch_downloads(monkeypatch, *_recorded(tmp_path, ["f1", "f2"], ids))

    with pytest.raises(SystemExit, match="f2"):
        ep.check_for_drift(
            "parent",
            pd.DataFrame({"f1": [0.0] * 2, "f3": [0.0] * 2}),
            pd.Series(ids),
            allow_population_drift=False,
            logger=_logger(),
        )


def test_the_feature_check_is_not_relaxed_by_the_drift_flag(monkeypatch, tmp_path):
    ids = ["a", "b"]
    _patch_downloads(monkeypatch, *_recorded(tmp_path, ["f1", "f2"], ids))

    with pytest.raises(SystemExit, match="cannot predict"):
        ep.check_for_drift(
            "parent",
            pd.DataFrame({"f1": [0.0] * 2}),
            pd.Series(ids),
            allow_population_drift=True,
            logger=_logger(),
        )


def test_an_added_point_aborts_under_strict(monkeypatch, tmp_path):
    _patch_downloads(monkeypatch, *_recorded(tmp_path, ["f1"], ["a", "b"]))

    with pytest.raises(SystemExit, match="1 point\\(s\\) added"):
        ep.check_for_drift(
            "parent",
            pd.DataFrame({"f1": [0.0] * 3}),
            pd.Series(["a", "b", "c"]),
            allow_population_drift=False,
            logger=_logger(),
        )


def test_an_added_point_is_only_a_warning_once_drift_is_allowed(monkeypatch, tmp_path):
    _patch_downloads(monkeypatch, *_recorded(tmp_path, ["f1"], ["a", "b"]))

    report = ep.check_for_drift(
        "parent",
        pd.DataFrame({"f1": [0.0] * 3}),
        pd.Series(["a", "b", "c"]),
        allow_population_drift=True,
        logger=_logger(),
    )
    assert report["n_points_added"] == 1


def test_a_run_that_cannot_be_verified_stops_under_strict(monkeypatch):
    """Strict means "prove it matches". A legacy run with no split_assignments cannot."""
    monkeypatch.setattr(ep, "_download", lambda run_id, artifact_path: None)

    with pytest.raises(SystemExit, match="cannot be checked"):
        ep.check_for_drift(
            "parent",
            pd.DataFrame({"f1": [0.0]}),
            pd.Series(["a"]),
            allow_population_drift=False,
            logger=_logger(),
        )


def test_an_unverifiable_run_can_be_forced(monkeypatch):
    monkeypatch.setattr(ep, "_download", lambda run_id, artifact_path: None)

    report = ep.check_for_drift(
        "parent",
        pd.DataFrame({"f1": [0.0]}),
        pd.Series(["a"]),
        allow_population_drift=True,
        logger=_logger(),
    )
    assert report["checked"] is False


# --- per-child dispatch ----------------------------------------------------


def _run(**tags):
    params = tags.pop("params", {})
    return SimpleNamespace(
        info=SimpleNamespace(run_id=tags.pop("run_id", "child")),
        data=SimpleNamespace(tags=tags, params=params),
    )


def test_a_single_lightning_model_is_not_skipped_as_an_ensemble(monkeypatch):
    # No uncertainty_n_members means one model, which IS recoverable from its checkpoint.
    monkeypatch.setattr(
        ep, "lightning_predictor", lambda run, config, logger: ((lambda: np.zeros((2, 1))), ["a", "b"], ["clay_pct"])
    )
    recorded = {}
    monkeypatch.setattr(
        ep.ChildRunLogger,
        "_log_point_predictions",
        lambda self, **kwargs: recorded.update(kwargs) or {"n_points": 2},
    )
    monkeypatch.setattr(ep.mlflow, "start_run", lambda **kwargs: __import__("contextlib").nullcontext())

    outcome = ep.backfill_child(
        _run(model_name="soil_cnn", framework="lightning", target="clay_pct"),
        client=None,
        config=SimpleNamespace(),
        export_config=SimpleNamespace(),
        features=pd.DataFrame(),
        point_ids=pd.Series(dtype=object),
        logger=_logger(),
        checkpoint_dir="lightning_logs",
    )
    assert "skipped" not in outcome
    assert recorded["point_ids"] == ["a", "b"]


def test_a_sklearn_child_predicts_on_the_rebuilt_features(monkeypatch):
    features = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    point_ids = pd.Series(["a", "b", "c"])
    monkeypatch.setattr(
        ep, "sklearn_predictor", lambda run, feats: ((lambda: np.arange(len(feats), dtype=float)), len(feats))
    )
    recorded = {}
    monkeypatch.setattr(
        ep.ChildRunLogger,
        "_log_point_predictions",
        lambda self, **kwargs: recorded.update(kwargs) or {"n_points": 3},
    )
    monkeypatch.setattr(ep.mlflow, "start_run", lambda **kwargs: __import__("contextlib").nullcontext())

    outcome = ep.backfill_child(
        _run(model_name="Ridge", framework="sklearn", target="clay_pct"),
        client=None,
        config=SimpleNamespace(),
        export_config=SimpleNamespace(),
        features=features,
        point_ids=point_ids,
        logger=_logger(),
        checkpoint_dir="lightning_logs",
    )
    assert outcome["n_points"] == 3
    assert recorded["target_names"] == ["clay_pct"]
    assert list(recorded["point_ids"]) == ["a", "b", "c"]


def test_a_joint_child_reports_every_target_in_its_group(monkeypatch):
    monkeypatch.setattr(ep, "sklearn_predictor", lambda run, feats: ((lambda: np.zeros((1, 2))), 1))
    recorded = {}
    monkeypatch.setattr(
        ep.ChildRunLogger,
        "_log_point_predictions",
        lambda self, **kwargs: recorded.update(kwargs) or {"n_points": 1},
    )
    monkeypatch.setattr(ep.mlflow, "start_run", lambda **kwargs: __import__("contextlib").nullcontext())

    ep.backfill_child(
        _run(model_name="Ridge", framework="sklearn", target="clay_pct__sand_pct"),
        client=None,
        config=SimpleNamespace(),
        export_config=SimpleNamespace(),
        features=pd.DataFrame({"f1": [0.0]}),
        point_ids=pd.Series(["a"]),
        logger=_logger(),
        checkpoint_dir="lightning_logs",
    )
    assert recorded["target_names"] == ["clay_pct", "sand_pct"]


def test_a_model_that_cannot_be_reloaded_says_which_uris_were_tried(monkeypatch):
    import mlflow.sklearn

    monkeypatch.setattr(ep, "_run_summary", lambda run_id: {})
    monkeypatch.setattr(
        mlflow.sklearn, "load_model", lambda uri: (_ for _ in ()).throw(OSError("not there"))
    )

    with pytest.raises(SystemExit, match="runs:/child/"):
        ep.sklearn_predictor(
            _run(model_name="Ridge", framework="sklearn", target="clay_pct"), pd.DataFrame()
        )


# --- recovering a Lightning ensemble's members ------------------------------


def _version_dir(root, version, target, val_losses):
    """A lightning_logs/version_N with the two files the matcher reads, plus a stub checkpoint."""
    import yaml

    directory = root / f"version_{version}"
    (directory / "checkpoints").mkdir(parents=True)
    (directory / "checkpoints" / "epoch=1-step=2.ckpt").write_bytes(b"stub")
    (directory / "hparams.yaml").write_text(yaml.safe_dump({"target_names": [target]}))
    pd.DataFrame({"epoch": range(len(val_losses)), "val_loss": val_losses}).to_csv(
        directory / "metrics.csv", index=False
    )
    return directory


class _FakeClient:
    """Just the two calls match_member_checkpoints makes."""

    def __init__(self, members, histories):
        self._members = members
        self._histories = histories

    def search_runs(self, experiment_ids, filter_string):
        return self._members

    def get_metric_history(self, run_id, key):
        return [SimpleNamespace(value=v) for v in self._histories[run_id]]


def _member(index, run_id):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, experiment_id="0"),
        data=SimpleNamespace(
            tags={"run_kind": "ensemble_member"},
            params={"ensemble_member": str(index), "ensemble_seed": str(42 + index * 1000)},
        ),
    )


def _child(run_id="child"):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, experiment_id="0"),
        data=SimpleNamespace(tags={"target": "clay_pct", "model_name": "soil_cnn"}, params={}),
    )


def test_members_are_matched_by_validation_loss_not_by_position(monkeypatch, tmp_path):
    """Ordering is an accident of when the run happened; val_loss is a property of the fit.

    The two version dirs here are deliberately in the OPPOSITE order to the members, so a matcher
    that paired them by position would get both wrong and one that reads val_loss gets both right.
    """
    _version_dir(tmp_path, 10, "clay_pct", [0.9, 0.55])   # -> member 1
    _version_dir(tmp_path, 11, "clay_pct", [0.8, 0.44])   # -> member 0
    monkeypatch.setattr(ep, "_download", lambda run_id, artifact_path: None)

    client = _FakeClient(
        [_member(0, "m0"), _member(1, "m1")],
        {"m0": [0.7, 0.44], "m1": [0.9, 0.55]},
    )
    matched = ep.match_member_checkpoints(client, _child(), str(tmp_path), ["clay_pct"])

    assert [r["ensemble_member"] for r in matched] == [0, 1]
    assert "version_11" in matched[0]["version_dir"]
    assert "version_10" in matched[1]["version_dir"]


def test_a_version_dir_for_another_target_is_never_a_candidate(monkeypatch, tmp_path):
    _version_dir(tmp_path, 10, "sand_pct", [0.44])        # same loss, wrong target
    monkeypatch.setattr(ep, "_download", lambda run_id, artifact_path: None)

    client = _FakeClient([_member(0, "m0")], {"m0": [0.44]})
    matched = ep.match_member_checkpoints(client, _child(), str(tmp_path), ["clay_pct"])
    assert matched[0]["checkpoint"] is None


def test_a_member_matching_nothing_is_reported_not_guessed(monkeypatch, tmp_path):
    _version_dir(tmp_path, 10, "clay_pct", [0.44])
    monkeypatch.setattr(ep, "_download", lambda run_id, artifact_path: None)

    client = _FakeClient([_member(0, "m0")], {"m0": [0.99]})
    matched = ep.match_member_checkpoints(client, _child(), str(tmp_path), ["clay_pct"])
    assert matched[0]["checkpoint"] is None


def test_one_checkpoint_is_never_claimed_by_two_members(monkeypatch, tmp_path):
    # Two members that genuinely converged to the same loss: the first takes the dir, the second
    # must come up empty rather than double-count it.
    _version_dir(tmp_path, 10, "clay_pct", [0.44])
    monkeypatch.setattr(ep, "_download", lambda run_id, artifact_path: None)

    client = _FakeClient([_member(0, "m0"), _member(1, "m1")], {"m0": [0.44], "m1": [0.44]})
    matched = ep.match_member_checkpoints(client, _child(), str(tmp_path), ["clay_pct"])
    assert matched[0]["checkpoint"] is not None
    assert matched[1]["checkpoint"] is None


def test_a_member_that_logged_its_own_checkpoint_needs_no_scavenging(monkeypatch, tmp_path):
    """Runs trained after the structural fix take this path and never touch lightning_logs."""
    monkeypatch.setattr(ep, "_download", lambda run_id, artifact_path: "/from/mlflow.ckpt")

    client = _FakeClient([_member(0, "m0")], {"m0": [0.44]})
    matched = ep.match_member_checkpoints(client, _child(), str(tmp_path), ["clay_pct"])
    assert matched[0]["source"] == "mlflow"
    assert matched[0]["checkpoint"] == "/from/mlflow.ckpt"


def test_an_incomplete_ensemble_is_skipped_with_the_counts(monkeypatch):
    monkeypatch.setattr(
        ep,
        "match_member_checkpoints",
        lambda client, run, directory, targets: [
            {"ensemble_member": 0, "checkpoint": "a.ckpt"},
            {"ensemble_member": 1, "checkpoint": None},
        ],
    )
    outcome = ep.backfill_child(
        _run(
            model_name="soil_cnn",
            framework="lightning",
            target="clay_pct",
            params={"uncertainty_n_members": "5"},
        ),
        client=None,
        config=SimpleNamespace(),
        export_config=SimpleNamespace(),
        features=pd.DataFrame(),
        point_ids=pd.Series(dtype=object),
        logger=_logger(),
        checkpoint_dir="lightning_logs",
    )
    assert "only 1 checkpoint(s) could be recovered" in outcome["skipped"]
    assert "would not be the ensemble" in outcome["skipped"]


def test_member_recovery_can_be_turned_off():
    outcome = ep.backfill_child(
        _run(
            model_name="soil_cnn",
            framework="lightning",
            target="clay_pct",
            params={"uncertainty_n_members": "5"},
        ),
        client=None,
        config=SimpleNamespace(),
        export_config=SimpleNamespace(),
        features=pd.DataFrame(),
        point_ids=pd.Series(dtype=object),
        logger=_logger(),
        checkpoint_dir=None,
    )
    assert "member recovery is disabled" in outcome["skipped"]


def test_the_recovered_value_is_the_mean_of_the_members(monkeypatch):
    """Three members predicting 1, 2 and 3 must export 2 - not one member, not a sum."""
    constants = iter([1.0, 2.0, 3.0])

    class _Predictor:
        def __init__(self, model):
            self.preprocessing_state = {"target_names": ["clay_pct"]}
            self._value = next(constants)

        def predict(self, bundle):
            return np.full((4, 1), self._value)

    import yg_eo_soilnet.serving.sequence_predictor as sp

    monkeypatch.setattr(sp, "SoilSequencePredictor", _Predictor)
    monkeypatch.setattr(ep, "_restore_lightning_model", lambda ckpt, name, config: object())
    monkeypatch.setattr(
        ep, "sequence_bundle_for", lambda name, config, logger: SimpleNamespace(point_ids=list("abcd"))
    )

    predict, ids, targets = ep.lightning_ensemble_predictor(
        _run(model_name="soil_cnn", framework="lightning", target="clay_pct"),
        SimpleNamespace(),
        _logger(),
        [{"checkpoint": f"{i}.ckpt"} for i in range(3)],
    )
    assert targets == ["clay_pct"] and ids == list("abcd")
    assert np.allclose(predict(), 2.0)


def test_the_sequence_bundle_is_built_once_and_reused(monkeypatch):
    builds = []

    class _Builder:
        def __init__(self, config, logger, data_manager):
            pass

        def build(self, sequence_data_args=None):
            builds.append(1)
            return SimpleNamespace(point_ids=["a"])

    import yg_eo_soilnet.datamodules.sequence.sequence_builder as sb

    monkeypatch.setattr(sb, "SoilSequenceBuilder", _Builder)
    monkeypatch.setattr(ep, "_BUNDLE_CACHE", {})
    config = SimpleNamespace(LIGHTNING_MODEL_REGISTRY={"soil_cnn": {}})

    for _ in range(5):
        ep.sequence_bundle_for("soil_cnn", config, _logger())
    assert len(builds) == 1
