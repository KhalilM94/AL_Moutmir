"""Registration and champion promotion.

The registry was empty: models were logged against runs but never registered, so there were no
versions, no aliases and no `models:/Name@champion` URIs. Registration is now on by default and the
champion alias moves only on a measured improvement.

The promotion rules are where this can go quietly wrong - a rule that promotes on a missing metric,
or churns the alias on a tie, is not visible in any artifact - so each one has its own test.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from yg_eo_soilnet.artifacts import ArtifactLayout
from yg_eo_soilnet.tracking import CHAMPION_METRIC, promote_if_better


class _Client:
    """A registry stub: an aliased version, and the metric on the run behind it."""

    def __init__(self, alias_version=None, incumbent_metric=None, run_missing=False):
        self._alias_version = alias_version
        self._incumbent_metric = incumbent_metric
        self._run_missing = run_missing
        self.alias_calls: list[tuple] = []

    def get_model_version_by_alias(self, name, alias):
        if self._alias_version is None:
            raise RuntimeError(f"no version aliased {alias}")
        return SimpleNamespace(version=self._alias_version, run_id="run-incumbent")

    def get_run(self, run_id):
        if self._run_missing:
            raise RuntimeError("run deleted")
        metrics = {} if self._incumbent_metric is None else {CHAMPION_METRIC: self._incumbent_metric}
        return SimpleNamespace(data=SimpleNamespace(metrics=metrics))

    def set_registered_model_alias(self, name, alias, version):
        self.alias_calls.append((name, alias, version))


def test_the_first_version_becomes_champion() -> None:
    client = _Client(alias_version=None)

    decision = promote_if_better("om_soil_cnn", "1", 4.2, client=client)

    assert decision["promoted"] is True
    assert client.alias_calls == [("om_soil_cnn", "champion", "1")]


def test_a_better_score_promotes() -> None:
    client = _Client(alias_version="1", incumbent_metric=5.30)

    decision = promote_if_better("om_soil_cnn", "2", 4.81, client=client)

    assert decision["promoted"] is True
    assert decision["candidate"] == 4.81
    assert decision["incumbent"] == 5.30
    assert client.alias_calls == [("om_soil_cnn", "champion", "2")]


def test_a_worse_score_keeps_the_incumbent() -> None:
    client = _Client(alias_version="1", incumbent_metric=4.81)

    decision = promote_if_better("om_soil_cnn", "2", 5.62, client=client)

    assert decision["promoted"] is False
    assert client.alias_calls == []
    assert "does not beat" in decision["reason"]


def test_an_equal_score_keeps_the_incumbent() -> None:
    """Re-running the same config must not churn the alias."""
    client = _Client(alias_version="1", incumbent_metric=4.81)

    decision = promote_if_better("om_soil_cnn", "2", 4.81, client=client)

    assert decision["promoted"] is False
    assert client.alias_calls == []


def test_a_version_without_the_metric_is_never_promoted() -> None:
    """A degenerate fit produces no metrics; shipping it because it has no worse score is the
    failure this guards against."""
    client = _Client(alias_version="1", incumbent_metric=4.81)

    decision = promote_if_better("om_soil_cnn", "2", None, client=client)

    assert decision["promoted"] is False
    assert client.alias_calls == []
    assert CHAMPION_METRIC in decision["reason"]


def test_a_non_finite_score_is_never_promoted() -> None:
    client = _Client(alias_version="1", incumbent_metric=4.81)

    assert promote_if_better("om_soil_cnn", "2", float("nan"), client=client)["promoted"] is False
    assert client.alias_calls == []


def test_an_unreadable_incumbent_is_replaced_and_said_so() -> None:
    """A candidate we can score beats one we cannot - but the reason has to be recorded."""
    client = _Client(alias_version="1", run_missing=True)

    decision = promote_if_better("om_soil_cnn", "2", 4.81, client=client)

    assert decision["promoted"] is True
    assert "no readable" in decision["reason"]
    assert client.alias_calls == [("om_soil_cnn", "champion", "2")]


def test_an_incumbent_whose_run_lacks_the_metric_is_replaced() -> None:
    client = _Client(alias_version="1", incumbent_metric=None)

    assert promote_if_better("om_soil_cnn", "2", 4.81, client=client)["promoted"] is True


def test_an_unregistered_model_is_not_promoted() -> None:
    client = _Client()

    decision = promote_if_better("om_soil_cnn", None, 4.81, client=client)

    assert decision["promoted"] is False
    assert client.alias_calls == []


def test_the_decision_records_both_scores_for_audit() -> None:
    client = _Client(alias_version="3", incumbent_metric=5.0)

    decision = promote_if_better("om_soil_cnn", "4", 4.0, client=client)

    assert decision["metric"] == CHAMPION_METRIC
    assert decision["candidate"] == 4.0
    assert decision["incumbent"] == 5.0
    assert decision["candidate_version"] == "4"
    assert decision["incumbent_version"] == "3"


def test_a_higher_is_better_metric_flips_the_comparison() -> None:
    """The direction comes from yg_eo_soilnet.metrics, not from a second copy of the rule here."""
    client = _Client(alias_version="1", incumbent_metric=0.30)

    decision = promote_if_better("om_soil_cnn", "2", 0.55, client=client, metric_name="r2_test")

    assert decision["promoted"] is True


# --- the logger's use of it -------------------------------------------------


def test_the_logger_records_the_decision_without_raising() -> None:
    """A registry that will not take the alias must not lose a finished training run."""
    from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger

    decision = ChildRunLogger()._promote_champion("om", "soil_cnn", {CHAMPION_METRIC: 4.0}, None)

    assert decision["promoted"] is False
    assert "not registered" in decision["reason"]


def test_the_registered_name_matches_the_logged_model_name() -> None:
    """Both families register under the same convention, so a target's versions accumulate in one
    place regardless of which family produced them."""
    assert ArtifactLayout.logged_model_name("organic_matter_g_kg", "soil_cnn") == (
        "organic_matter_g_kg_soil_cnn"
    )


# --- registration wiring ----------------------------------------------------


@pytest.mark.parametrize("enabled", [True, False])
def test_the_lightning_path_honours_the_registration_switch(monkeypatch, tmp_path, enabled) -> None:
    import mlflow.pyfunc
    import mlflow.pytorch

    import yg_eo_soilnet.logger.mlflow_loggers as loggers

    captured: dict = {}
    monkeypatch.setattr(mlflow.pytorch, "save_model", lambda *a, **k: None)
    monkeypatch.setattr(loggers.mlflow, "set_tags", MagicMock())
    monkeypatch.setattr(
        mlflow.pyfunc,
        "log_model",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(registered_model_version="1"),
    )

    checkpoint = tmp_path / "a.ckpt"
    checkpoint.write_bytes(b"x")

    loggers.ChildRunLogger()._log_lightning_serialized_model(
        model=SimpleNamespace(),
        model_name="soil_cnn",
        target="om",
        bundle=None,
        best_model_path=str(checkpoint),
        config=SimpleNamespace(MLFLOW_REGISTER_MODELS=enabled),
    )

    expected = "om_soil_cnn" if enabled else None
    assert captured["registered_model_name"] == expected


# --- a failed model log cannot look like a success ---------------------------
# This is what let the auxiliary-column regression run unnoticed: the model failed to log, nothing
# was registered, and the run still finished looking healthy because the only evidence was inside
# meta/run_summary.json.


def _run_lightning_child(monkeypatch, *, log_model_raises: bool):
    import mlflow.pyfunc
    import mlflow.pytorch
    import pandas as pd

    import yg_eo_soilnet.logger.mlflow_loggers as loggers

    tags: dict = {}
    monkeypatch.setattr(loggers.mlflow, "set_tags", lambda values: tags.update(values))
    monkeypatch.setattr(loggers.mlflow, "log_params", MagicMock())
    monkeypatch.setattr(loggers.mlflow, "log_metric", MagicMock())
    monkeypatch.setattr(loggers.mlflow, "log_artifact", MagicMock())
    monkeypatch.setattr("yg_eo_soilnet.artifacts.mlflow.log_artifact", MagicMock())
    monkeypatch.setattr(mlflow.pytorch, "save_model", lambda *a, **k: None)

    def log_model(**kwargs):
        if log_model_raises:
            raise ValueError("Batch carries 0 lab column(s) but this model resolved 24")
        return SimpleNamespace(registered_model_version="1")

    monkeypatch.setattr(mlflow.pyfunc, "log_model", log_model)

    summaries: list[dict] = []
    logger = loggers.ChildRunLogger()
    monkeypatch.setattr(logger, "_write_json_artifact", lambda payload, *a, **k: summaries.append(payload))
    monkeypatch.setattr(logger, "_promote_champion", lambda *a, **k: {"promoted": False})

    logger.log_lightning_child_run(
        config=SimpleNamespace(EXPLAIN_ENABLED=False),
        target="om",
        model_name="soil_cnn",
        evaluation_df=pd.DataFrame({"om": [1.0, 2.0, 3.0], "prediction": [1.1, 2.1, 2.9]}),
        validation_metrics={},
        test_metrics={},
        model=SimpleNamespace(),
    )
    return tags, summaries[-1]


def test_a_failed_model_log_is_tagged_on_the_run(monkeypatch) -> None:
    """Visible in the MLflow run list, not only inside an artifact nobody opens."""
    tags, summary = _run_lightning_child(monkeypatch, log_model_raises=True)

    assert tags["model_logged"] == "false"
    assert "Batch carries 0 lab column" in tags["model_logging_error"]
    assert summary["serialized_model_logged"] is False


def test_a_failed_model_log_does_not_kill_the_run(monkeypatch) -> None:
    """A fitted model should not be lost to a packaging problem."""
    _tags, summary = _run_lightning_child(monkeypatch, log_model_raises=True)

    assert summary["metrics"]["rmse_test"] > 0  # the run still recorded its results


def test_a_successful_model_log_is_tagged_too(monkeypatch) -> None:
    """The absence of a tag is not evidence, so the success case is tagged as well."""
    tags, summary = _run_lightning_child(monkeypatch, log_model_raises=False)

    assert tags["model_logged"] == "true"
    assert "model_logging_error" not in tags
    assert summary["serialized_model_logged"] is True


def test_a_failed_model_log_warns(monkeypatch, caplog) -> None:
    import logging

    with caplog.at_level(logging.WARNING):
        _run_lightning_child(monkeypatch, log_model_raises=True)

    assert any("NO servable model" in record.getMessage() for record in caplog.records)
    # The actual cause must be in the line, not just "something failed".
    assert any("Batch carries 0 lab column" in record.getMessage() for record in caplog.records)
