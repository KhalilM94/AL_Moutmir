"""EXPLAIN_ENABLED has to be a real off-switch, not a plot suppressor.

shap 0.48 pulls in numba and is slow to import, and the suite runs with
``filterwarnings = ["error"]``, so a run that asked for no explanations must not import it at all.
The `"shap" not in sys.modules` assertion here is what fails if anyone hoists the import to module
scope in mlflow_loggers or explain/__init__.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger


@pytest.fixture
def logger() -> ChildRunLogger:
    return ChildRunLogger()


def test_disabled_switch_returns_without_logging_anything(logger, monkeypatch) -> None:
    log_artifact = MagicMock()
    monkeypatch.setattr("yg_eo_soilnet.artifacts.mlflow.log_artifact", log_artifact)

    summary = logger._log_shap_artifacts(
        config=SimpleNamespace(EXPLAIN_ENABLED=False),
        target="om",
        model_name="soil_cnn",
        backend="lightning",
        payload={},
    )

    assert summary == {"enabled": False, "reason": "EXPLAIN_ENABLED is false"}
    log_artifact.assert_not_called()


def test_disabled_switch_does_not_import_shap(logger, monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "shap", raising=False)

    logger._log_shap_artifacts(
        config=SimpleNamespace(EXPLAIN_ENABLED=False),
        target="om",
        model_name="soil_cnn",
        backend="lightning",
        payload={"model": object(), "bundle": object(), "target": "om"},
    )

    assert "shap" not in sys.modules, (
        "shap was imported on a run with EXPLAIN_ENABLED false; keep the import inside the "
        "explainer functions and the guard ahead of it"
    )


def test_explain_models_allowlist_skips_models_it_does_not_name(logger, monkeypatch) -> None:
    log_artifact = MagicMock()
    monkeypatch.setattr("yg_eo_soilnet.artifacts.mlflow.log_artifact", log_artifact)

    summary = logger._log_shap_artifacts(
        config=SimpleNamespace(EXPLAIN_ENABLED=True, EXPLAIN_MODELS=["XGBoost"]),
        target="om",
        model_name="TabICL",
        backend="sklearn",
        payload={},
    )

    assert summary["enabled"] is False
    assert "TabICL" in summary["reason"]
    log_artifact.assert_not_called()


def test_a_named_model_is_not_skipped(logger) -> None:
    """The allowlist must not skip the model it names; the explainer then fails on the empty
    payload, which is recorded rather than raised."""
    summary = logger._log_shap_artifacts(
        config=SimpleNamespace(EXPLAIN_ENABLED=True, EXPLAIN_MODELS=["XGBoost"]),
        target="om",
        model_name="XGBoost",
        backend="sklearn",
        payload={},
    )

    assert summary["enabled"] is True
    assert summary["logged"] is False


def test_explainer_failure_is_recorded_not_raised(logger) -> None:
    """Losing a finished training run because an explainer choked is a bad trade."""
    summary = logger._log_shap_artifacts(
        config=SimpleNamespace(EXPLAIN_ENABLED=True),
        target="om",
        model_name="soil_cnn",
        backend="not_a_backend",
        payload={},
    )

    assert summary["enabled"] is True
    assert summary["logged"] is False
    assert "ValueError" in summary["error"]


def test_explain_fail_on_error_re_raises(logger) -> None:
    with pytest.raises(ValueError, match="Unknown explain backend"):
        logger._log_shap_artifacts(
            config=SimpleNamespace(EXPLAIN_ENABLED=True, EXPLAIN_FAIL_ON_ERROR=True),
            target="om",
            model_name="soil_cnn",
            backend="not_a_backend",
            payload={},
        )


def test_switch_defaults_to_on_when_the_config_predates_the_key(logger) -> None:
    """A SimpleNamespace config with no EXPLAIN_* attributes at all must not crash."""
    summary = logger._log_shap_artifacts(
        config=SimpleNamespace(),
        target="om",
        model_name="soil_cnn",
        backend="not_a_backend",
        payload={},
    )

    assert summary["enabled"] is True


# The switch's config plumbing is covered in tests/test_config.py, beside the base_config_paths
# fixture that builds a complete config tree.
