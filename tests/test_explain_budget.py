"""The sklearn explainer must survive a model shap cannot parse, and refuse one it cannot afford.

Both failures were observed on a real run:

* **XGBoost** raised ``ValueError: could not convert string to float: '[2.7789434E1]'`` - shap 0.48
  cannot parse xgboost 3.3's ``base_score``, now serialized as a bracketed string. The failure comes
  out of ``shap_values``, not the constructor, which is why the fallback has to wrap the call.
* **TabICL** matched no tree or linear marker and fell to KernelExplainer at ~2166 coalitions per
  row. At 500 rows that is ~1.08M forward passes; the run never terminated.
"""

import pathlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from yg_eo_soilnet.explain import build_shap_results
from yg_eo_soilnet.explain.sklearn_explainer import DEFAULT_MAX_EVALS, ExplainBudgetExceeded
from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger


class _UnparsableBooster(BaseEstimator, RegressorMixin):
    """Stands in for XGBoost 3.3 under shap 0.48.

    Named to match the tree marker list so the explainer picks the fast path, then fails the way the
    real one does: from inside shap_values rather than at construction.
    """

    def __init__(self):
        self.fitted_ = False

    def fit(self, X, y):
        self._mean = float(np.mean(y))
        self.fitted_ = True
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self._mean + X[:, 0] * 0.5


# The marker test keys off the class name, so this is what routes it to TreeExplainer.
_UnparsableBooster.__name__ = "XGBRegressorLookalikeBoost"


@pytest.fixture
def frame():
    generator = np.random.default_rng(0)
    features = pd.DataFrame({f"f{index}": generator.normal(0, 1, 80) for index in range(6)})
    target = pd.Series(features["f0"] * 2 + generator.normal(0, 0.1, 80), name="om")
    return features, target


def _pipeline(model, features, target):
    from yg_eo_soilnet.datamodules.scikit.scikit_trainer_utils import PipelineBuilder

    pipeline = PipelineBuilder().build(
        model, False, categorical_cols=[], numeric_cols=list(features.columns)
    )
    return pipeline.fit(features, target)


def _config(**overrides):
    settings = {
        "RANDOM_SEED": 42,
        "EXPLAIN_MAX_SAMPLES": 20,
        "EXPLAIN_BACKGROUND_SAMPLES": 10,
        "EXPLAIN_MAX_EVALS": DEFAULT_MAX_EVALS,
    }
    settings.update(overrides)
    return SimpleNamespace(**settings)


def test_a_model_the_fast_explainer_cannot_parse_still_gets_explained(monkeypatch, frame) -> None:
    """The XGBoost case: TreeExplainer raises, and the result comes from the fallback instead."""
    import shap

    def exploding_tree_explainer(*args, **kwargs):
        explainer = MagicMock()
        explainer.shap_values.side_effect = ValueError(
            "could not convert string to float: '[2.7789434E1]'"
        )
        return explainer

    monkeypatch.setattr(shap, "TreeExplainer", exploding_tree_explainer)

    features, target = frame
    result = build_shap_results(
        config=_config(),
        backend="sklearn",
        fitted_estimator=_pipeline(_UnparsableBooster(), features, target),
        X_train=features,
        X_test=features,
        target="om",
    )[0]

    assert result.values.shape[0] == 20
    assert result.explainer != "TreeExplainer"
    # The downgrade is recorded, so a reader of the plot knows it is an approximation.
    assert result.summary()["explainer"] == result.explainer


def test_the_fast_explainer_is_still_preferred_when_it_works(frame) -> None:
    from sklearn.ensemble import RandomForestRegressor

    features, target = frame
    result = build_shap_results(
        config=_config(),
        backend="sklearn",
        fitted_estimator=_pipeline(RandomForestRegressor(n_estimators=5, random_state=0), features, target),
        X_train=features,
        X_test=features,
        target="om",
    )[0]

    assert result.explainer == "TreeExplainer"


def test_an_unaffordable_model_is_refused_rather_than_run(frame) -> None:
    """The TabICL case. Raised, not silently downscaled, so the skip is visible in the run."""
    features, target = frame

    with pytest.raises(ExplainBudgetExceeded) as excinfo:
        build_shap_results(
            config=_config(EXPLAIN_MAX_EVALS=10),
            backend="sklearn",
            fitted_estimator=_pipeline(_slow_model(), features, target),
            X_train=features,
            X_test=features,
            target="om",
        )

    message = str(excinfo.value)
    assert "EXPLAIN_MAX_EVALS" in message
    assert "model evaluations" in message


def _slow_model():
    class _Opaque(BaseEstimator, RegressorMixin):
        """Matches no tree or linear marker, exactly like TabICLRegressor."""

        def fit(self, X, y):
            self._mean = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(np.asarray(X).shape[0], self._mean)

    return _Opaque()


def test_raising_the_budget_lets_an_expensive_model_through(frame) -> None:
    features, target = frame

    result = build_shap_results(
        config=_config(EXPLAIN_MAX_EVALS=1_000_000),
        backend="sklearn",
        fitted_estimator=_pipeline(_slow_model(), features, target),
        X_train=features,
        X_test=features,
        target="om",
    )[0]

    assert result.values.shape[0] == 20


def test_the_logger_reports_a_budget_skip_as_a_decision_not_a_crash() -> None:
    """A skipped explanation must not read like a failed run, and must not be escalated."""
    summary = ChildRunLogger()._log_shap_artifacts(
        config=SimpleNamespace(EXPLAIN_ENABLED=True, EXPLAIN_FAIL_ON_ERROR=True),
        target="om",
        model_name="TabICL",
        backend="sklearn",
        payload={"fitted_estimator": _Exploder(), "X_train": None, "X_test": None, "target": "om"},
    )

    assert summary["skipped"] is True
    assert summary["logged"] is False
    assert "error" not in summary
    assert "EXPLAIN_MAX_EVALS" in summary["reason"]


class _Exploder:
    """Raises the budget error the moment the explainer touches it."""

    @property
    def named_steps(self):
        raise ExplainBudgetExceeded(
            "model-agnostic SHAP would need about 1,080,000 model evaluations, over the "
            "EXPLAIN_MAX_EVALS budget of 200,000."
        )


# --- the TabICL denylist ----------------------------------------------------


def _skip_summary(config, model_name: str) -> dict:
    """Run the guard alone; the payload is never reached when a model is skipped."""
    return ChildRunLogger()._log_shap_artifacts(
        config=config,
        target="om",
        model_name=model_name,
        backend="sklearn",
        payload={},
    )


def test_tabicl_is_skipped_by_name_with_a_reason() -> None:
    summary = _skip_summary(SimpleNamespace(EXPLAIN_ENABLED=True, EXPLAIN_SKIP_MODELS=["TabICL"]), "TabICL")

    assert summary["skipped"] is True
    assert summary["logged"] is False
    assert "EXPLAIN_SKIP_MODELS" in summary["reason"]
    # It reads as a decision, not a crash.
    assert "error" not in summary


@pytest.mark.parametrize("model_name", ["XGBoost", "GradientBoosting", "Ridge", "PLSRegression"])
def test_every_other_model_is_still_explained(model_name: str) -> None:
    """'Explainability for all models except TabICL' stated directly.

    These get past the guard and fail later on the empty payload, which is what distinguishes
    "was not skipped" from "was skipped".
    """
    summary = _skip_summary(
        SimpleNamespace(EXPLAIN_ENABLED=True, EXPLAIN_SKIP_MODELS=["TabICL"]), model_name
    )

    assert summary.get("skipped") is not True
    assert summary["enabled"] is True


def test_the_allowlist_overrides_the_denylist() -> None:
    """Naming a model explicitly is a deliberate request and must win.

    Otherwise the two settings contradict each other and EXPLAIN_MODELS silently does nothing -
    the user asks for the expensive explanation and gets neither a plot nor a reason.
    """
    summary = _skip_summary(
        SimpleNamespace(
            EXPLAIN_ENABLED=True, EXPLAIN_SKIP_MODELS=["TabICL"], EXPLAIN_MODELS=["TabICL"]
        ),
        "TabICL",
    )

    assert summary.get("skipped") is not True
    assert summary["enabled"] is True


def test_the_off_switch_still_wins_over_everything() -> None:
    summary = _skip_summary(
        SimpleNamespace(EXPLAIN_ENABLED=False, EXPLAIN_MODELS=["TabICL"], EXPLAIN_SKIP_MODELS=[]),
        "TabICL",
    )

    assert summary["enabled"] is False


def test_a_config_without_the_key_skips_nothing() -> None:
    """A SimpleNamespace config predating EXPLAIN_SKIP_MODELS must not start skipping models."""
    summary = _skip_summary(SimpleNamespace(EXPLAIN_ENABLED=True), "TabICL")

    assert summary.get("skipped") is not True


def test_the_shipped_default_excludes_tabicl_and_nothing_else(tmp_path) -> None:
    """No per-run configuration should be needed for the behaviour the user asked for."""
    import yaml

    document = yaml.safe_load(
        (pathlib.Path(__file__).resolve().parents[1] / "configs" / "main_config.yml").read_text()
    )

    assert document["common"]["EXPLAIN_SKIP_MODELS"] == ["TabICL"]
    assert document["common"]["EXPLAIN_MODELS"] == []
