"""SHAP for the sklearn path: explain the fitted estimator on POST-preprocessing features.

Two things here are easy to get wrong and both change what the plot means.

**Explain the estimator, not the pipeline.** One-hot and ordinal encoding happen inside the
pipeline's ColumnTransformer, so the estimator never sees the raw column names. Feature names have
to come from ``preprocessor.get_feature_names_out()`` and the data has to be the transformed matrix,
or a categorical column with eight levels silently gets attributed to whatever numeric column now
sits at its index.

**Mind the target space.** When the target is log-transformed the pipeline's final step is a
``TransformedTargetRegressor``, and the thing that can actually be explained is its inner
``regressor_``, which predicts ``10 * log1p(y)``. The SHAP values are contributions in that space,
which is why every result carries an ``output_space``.

``import shap`` is deliberately inside the function. The module must stay importable - and cheap -
when EXPLAIN_ENABLED is false.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from yg_eo_soilnet.explain.result import LOG1P_X10, ORIGINAL_UNITS, ShapResult

_TREE_MARKERS = ("tree", "forest", "boost", "xgb", "lightgbm", "catboost")
_LINEAR_MARKERS = ("linear", "ridge", "lasso", "elasticnet", "ols", "bayesianridge")

# Ceiling on model evaluations for the model-agnostic path, in units of "rows scored".
# The number that motivated it: TabICL matches no marker, so it fell to KernelExplainer, whose
# default is 2*n_features + 2048 coalitions PER ROW. At 59 features and 500 explained rows that is
# ~1.08M forward passes through a transformer - the run never finished.
DEFAULT_MAX_EVALS = 200_000

# Coalitions per row for the agnostic explainer. Permutation SHAP needs 2*n_features + 1 to be
# exact, so this is expressed relative to the feature count and only clamped by the budget.
_AGNOSTIC_EVALS_PER_ROW = lambda n_features: 2 * n_features + 1  # noqa: E731


class ExplainBudgetExceeded(RuntimeError):
    """The model-agnostic explainer would cost more evaluations than the budget allows.

    Raised rather than silently degrading, so ChildRunLogger records the estimate and the budget in
    meta/run_summary.json and the skip is visible in the run rather than inferred from a gap.
    """


def _looks_like(estimator: Any, markers: tuple[str, ...]) -> bool:
    """Same name/module marker test PipelineBuilder._is_tree_based_model uses to pick an encoder.

    Matched by name rather than by isinstance so that xgboost and catboost, which are not sklearn
    subclasses, are recognised without importing them.
    """
    name = estimator.__class__.__name__.lower()
    module = estimator.__class__.__module__.lower()
    return any(marker in name or marker in module for marker in markers)


def _densify(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float64)


def _subsample(matrix: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or matrix.shape[0] <= limit:
        return matrix
    generator = np.random.default_rng(seed)
    picked = generator.choice(matrix.shape[0], size=limit, replace=False)
    return matrix[np.sort(picked)]


def _unwrap(fitted_estimator: Any) -> tuple[Any, Any, str]:
    """``(preprocessor, estimator, output_space)`` from a pipeline built by PipelineBuilder."""
    from sklearn.compose import TransformedTargetRegressor

    named_steps = getattr(fitted_estimator, "named_steps", None)
    if named_steps is None or "model" not in named_steps:
        raise TypeError(
            "Expected a fitted Pipeline with a 'model' step as built by PipelineBuilder.build, got "
            f"{type(fitted_estimator).__name__}"
        )

    preprocessor = named_steps.get("preprocessor")
    estimator = named_steps["model"]
    output_space = ORIGINAL_UNITS

    if isinstance(estimator, TransformedTargetRegressor):
        # regressor_ (fitted) rather than regressor (the unfitted template).
        estimator = estimator.regressor_
        output_space = LOG1P_X10

    return preprocessor, estimator, output_space


def _as_values(raw: Any) -> np.ndarray:
    """SHAP output as ``(n_samples, n_features)`` or ``(n_samples, n_features, n_outputs)``.

    A multi-output explainer appends an output axis. It is kept: under a joint fit the estimator
    has one output per target, and taking ``[..., 0]`` - which this used to do unconditionally -
    reported the first target's attributions for every target. ``shap.Explainer`` returns an
    Explanation object rather than an array, hence the ``.values`` probe.
    """
    return np.asarray(getattr(raw, "values", raw), dtype=np.float64)


def _output_slice(values: np.ndarray, index: int) -> np.ndarray:
    """One output's ``(n_samples, n_features)`` block."""
    return values[..., index] if values.ndim == 3 else values


def _base_value_at(base_value: Any, index: int) -> float:
    array = np.asarray(base_value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return 0.0
    return float(array[index]) if index < array.size else float(array[0])


def _agnostic(shap, estimator, background_matrix, explain_matrix, max_evals):
    """Model-agnostic SHAP, budgeted.

    Used for anything that is neither a tree nor a linear model, and as the fallback when the fast
    exact explainer refuses a model it should have handled.
    """
    n_rows, n_features = explain_matrix.shape
    evals_per_row = _AGNOSTIC_EVALS_PER_ROW(n_features)
    estimated = n_rows * evals_per_row

    if estimated > max_evals:
        raise ExplainBudgetExceeded(
            f"model-agnostic SHAP would need about {estimated:,} model evaluations "
            f"({n_rows} rows x {evals_per_row} per row at {n_features} features), over the "
            f"EXPLAIN_MAX_EVALS budget of {max_evals:,}. Lower EXPLAIN_MAX_SAMPLES, or raise "
            f"EXPLAIN_MAX_EVALS if you want to pay for it."
        )

    # PermutationExplainer explicitly, not shap.Explainer's auto-selection. Auto-selection picks the
    # Exact explainer when the feature count is small, and Exact needs 2**n_features evaluations per
    # row - so the estimate above would be wrong by orders of magnitude in exactly the case it looks
    # safest, and shap would then refuse the budget we handed it. A budget is only meaningful
    # against a known cost model.
    explainer = shap.PermutationExplainer(estimator.predict, background_matrix)
    # silent=True, or shap writes one progress line per explained row to stderr - several hundred
    # per model, which buries the training log. It is a __call__ argument, not a constructor one.
    values = explainer(explain_matrix, max_evals=evals_per_row, silent=True)
    return _as_values(values), explainer


def _explain(shap, *, estimator, explain_matrix, background_matrix, max_evals):
    """``(values, base_value, explainer_name)`` from the cheapest explainer that actually works.

    TreeExplainer and LinearExplainer are exact and fast, so they are tried first. The catch is that
    they can fail on a model they nominally support: shap 0.48 cannot parse xgboost 3.3's
    ``base_score``, which is now serialized as a bracketed string like ``'[2.7789434E1]'``, and it
    raises ``ValueError: could not convert string to float`` - **inside shap_values, not at
    construction**, which is why the call is inside the try and not just the constructor.

    Falling back keeps XGBoost explainable across that version skew. The explainer actually used is
    returned and recorded in shap_summary.json, so the downgrade is visible rather than silent.
    """
    fast = None
    if _looks_like(estimator, _TREE_MARKERS):
        fast = ("TreeExplainer", lambda: shap.TreeExplainer(estimator))
    elif _looks_like(estimator, _LINEAR_MARKERS):
        fast = ("LinearExplainer", lambda: shap.LinearExplainer(estimator, background_matrix))

    if fast is not None:
        name, build = fast
        try:
            explainer = build()
            values = _as_values(explainer.shap_values(explain_matrix))
            return values, _base_value(explainer), name
        except ExplainBudgetExceeded:
            raise
        except Exception:
            pass  # fall through; the reason is recorded via the explainer name that ends up used

    values, explainer = _agnostic(shap, estimator, background_matrix, explain_matrix, max_evals)
    return values, _base_value(explainer), type(explainer).__name__


def _blocks_from_feature_names(feature_names: list[str]) -> list[str]:
    """Group post-preprocessing features by the ColumnTransformer prefix they already carry.

    ``get_feature_names_out`` emits ``num__clay_pct`` and ``cat__texture_loam`` because
    PipelineBuilder names its two transformers 'num' and 'cat'. Reusing that split gives the block
    plot something to say - previously every sklearn feature was labelled "features" and the chart
    was a single bar.

    Worth having because the two families encode categoricals differently: a tree model ordinal-
    encodes texture into ONE column while a linear model one-hot encodes it into several, so
    per-feature bars are not comparable across them but the block totals are. ShapResult
    .block_mean_abs sums within a block before taking the absolute value, so this answers "how much
    do the categorical covariates move the prediction, jointly".
    """
    labels = {"num": "continuous", "cat": "categorical"}
    blocks = []
    for name in feature_names:
        prefix = name.split("__", 1)[0] if "__" in name else ""
        blocks.append(labels.get(prefix, "features"))
    return blocks


def _base_value(explainer):
    """The explainer's expected value, kept per output when it has several."""
    expected = getattr(explainer, "expected_value", None)
    if expected is None:
        return 0.0
    array = np.asarray(expected, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return 0.0
    return array if array.size > 1 else float(array[0])


def sklearn_shap_results(
    *,
    config,
    fitted_estimator: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    target: str,
    target_names: Any = None,
) -> list[ShapResult]:
    import shap

    preprocessor, estimator, output_space = _unwrap(fitted_estimator)

    if preprocessor is None:
        feature_names = list(X_test.columns)
        explain_matrix = _densify(X_test.to_numpy())
        background_matrix = _densify(X_train.to_numpy())
    else:
        feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
        explain_matrix = _densify(preprocessor.transform(X_test))
        background_matrix = _densify(preprocessor.transform(X_train))

    seed = int(getattr(config, "RANDOM_SEED", 42) or 42)
    explain_matrix = _subsample(explain_matrix, int(getattr(config, "EXPLAIN_MAX_SAMPLES", 500)), seed)
    background_matrix = _subsample(
        background_matrix, int(getattr(config, "EXPLAIN_BACKGROUND_SAMPLES", 100)), seed
    )

    if explain_matrix.size == 0 or not feature_names:
        return []

    values, base_value, explainer_name = _explain(
        shap,
        estimator=estimator,
        explain_matrix=explain_matrix,
        background_matrix=background_matrix,
        max_evals=int(getattr(config, "EXPLAIN_MAX_EVALS", DEFAULT_MAX_EVALS)),
    )

    # A joint fit has one output per target. `_as_values` keeps that axis - it used to be dropped
    # with `[..., 0]`, so every target's plots showed the FIRST target's attributions under its own
    # name. EVERY output is returned, in output order: the caller explains the joint model once, at
    # model-run scope, and routes each output to the run that holds that target's evaluation. This
    # used to slice down to the single output matching `target`, which paid for the full multi-output
    # explanation N times over to throw away N-1 of it each time.
    n_outputs = values.shape[2] if values.ndim == 3 else 1
    names = [str(name) for name in (target_names or [])]
    if len(names) != n_outputs:
        names = [str(target)] if n_outputs == 1 else [f"{target}_{index}" for index in range(n_outputs)]

    return [
        ShapResult(
            values=_output_slice(values, index),
            data=explain_matrix,
            feature_names=feature_names,
            blocks=_blocks_from_feature_names(feature_names),
            target_name=names[index],
            output_space=output_space,
            base_value=_base_value_at(base_value, index),
            explainer=explainer_name,
        )
        for index in range(n_outputs)
    ]
