"""The one regression metric set both training families report.

Before this module the two paths logged different numbers under the same name. sklearn logged a
POSITIVE cross-validated RMSE as ``mean_test_score`` (its ``GridSearchCV`` runs
``neg_root_mean_squared_error``, and the logger negated the already-negative value), while the
Lightning path logged ``-test_loss`` - a negated MSE in standardized log1p space, so a NEGATIVE
number - under that same name. The leaderboard then plotted both on one axis, mixing three units and
two signs.

The fix is this module. Both families call :func:`regression_metrics` on the same object - the test
split's prediction frame, in ORIGINAL target units - so every name below means exactly one thing no
matter which framework produced it, and no caller ever flips a sign.

Two rules keep it that way:

* **No metric here is ever negative by convention.** ``r2_test`` and ``bias_test`` can be negative
  because the quantity genuinely is; nothing is negated to express "higher is better". Direction
  lives in :data:`METRIC_DIRECTION`, not in the sign.
* **:func:`cv_rmse_from_search` is the only place a ``neg_*`` scorer sign is flipped.** If you find
  yourself writing a unary minus on a metric anywhere else, that is the bug this module exists to
  prevent.

Deliberately NOT covered here: the Lightning module's own ``train_loss`` / ``val_loss`` /
``test_loss`` / ``{stage}_r2`` / ``{stage}_pred_std_ratio``. Those are computed in STANDARDIZED LOG1P
space (``_shared_step`` compares against the transformed target; only ``predict_step`` inverts), and
they are wired into EarlyStopping monitors, ModelCheckpoint, the LR scheduler, the HPO objective
whitelist and every exported tuned config. They keep their names and their space. The metrics here
are logged ALONGSIDE them, and :data:`METRIC_SPACE` records which is which so a reader of an MLflow
run never has to guess.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from yg_eo_soilnet.utils import rpd_score, rpiq_score

# name stem -> which way is better. "zero" means a bias-like metric best read as |value|.
# None means the metric is a count, not a score.
METRIC_DIRECTION: dict[str, str | None] = {
    "rmse": "lower",
    "mae": "lower",
    "r2": "higher",
    "rpd": "higher",
    "rpiq": "higher",
    "bias": "zero",
    "n": None,
}

# The unified set, in the order they are logged and reported.
METRIC_STEMS: tuple[str, ...] = ("rmse", "mae", "r2", "rpd", "rpiq", "bias", "n")

# Which numeric space a metric name lives in. Recorded in every run summary so that a reader
# comparing test_loss against rmse_test knows they are not the same quantity in different units -
# they are different quantities in different spaces.
ORIGINAL_UNITS = "original_units"
STANDARDIZED_LOG1P = "standardized_log1p"

METRIC_SPACE: dict[str, str] = {
    # computed by this module, from the prediction frame, after inverse_transform_targets
    **{f"{stem}_test": ORIGINAL_UNITS for stem in METRIC_STEMS},
    "rmse_cv_mean": ORIGINAL_UNITS,
    "rmse_cv_std": ORIGINAL_UNITS,
    "rmse_cv_train_mean": ORIGINAL_UNITS,
    "r2_train_fit": ORIGINAL_UNITS,
    # logged by SoilRegressionLightningBase, against the standardized log1p target
    "train_loss": STANDARDIZED_LOG1P,
    "val_loss": STANDARDIZED_LOG1P,
    "test_loss": STANDARDIZED_LOG1P,
    "train_r2": STANDARDIZED_LOG1P,
    "val_r2": STANDARDIZED_LOG1P,
    "test_r2": STANDARDIZED_LOG1P,
    "train_pred_std_ratio": STANDARDIZED_LOG1P,
    "val_pred_std_ratio": STANDARDIZED_LOG1P,
    "test_pred_std_ratio": STANDARDIZED_LOG1P,
}

# Retired names, kept only so readers of pre-existing mlruns can still be interpreted. Nothing
# logs these any more. `mean_test_score` meant a POSITIVE cv RMSE on the sklearn side and a
# NEGATIVE -test_loss on the Lightning side, which is the whole reason this module exists.
LEGACY_METRIC_NAMES: tuple[str, ...] = ("mean_test_score", "mean_train_score", "r2_test_legacy")


def _reject_multi_column(name: str, values: Any) -> None:
    """Raise when `values` carries more than one target column."""
    array = np.asarray(values)
    if array.ndim > 1 and array.shape[-1] > 1:
        raise ValueError(
            f"{name} has {array.shape[-1]} columns; regression_metrics scores ONE target at a time. "
            "Pooling several would mix their units into a single meaningless number. Call it once "
            "per target with suffix=f'_{target_name}'."
        )


def _finite_pairs(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    """Both series as float arrays, keeping only positions where BOTH are finite.

    Dropping per-array rather than pairwise would misalign them, which is why the mask is built
    from the conjunction.

    Refuses a multi-column input. The reshape below would otherwise flatten several targets into
    one pooled score - an RMSE mixing pH with g/kg, reported under a name that claims to describe
    one target. Callers with several targets must call once per target with ``suffix``.
    """
    _reject_multi_column("y_true", y_true)
    _reject_multi_column("y_pred", y_pred)
    true_values = pd.to_numeric(pd.Series(np.asarray(y_true).reshape(-1)), errors="coerce").to_numpy(dtype=float)
    predicted_values = pd.to_numeric(pd.Series(np.asarray(y_pred).reshape(-1)), errors="coerce").to_numpy(dtype=float)

    if true_values.shape != predicted_values.shape:
        raise ValueError(
            f"y_true has {true_values.shape[0]} value(s) but y_pred has {predicted_values.shape[0]}"
        )

    keep = np.isfinite(true_values) & np.isfinite(predicted_values)
    return true_values[keep], predicted_values[keep]


def regression_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    split: str = "test",
    suffix: str = "",
) -> dict[str, float]:
    """The unified metric set for one (observed, predicted) pair, in original target units.

    Args:
        y_true: observed values.
        y_pred: predicted values, same length and order.
        split: name stitched into every key, e.g. ``"test"`` -> ``rmse_test``.
        suffix: appended after the split, used for per-target keys on a multi-target run,
            e.g. ``suffix="_organic_matter_pct"`` -> ``rmse_test_organic_matter_pct``.

    Returns an empty dict when there are fewer than two finite pairs, rather than emitting NaN
    metrics that would then poison the leaderboard. Metrics whose denominator is degenerate
    (zero target variance for R2, zero RMSE for RPD/RPIQ) are individually omitted for the same
    reason - the same policy `_log_epoch_metrics` already applies on the Lightning side.
    """
    true_values, predicted_values = _finite_pairs(y_true, y_pred)

    count = int(true_values.shape[0])
    if count < 2:
        return {}

    def key(stem: str) -> str:
        return f"{stem}_{split}{suffix}"

    rmse = float(root_mean_squared_error(true_values, predicted_values))
    metrics: dict[str, float] = {
        key("rmse"): rmse,
        key("mae"): float(mean_absolute_error(true_values, predicted_values)),
        # Signed on purpose: the magnitude says how far off the model is on average, the sign says
        # in which direction. A model collapsing toward the target mean shows near-zero bias with a
        # large rmse, which is exactly the pair that identifies it.
        key("bias"): float(np.mean(predicted_values - true_values)),
        key("n"): float(count),
    }

    target_variance = float(np.var(true_values))
    if target_variance > 1e-12:
        metrics[key("r2")] = float(r2_score(true_values, predicted_values))

    if rmse > 0.0:
        # Argument order matters: both take (predictions, targets), and the numerator (the spread)
        # is read off the SECOND argument. Passing them the other way round measures the spread of
        # the predictions instead of the observations, which under-reports both scores because
        # predictions are systematically under-dispersed.
        metrics[key("rpd")] = float(rpd_score(predicted_values, true_values))
        metrics[key("rpiq")] = float(rpiq_score(predicted_values, true_values))

    return metrics


def cv_rmse_from_search(cv_results: Any, best_index: int) -> dict[str, float]:
    """Cross-validated RMSE at the best grid point, as a POSITIVE number.

    This is the only sign flip in the codebase. ``GridSearchCV`` is built with
    ``scoring="neg_root_mean_squared_error"``, so ``mean_test_score`` in ``cv_results_`` is a
    negative RMSE and reads "higher is better". One negation here turns it into the plain RMSE
    every other metric in this module is expressed as.

    ``std_test_score`` is NOT flipped: negating every fold score leaves their standard deviation
    unchanged, so flipping it would make the spread negative.
    """
    if isinstance(cv_results, pd.DataFrame):
        columns: Mapping[str, Any] = {name: cv_results[name].to_numpy() for name in cv_results.columns}
    else:
        columns = cv_results

    def value_at(column_name: str) -> float | None:
        column = columns.get(column_name)
        if column is None:
            return None
        try:
            scalar = float(np.asarray(column)[best_index])
        except (IndexError, TypeError, ValueError):
            return None
        return scalar if np.isfinite(scalar) else None

    metrics: dict[str, float] = {}

    mean_test_score = value_at("mean_test_score")
    if mean_test_score is not None:
        metrics["rmse_cv_mean"] = -mean_test_score

    std_test_score = value_at("std_test_score")
    if std_test_score is not None:
        metrics["rmse_cv_std"] = std_test_score

    mean_train_score = value_at("mean_train_score")
    if mean_train_score is not None:
        metrics["rmse_cv_train_mean"] = -mean_train_score

    return metrics


def metric_space_for(metric_names: Any) -> dict[str, str]:
    """The subset of :data:`METRIC_SPACE` covering the names actually logged on a run.

    Unknown names are reported as ``"unknown"`` rather than dropped, so a metric someone adds
    without registering it here shows up as a gap instead of silently looking like original units.
    """
    return {str(name): _space_of(str(name)) for name in metric_names}


def _space_of(name: str) -> str:
    """The space a metric name lives in, resolving per-target suffixes to their stem.

    A joint run logs ``rmse_test_clay_pct`` and ``val_r2_clay_pct`` alongside the unsuffixed pair.
    Those are the same quantity in the same space as their stem, so they are resolved by prefix
    rather than needing one registry entry per configured target.
    """
    known = METRIC_SPACE.get(name)
    if known is not None:
        return known
    # Longest first: "test_r2" must not match a name that "test_r2_..." also starts with by way of
    # some shorter stem like "test_r".
    for stem in sorted(METRIC_SPACE, key=len, reverse=True):
        if name.startswith(f"{stem}_"):
            return METRIC_SPACE[stem]
    return "unknown"
