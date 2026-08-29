"""The metric set that says whether a prediction interval is honest.

Companion to yg_eo_soilnet.metrics and bound by the same three rules: one target at a time, no
metric negated to express "higher is better", and every name registered in METRIC_DIRECTION and
METRIC_SPACE so a reader of an MLflow run never has to guess what a number means.

The reason this module exists separately is that an interval can fail in two independent ways and a
single number cannot catch both. An interval spanning the entire target range covers 100% of
observations and tells you nothing; an interval of width zero is maximally sharp and covers nothing.
So coverage (picp) and width (mpiw) are always read together, and interval_score is the one number
that penalises both at once when a single ranking is needed.

Everything here is in ORIGINAL TARGET UNITS, computed from the same evaluation frame the point
metrics come from - so a sigma is comparable to an rmse on the same target, which is the comparison
that makes uncertainty interpretable at all.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

# The name table lives in yg_eo_soilnet.metrics, which is the single registry of what a metric name
# means and which way is better. Imported rather than redeclared so the two cannot drift, and in
# this direction so the dependency stays acyclic.
from yg_eo_soilnet.metrics import (  # noqa: F401  (re-exported for callers of this module)
    UNCERTAINTY_METRIC_DIRECTION,
    UNCERTAINTY_METRIC_STEMS,
    _reject_multi_column,
)

# Bins used by the expected normalized calibration error. Ten is the usual choice and is about as
# fine as a ~900-row test split supports: at 20 bins each holds ~45 points and the per-bin RMSE is
# noise.
ENCE_BINS = 10


def uncertainty_metrics(
    y_true: Any,
    y_pred: Any,
    sigma: Any,
    lower: Optional[Any] = None,
    upper: Optional[Any] = None,
    *,
    alpha: float = 0.05,
    split: str = "test",
    suffix: str = "",
) -> dict[str, float]:
    """Interval and distributional metrics for one target, in original units.

    Args:
        y_true: observed values.
        y_pred: predicted mean, same length and order.
        sigma: predictive standard deviation, same length and order.
        lower / upper: the calibrated interval. When omitted, the interval metrics are skipped and
            only the distributional ones (crps, nll, ence, sigma_error_corr) are returned - which is
            what an uncalibrated run should report rather than inventing a Gaussian interval and
            grading itself against it.
        alpha: the miscoverage rate the interval was built at, used by coverage_error.
        split / suffix: key construction, identical to regression_metrics.

    Returns an empty dict when there are fewer than two finite rows, matching regression_metrics
    rather than emitting NaNs that would poison the leaderboard.
    """
    observed, predicted, sigma_values, lower_values, upper_values = _finite_rows(
        y_true, y_pred, sigma, lower, upper
    )

    count = int(observed.shape[0])
    if count < 2:
        return {}

    def key(stem: str) -> str:
        return f"{stem}_{split}{suffix}"

    residuals = observed - predicted
    absolute_residuals = np.abs(residuals)
    metrics: dict[str, float] = {key("mean_sigma"): float(np.mean(sigma_values))}

    if lower_values is not None and upper_values is not None:
        covered = (observed >= lower_values) & (observed <= upper_values)
        picp = float(np.mean(covered))
        widths = upper_values - lower_values
        target_range = float(np.max(observed) - np.min(observed))

        metrics[key("picp")] = picp
        # Signed, like `bias` in the point metrics: the magnitude says how far off the coverage is,
        # the sign says whether the model is over- or under-confident. Those call for opposite fixes.
        metrics[key("coverage_error")] = picp - (1.0 - alpha)
        metrics[key("mpiw")] = float(np.mean(widths))
        if target_range > 1e-12:
            metrics[key("nmpiw")] = float(np.mean(widths) / target_range)
        metrics[key("interval_score")] = _interval_score(
            observed, lower_values, upper_values, alpha
        )

    # Distributional scores need a non-degenerate sigma; a deterministic ensemble has none and gets
    # the interval metrics only. Omitted individually rather than reported as inf, the same policy
    # regression_metrics applies to r2 on a constant target.
    if np.any(sigma_values > 0.0):
        safe_sigma = np.maximum(sigma_values, np.finfo(float).tiny)
        metrics[key("crps")] = _gaussian_crps(residuals, safe_sigma)
        metrics[key("nll")] = _gaussian_nll(residuals, safe_sigma)
        ence = _ence(absolute_residuals, sigma_values)
        if ence is not None:
            metrics[key("ence")] = ence
        correlation = _sigma_error_correlation(absolute_residuals, sigma_values)
        if correlation is not None:
            metrics[key("sigma_error_corr")] = correlation

    return metrics


def _interval_score(observed: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    """Winkler / interval score: width, plus a penalty for each miss proportional to how far.

    The one number that ranks intervals honestly. Coverage alone rewards a band spanning the whole
    target range; width alone rewards a band of zero width. This charges for width always and for
    misses at rate 2/alpha, so at alpha=0.05 an observation just outside the band costs 40x its
    distance. Lower is better.
    """
    widths = upper - lower
    below = np.maximum(lower - observed, 0.0)
    above = np.maximum(observed - upper, 0.0)
    return float(np.mean(widths + (2.0 / alpha) * (below + above)))


def _gaussian_crps(residuals: np.ndarray, sigma: np.ndarray) -> float:
    """Continuous ranked probability score under a Gaussian predictive law, in closed form.

    CRPS grades the WHOLE predicted distribution against the single observed value, and unlike NLL
    it is bounded in how badly one point can score - a single observation far in the tail sends NLL
    to a number that dominates the mean, while CRPS grows only linearly. That makes it the more
    readable of the two on real soil data, where a handful of outliers is normal.

    Closed form for a Gaussian: sigma * [ z(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ], z = (y - mu)/sigma.
    """
    standardized = residuals / sigma
    return float(
        np.mean(
            sigma
            * (
                standardized * (2.0 * stats.norm.cdf(standardized) - 1.0)
                + 2.0 * stats.norm.pdf(standardized)
                - 1.0 / np.sqrt(np.pi)
            )
        )
    )


def _gaussian_nll(residuals: np.ndarray, sigma: np.ndarray) -> float:
    """Mean negative log likelihood under a Gaussian predictive law.

    Positive by construction here in the sense that nothing is negated to change its direction: it
    is the NLL itself, lower is better, and it may legitimately be negative when the intervals are
    genuinely tight. That is the quantity being negative, not a sign flipped for convenience - the
    distinction yg_eo_soilnet.metrics draws in its header.
    """
    return float(np.mean(0.5 * np.log(2.0 * np.pi * sigma**2) + (residuals**2) / (2.0 * sigma**2)))


def _ence(absolute_residuals: np.ndarray, sigma: np.ndarray) -> Optional[float]:
    """Expected normalized calibration error: does sigma match the error AT EACH LEVEL of sigma?

    picp is a single global number and a model can hit it while being badly wrong locally - too
    confident on its easy points and too humble on its hard ones, with the two errors cancelling.
    This bins the points by predicted sigma and compares each bin's RMSE against its mean sigma, so
    that cancellation cannot hide. Zero is perfect. Returns None when there are too few points to
    bin meaningfully.
    """
    count = int(sigma.shape[0])
    if count < ENCE_BINS * 2:
        return None

    order = np.argsort(sigma)
    bins = np.array_split(order, ENCE_BINS)
    errors = []
    for indices in bins:
        if indices.size == 0:
            continue
        bin_rmse = float(np.sqrt(np.mean(absolute_residuals[indices] ** 2)))
        bin_sigma = float(np.mean(sigma[indices]))
        if bin_sigma <= 0.0:
            continue
        errors.append(abs(bin_sigma - bin_rmse) / bin_sigma)

    return float(np.mean(errors)) if errors else None


def _sigma_error_correlation(absolute_residuals: np.ndarray, sigma: np.ndarray) -> Optional[float]:
    """Spearman correlation between predicted sigma and realised |error|.

    The question every uncertainty estimate should have to answer: is the model actually less
    accurate where it says it is less certain? A conformal interval can be perfectly calibrated on
    average and still rank its points at random, in which case this sits near zero and the per-point
    bar carries no information even though picp looks excellent. Rank correlation rather than
    Pearson because only the ORDERING is claimed, not a linear relationship.

    None when either input is constant, where the correlation is undefined rather than zero.
    """
    if np.ptp(sigma) <= 0.0 or np.ptp(absolute_residuals) <= 0.0:
        return None
    correlation = stats.spearmanr(sigma, absolute_residuals).statistic
    return None if not np.isfinite(correlation) else float(correlation)


def _finite_rows(
    y_true: Any,
    y_pred: Any,
    sigma: Any,
    lower: Optional[Any],
    upper: Optional[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Every input as a float array, keeping only rows finite in ALL of them.

    Masking on the conjunction, like metrics._finite_pairs: dropping per-array would misalign the
    columns and silently score one point's observation against another's interval.
    """
    for name, values in (("y_true", y_true), ("y_pred", y_pred), ("sigma", sigma)):
        _reject_multi_column(name, values)

    columns = {
        "observed": _to_float(y_true),
        "predicted": _to_float(y_pred),
        "sigma": _to_float(sigma),
    }
    has_interval = lower is not None and upper is not None
    if has_interval:
        _reject_multi_column("lower", lower)
        _reject_multi_column("upper", upper)
        columns["lower"] = _to_float(lower)
        columns["upper"] = _to_float(upper)

    lengths = {name: values.shape[0] for name, values in columns.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"uncertainty_metrics inputs have mismatched lengths: {lengths}")

    keep = np.ones(next(iter(lengths.values())), dtype=bool)
    for values in columns.values():
        keep &= np.isfinite(values)
    # A negative sigma is not a small sigma, it is a bug upstream - most likely a variance handed
    # over where a standard deviation was expected. Dropping those rows rather than taking their
    # absolute value keeps the mistake visible in the row count.
    keep &= columns["sigma"] >= 0.0

    return (
        columns["observed"][keep],
        columns["predicted"][keep],
        columns["sigma"][keep],
        columns["lower"][keep] if has_interval else None,
        columns["upper"][keep] if has_interval else None,
    )


def _to_float(values: Any) -> np.ndarray:
    return pd.to_numeric(
        pd.Series(np.asarray(values).reshape(-1)), errors="coerce"
    ).to_numpy(dtype=float)
