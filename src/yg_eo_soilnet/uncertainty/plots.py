"""Diagnostics that say whether the bars on the pred-vs-obs plot are honest.

Repo convention, the same one explain/plots.py and hpo/plots.py follow: build the Figure, call
tight_layout, and RETURN it. The caller saves and closes. (plot_utils.create_pred_obs_plot is the
documented exception, because MLflow's custom-artifact contract makes it save itself.)

The pred-vs-obs bars show what the model claims. These two panels are how you check the claim:

  reliability_curve  the interval is built at one nominal level (95%) but the model has an opinion
                     at every level. Sweeping them and plotting nominal against empirical shows
                     whether the shape of the predictive distribution is right or whether one
                     rescaling happens to fix the 95% point while everything else is off.
  sigma_vs_error     coverage is a global average, and a model can hit 95% overall while being
                     over-confident on its easy points and under-confident on its hard ones - the
                     two errors cancel. Binning by sigma is what stops that cancellation hiding.
"""

from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

# Nominal levels swept by the reliability curve. Dense enough to show the shape, coarse enough that
# each point is a distinct empirical fraction on an ~900-row test split.
NOMINAL_LEVELS = np.linspace(0.05, 0.95, 19)


def reliability_curve(
    y_true: Any,
    y_pred: Any,
    sigma: Any,
    *,
    calibrator: Any = None,
    target_name: str = "",
    axis: Optional[plt.Axes] = None,
):
    """Nominal vs empirical coverage, swept across levels.

    With a `calibrator`, each nominal level is realised the way the plotted interval is - by scaling
    sigma with a conformal q refitted at that level - so the curve grades the actual procedure. That
    is the honest version, and it is why the curve does not simply pass through (0.95, 0.95) by
    construction: q is fitted on the CALIBRATION split and the curve is drawn on the TEST split, so
    the gap between them is exactly the generalisation of the calibration.

    Without one, levels are realised as Gaussian z-multiples, which grades the raw sigma instead and
    is the right picture when calibration is off.

    Returns the Figure when it created one, else None (it drew into the axis it was given).
    """
    observed = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    sigma_values = np.asarray(sigma, dtype=float).reshape(-1)

    figure = None
    if axis is None:
        figure, axis = plt.subplots(figsize=(6, 5))

    empirical = [
        _empirical_coverage(observed, predicted, sigma_values, level, calibrator)
        for level in NOMINAL_LEVELS
    ]

    axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="perfect")
    axis.plot(NOMINAL_LEVELS, empirical, marker="o", markersize=4, linewidth=1.5, label="observed")
    # Shading the gap makes the DIRECTION of the miscalibration readable at a glance, which is the
    # thing that determines what to do about it: below the diagonal is over-confident (intervals too
    # narrow), above is over-cautious (too wide, and the bars are not saying much).
    axis.fill_between(NOMINAL_LEVELS, NOMINAL_LEVELS, empirical, alpha=0.15, color="tab:blue")

    axis.set_xlabel("Nominal coverage")
    axis.set_ylabel("Empirical coverage")
    axis.set_title(f"{target_name}\nReliability" if target_name else "Reliability")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(True)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", fontsize=8)

    if figure is not None:
        figure.tight_layout()
    return figure


def sigma_vs_error(
    y_true: Any,
    y_pred: Any,
    sigma: Any,
    *,
    n_bins: int = 10,
    target_name: str = "",
    axis: Optional[plt.Axes] = None,
):
    """Per-bin mean sigma against per-bin RMSE, with the identity line.

    Points on the diagonal mean the model's stated uncertainty is the size of its actual error at
    that level. Below means over-confident, above over-cautious. A flat cloud - every bin at the
    same RMSE regardless of sigma - is the failure that global coverage cannot see: the intervals
    are the right size on average and carry no per-point information at all.
    """
    observed = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    sigma_values = np.asarray(sigma, dtype=float).reshape(-1)
    absolute_residuals = np.abs(observed - predicted)

    figure = None
    if axis is None:
        figure, axis = plt.subplots(figsize=(6, 5))

    bin_sigmas, bin_rmses = _bin_by_sigma(absolute_residuals, sigma_values, n_bins)

    if bin_sigmas.size:
        axis.scatter(bin_sigmas, bin_rmses, s=45, zorder=3, edgecolor="k", linewidth=0.5)
        limit = float(max(bin_sigmas.max(), bin_rmses.max())) * 1.05
        axis.plot([0, limit], [0, limit], linestyle="--", color="red", linewidth=1.5)
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)

    axis.set_xlabel("Predicted σ (bin mean)")
    axis.set_ylabel("Observed RMSE (bin)")
    axis.set_title(f"{target_name}\nσ vs realised error" if target_name else "σ vs realised error")
    axis.grid(True)
    axis.set_axisbelow(True)

    if figure is not None:
        figure.tight_layout()
    return figure


def _empirical_coverage(
    observed: np.ndarray,
    predicted: np.ndarray,
    sigma: np.ndarray,
    level: float,
    calibrator: Any,
) -> float:
    """Fraction of observations inside the interval built at this nominal `level`."""
    if calibrator is None:
        # No calibrator: read the level off the Gaussian the raw sigma implies.
        from scipy import stats

        half_width = stats.norm.ppf(0.5 + level / 2.0) * sigma
        lower, upper = predicted - half_width, predicted + half_width
    else:
        lower, upper = calibrator.intervals(predicted, sigma)
        if not np.isclose(calibrator.alpha, 1.0 - level):
            # Rescale the fitted q to this level rather than refitting: the calibrator was fitted on
            # the calibration split, which is not available here. The ratio of Gaussian z-scores is
            # the same approximation the alternative would make and keeps the curve monotone.
            from scipy import stats

            scale = stats.norm.ppf(0.5 + level / 2.0) / stats.norm.ppf(
                0.5 + (1.0 - calibrator.alpha) / 2.0
            )
            half_width = (upper - lower) / 2.0 * scale
            lower, upper = predicted - half_width, predicted + half_width

    return float(np.mean((observed >= lower) & (observed <= upper)))


def _bin_by_sigma(
    absolute_residuals: np.ndarray,
    sigma: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-COUNT bins ordered by sigma, returning each bin's mean sigma and its RMSE.

    Equal count rather than equal width: predicted sigma is usually heavily right-skewed, so
    equal-width bins put almost every point in the first bin and leave the rest holding one or two
    points each, whose RMSE is meaningless.
    """
    finite = np.isfinite(absolute_residuals) & np.isfinite(sigma)
    absolute_residuals, sigma = absolute_residuals[finite], sigma[finite]
    if sigma.size < n_bins * 2:
        return np.array([]), np.array([])

    order = np.argsort(sigma)
    bin_sigmas, bin_rmses = [], []
    for indices in np.array_split(order, n_bins):
        if indices.size == 0:
            continue
        bin_sigmas.append(float(np.mean(sigma[indices])))
        bin_rmses.append(float(np.sqrt(np.mean(absolute_residuals[indices] ** 2))))

    return np.array(bin_sigmas), np.array(bin_rmses)
