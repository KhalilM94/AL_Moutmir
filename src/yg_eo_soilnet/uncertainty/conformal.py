"""Split-conformal calibration: turning a model's sigma into an interval that actually covers.

A raw ensemble standard deviation is not a prediction interval. Five members trained on the same
data agree with each other far more than they agree with reality, so `mu +- 1.96 sigma` typically
covers well under the 95% it claims - the number is a measure of disagreement, not of error.

Split conformal fixes that with one scalar. Score every calibration point by how many of its own
sigmas it missed by, take the appropriate order statistic of those scores, and scale every interval
by it. The result covers at least 1-alpha of future points under exchangeability alone: no
assumption that the residuals are Gaussian, and no assumption that the model is any good. A badly
calibrated model gets wide intervals rather than wrong ones.

This is also what makes a sklearn interval and a Lightning interval the same kind of object. The
two families produce sigma by completely different mechanisms, but both are rescaled against their
own held-out residuals, so `prediction_lower`/`prediction_upper` mean one thing across the run -
the same property yg_eo_soilnet.metrics exists to protect for the point metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

# Below this the interval is treated as coming from a model with no usable sigma and the calibrator
# falls back to a constant width. Not zero: a float sigma that underflows to 1e-300 would otherwise
# produce a conformity score of 1e300 and a q that swallows the whole target range.
SIGMA_FLOOR = 1e-12

# The guarantee is asymptotic in n_calib, and the finite-sample correction below cannot produce a
# valid quantile at all when n_calib < 1/alpha - 1 (at alpha=0.05, fewer than 19 points). Warn well
# above that, because coverage at n=25 is still extremely noisy run to run.
MIN_RELIABLE_CALIBRATION_ROWS = 50


@dataclass(frozen=True)
class ConformalCalibrator:
    """The scalar multiplier that makes an interval cover, plus how it was arrived at.

    Frozen and made of plain floats so it pickles into a model artifact without dragging a fitted
    estimator or a dataframe along with it.
    """

    q: float
    alpha: float
    n_calib: int
    # False when the calibration sigma was degenerate and q is an ABSOLUTE residual quantile rather
    # than a multiplier on sigma. The interval is then a constant width, which is honest for a
    # deterministic ensemble and would be silently wrong if read as `q * sigma`.
    normalized: bool = True

    @property
    def nominal_coverage(self) -> float:
        """The fraction of observations this interval claims to contain.

        Read by the metrics so ``coverage_error`` compares picp against what the interval actually
        claims. Every interval estimator exposes this; for a sigma band it is NOT 1 - alpha, which
        is the whole reason it is a property rather than a config lookup.
        """
        return 1.0 - float(self.alpha)

    def intervals(self, mean: Any, sigma: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(lower, upper)`` for predictions in the SAME units the calibrator was fitted on."""
        mean_array = np.asarray(mean, dtype=float)
        if self.normalized:
            half_width = self.q * np.maximum(np.asarray(sigma, dtype=float), 0.0)
        else:
            half_width = np.full_like(mean_array, self.q)
        return mean_array - half_width, mean_array + half_width

    def to_dict(self) -> dict[str, Any]:
        """Log-friendly provenance; every value is an MLflow-loggable scalar."""
        return {
            "conformal_q": float(self.q),
            "conformal_alpha": float(self.alpha),
            "conformal_n_calib": int(self.n_calib),
            "conformal_normalized": bool(self.normalized),
        }


def fit_conformal(
    y_calib: Any,
    mean_calib: Any,
    sigma_calib: Optional[Any] = None,
    *,
    alpha: float = 0.05,
    logger: Any = None,
) -> ConformalCalibrator:
    """Fit the calibrator on one target's held-out residuals.

    One target at a time, like ``regression_metrics``. Pooling several would mix pH with g/kg into
    a single multiplier that is correct for neither.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    y_array = np.asarray(y_calib, dtype=float).reshape(-1)
    mean_array = np.asarray(mean_calib, dtype=float).reshape(-1)
    if y_array.shape != mean_array.shape:
        raise ValueError(
            f"y_calib has {y_array.shape[0]} rows and mean_calib has {mean_array.shape[0]}; "
            "conformal calibration pairs them row for row."
        )

    if sigma_calib is None:
        sigma_array = np.zeros_like(y_array)
    else:
        sigma_array = np.asarray(sigma_calib, dtype=float).reshape(-1)
        if sigma_array.shape != y_array.shape:
            raise ValueError(
                f"sigma_calib has {sigma_array.shape[0]} rows but y_calib has {y_array.shape[0]}."
            )

    finite = np.isfinite(y_array) & np.isfinite(mean_array) & np.isfinite(sigma_array)
    y_array, mean_array, sigma_array = y_array[finite], mean_array[finite], sigma_array[finite]

    n_calib = int(y_array.shape[0])
    if n_calib < 2:
        raise ValueError(
            f"Conformal calibration needs at least 2 finite calibration rows; got {n_calib}. "
            "Check that the calibration split is non-empty and that this target is measured on it."
        )
    if n_calib < MIN_RELIABLE_CALIBRATION_ROWS and logger is not None:
        logger.warning(
            f"Conformal calibration has only {n_calib} rows (alpha={alpha}). The coverage guarantee "
            f"holds in expectation but is very noisy below ~{MIN_RELIABLE_CALIBRATION_ROWS} rows; "
            "treat picp_test from this run as indicative rather than as a measurement."
        )

    absolute_residuals = np.abs(y_array - mean_array)

    # A sigma that is degenerate on the calibration set cannot normalize anything, and dividing by
    # it would turn the scores into noise amplified by 1/eps. Fall back to an ABSOLUTE residual
    # quantile: a constant-width band, which is exactly the right answer for a deterministic
    # ensemble and is flagged so no reader mistakes q for a multiplier.
    normalized = bool(np.any(sigma_array > SIGMA_FLOOR))
    if normalized:
        scores = absolute_residuals / np.maximum(sigma_array, SIGMA_FLOOR)
    else:
        scores = absolute_residuals

    return ConformalCalibrator(
        q=_conformal_quantile(scores, alpha),
        alpha=float(alpha),
        n_calib=n_calib,
        normalized=normalized,
    )


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ``ceil((n+1)(1-alpha)) / n`` empirical quantile of the conformity scores.

    NOT ``np.quantile(scores, 1 - alpha)``. The finite-sample correction is what the coverage proof
    rests on: the guarantee is about where a NEW point falls among the n calibration points, so the
    rank is taken out of n+1, not n. At n=100 and alpha=0.05 the plain quantile takes rank 95 and
    the corrected one takes rank 96 - a small difference in width, and the difference between a
    proven bound and an approximation that under-covers slightly and always in the same direction.

    When ``ceil((n+1)(1-alpha))`` exceeds n - too few calibration points to place the bound at all -
    the level saturates at the maximum observed score, which is the widest honest answer available.
    """
    n_scores = int(scores.shape[0])
    rank = math.ceil((n_scores + 1) * (1.0 - alpha))
    if rank > n_scores:
        # Too few calibration points to place the bound at all: at alpha=0.05 this is n < 19. The
        # widest honest answer available is the largest score observed.
        return float(np.max(scores))
    # Indexed directly into the sorted scores rather than going through np.quantile. Even with
    # method="higher", np.quantile maps a probability onto the position (n-1)*p and then rounds,
    # which is NOT the rank-th order statistic: at n=99 and alpha=0.05 the rank is 95 and
    # np.quantile(scores, 95/99, method="higher") returns the 96th. One rank too wide is harmless
    # for coverage and wrong for the quantity being reported, and the discrepancy grows with alpha.
    return float(np.sort(scores)[rank - 1])
