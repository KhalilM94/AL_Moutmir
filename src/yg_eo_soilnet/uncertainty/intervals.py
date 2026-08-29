"""How a predictive standard deviation becomes a lower and an upper bound.

Three ways, chosen by ``uncertainty.interval.method``, and they are not interchangeable:

``conformal``
    ``mean +- q * sigma`` with ``q`` fitted on held-out residuals. The only one whose coverage is
    guaranteed rather than assumed, and the only one whose intervals mean the same thing across the
    two training families - which is why it is the default. See :mod:`yg_eo_soilnet.uncertainty.conformal`.

``gaussian``
    ``mean +- z(1 - alpha/2) * sigma``. Looks like a confidence interval and is one only if the
    predictive distribution really is Normal AND sigma is on the right scale. On a run where the
    ensemble spread understates the error - the Ridge case, where the fitted conformal q was 50.3
    against a Gaussian 1.96 - this covers a small fraction of what it claims.

``sigma``
    ``mean +- k * sigma``. Makes no claim beyond "this is k standard deviations", which is the
    honest thing to plot when you want to see the model's own spread rather than a calibrated band.
    Its nominal coverage is the Gaussian implication of k (0.683 at k=1), not 1 - alpha.

All three present the interface ``attach_uncertainty_columns`` already consumes - ``intervals``,
``to_dict`` and ``nominal_coverage`` - so choosing between them changes nothing downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

CONFORMAL = "conformal"
GAUSSIAN = "gaussian"
SIGMA = "sigma"
NONE = "none"
INTERVAL_METHODS = (CONFORMAL, GAUSSIAN, SIGMA, NONE)

# The methods that need rows the model never saw. Only conformal does, and that single fact decides
# whether the sklearn family has to give up its val split - see ModelTrainer._resolve_fit_pool.
METHODS_NEEDING_CALIBRATION = (CONFORMAL,)


def needs_calibration_set(method: str) -> bool:
    """Whether this method has to be fitted on held-out data before it can produce an interval."""
    return normalize_method(method) in METHODS_NEEDING_CALIBRATION


def normalize_method(method: Any) -> str:
    """Resolve a configured name, including the legacy ``split_conformal`` spelling."""
    name = str(method or CONFORMAL).strip().lower()
    if name == "split_conformal":
        return CONFORMAL
    if name not in INTERVAL_METHODS:
        raise ValueError(
            f"Unknown uncertainty.interval.method {method!r}; expected one of "
            f"{', '.join(INTERVAL_METHODS)}."
        )
    return name


@dataclass(frozen=True)
class GaussianInterval:
    """``mean +- z(1 - alpha/2) * sigma``: a normal-theory band, calibrated by nothing.

    Frozen and made of plain floats for the same reason ConformalCalibrator is: it is pickled into
    the logged model.
    """

    alpha: float = 0.05

    @property
    def z(self) -> float:
        return float(stats.norm.ppf(1.0 - float(self.alpha) / 2.0))

    @property
    def nominal_coverage(self) -> float:
        return 1.0 - float(self.alpha)

    def intervals(self, mean: Any, sigma: Any) -> tuple[np.ndarray, np.ndarray]:
        mean_array = np.asarray(mean, dtype=float)
        half_width = self.z * np.maximum(np.asarray(sigma, dtype=float), 0.0)
        return mean_array - half_width, mean_array + half_width

    def to_dict(self) -> dict[str, Any]:
        return {
            "interval_method": GAUSSIAN,
            "interval_alpha": float(self.alpha),
            "interval_z": self.z,
        }


@dataclass(frozen=True)
class SigmaInterval:
    """``mean +- k * sigma``: the model's own spread, with no coverage claim attached.

    ``nominal_coverage`` is what k would mean under a Normal - 0.683 at k=1, 0.954 at k=2 - because
    the metrics need SOMETHING to compare picp against, and 1 - alpha is emphatically not it. Read
    it as "what this band would cover if sigma were right", which is exactly the question a
    coverage_error against it answers.
    """

    k: float = 1.0

    @property
    def nominal_coverage(self) -> float:
        return float(2.0 * stats.norm.cdf(float(self.k)) - 1.0)

    def intervals(self, mean: Any, sigma: Any) -> tuple[np.ndarray, np.ndarray]:
        mean_array = np.asarray(mean, dtype=float)
        half_width = float(self.k) * np.maximum(np.asarray(sigma, dtype=float), 0.0)
        return mean_array - half_width, mean_array + half_width

    def to_dict(self) -> dict[str, Any]:
        return {
            "interval_method": SIGMA,
            "interval_k": float(self.k),
            "interval_nominal_coverage": self.nominal_coverage,
        }


def effective_alpha(method: Any, alpha: float = 0.05, k: float = 1.0) -> float:
    """The miscoverage the configured interval actually implies.

    ``alpha`` for conformal and gaussian; ``1 - (2*Phi(k) - 1)`` for sigma. The metrics take this
    rather than the configured alpha, so a ±1σ band is graded against the ~0.683 it claims instead
    of against 0.95 - which would report a perfectly good band as under-covering by 0.27.
    """
    return 1.0 - SigmaInterval(k).nominal_coverage if normalize_method(method) == SIGMA else float(alpha)


def describe(estimator: Any) -> str:
    """A short label for the plot, so a picture says which claim its bars are making."""
    if estimator is None:
        return ""
    if isinstance(estimator, SigmaInterval):
        return f"±{float(estimator.k):g}σ"
    if isinstance(estimator, GaussianInterval):
        return f"gaussian {estimator.nominal_coverage:.0%}"
    coverage = getattr(estimator, "nominal_coverage", None)
    return f"conformal {coverage:.0%}" if coverage is not None else "interval"


def build_interval_estimators(
    method: Any,
    target_names: Sequence[str],
    *,
    alpha: float = 0.05,
    k: float = 1.0,
    prediction: Any = None,
    y_calib: Any = None,
    logger: Any = None,
) -> Mapping[str, Any]:
    """One interval estimator per target, for whichever method the run asked for.

    ``prediction`` and ``y_calib`` are read only by the conformal branch - the other methods need no
    data at all, which is what lets the sklearn family keep its val split under them.
    """
    resolved = normalize_method(method)
    if resolved == NONE:
        return {}
    if resolved == GAUSSIAN:
        return {str(name): GaussianInterval(alpha=float(alpha)) for name in target_names}
    if resolved == SIGMA:
        return {str(name): SigmaInterval(k=float(k)) for name in target_names}

    from yg_eo_soilnet.uncertainty import fit_calibrators

    if prediction is None or y_calib is None:
        raise ValueError(
            "The conformal interval needs held-out predictions and observations to fit against. "
            "Either supply a calibration set or choose uncertainty.interval.method: gaussian|sigma, "
            "which need none."
        )
    return fit_calibrators(prediction, y_calib, target_names, alpha=float(alpha), logger=logger)


def estimator_from_config(config: Any, target_names: Sequence[str], **kwargs) -> Mapping[str, Any]:
    """``build_interval_estimators`` with the method, alpha and k read off the run's config."""
    return build_interval_estimators(
        getattr(config, "UNCERTAINTY_INTERVAL_METHOD", CONFORMAL),
        target_names,
        alpha=float(getattr(config, "UNCERTAINTY_ALPHA", 0.05)),
        k=float(getattr(config, "UNCERTAINTY_INTERVAL_K", 1.0)),
        **kwargs,
    )
