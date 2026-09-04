"""Predictive uncertainty for both training families.

Every model in this project emits a point estimate. This package turns an ensemble of them into a
mean, a standard deviation split into its epistemic and aleatoric parts, and a conformally
calibrated interval - one contract, whichever framework produced the members.

The entry points the logger and the trainers use:

* :func:`uncertainty_enabled_for` - the switch, checked before any of the work below is set up;
* :func:`attach_uncertainty_columns` - write a group's predictions and intervals onto an eval frame;
* :func:`log_uncertainty_artifacts` - the ``uncertainty/`` artifacts and the run-summary block.

Unlike ``explain``, this package is safe to import at module scope: it depends on numpy, pandas,
scipy and matplotlib, all of which are already imported by the training path. The switch exists
because the members cost N training runs, not because the import costs anything.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from yg_eo_soilnet.artifacts import ArtifactLayout, log_figure, log_json
from yg_eo_soilnet.uncertainty.columns import (
    ALEATORIC_STD,
    EPISTEMIC_STD,
    LOWER,
    STD,
    UNCERTAINTY_STEMS,
    UPPER,
    column_name,
    interval_columns,
    is_prediction_column,
    sigma_column,
)
from yg_eo_soilnet.uncertainty.conformal import ConformalCalibrator, fit_conformal
from yg_eo_soilnet.uncertainty.ensemble import (
    BOOTSTRAP_AUTO,
    EnsemblePrediction,
    aggregate,
    bootstrap_indices,
    member_seeds,
    should_bootstrap,
)
from yg_eo_soilnet.uncertainty.metrics import uncertainty_metrics

__all__ = [
    "ALEATORIC_STD",
    "BOOTSTRAP_AUTO",
    "ConformalCalibrator",
    "EPISTEMIC_STD",
    "EnsemblePrediction",
    "LOWER",
    "STD",
    "UNCERTAINTY_STEMS",
    "UPPER",
    "aggregate",
    "attach_uncertainty_columns",
    "bootstrap_indices",
    "column_name",
    "fit_calibrators",
    "fit_conformal",
    "interval_columns",
    "is_prediction_column",
    "log_uncertainty_artifacts",
    "member_seeds",
    "should_bootstrap",
    "sigma_column",
    "uncertainty_enabled_for",
    "uncertainty_metrics",
]


def uncertainty_enabled_for(config: Any, model_name: str) -> bool:
    """Whether this registry entry should be trained as an ensemble.

    Same allowlist-beats-denylist rule ``ChildRunLogger._shap_gate`` applies, and for the
    same reason: the cost of uncertainty is per model, not per run, so naming an entry explicitly
    has to be able to override a blanket exclusion.
    """
    if not bool(getattr(config, "UNCERTAINTY_ENABLED", False)):
        return False

    allowed = [str(name) for name in (getattr(config, "UNCERTAINTY_MODELS", None) or [])]
    if allowed:
        return str(model_name) in allowed

    skipped = [str(name) for name in (getattr(config, "UNCERTAINTY_SKIP_MODELS", None) or [])]
    return str(model_name) not in skipped


def fit_calibrators(
    prediction: EnsemblePrediction,
    y_calib: Any,
    target_names: Sequence[str],
    *,
    alpha: float = 0.05,
    logger: Any = None,
) -> dict[str, ConformalCalibrator]:
    """One calibrator per target, fitted on the held-out calibration split.

    Per target rather than pooled: the multiplier that makes a pH interval cover is not the one that
    makes a g/kg interval cover, and a single q fitted across both would be wrong for each. Same
    rule ``regression_metrics`` follows for the point metrics.
    """
    calibrators: dict[str, ConformalCalibrator] = {}
    for index, target_name in enumerate(target_names):
        observed = (
            y_calib[target_name]
            if isinstance(y_calib, pd.DataFrame) and target_name in y_calib.columns
            else y_calib
        )
        calibrators[target_name] = fit_conformal(
            observed,
            prediction.mean[:, index],
            prediction.total_std[:, index],
            alpha=alpha,
            logger=logger,
        )
    return calibrators


def attach_uncertainty_columns(
    frame: pd.DataFrame,
    prediction: EnsemblePrediction,
    target_names: Sequence[str],
    calibrators: Optional[Mapping[str, ConformalCalibrator]] = None,
) -> pd.DataFrame:
    """Write the sigma and interval columns onto an evaluation frame, in place.

    ``prediction`` carries ``(n_rows, n_targets)`` arrays whose column order matches
    ``target_names``. The frame keeps whatever ``prediction`` / ``prediction_<t>`` columns it
    already has - those are written by the family's own frame builder and are the ensemble mean.

    The naming follows :mod:`yg_eo_soilnet.uncertainty.columns`: unsuffixed for a lone target,
    suffixed for one of several, matching the prediction columns exactly so one reader handles both.
    """
    multi_target = len(target_names) > 1
    calibrators = calibrators or {}

    for index, target_name in enumerate(target_names):
        def name(stem: str) -> str:
            return column_name(stem, target_name, multi_target=multi_target)

        total_std = prediction.total_std[:, index]
        frame[name(STD)] = total_std
        frame[name(EPISTEMIC_STD)] = prediction.epistemic_std[:, index]
        frame[name(ALEATORIC_STD)] = prediction.aleatoric_std[:, index]

        calibrator = calibrators.get(target_name)
        if calibrator is not None:
            lower, upper = calibrator.intervals(prediction.mean[:, index], total_std)
            frame[name(LOWER)] = lower
            frame[name(UPPER)] = upper

    return frame


def log_uncertainty_artifacts(
    frame: pd.DataFrame,
    target_name: str,
    *,
    calibrator: Optional[ConformalCalibrator] = None,
    artifact_path: Optional[str] = None,
) -> dict:
    """Write the ``uncertainty/`` diagnostics for one target and describe what was written.

    Returns the dict embedded in ``run_summary.json`` under ``"uncertainty"``, in the same shape
    ``log_shap_artifacts`` returns for ``"explain"``. An empty dict when the frame carries no sigma,
    so a caller does not have to check first.
    """
    from yg_eo_soilnet.uncertainty.plots import reliability_curve, sigma_vs_error

    sigma = sigma_column(frame, target_name)
    if sigma is None or target_name not in frame.columns or "prediction" not in frame.columns:
        return {}

    observed = frame[target_name]
    predicted = frame["prediction"]
    destination = artifact_path or ArtifactLayout.uncertainty_path()

    written: dict = {"artifacts": []}
    figures = [
        (
            reliability_curve(
                observed, predicted, sigma, calibrator=calibrator, target_name=target_name
            ),
            ArtifactLayout.RELIABILITY_FILE,
        ),
        (
            sigma_vs_error(observed, predicted, sigma, target_name=target_name),
            ArtifactLayout.SIGMA_ERROR_FILE,
        ),
    ]
    for figure, filename in figures:
        log_figure(figure, filename, destination)
        written["artifacts"].append(f"{destination}/{filename}")

    summary = {
        "target": target_name,
        "n_rows": int(len(frame)),
        "mean_sigma": float(sigma.mean()),
        **(calibrator.to_dict() if calibrator is not None else {}),
    }
    log_json(summary, ArtifactLayout.UNCERTAINTY_SUMMARY_FILE, destination)
    written["artifacts"].append(f"{destination}/{ArtifactLayout.UNCERTAINTY_SUMMARY_FILE}")
    written.update(summary)

    return written
