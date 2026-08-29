"""The fitted ensemble, as one estimator that behaves like the single model it replaces.

The design constraint that shapes this whole class: ``predict`` must keep returning a plain array of
means, in exactly the shape one member returns. Three separate things downstream depend on it -
``infer_signature`` when the model is logged, ``mlflow.models.evaluate`` when the sklearn family is
scored, and champion promotion when ``rmse_test`` is compared across versions. Returning a wide
frame of pred/std/lower/upper here would have broken all three, and none of them would have failed
loudly. The uncertainty is reached through the extra methods instead.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from yg_eo_soilnet.uncertainty.conformal import ConformalCalibrator
from yg_eo_soilnet.uncertainty.ensemble import EnsemblePrediction, aggregate


class EnsembleRegressor(BaseEstimator, RegressorMixin):
    """N fitted pipelines and their conformal calibrators, presented as one regressor.

    Already fitted on construction: the members are trained by the caller, which is what lets each
    one open its own MLflow child run and be seeded and resampled independently. ``fit`` therefore
    does nothing but return self, so that a stray ``clone().fit()`` somewhere in sklearn's machinery
    cannot silently discard the ensemble and leave one untrained estimator behind.
    """

    def __init__(
        self,
        members: Sequence[Any],
        target_names: Sequence[str],
        calibrators: Optional[Mapping[str, ConformalCalibrator]] = None,
        member_seeds: Optional[Sequence[int]] = None,
        bootstrapped: bool = False,
    ):
        if not len(members):
            raise ValueError("An EnsembleRegressor needs at least one fitted member")
        self.members = list(members)
        self.target_names = [str(name) for name in target_names]
        self.calibrators = dict(calibrators or {})
        self.member_seeds = list(member_seeds or [])
        self.bootstrapped = bool(bootstrapped)

    # --- the single-model contract -----------------------------------------

    def fit(self, X, y=None):  # noqa: N803 - sklearn's argument name
        """No-op: the members arrive fitted. Present so the estimator API is complete."""
        return self

    def predict(self, X):  # noqa: N803 - sklearn's argument name
        """The ensemble mean, in the same shape a single member returns.

        1-D for a single target and ``(n_rows, n_targets)`` for a group, matching what the wrapped
        pipeline would have returned on its own. Every existing reader of this model's output -
        the MLflow signature, the evaluator, the eval frame builder - sees no difference.
        """
        mean = self.predict_uncertainty(X).mean
        return mean.reshape(-1) if len(self.target_names) <= 1 else mean

    # --- the uncertainty API -----------------------------------------------

    def predict_members(self, X) -> list[np.ndarray]:  # noqa: N803
        """Each member's own prediction, in member order."""
        return [np.asarray(member.predict(X)) for member in self.members]

    def predict_uncertainty(self, X) -> EnsemblePrediction:  # noqa: N803
        """Mean plus the epistemic/aleatoric decomposition, always ``(n_rows, n_targets)``.

        No aleatoric component here: a sklearn regressor predicts a point, not a distribution, so
        the only uncertainty an ensemble of them can measure is the members' disagreement. The
        conformal calibrator is what turns that into an interval that nevertheless covers - see
        yg_eo_soilnet.uncertainty.conformal.
        """
        return aggregate(self.predict_members(X))

    def predict_frame(self, X) -> pd.DataFrame:  # noqa: N803
        """The wide output: mean, sigma and interval per target, as named columns.

        This is what serving returns when asked for uncertainty, and it is deliberately NOT what
        ``predict`` returns.
        """
        prediction = self.predict_uncertainty(X)
        frame = pd.DataFrame(index=getattr(X, "index", None))

        for index, target_name in enumerate(self.target_names):
            total_std = prediction.total_std[:, index]
            frame[f"{target_name}_pred"] = prediction.mean[:, index]
            frame[f"{target_name}_std"] = total_std
            frame[f"{target_name}_epistemic_std"] = prediction.epistemic_std[:, index]
            frame[f"{target_name}_aleatoric_std"] = prediction.aleatoric_std[:, index]

            calibrator = self.calibrators.get(target_name)
            if calibrator is not None:
                lower, upper = calibrator.intervals(prediction.mean[:, index], total_std)
                frame[f"{target_name}_lower"] = lower
                frame[f"{target_name}_upper"] = upper

        return frame

    # --- provenance ---------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Log-friendly provenance; every value is an MLflow-loggable scalar."""
        payload: dict[str, Any] = {
            "uncertainty_n_members": len(self.members),
            "uncertainty_bootstrapped": self.bootstrapped,
            "uncertainty_member_seeds": ",".join(str(seed) for seed in self.member_seeds),
        }
        for target_name, calibrator in self.calibrators.items():
            suffix = "" if len(self.target_names) <= 1 else f"_{target_name}"
            for key, value in calibrator.to_dict().items():
                payload[f"{key}{suffix}"] = value
        return payload
