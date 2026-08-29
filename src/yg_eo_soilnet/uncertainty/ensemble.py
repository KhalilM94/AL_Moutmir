"""Ensemble mechanics: which seeds the members train at, and how their outputs combine.

Deliberately free of any sklearn or torch import. Both families call the same three functions on
plain arrays, which is what keeps the epistemic/aleatoric decomposition identical no matter which
framework produced the members - the same discipline yg_eo_soilnet.metrics applies to the point
metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

# `auto` bootstraps only estimators with no random_state; see should_bootstrap.
BOOTSTRAP_AUTO = "auto"
BOOTSTRAP_ALWAYS = "always"
BOOTSTRAP_NEVER = "never"
BOOTSTRAP_MODES = (BOOTSTRAP_AUTO, BOOTSTRAP_ALWAYS, BOOTSTRAP_NEVER)


def member_seeds(base_seed: int, n_members: int, stride: int = 1000) -> list[int]:
    """The seed each ensemble member trains at.

    A STRIDE rather than consecutive integers. Member seeds share a namespace with split.seed and
    with every registry entry's `random_seed`, and `base + 1` collides with the next entry's seed
    often enough to matter: two members drawing the same initialization are not two samples of the
    posterior, they are one sample counted twice, and the variance they report is too small with no
    sign that anything went wrong.
    """
    if n_members < 1:
        raise ValueError(f"n_members must be at least 1; got {n_members}")
    return [int(base_seed) + index * int(stride) for index in range(int(n_members))]


def should_bootstrap(estimator: Any, mode: str = BOOTSTRAP_AUTO) -> bool:
    """Whether this estimator's members must differ by a resample of the training rows.

    ``auto`` resamples for EVERY sklearn estimator, and the reason is worth stating because the
    obvious cheaper rule is wrong.

    The tempting test is "does it expose a ``random_state``?" - the same test
    ``ModelConfigFactory._seed_estimator`` uses to decide whether a seed can be pushed. It does not
    work here, because exposing a seed is not the same as using one. ``sklearn.linear_model.Ridge``
    reports ``random_state`` in ``get_params``, but only its ``sag``/``saga`` solvers consult it;
    under the default ``solver="auto"`` the fit is a closed-form solve and the seed is inert. That
    test therefore classifies Ridge as stochastic, five members train to identical coefficients, the
    ensemble standard deviation is exactly zero, and the run reports total confidence in every
    prediction - with nothing anywhere saying that no ensemble was formed. A name list has the same
    problem one registry entry later.

    Resampling every entry costs each member the ~36.8% of rows a bootstrap leaves out, which is a
    real if modest hit to each member's fit. It buys an ensemble that is an ensemble for every
    estimator, including ones nobody has classified yet - and for a convex model, bagging IS the
    classical epistemic estimate, not a substitute for one.

    ``never`` exists for the Lightning family, where a different weight initialization genuinely
    produces a different model and resampling on top of it would shrink the training set for no
    additional spread.
    """
    normalized = str(mode).lower()
    if normalized not in BOOTSTRAP_MODES:
        raise ValueError(f"bootstrap must be one of {BOOTSTRAP_MODES}; got {mode!r}")
    if normalized == BOOTSTRAP_NEVER:
        return False
    return True


def bootstrap_indices(n_rows: int, seed: int) -> np.ndarray:
    """Row positions for one bootstrap resample: `n_rows` draws with replacement.

    Its own function so the sklearn trainer and the tests draw the same rows for the same seed.
    """
    generator = np.random.default_rng(int(seed))
    return generator.integers(0, int(n_rows), size=int(n_rows))


@dataclass(frozen=True)
class EnsemblePrediction:
    """One ensemble's output for one array of inputs, shaped ``(n_rows, n_targets)`` throughout.

    The two variance components are kept apart rather than pre-summed because they answer different
    questions and have different remedies: epistemic shrinks if you collect more training data,
    aleatoric does not. A wide bar means nothing until you know which of the two produced it.
    """

    mean: np.ndarray
    epistemic_std: np.ndarray
    aleatoric_std: np.ndarray

    @property
    def total_std(self) -> np.ndarray:
        """The predictive standard deviation the interval is built from.

        Variances add, standard deviations do not - hence the sum under the root. Writing
        ``epistemic_std + aleatoric_std`` instead would overstate the width by up to 41%.
        """
        return np.sqrt(self.epistemic_std**2 + self.aleatoric_std**2)


def aggregate(
    member_predictions: Sequence[Any],
    member_sigmas: Optional[Sequence[Any]] = None,
) -> EnsemblePrediction:
    """Combine per-member outputs into a mean and its two variance components.

    ``member_predictions`` is one entry per member, each ``(n_rows,)`` or ``(n_rows, n_targets)``.
    ``member_sigmas``, when given, is the per-member ALEATORIC standard deviation from a
    heteroscedastic head, in the same shapes and the same units.

    The decomposition is the deep-ensemble one (Lakshminarayanan et al. 2017): the mixture of the
    members' Gaussians has mean ``mean(mu_k)`` and variance ``mean(sigma_k^2) + var(mu_k)``. Note
    that the aleatoric term averages VARIANCES, not standard deviations; averaging the standard
    deviations understates a mixture whose members disagree about the noise level.

    The population variance (``ddof=0``) is used for the epistemic term deliberately: this is the
    variance of the mixture that was actually fitted, not an estimate of the variance of a larger
    population of models it was sampled from.
    """
    if not len(member_predictions):
        raise ValueError("aggregate needs at least one member prediction")

    stacked = np.stack([_as_2d(values) for values in member_predictions], axis=0)
    mean = stacked.mean(axis=0)
    epistemic_std = stacked.std(axis=0, ddof=0)

    if member_sigmas is None:
        aleatoric_std = np.zeros_like(mean)
    else:
        if len(member_sigmas) != len(member_predictions):
            raise ValueError(
                f"member_sigmas has {len(member_sigmas)} entries but there are "
                f"{len(member_predictions)} members; they must correspond one to one."
            )
        sigmas = np.stack([_as_2d(values) for values in member_sigmas], axis=0)
        if sigmas.shape != stacked.shape:
            raise ValueError(
                f"member_sigmas shape {sigmas.shape} does not match member predictions "
                f"{stacked.shape}."
            )
        aleatoric_std = np.sqrt((sigmas**2).mean(axis=0))

    return EnsemblePrediction(
        mean=mean,
        epistemic_std=epistemic_std,
        aleatoric_std=aleatoric_std,
    )


def _as_2d(values: Any) -> np.ndarray:
    """A member's output as ``(n_rows, n_targets)`` float array.

    A single-target member is ``(n_rows,)``; widening it here means every caller downstream sees
    one shape and the single-target path is not a special case in five different places.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array
    raise ValueError(f"Member predictions must be 1-D or 2-D; got shape {array.shape}")
