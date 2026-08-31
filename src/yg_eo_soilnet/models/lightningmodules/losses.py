"""Training objectives for the soil regression heads, including the structure-aware ones.

The point losses - MSE, Huber, SmoothL1 - score every target independently and weight them
equally. Under ``MULTI_TARGET_MODE: joint`` that throws away the one thing a joint head knows
that a stack of per-target heads does not: the targets are correlated, and some combinations of
them are pedologically impossible. A prediction of high CEC with low organic matter costs exactly
as much as a plausible one, as long as the squared errors match.

The three losses below put that structure back into the objective:

``mahalanobis``            whitened MSE - errors are scored against the inverse target covariance,
                          so an error direction the training data never exhibits costs more than
                          an equally large one along a natural axis.
``correlation_penalty``    a point loss plus a regularizer pulling the batch's PREDICTED
                          correlation matrix toward the one measured on the training split.
``cosine``                 a point loss plus a penalty on the angle between the predicted and
                          measured target vectors, i.e. on their ratios rather than magnitudes.

THE SPACE THESE OPERATE IN is the thing to keep in mind while reading. The datamodule hands the
model targets that are already log1p-transformed (optionally) and then per-target standardized, and
only ``predict_step`` inverts that. So a loss here sees zero-mean, unit-variance columns, which has
two consequences worth stating rather than rediscovering:

* The covariance of the standardized training targets IS their Pearson correlation matrix. That is
  what makes the Mahalanobis form useful here rather than redundant - the per-target scale weighting
  it would otherwise contribute has already been applied by the standardizer, so what is left is
  purely the cross-target decorrelation.
* A sample sitting at the target mean is the ZERO VECTOR, and its direction is meaningless. The
  cosine loss therefore defaults to inverting back to original units, where the targets are positive
  quantities and an angle between them is the ratio the geochemistry actually constrains.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

# Point losses, by the name the registry spells. Also the set of values `loss_base` accepts, since a
# composite loss's accuracy term is exactly one of these.
BASE_LOSSES = {"mse", "l2", "huber", "smooth_l1", "smoothl1"}

# The losses that read across targets, and are therefore undefined on a single one.
STRUCTURAL_LOSSES = {"mahalanobis", "correlation_penalty", "cosine"}

LOSS_NAMES = BASE_LOSSES | STRUCTURAL_LOSSES

COSINE_SPACES = {"original", "standardized"}


def inverse_transform_targets(
    values: torch.Tensor,
    *,
    mean: Optional[torch.Tensor],
    scale: Optional[torch.Tensor],
    standardized: bool,
    log1p: bool,
) -> torch.Tensor:
    """Map standardized values back to the target's original units.

    Un-standardize first, then undo log1p: the datamodule fits the standardization stats on
    already-transformed targets, so the two must be inverted in the opposite order.

    A free function rather than a method because two callers need it - the LightningModule's
    ``predict_step`` and ``CosineStructureLoss`` - and the loss must not hold a reference back to
    the module that owns it. Duplicating the arithmetic instead is how the two silently drift.
    """
    if standardized and mean is not None and scale is not None:
        values = values * scale.to(values.device) + mean.to(values.device)
    if log1p:
        # Mirrors LogTransformer in yg_eo_soilnet.utils: forward is 10 * log1p(y).
        values = torch.expm1(values / 10.0)
    return values


def build_base_loss(loss_name: str, huber_delta: float) -> nn.Module:
    """One of the point losses, by name. Raises on anything else."""
    if loss_name in {"mse", "l2"}:
        return nn.MSELoss()
    if loss_name == "huber":
        return nn.HuberLoss(delta=huber_delta)
    if loss_name in {"smooth_l1", "smoothl1"}:
        return nn.SmoothL1Loss(beta=huber_delta)
    raise ValueError(f"Unknown loss_name '{loss_name}'; expected one of mse, huber, smooth_l1")


def _prepare_covariance(covariance: Any, target_dim: int, shrinkage: float) -> np.ndarray:
    """A symmetric, positive-definite covariance of the right shape, ready to be whitened."""
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (target_dim, target_dim):
        raise ValueError(
            f"target_covariance must be a {target_dim}x{target_dim} matrix; got shape {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("target_covariance contains non-finite values.")
    shrinkage = float(shrinkage)
    if not 0.0 <= shrinkage < 1.0:
        raise ValueError(f"loss_shrinkage must be in [0, 1); got {shrinkage}.")
    # Ridge toward the identity BEFORE inverting. Two nearly collinear targets - organic matter and
    # the C/N ratio are a plausible pair - make the covariance near-singular, and an uncorrected
    # inverse then puts almost all the loss on one nearly-unobservable contrast.
    matrix = (1.0 - shrinkage) * matrix + shrinkage * np.eye(target_dim)
    # np.cov is symmetric in exact arithmetic but not always in floating point, and eigh reads only
    # one triangle. Symmetrizing makes which triangle irrelevant.
    return 0.5 * (matrix + matrix.T)


class MahalanobisLoss(nn.Module):
    """``mean_batch[ d^T Sigma^-1 d ] / D`` for ``d = prediction - target``.

    Scores an error by how surprising its DIRECTION is under the training targets' covariance, not
    just by its length. An error that raises CEC while lowering organic matter runs against the
    correlation the data shows and is charged for it; an error that moves both together is charged
    close to what plain MSE would charge.

    Two implementation choices worth their lines:

    * The inverse is taken through an eigendecomposition with clipped eigenvalues, and kept as the
      whitening matrix ``W`` with ``Sigma^-1 = W W`` rather than as ``Sigma^-1`` itself. The loss is
      then ``||W d||^2``, which cannot come out negative however ill-conditioned the input was -
      whereas a directly inverted near-singular matrix can, and a negative loss trips the finiteness
      check in ``_shared_step`` far downstream of the actual cause.
    * The ``/ D`` normalization. Without it the loss has expectation ``D`` where MSE has ``1``, so
      switching to this loss would silently rescale every threshold tuned against ``val_loss`` -
      the LR-plateau factor, the early-stopping min_delta. With it, an identity covariance makes
      this loss numerically identical to ``nn.MSELoss``, which is also what the tests pin.
    """

    def __init__(self, covariance: Any, *, target_dim: int, shrinkage: float = 0.05):
        super().__init__()
        self.target_dim = int(target_dim)
        matrix = _prepare_covariance(covariance, self.target_dim, shrinkage)

        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        largest = float(eigenvalues.max())
        if largest <= 0.0:
            raise ValueError("target_covariance has no positive eigenvalue; it is not a covariance.")
        # A relative floor, not an absolute one: what counts as a degenerate direction depends on the
        # scale of the matrix, and in standardized space that scale is ~1 but need not be exactly.
        floor = largest * 1e-6
        clipped = np.maximum(eigenvalues, floor)
        self.condition_number = float(clipped.max() / clipped.min())
        if not np.allclose(clipped, eigenvalues):
            logger.warning(
                "target_covariance is rank-deficient; %d of %d eigenvalues were clipped to %.3e. "
                "Consider raising loss_shrinkage.",
                int((clipped != eigenvalues).sum()),
                self.target_dim,
                floor,
            )
        logger.info(
            "MahalanobisLoss built over %d targets; condition number %.1f after shrinkage.",
            self.target_dim,
            self.condition_number,
        )
        whitening = (eigenvectors * (clipped**-0.5)) @ eigenvectors.T

        # Not persistent: it is derived from the `target_covariance` hyperparameter, which
        # save_hyperparameters() already puts in the checkpoint, and the loss is a training-time
        # object that predict_step never touches. Non-persistent buffers still follow .to(device).
        self.register_buffer(
            "whitening", torch.as_tensor(whitening, dtype=torch.float32), persistent=False
        )

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        difference = predictions - targets
        # The whitening matrix is symmetric, so `d @ W` and `W @ d` agree and the transpose is noise.
        whitened = difference @ self.whitening.to(difference.dtype)
        return (whitened**2).sum(dim=-1).mean() / self.target_dim


class CorrelationPenaltyLoss(nn.Module):
    """``base(pred, target) + lambda * mean_{i<j} (rho(pred_i, pred_j) - R_train_ij)^2``.

    Keeps a point loss for accuracy and adds a regularizer that pushes the correlation structure of
    the model's OUTPUTS toward the structure measured on the training split. A joint head trained on
    MSE alone will happily produce predictions that are more correlated than reality (they all track
    the same dominant feature) or less (each target is fit independently); neither shows up in a
    per-target R2, and both are visible here.

    The reference is the fixed training-set correlation matrix rather than the batch's own
    ``rho(y)``. A batch of 64 estimates a correlation coefficient with a standard error around 0.12,
    so a batch-computed reference would spend most of its gradient chasing sampling noise.

    Below ``min_batch`` rows the penalty is SKIPPED rather than computed from an unusable estimate.
    That is what makes the trailing partial validation batch harmless - drop_last is train-only.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        reference_correlation: Any,
        *,
        target_dim: int,
        weight: float = 0.1,
        min_batch: int = 16,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.target_dim = int(target_dim)
        self.weight = float(weight)
        self.min_batch = max(2, int(min_batch))
        self.last_components: dict[str, float] = {}

        matrix = np.asarray(reference_correlation, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape != (self.target_dim, self.target_dim):
            raise ValueError(
                f"target_covariance must be a {self.target_dim}x{self.target_dim} matrix; "
                f"got shape {matrix.shape}."
            )
        if not np.isfinite(matrix).all():
            raise ValueError("target_covariance contains non-finite values.")
        # The datamodule fits this on STANDARDIZED targets, so it already is a correlation matrix.
        # Re-normalizing by the diagonal anyway costs nothing and makes the class correct if it is
        # ever handed a covariance in raw units.
        deviation = np.sqrt(np.clip(np.diag(matrix), 1e-12, None))
        correlation = matrix / np.outer(deviation, deviation)
        self.register_buffer(
            "reference", torch.as_tensor(correlation, dtype=torch.float32), persistent=False
        )
        # Upper-triangle positions, so each pair is counted once. The diagonal is 1 on both sides by
        # construction and would only dilute the mean with a term that is always zero.
        rows, columns = np.triu_indices(self.target_dim, k=1)
        self.register_buffer("pair_rows", torch.as_tensor(rows, dtype=torch.long), persistent=False)
        self.register_buffer(
            "pair_columns", torch.as_tensor(columns, dtype=torch.long), persistent=False
        )

    @staticmethod
    def _batch_correlation(values: torch.Tensor) -> torch.Tensor:
        """Pearson correlation across the columns of ``[B, D]``, differentiable and clamped."""
        centered = values - values.mean(dim=0, keepdim=True)
        # ddof=0, matching how the datamodule standardizes. The floor keeps a collapsed column - a
        # real early-training state, and exactly what pred_std_ratio reports - from dividing by zero.
        deviation = centered.pow(2).mean(dim=0).sqrt().clamp_min(1e-6)
        covariance = (centered.T @ centered) / centered.shape[0]
        correlation = covariance / torch.outer(deviation, deviation)
        # Floating point can put a coefficient a hair outside [-1, 1]; squaring the difference then
        # rewards the model for overshooting.
        return correlation.clamp(-1.0, 1.0)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base = self.base_loss(predictions, targets)
        if predictions.shape[0] < self.min_batch:
            self.last_components = {"base": float(base.detach()), "penalty": 0.0}
            return base

        predicted = self._batch_correlation(predictions)
        reference = self.reference.to(predictions.dtype)
        difference = (
            predicted[self.pair_rows, self.pair_columns]
            - reference[self.pair_rows, self.pair_columns]
        )
        # Mean over pairs, not sum: lambda then keeps its meaning when TARGET_COLUMNS changes length,
        # where a sum would scale with D^2 and quietly re-tune itself.
        penalty = difference.pow(2).mean()
        self.last_components = {"base": float(base.detach()), "penalty": float(penalty.detach())}
        return base + self.weight * penalty


class CosineStructureLoss(nn.Module):
    """``base(pred, target) + lambda * mean_batch[ 1 - cos(pred, target) ]``.

    Penalizes getting the RATIOS between targets wrong, independently of magnitude. Where the
    correlation penalty is a batch statistic and needs a batch large enough to estimate one, this is
    per-row and works at any batch size.

    ``space`` decides which vectors the angle is measured between:

    ``original``      invert the standardization and log1p first, so the angle is between vectors of
                      positive physical quantities and 1 - cos is the ratio distortion the soil
                      chemistry actually constrains. The default. Because every component is then
                      positive, cos lives in a narrow band near 1 and lambda has to be larger than
                      it would be for the other losses to have comparable effect.
    ``standardized``  measure in the space the loss already runs in. The angle is then between
                      vectors of z-scores: it asks whether the model gets the SHAPE of the joint
                      anomaly right (all targets high, versus high carbon with low carbonate). A
                      legitimate quantity, but not the ratio, and rows near the target mean are
                      near-zero vectors whose direction is noise - hence the norm floor below.
    """

    NORM_FLOOR = 1e-3

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        space: str = "original",
        weight: float = 0.1,
        target_mean: Optional[Sequence[float]] = None,
        target_scale: Optional[Sequence[float]] = None,
        target_transform: Optional[str] = None,
    ):
        super().__init__()
        space = str(space).lower()
        if space not in COSINE_SPACES:
            raise ValueError(
                f"Unknown cosine_space {space!r}; expected one of: {', '.join(sorted(COSINE_SPACES))}."
            )
        self.base_loss = base_loss
        self.space = space
        self.weight = float(weight)
        self.last_components: dict[str, float] = {}

        self.standardized = target_mean is not None and target_scale is not None
        self.log1p = str(target_transform).lower() == "log1p"
        if self.space == "original" and not (self.standardized or self.log1p):
            # Nothing to invert means the two spaces coincide. Not an error - a run with no target
            # transform at all is legal - but it is worth saying so rather than letting the config
            # read as though it selected something it did not.
            logger.info(
                "cosine_space='original' with untransformed targets: identical to 'standardized'."
            )
        # Non-persistent copies of statistics the owning module already carries as its own buffers.
        # Duplicated rather than shared so the loss never reaches back into its parent.
        self.register_buffer(
            "target_mean",
            torch.as_tensor(list(target_mean or []), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "target_scale",
            torch.as_tensor(list(target_scale or []), dtype=torch.float32),
            persistent=False,
        )

    def _to_scored_space(self, values: torch.Tensor) -> torch.Tensor:
        if self.space != "original":
            return values
        return inverse_transform_targets(
            values,
            mean=self.target_mean if self.standardized else None,
            scale=self.target_scale if self.standardized else None,
            standardized=self.standardized,
            log1p=self.log1p,
        )

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base = self.base_loss(predictions, targets)
        scored_predictions = self._to_scored_space(predictions)
        scored_targets = self._to_scored_space(targets)

        similarity = torch.nn.functional.cosine_similarity(
            scored_predictions, scored_targets, dim=-1, eps=1e-8
        )
        deviation = 1.0 - similarity
        # A row whose target vector is essentially zero has no direction to reproduce. In
        # standardized space that is a sample sitting at the mean of every target, which is common;
        # scoring its angle would inject pure noise into the gradient.
        measurable = scored_targets.norm(dim=-1) > self.NORM_FLOOR
        if bool(measurable.all()):
            penalty = deviation.mean()
        elif bool(measurable.any()):
            penalty = (deviation * measurable).sum() / measurable.sum()
        else:
            penalty = deviation.sum() * 0.0

        self.last_components = {"base": float(base.detach()), "penalty": float(penalty.detach())}
        return base + self.weight * penalty


def build_loss_fn(
    loss_name: str,
    *,
    huber_delta: float = 1.0,
    loss_base: str = "mse",
    loss_lambda: float = 0.1,
    loss_shrinkage: float = 0.05,
    loss_min_batch: int = 16,
    cosine_space: str = "original",
    target_dim: int = 1,
    target_covariance: Any = None,
    target_mean: Optional[Sequence[float]] = None,
    target_scale: Optional[Sequence[float]] = None,
    target_transform: Optional[str] = None,
) -> nn.Module:
    """The training objective named by ``loss_name``.

    Every failure mode is raised HERE, at model construction, rather than at the first training
    step: a run that dies after the data has been assembled and the first epoch has started has
    already cost minutes, and a structural loss that silently degenerates to its base term costs an
    entire experiment because nothing downstream looks wrong.

    THE KEYWORD ARGUMENTS ARE PER-LOSS, NOT UNIVERSAL. Each branch below reads only its own subset
    and the rest are ignored in silence, which the registry comment spells out for whoever is
    editing the YAML:

        mse                  (none)
        huber | smooth_l1    huber_delta
        mahalanobis          loss_shrinkage, target_covariance
        correlation_penalty  loss_base (+ huber_delta through it), loss_lambda, loss_min_batch,
                             target_covariance
        cosine               loss_base (+ huber_delta through it), loss_lambda, cosine_space,
                             target_mean / target_scale / target_transform

    One asymmetry is deliberate and worth naming, because it reads as a bug from either side.
    ``loss_base`` is validated for EVERY structural loss, ``mahalanobis`` included, even though
    mahalanobis never uses it - the check sits above the branch. ``cosine_space`` is validated only
    inside CosineStructureLoss, so a nonsense value passes silently under the other two. Neither is
    worth tightening on its own: moving the loss_base check down would let a typo through on the
    losses that DO use it if the branches are ever reordered, and hoisting the cosine_space check up
    would reject a config key that has no effect on the selected loss. If you change one, change
    both, and say which way you went.
    """
    loss_name = str(loss_name).lower()
    if loss_name in BASE_LOSSES:
        return build_base_loss(loss_name, huber_delta)
    if loss_name not in STRUCTURAL_LOSSES:
        raise ValueError(
            f"Unknown loss_name '{loss_name}'; expected one of "
            f"{', '.join(sorted(BASE_LOSSES | STRUCTURAL_LOSSES))}"
        )

    if int(target_dim) < 2:
        raise ValueError(
            f"loss_name '{loss_name}' reads across targets and needs at least 2 of them, but this "
            f"model has {int(target_dim)}. Set MULTI_TARGET_MODE: joint in configs/data_spec.yml, or "
            f"use a point loss (mse, huber, smooth_l1)."
        )

    loss_base = str(loss_base).lower()
    if loss_base not in BASE_LOSSES:
        raise ValueError(
            f"Unknown loss_base '{loss_base}'; expected one of {', '.join(sorted(BASE_LOSSES))}"
        )

    if loss_name == "cosine":
        return CosineStructureLoss(
            build_base_loss(loss_base, huber_delta),
            space=cosine_space,
            weight=loss_lambda,
            target_mean=target_mean,
            target_scale=target_scale,
            target_transform=target_transform,
        )

    if target_covariance is None:
        raise ValueError(
            f"loss_name '{loss_name}' needs the training targets' covariance, but none was supplied. "
            f"LightningConfigFactory injects it from SoilSequenceDataModule.target_covariance_, "
            f"which _fit_normalization computes in setup(); a hand-built module must pass "
            f"target_covariance itself."
        )

    if loss_name == "mahalanobis":
        return MahalanobisLoss(
            target_covariance, target_dim=int(target_dim), shrinkage=loss_shrinkage
        )
    return CorrelationPenaltyLoss(
        build_base_loss(loss_base, huber_delta),
        target_covariance,
        target_dim=int(target_dim),
        weight=loss_lambda,
        min_batch=loss_min_batch,
    )
