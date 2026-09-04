"""Training plumbing shared by the soil regression LightningModules.

Loss selection, target de-standardization, the train/val/test steps and the optimizer are identical
across architectures; only ``forward`` differs. Subclasses implement ``forward(batch)`` and call
``_init_regression_targets`` from their constructor.

``SoilGraphLightningModule`` deliberately keeps its own copy rather than inheriting from this: it is
covered by a large test suite and refactoring it is a separate change.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from lightning.pytorch import LightningModule

from yg_eo_soilnet.models.lightningmodules.losses import (
    STRUCTURAL_LOSSES,
    build_loss_fn,
    inverse_transform_targets,
)


def batch_get(batch: Any, key: str, default=None):
    """Read a key from a batch that may be a dict or a dataclass with Mapping-style access."""
    if isinstance(batch, Mapping):
        return batch.get(key, default)
    if hasattr(batch, "get"):
        return batch.get(key, default)
    if hasattr(batch, "__getitem__"):
        try:
            return batch[key]
        except Exception:
            return default
    return getattr(batch, key, default)


def as_float_list(value: Any) -> Optional[list[float]]:
    """Normalize array-likes to plain Python floats so hyperparameters stay pickle-safe."""
    if value is None:
        return None
    return [float(item) for item in torch.as_tensor(value, dtype=torch.float32).flatten().tolist()]


def as_float_matrix(value: Any) -> Optional[list[list[float]]]:
    """``as_float_list`` for a 2-D statistic, keeping its rows. Same pickle-safety reason."""
    if value is None:
        return None
    rows = torch.as_tensor(value, dtype=torch.float32)
    if rows.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix; got shape {tuple(rows.shape)}.")
    return [[float(item) for item in row] for row in rows.tolist()]


class SoilRegressionLightningBase(LightningModule):
    """Loss, target inversion, steps, metrics and optimizer for a soil regression head."""

    def _init_regression_targets(
        self,
        *,
        target_dim: int,
        target_mean: Optional[Any],
        target_scale: Optional[Any],
        target_transform: Optional[str],
        loss_name: str,
        huber_delta: float,
        # --- structure-aware losses (see lightningmodules/losses.py) -----------------------
        # Inert unless loss_name is one of mahalanobis / correlation_penalty / cosine.
        loss_base: str = "mse",
        loss_lambda: float = 0.1,
        loss_shrinkage: float = 0.05,
        loss_min_batch: int = 16,
        cosine_space: str = "original",
        target_covariance: Optional[Any] = None,
        learning_rate: float,
        optimizer_name: str,
        weight_decay: float,
        scheduler_type: str,
        scheduler_factor: float,
        scheduler_patience: int,
        scheduler_min_lr: float,
        scheduler_monitor: str,
        target_names: Optional[Any] = None,
        predict_variance: bool = False,
        beta_nll: float = 0.5,
    ) -> None:
        self.target_dim = int(target_dim)
        # --- heteroscedastic head ---------------------------------------------------------
        # With predict_variance the readout emits TWO numbers per target - a mean and a log
        # variance - and the loss becomes beta-NLL instead of the point loss. That is what makes a
        # prediction interval narrow where the data is clean and wide where it is noisy; an
        # ensemble on its own can only measure where its members disagree, which is a different
        # quantity and is often near-constant across the test set.
        #
        # Subclasses build their readout at `self.head_output_dim` rather than `self.target_dim`,
        # so the extra width is decided in exactly one place.
        self.predict_variance = bool(predict_variance)
        self.head_output_dim = self.target_dim * (2 if self.predict_variance else 1)
        self.beta_nll = float(beta_nll)
        # Only set when the subclass has not already: SoilCNNLightningModule assigns its own after
        # this call. Used to NAME the per-target metrics below, so a joint run's r2 can be read per
        # target instead of only in aggregate.
        if not getattr(self, "target_names", None):
            self.target_names = [str(name) for name in (target_names or [])]
        self.learning_rate = float(learning_rate)
        self.optimizer_name = str(optimizer_name).lower()
        self.weight_decay = float(weight_decay)
        self.scheduler_type = str(scheduler_type).lower()
        self.scheduler_factor = float(scheduler_factor)
        self.scheduler_patience = max(0, int(scheduler_patience))
        self.scheduler_min_lr = float(scheduler_min_lr)
        self.scheduler_monitor = str(scheduler_monitor)

        # Target standardization stats from the datamodule. The loss is computed in standardized
        # space; predict_step inverts so downstream evaluation sees original units.
        self.register_buffer(
            "target_mean",
            torch.zeros(self.target_dim) if target_mean is None else torch.as_tensor(target_mean, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "target_scale",
            torch.ones(self.target_dim) if target_scale is None else torch.as_tensor(target_scale, dtype=torch.float32),
            persistent=True,
        )
        # Buffers, not plain attributes: they must round-trip through state_dict. Derived from init
        # args alone, a checkpoint restore that lost the hyperparameters would silently skip the
        # inverse transform and report predictions in standardized units.
        self.register_buffer(
            "targets_are_standardized",
            torch.tensor(target_mean is not None and target_scale is not None),
            persistent=True,
        )
        self.target_transform = None if target_transform is None else str(target_transform).lower()
        if self.target_transform not in {None, "none", "log1p"}:
            raise ValueError("target_transform must be None or 'log1p'")
        self.register_buffer(
            "targets_are_log1p",
            torch.tensor(self.target_transform == "log1p"),
            persistent=True,
        )
        # A buffer for the same reason the two above are: a checkpoint restored without its
        # hyperparameters would otherwise read a 2*target_dim head as 2*target_dim TARGETS, and
        # report log variances as if they were predictions of targets that do not exist.
        self.register_buffer(
            "head_predicts_variance",
            torch.tensor(self.predict_variance),
            persistent=True,
        )

        # NOTE: val_loss is only comparable across runs that share loss_name - it is the monitor for
        # early stopping, checkpoint selection and the LR scheduler.
        self.loss_name = str(loss_name).lower()
        self.huber_delta = float(huber_delta)
        # A structural loss would be SILENTLY IGNORED on a variance head: _shared_step routes to
        # beta-NLL whenever the readout emits a log variance and never consults self.loss_fn. Refuse
        # here rather than let a run report a mahalanobis loss_name it never optimized.
        if self.predict_variance and self.loss_name in STRUCTURAL_LOSSES:
            raise ValueError(
                f"loss_name '{self.loss_name}' cannot be combined with a heteroscedastic head: "
                f"predict_variance replaces the point loss with beta-NLL. Set "
                f"uncertainty.heteroscedastic to false, or use a point loss."
            )
        # An nn.Module assigned here becomes a submodule, so the loss's own buffers - the whitening
        # matrix, the reference correlation - follow the model onto the accelerator.
        self.loss_fn = self._build_loss_fn(
            self.loss_name,
            self.huber_delta,
            loss_base=loss_base,
            loss_lambda=loss_lambda,
            loss_shrinkage=loss_shrinkage,
            loss_min_batch=loss_min_batch,
            cosine_space=cosine_space,
            target_dim=self.target_dim,
            target_covariance=target_covariance,
            target_mean=target_mean,
            target_scale=target_scale,
            target_transform=self.target_transform,
        )
        # stage -> {"n": float, and one length-target_dim float64 tensor per running sum}
        self._metric_state: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _build_loss_fn(loss_name: str, huber_delta: float, **kwargs):
        """The objective named by ``loss_name``; see ``lightningmodules.losses.build_loss_fn``.

        Kept as a method so the two-argument call every existing caller and test makes still selects
        the point losses exactly as it did.
        """
        return build_loss_fn(loss_name, huber_delta=huber_delta, **kwargs)

    # --- steps -------------------------------------------------------------

    # Bounds on the predicted log variance. Not cosmetic: an unclamped head can drive the variance
    # toward zero on a point it happens to fit early, at which point the NLL's 1/var term explodes
    # and the run dies with a non-finite loss. exp(-10) ~ 4.5e-5 and exp(10) ~ 2.2e4, which spans
    # every plausible noise level in STANDARDIZED space, where the target has unit variance.
    LOG_VARIANCE_MIN = -10.0
    LOG_VARIANCE_MAX = 10.0

    def _split_head_output(self, raw: torch.Tensor):
        """``(mean, log_variance)`` from the readout, with log_variance None on a point head."""
        if not self.predict_variance:
            return raw, None
        mean, log_variance = raw[..., : self.target_dim], raw[..., self.target_dim :]
        return mean, log_variance.clamp(self.LOG_VARIANCE_MIN, self.LOG_VARIANCE_MAX)

    def _beta_nll_loss(self, mean, log_variance, targets):
        """beta-NLL (Seitzer et al. 2022), reducing to Gaussian NLL at beta = 0.

        Plain Gaussian NLL has a well-known failure that matters here. The 1/var factor weights each
        point's mean-error by how certain the model already is, so a point it starts out uncertain
        about contributes almost nothing to the mean's gradient - and it therefore never learns to
        fit it, which retroactively justifies the large variance. The result is a model that has
        given up on its hard points while scoring well on NLL.

        beta-NLL multiplies each term by ``var^beta`` with the gradient stopped, which cancels that
        weighting back out. beta = 0 is plain NLL and beta = 1 recovers MSE-like weighting on the
        mean; 0.5 is the paper's recommendation and the default here.
        """
        variance = torch.exp(log_variance)
        negative_log_likelihood = 0.5 * (log_variance + (targets - mean) ** 2 / variance)
        if self.beta_nll > 0.0:
            negative_log_likelihood = negative_log_likelihood * variance.detach() ** self.beta_nll
        return negative_log_likelihood.mean()

    def _shared_step(self, batch: Any, stage: str):
        raw_output = self.forward(batch)
        predictions, log_variance = self._split_head_output(raw_output)
        targets = batch_get(batch, "y")
        if targets is None:
            raise KeyError("Batch is missing 'y'")
        targets = targets.to(device=predictions.device, dtype=predictions.dtype)

        if not torch.isfinite(targets).all():
            raise ValueError(f"Non-finite target values encountered during {stage} step.")
        if not torch.isfinite(predictions).all():
            raise ValueError(f"Non-finite prediction values encountered during {stage} step.")
        if log_variance is None:
            loss = self.loss_fn(predictions, targets)
        else:
            if not torch.isfinite(log_variance).all():
                raise ValueError(f"Non-finite predicted variance encountered during {stage} step.")
            loss = self._beta_nll_loss(predictions, log_variance, targets)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss encountered during {stage} step.")

        # The real sample count, not 1: Lightning weights the epoch mean by batch_size, so a constant
        # 1 makes the trailing partial batch count as much as a full one. This metric drives early
        # stopping, checkpoint selection and the LR scheduler.
        batch_size = int(targets.shape[0]) if targets.ndim else 1
        self.log(
            f"{stage}_loss",
            loss,
            batch_size=batch_size,
            prog_bar=stage != "train",
            on_step=False,
            on_epoch=True,
        )
        # A composite loss reports its two halves separately. Without them there is no way to tell a
        # lambda that is doing nothing from one that has swamped the accuracy term - both look like
        # a single number that went down.
        for name, value in getattr(self.loss_fn, "last_components", {}).items():
            self.log(
                f"{stage}_loss_{name}",
                value,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
            )
        self._accumulate_metrics(stage, predictions.detach(), targets.detach())
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int):
        return self._shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int):
        return self._shared_step(batch, "test")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        """Predictions in the target's original units, with sigma alongside on a variance head.

        The return type deliberately changes shape with the head: a bare tensor for a point head,
        so every existing caller is untouched, and a ``(mean, sigma)`` tuple when there is a sigma
        to report. LightningTrainer._flatten_predictions handles both.
        """
        mean, log_variance = self._split_head_output(self.forward(batch))
        if log_variance is None:
            return self.inverse_transform_targets(mean)
        sigma = torch.exp(0.5 * log_variance)
        # Order matters: the sigma inversion reads the STANDARDIZED mean, so it has to be computed
        # before `mean` is overwritten with the inverted one.
        return (
            self.inverse_transform_targets(mean),
            self.inverse_transform_sigma(sigma, mean),
        )

    # --- epoch metrics -----------------------------------------------------
    # R2 and the prediction/target standard-deviation ratio are the two numbers that say whether a
    # run has collapsed toward the target mean. Accumulated by hand because torchmetrics is not a
    # dependency of this project.

    def _metric_target_names(self) -> list[str]:
        """Names for the head's outputs, one per column, for metric keys."""
        names = [str(name) for name in (getattr(self, "target_names", None) or [])]
        if len(names) == self.target_dim:
            return names
        return [f"target_{index}" for index in range(self.target_dim)]

    def _accumulate_metrics(self, stage: str, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        # Summed over the BATCH axis only, so every accumulator is one value per target. Flattening
        # both axes - which is what this used to do - pools every target into a single r2. That is
        # not the mean of the per-target scores and it hides a target that has collapsed behind one
        # that has not, which on a joint run is exactly the failure worth seeing.
        predictions = predictions.double().reshape(-1, self.target_dim)
        targets = targets.double().reshape(-1, self.target_dim)
        state = self._metric_state.setdefault(stage, {})
        if not state:
            # Kept on the CPU so the accumulator does not pin accelerator memory for the epoch, and
            # so the addends below can be moved to it unconditionally.
            zeros = torch.zeros(self.target_dim, dtype=torch.float64)
            state.update(
                {
                    "n": 0.0,
                    "sum_y": zeros.clone(),
                    "sum_y2": zeros.clone(),
                    "sum_p": zeros.clone(),
                    "sum_p2": zeros.clone(),
                    "sse": zeros.clone(),
                }
            )
        state["n"] += float(targets.shape[0])
        state["sum_y"] += targets.sum(dim=0).cpu()
        state["sum_y2"] += (targets**2).sum(dim=0).cpu()
        state["sum_p"] += predictions.sum(dim=0).cpu()
        state["sum_p2"] += (predictions**2).sum(dim=0).cpu()
        state["sse"] += ((predictions - targets) ** 2).sum(dim=0).cpu()

    def _log_epoch_metrics(self, stage: str) -> None:
        state = self._metric_state.pop(stage, None)
        if not state or state["n"] < 2:
            return

        count = state["n"]
        target_variance = state["sum_y2"] / count - (state["sum_y"] / count) ** 2
        prediction_variance = state["sum_p2"] / count - (state["sum_p"] / count) ** 2

        names = self._metric_target_names()
        r2_scores: list[float] = []
        std_ratios: list[float] = []
        for index, name in enumerate(names):
            variance = float(target_variance[index])
            # A constant target has no variance to explain, so r2 is undefined rather than zero.
            # Skipped per column: one degenerate target must not suppress the others' metrics.
            if variance <= 1e-12:
                continue
            # In standardized space this is exactly 1 - MSE, which is how a run's health can be read
            # straight off test_loss.
            r2 = 1.0 - (float(state["sse"][index]) / count) / variance
            std_ratio = (max(float(prediction_variance[index]), 0.0) ** 0.5) / (variance**0.5)
            r2_scores.append(r2)
            std_ratios.append(std_ratio)
            if len(names) > 1:
                self.log(f"{stage}_r2_{name}", r2, on_step=False, on_epoch=True)
                self.log(f"{stage}_pred_std_ratio_{name}", std_ratio, on_step=False, on_epoch=True)

        if not r2_scores:
            return
        # The unsuffixed pair is the MEAN across targets, which is what makes it comparable with a
        # per-target run's single value. A single target leaves it numerically unchanged.
        self.log(
            f"{stage}_r2",
            sum(r2_scores) / len(r2_scores),
            on_step=False,
            on_epoch=True,
            prog_bar=stage == "val",
        )
        self.log(
            f"{stage}_pred_std_ratio",
            sum(std_ratios) / len(std_ratios),
            on_step=False,
            on_epoch=True,
        )

    def on_train_epoch_start(self) -> None:
        self._metric_state.pop("train", None)

    def on_validation_epoch_start(self) -> None:
        self._metric_state.pop("val", None)

    def on_test_epoch_start(self) -> None:
        self._metric_state.pop("test", None)

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics("test")

    # --- preprocessing state ------------------------------------------------

    def attach_preprocessing_state(self, state: Mapping[str, Any] | None) -> None:
        """Record the datamodule's fitted input statistics on this module.

        The target statistics were always buffers, so a restored checkpoint could invert its own
        predictions. The INPUT statistics were not: they lived only on the datamodule, so a
        restored model could not standardize raw data and was therefore not servable on its own.
        This carries them across.

        Carried in the checkpoint by :meth:`on_save_checkpoint` rather than in ``hparams``. It is
        deliberately NOT a hyperparameter: Lightning replays ``hparams`` as constructor keyword
        arguments on restore, so an entry that is not an ``__init__`` parameter would break
        ``load_from_checkpoint``. It is not a buffer either - the payload is ragged (per-modality
        dicts of differing widths, string vocabularies, feature names), which is not a tensor shape.

        Also the source of the feature names an explainer labels its rows with; without it the
        attribution seam falls back to positional names like ``static_0``.
        """
        if not state:
            return

        payload = dict(state)
        self.preprocessing_state = payload

        # Promoted to attributes because the attribution seam and the serving wrapper read them on
        # every call and should not have to know they came from a dict.
        self.static_feature_names = list(payload.get("static_feature_names") or [])
        self.modality_column_names = {
            str(name): list(columns) for name, columns in (payload.get("modality_column_names") or {}).items()
        }
        # The declared spatial-context subset of static_feature_names, and the coordinate column
        # names. Both ride this channel rather than being constructor arguments, for the same reason
        # static_feature_names does: they label attributions and change no shape, so making them
        # hyperparameters would put a purely cosmetic list in the load_from_checkpoint contract.
        self.context_feature_names = list(payload.get("context_feature_names") or [])
        self.coord_names = list(payload.get("coord_names") or [])

    def get_preprocessing_state(self) -> dict:
        """The attached state, empty when this module was never given one."""
        return dict(getattr(self, "preprocessing_state", None) or {})

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Put the fitted input statistics in the checkpoint, beside the weights.

        Plain builtins by contract (see SoilSequenceDataModule.preprocessing_state), so a checkpoint
        carrying them still loads under ``torch.load``'s ``weights_only=True`` default.
        """
        state = self.get_preprocessing_state()
        if state:
            checkpoint["preprocessing_state"] = state

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.attach_preprocessing_state(checkpoint.get("preprocessing_state"))

    # --- target inversion and optimizer ------------------------------------

    def inverse_transform_targets(self, predictions):
        """Map standardized predictions back to the target's original units.

        Un-standardize first, then undo log1p: the datamodule fits the standardization stats on
        already-transformed targets, so the two must be inverted in the opposite order.

        The arithmetic lives in ``losses.inverse_transform_targets`` because CosineStructureLoss
        needs the same inversion and must not reach back into the module that owns it.
        """
        return inverse_transform_targets(
            predictions,
            mean=self.target_mean,
            scale=self.target_scale,
            standardized=bool(self.targets_are_standardized),
            log1p=bool(self.targets_are_log1p),
        )

    def inverse_transform_sigma(self, sigma, predictions):
        """Map a standardized predictive sigma back to the target's original units.

        ``predictions`` must be the STANDARDIZED mean - the same tensor that goes into
        ``inverse_transform_targets``, not its output.

        Two steps, and only the first is the one people expect:

        1. Un-standardize. Scaling is linear, so the sigma scales with it: ``sigma * target_scale``.

        2. Undo log1p. This one is NOT linear, so there is no single factor that maps a standard
           deviation across it - the transform stretches the axis by a different amount at every
           point. The delta method takes the local slope: the forward transform is
           ``z = 10 * log1p(y)``, so ``y = expm1(z / 10)`` and ``dy/dz = exp(z / 10) / 10``,
           evaluated at this row's own predicted ``z``.

        The consequence worth stating: on a log1p target the returned sigma is asymmetric in
        substance even though it is reported as one number, and it grows with the prediction. A
        version of this that reused ``inverse_transform_targets`` on the sigma - the obvious
        shortcut - would produce a number in no units at all, and nothing downstream would catch it
        because it would still be positive and roughly the right magnitude.
        """
        if bool(self.targets_are_standardized):
            scale = self.target_scale.to(sigma.device)
            sigma = sigma * scale
            transformed = predictions * scale + self.target_mean.to(predictions.device)
        else:
            transformed = predictions

        if bool(self.targets_are_log1p):
            sigma = sigma * torch.exp(transformed / 10.0) / 10.0
        return sigma

    def configure_optimizers(self):
        if self.optimizer_name in {"adamw", "adam_w"}:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.scheduler_type in {"plateau", "reducelronplateau", "reduce_on_plateau"}:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                min_lr=self.scheduler_min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.scheduler_monitor,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return optimizer
