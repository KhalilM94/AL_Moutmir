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
from torch import nn

from lightning.pytorch import LightningModule


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
        learning_rate: float,
        optimizer_name: str,
        weight_decay: float,
        scheduler_type: str,
        scheduler_factor: float,
        scheduler_patience: int,
        scheduler_min_lr: float,
        scheduler_monitor: str,
    ) -> None:
        self.target_dim = int(target_dim)
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

        # NOTE: val_loss is only comparable across runs that share loss_name - it is the monitor for
        # early stopping, checkpoint selection and the LR scheduler.
        self.loss_name = str(loss_name).lower()
        self.huber_delta = float(huber_delta)
        self.loss_fn = self._build_loss_fn(self.loss_name, self.huber_delta)
        self._metric_state: dict[str, dict[str, float]] = {}

    @staticmethod
    def _build_loss_fn(loss_name: str, huber_delta: float):
        if loss_name in {"mse", "l2"}:
            return nn.MSELoss()
        if loss_name == "huber":
            return nn.HuberLoss(delta=huber_delta)
        if loss_name in {"smooth_l1", "smoothl1"}:
            return nn.SmoothL1Loss(beta=huber_delta)
        raise ValueError(f"Unknown loss_name '{loss_name}'; expected one of mse, huber, smooth_l1")

    # --- steps -------------------------------------------------------------

    def _shared_step(self, batch: Any, stage: str):
        predictions = self.forward(batch)
        targets = batch_get(batch, "y")
        if targets is None:
            raise KeyError("Batch is missing 'y'")
        targets = targets.to(device=predictions.device, dtype=predictions.dtype)

        if not torch.isfinite(targets).all():
            raise ValueError(f"Non-finite target values encountered during {stage} step.")
        if not torch.isfinite(predictions).all():
            raise ValueError(f"Non-finite prediction values encountered during {stage} step.")
        loss = self.loss_fn(predictions, targets)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss encountered during {stage} step.")

        # The real sample count, not 1: Lightning weights the epoch mean by batch_size, so a constant
        # 1 makes the trailing partial batch count as much as a full one. This metric drives early
        # stopping, checkpoint selection and the LR scheduler.
        self.log(
            f"{stage}_loss",
            loss,
            batch_size=int(targets.shape[0]) if targets.ndim else 1,
            prog_bar=stage != "train",
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
        return self.inverse_transform_targets(self.forward(batch))

    # --- epoch metrics -----------------------------------------------------
    # R2 and the prediction/target standard-deviation ratio are the two numbers that say whether a
    # run has collapsed toward the target mean. Accumulated by hand because torchmetrics is not a
    # dependency of this project.

    def _accumulate_metrics(self, stage: str, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        state = self._metric_state.setdefault(
            stage, {"n": 0.0, "sum_y": 0.0, "sum_y2": 0.0, "sum_p": 0.0, "sum_p2": 0.0, "sse": 0.0}
        )
        predictions = predictions.reshape(-1).double()
        targets = targets.reshape(-1).double()
        state["n"] += float(targets.numel())
        state["sum_y"] += float(targets.sum())
        state["sum_y2"] += float((targets**2).sum())
        state["sum_p"] += float(predictions.sum())
        state["sum_p2"] += float((predictions**2).sum())
        state["sse"] += float(((predictions - targets) ** 2).sum())

    def _log_epoch_metrics(self, stage: str) -> None:
        state = self._metric_state.pop(stage, None)
        if not state or state["n"] < 2:
            return

        count = state["n"]
        target_variance = state["sum_y2"] / count - (state["sum_y"] / count) ** 2
        prediction_variance = state["sum_p2"] / count - (state["sum_p"] / count) ** 2
        if target_variance <= 1e-12:
            return

        # In standardized space this is exactly 1 - MSE, which is how a run's health can be read
        # straight off test_loss.
        r2 = 1.0 - (state["sse"] / count) / target_variance
        std_ratio = (max(prediction_variance, 0.0) ** 0.5) / (target_variance**0.5)
        self.log(f"{stage}_r2", r2, on_step=False, on_epoch=True, prog_bar=stage == "val")
        self.log(f"{stage}_pred_std_ratio", std_ratio, on_step=False, on_epoch=True)

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

    # --- target inversion and optimizer ------------------------------------

    def inverse_transform_targets(self, predictions):
        """Map standardized predictions back to the target's original units.

        Un-standardize first, then undo log1p: the datamodule fits the standardization stats on
        already-transformed targets, so the two must be inverted in the opposite order.
        """
        if bool(self.targets_are_standardized):
            predictions = (
                predictions * self.target_scale.to(predictions.device)
                + self.target_mean.to(predictions.device)
            )
        if bool(self.targets_are_log1p):
            # Mirrors LogTransformer in yg_eo_soilnet.utils: forward is 10 * log1p(y).
            predictions = torch.expm1(predictions / 10.0)
        return predictions

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
