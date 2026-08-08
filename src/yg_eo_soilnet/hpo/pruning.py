from __future__ import annotations

from typing import Any

import optuna

try:  # pragma: no cover - optional dependency, mirrors lightning_trainer.py
    from lightning.pytorch.callbacks import Callback as LightningCallback
except ImportError:  # pragma: no cover
    LightningCallback = object  # type: ignore[assignment]


def metric_to_float(value: Any) -> float | None:
    """A logged Lightning metric as a plain float, or None when it is not a single number.

    ``trainer.callback_metrics`` holds 0-d tensors; a study stored in SQLite needs builtins.
    """
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        value = value.item() if getattr(value, "ndim", 0) == 0 else value.numpy()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    # A pruner fed NaN would rank it against real values; treat it as "no reading".
    return None if result != result else result


class OptunaPruningCallback(LightningCallback):
    """Reports the objective metric to an Optuna trial each epoch and prunes hopeless trials.

    Written here rather than taken from ``optuna-integration`` on purpose: that package's callback
    is built against the ``pytorch_lightning`` namespace, while this repo drives a
    ``lightning.pytorch`` Trainer (see LightningTrainer._get_lightning_module). Mixing the two
    trips Lightning's callback type check, so the twenty lines below buy independence from that
    coupling and from optuna-integration's release cadence.

    It also keeps the best value it has seen. ``trainer.callback_metrics`` after ``fit()`` holds the
    *last* epoch, which under early stopping is `patience` epochs past the best one - scoring a
    trial on that would systematically understate it and disagree with the checkpoint the production
    run would select.
    """

    def __init__(self, trial: optuna.Trial, monitor: str, mode: str = "min", *, report: bool = True):
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max'; got {mode!r}.")
        self.trial = trial
        self.monitor = monitor
        self.mode = mode
        # When a trial averages several seeds, only the first repeat reports: Optuna keeps one
        # intermediate value per step, so later repeats would overwrite the first one's curve.
        self.report = report
        self.best_value: float | None = None
        self.best_epoch: int | None = None
        self._reported_epoch: int | None = None

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        return value > self.best_value if self.mode == "max" else value < self.best_value

    def on_validation_end(self, trainer, pl_module) -> None:
        # on_validation_end, NOT on_validation_epoch_end. Lightning runs callback
        # on_validation_epoch_end hooks *before* the LightningModule's, and `val_r2` is logged in
        # the module's hook (_regression_base._log_epoch_metrics). Reading it there yields the
        # previous epoch's value - None on epoch 0 - so every report, the best value and the best
        # epoch would be off by one. `val_loss` is logged in validation_step and so is current
        # either way, which is what made this easy to miss.
        if getattr(trainer, "sanity_checking", False):
            return

        epoch = int(getattr(trainer, "current_epoch", 0))
        if self._reported_epoch == epoch:
            return

        value = metric_to_float(trainer.callback_metrics.get(self.monitor))
        if value is None:
            # `val_r2` is skipped by _log_epoch_metrics when the batch is degenerate (n < 2, or a
            # constant target). A gap in the series is not grounds to prune.
            return

        self._reported_epoch = epoch
        if self._is_better(value):
            self.best_value, self.best_epoch = value, epoch

        if not self.report:
            return
        self.trial.report(value, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Trial {self.trial.number} pruned at epoch {epoch} ({self.monitor}={value:.5f})")
