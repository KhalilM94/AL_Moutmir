"""Running one trial: a built bundle in, an objective value out.

Deliberately not LightningTrainer. That class is built for a production run - it validates, tests,
predicts, assembles an evaluation frame and logs an MLflow child run with artifacts, and writes a
checkpoint. Multiplied by a few hundred trials that is most of the wall clock and a lot of disk, and
none of it informs the search. A trial only needs fit() and the best value of one metric; the
winning configuration is retrained through the normal path afterwards.
"""

from __future__ import annotations

import gc
import importlib
import logging
from dataclasses import dataclass

import optuna

from yg_eo_soilnet.hpo.pruning import OptunaPruningCallback
from yg_eo_soilnet.hpo.search_space import Objective

# Everything a trial must not do, whatever the registry says.
TRIAL_TRAINER_OVERRIDES = {
    "enable_checkpointing": False,  # hundreds of trials must not each write a .ckpt
    "logger": False,  # no lightning_logs/version_N per trial
    "enable_progress_bar": False,
    "enable_model_summary": False,
}


@dataclass
class TrialResult:
    value: float
    best_epoch: int | None
    epochs_run: int


def release_dataloader_workers() -> None:
    """Collect finished Trainer/DataLoader cycles now, between trials.

    Lightning's Trainer <-> LightningModule <-> callbacks <-> loops graph is cyclic, so dropping the
    last name binding never frees it - it waits for a generational GC pass. Until that happens its
    DataLoader iterators stay alive, and the next trial's fork() copies them into every new worker.
    When such a worker exits it finalizes the inherited iterator and hits
    `assert self._parent_pid == os.getpid()`, because is_alive() is only valid in the process that
    started the workers. Collecting here runs that finalization in the main process instead, where
    it is valid, and stops workers accumulating across a long study.
    """
    gc.collect()


def silence_lightning() -> None:
    """Quiet the per-trial chatter that would otherwise scroll a study off the screen."""
    for name in (
        "lightning.pytorch",
        "lightning.pytorch.utilities.rank_zero",
        "lightning.pytorch.accelerators",
        # seed_everything logs "Seed set to N" per trial, from the fabric namespace.
        "lightning.fabric",
        "lightning.fabric.utilities.seed",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
    optuna.logging.set_verbosity(optuna.logging.WARNING)


class TrialRunner:
    def __init__(self, objective: Objective, *, logger=None, fail_fast: bool = False, extra_callbacks=None):
        self.objective = objective
        self.logger = logger
        self.fail_fast = fail_fast
        # Lightning callbacks that observe a trial without steering it - the progress display's
        # epoch bar. Kept separate from the pruning and early-stopping callbacks the runner owns.
        self.extra_callbacks = list(extra_callbacks or [])

    def _lightning(self):
        return importlib.import_module("lightning.pytorch")

    def build_trainer(self, bundle, pruning_callback):
        lightning = self._lightning()
        trainer_kwargs = {**bundle.trainer_kwargs, **TRIAL_TRAINER_OVERRIDES}

        callbacks = [pruning_callback, *self.extra_callbacks]
        early_stopping = dict(bundle.callback_specs.get("early_stopping") or {})
        if early_stopping:
            # Rewritten to the study objective so early stopping and Optuna never disagree about
            # which direction is better. strict=False for the same reason the pruning callback
            # tolerates a gap: a metric _log_epoch_metrics skipped is not a misconfiguration.
            early_stopping.update(monitor=self.objective.metric, mode=self.objective.mode, strict=False)
            callbacks.append(lightning.callbacks.EarlyStopping(**early_stopping))

        return lightning.Trainer(**trainer_kwargs, callbacks=callbacks)

    def run(self, bundle, trial: optuna.Trial, *, report: bool = True) -> TrialResult:
        pruning_callback = OptunaPruningCallback(
            trial, monitor=self.objective.metric, mode=self.objective.mode, report=report
        )
        trainer = self.build_trainer(bundle, pruning_callback)

        try:
            try:
                trainer.fit(bundle.model, datamodule=bundle.datamodule)
            except optuna.TrialPruned:
                raise
            except Exception as exc:
                if self.fail_fast:
                    raise
                # A single bad corner of the space - an OOM, a non-finite loss from _shared_step, an
                # architecture combination the model rejects - must not end a long study.
                if self.logger is not None:
                    self.logger.warning(f"Trial {trial.number} failed and was pruned: {exc!r}", exc_info=True)
                raise optuna.TrialPruned(f"Trial {trial.number} raised {type(exc).__name__}: {exc}") from exc

            if pruning_callback.best_value is None:
                raise optuna.TrialPruned(
                    f"Trial {trial.number} never logged {self.objective.metric!r}; nothing to optimize."
                )

            return TrialResult(
                value=pruning_callback.best_value,
                best_epoch=pruning_callback.best_epoch,
                epochs_run=int(getattr(trainer, "current_epoch", 0)),
            )
        finally:
            # Drop this frame's reference so the Trainer becomes collectable once the caller drops
            # the bundle - Lightning leaves model._trainer pointing back here, so the bundle is the
            # other half of the cycle. TrialObjective owns the actual collection.
            trainer = None  # noqa: F841
