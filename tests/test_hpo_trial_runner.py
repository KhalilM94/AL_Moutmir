import sys
from types import SimpleNamespace

import optuna
import pytest

from yg_eo_soilnet.hpo.search_space import Objective
from yg_eo_soilnet.hpo.trial_runner import (
    TRIAL_TRAINER_OVERRIDES,
    TrialRunner,
    UnrecoverableAcceleratorError,
    cuda_context_is_dead,
    release_dataloader_workers,
)


class FakeBundle:
    def __init__(self, trainer_kwargs=None, callback_specs=None):
        self.name = "fake_model"
        self.model = SimpleNamespace()
        self.datamodule = SimpleNamespace()
        self.trainer_kwargs = trainer_kwargs or {"max_epochs": 5, "enable_checkpointing": True, "deterministic": True}
        self.callback_specs = callback_specs if callback_specs is not None else {
            "early_stopping": {"monitor": "val_loss", "mode": "min", "patience": 30},
            "checkpoint": {"monitor": "val_loss", "mode": "min", "save_top_k": 1},
        }


class RecordingTrainer:
    """Stands in for lightning.pytorch.Trainer, replaying a metric series through the callbacks."""

    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.callbacks = kwargs.get("callbacks", [])
        self.callback_metrics = {}
        self.current_epoch = 0
        self.sanity_checking = False
        self.series: list[dict] = []
        RecordingTrainer.last = self

    def fit(self, model, datamodule=None):
        for epoch, metrics in enumerate(self.series):
            self.current_epoch = epoch
            self.callback_metrics = metrics
            for callback in self.callbacks:
                hook = getattr(callback, "on_validation_end", None)
                if hook is not None:
                    hook(self, model)


class FakeEarlyStopping:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _runner(monkeypatch, series, *, objective=None, bundle=None, **runner_kwargs):
    fake_lightning = SimpleNamespace(
        Trainer=RecordingTrainer,
        callbacks=SimpleNamespace(EarlyStopping=FakeEarlyStopping),
    )
    runner = TrialRunner(objective or Objective(metric="val_r2", direction="maximize"), **runner_kwargs)
    monkeypatch.setattr(runner, "_lightning", lambda: fake_lightning)

    original_init = RecordingTrainer.__init__

    def init_with_series(self, **kwargs):
        original_init(self, **kwargs)
        self.series = series

    monkeypatch.setattr(RecordingTrainer, "__init__", init_with_series)
    return runner, bundle or FakeBundle()


def _trial():
    return optuna.create_study(direction="maximize").ask()


# --- forced trainer settings -------------------------------------------------


def test_a_trial_never_checkpoints_or_writes_lightning_logs(monkeypatch):
    """Hundreds of trials must leave no .ckpt files and no lightning_logs/version_N directories."""
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.5}])
    runner.run(bundle, _trial())

    kwargs = RecordingTrainer.last.kwargs
    for key, expected in TRIAL_TRAINER_OVERRIDES.items():
        assert kwargs[key] is expected, key
    # Registry settings that are not about noise or disk survive untouched.
    assert kwargs["max_epochs"] == 5
    assert kwargs["deterministic"] is True


def test_early_stopping_is_rewired_to_the_study_objective(monkeypatch):
    """Otherwise early stopping could halt on val_loss while Optuna scores val_r2."""
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.5}])
    runner.run(bundle, _trial())

    early_stopping = [cb for cb in RecordingTrainer.last.callbacks if isinstance(cb, FakeEarlyStopping)][0]
    assert early_stopping.kwargs["monitor"] == "val_r2"
    assert early_stopping.kwargs["mode"] == "max"
    assert early_stopping.kwargs["strict"] is False
    assert early_stopping.kwargs["patience"] == 30  # the registry's value is kept


def test_the_checkpoint_spec_never_becomes_a_callback(monkeypatch):
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.5}])
    runner.run(bundle, _trial())

    assert len(RecordingTrainer.last.callbacks) == 2  # pruning + early stopping only


def test_a_bundle_without_an_early_stopping_spec_still_runs(monkeypatch):
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.5}], bundle=FakeBundle(callback_specs={}))
    result = runner.run(bundle, _trial())

    assert result.value == 0.5
    assert not any(isinstance(cb, FakeEarlyStopping) for cb in RecordingTrainer.last.callbacks)


# --- scoring -----------------------------------------------------------------


def test_the_objective_is_the_best_epoch_not_the_last(monkeypatch):
    """Under early stopping the last epoch is `patience` epochs past the best one."""
    series = [{"val_r2": 0.1}, {"val_r2": 0.9}, {"val_r2": 0.4}, {"val_r2": 0.2}]
    runner, bundle = _runner(monkeypatch, series)
    result = runner.run(bundle, _trial())

    assert result.value == 0.9
    assert result.best_epoch == 1


def test_best_means_lowest_when_minimizing(monkeypatch):
    series = [{"val_loss": 1.0}, {"val_loss": 0.3}, {"val_loss": 0.8}]
    runner, bundle = _runner(
        monkeypatch, series, objective=Objective(metric="val_loss", direction="minimize")
    )
    result = runner.run(bundle, _trial())

    assert result.value == 0.3


def test_a_metric_that_is_never_logged_prunes_the_trial(monkeypatch):
    """val_r2 is skipped by _log_epoch_metrics on a degenerate validation split."""
    runner, bundle = _runner(monkeypatch, [{"val_loss": 0.5}, {"val_loss": 0.4}])

    with pytest.raises(optuna.TrialPruned, match="never logged 'val_r2'"):
        runner.run(bundle, _trial())


# --- failure policy ----------------------------------------------------------


def test_a_raising_trial_is_pruned_rather_than_killing_the_study(monkeypatch):
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.5}])
    monkeypatch.setattr(RecordingTrainer, "fit", lambda self, model, datamodule=None: 1 / 0)

    with pytest.raises(optuna.TrialPruned, match="raised ZeroDivisionError"):
        runner.run(bundle, _trial())


def test_fail_fast_surfaces_the_original_exception(monkeypatch):
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.5}], fail_fast=True)
    monkeypatch.setattr(RecordingTrainer, "fit", lambda self, model, datamodule=None: 1 / 0)

    with pytest.raises(ZeroDivisionError):
        runner.run(bundle, _trial())


def test_a_pruned_trial_propagates_as_pruned(monkeypatch):
    """TrialPruned from the callback must not be swallowed by the generic failure handler."""
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0)
    )
    # A strong completed baseline, with the intermediates the median pruner compares against.
    baseline = study.ask()
    baseline.report(10.0, step=0)
    baseline.report(10.0, step=1)
    study.tell(baseline, 10.0)

    runner, bundle = _runner(monkeypatch, [{"val_r2": -5.0}, {"val_r2": -4.0}])
    with pytest.raises(optuna.TrialPruned, match="pruned at epoch"):
        runner.run(bundle, study.ask())


def test_the_trainer_is_collectable_once_the_bundle_is_dropped(monkeypatch):
    """run() must not leave a reference behind, or the Trainer's DataLoader iterators outlive it.

    Modelled on the real cycle: Lightning points model._trainer back at the Trainer, so the bundle
    is the other half. Uses its own trainer class because RecordingTrainer pins the last instance
    on a class attribute.
    """
    import gc
    import weakref

    created: list[weakref.ref] = []

    class UnpinnedTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.callbacks = kwargs.get("callbacks", [])
            self.callback_metrics = {"val_r2": 0.5}
            self.current_epoch = 0
            self.sanity_checking = False
            created.append(weakref.ref(self))

        def fit(self, model, datamodule=None):
            model.trainer = self  # the cycle Lightning creates
            for callback in self.callbacks:
                hook = getattr(callback, "on_validation_end", None)
                if hook is not None:
                    hook(self, model)

    fake_lightning = SimpleNamespace(
        Trainer=UnpinnedTrainer, callbacks=SimpleNamespace(EarlyStopping=FakeEarlyStopping)
    )
    runner = TrialRunner(Objective(metric="val_r2", direction="maximize"))
    monkeypatch.setattr(runner, "_lightning", lambda: fake_lightning)

    bundle = FakeBundle()
    runner.run(bundle, _trial())
    assert created[0]() is not None  # still reachable through bundle.model.trainer

    del bundle
    gc.collect()
    assert created[0]() is None


def test_report_false_still_tracks_the_best_without_reporting(monkeypatch):
    """Seed repeats past the first must not overwrite the first repeat's intermediate curve."""
    trial = _trial()
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.2}, {"val_r2": 0.7}])
    result = runner.run(bundle, trial, report=False)

    assert result.value == 0.7
    assert trial.storage.get_trial(trial._trial_id).intermediate_values == {}


# --- a dead accelerator is not a bad hyperparameter ---------------------------


class FakeCuda:
    """Stands in for torch.cuda: `initialized` gates the probe, `alive` decides its verdict."""

    def __init__(self, initialized=True, alive=True):
        self.initialized = initialized
        self.alive = alive
        self.empty_cache_calls = 0

    def is_initialized(self):
        return self.initialized

    def empty_cache(self):
        self.empty_cache_calls += 1


def _fake_torch(monkeypatch, initialized=True, alive=True):
    """Install a stand-in `torch` in sys.modules, which is where the probe looks it up."""
    cuda = FakeCuda(initialized=initialized, alive=alive)

    def zeros(*args, **kwargs):
        if not cuda.alive:
            raise RuntimeError("CUDA error: unknown error")
        return SimpleNamespace(add_=lambda *a: SimpleNamespace(cpu=lambda: None))

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, zeros=zeros))
    return cuda


def test_a_dead_context_aborts_the_study_instead_of_pruning(monkeypatch):
    """Pruning a dead GPU would spend the rest of the study reloading data to fail again."""
    _fake_torch(monkeypatch, alive=False)
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.5}])
    monkeypatch.setattr(
        RecordingTrainer, "fit", lambda self, model, datamodule=None: (_ for _ in ()).throw(RuntimeError("CUDA error"))
    )

    with pytest.raises(UnrecoverableAcceleratorError, match="died during trial"):
        runner.run(bundle, _trial())


def test_the_abort_message_says_how_to_recover(monkeypatch):
    _fake_torch(monkeypatch, alive=False)
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.5}])
    monkeypatch.setattr(RecordingTrainer, "fit", lambda self, model, datamodule=None: 1 / 0)

    with pytest.raises(UnrecoverableAcceleratorError, match="wsl --shutdown"):
        runner.run(bundle, _trial())


def test_a_live_context_still_prunes_the_trial(monkeypatch):
    """An OOM leaves the context usable: a too-large corner of the space stays one pruned trial."""
    _fake_torch(monkeypatch, alive=True)
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.5}])
    monkeypatch.setattr(
        RecordingTrainer,
        "fit",
        lambda self, model, datamodule=None: (_ for _ in ()).throw(MemoryError("CUDA out of memory")),
    )

    with pytest.raises(optuna.TrialPruned, match="raised MemoryError"):
        runner.run(bundle, _trial())


def test_a_process_with_no_cuda_context_never_probes(monkeypatch):
    """A CPU run must behave exactly as it did before the probe existed."""
    _fake_torch(monkeypatch, initialized=False, alive=False)
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.5}])
    monkeypatch.setattr(RecordingTrainer, "fit", lambda self, model, datamodule=None: 1 / 0)

    with pytest.raises(optuna.TrialPruned, match="raised ZeroDivisionError"):
        runner.run(bundle, _trial())


def test_fail_fast_still_wins_over_the_probe(monkeypatch):
    _fake_torch(monkeypatch, alive=False)
    runner, bundle = _runner(monkeypatch, [{"val_r2": 0.5}], fail_fast=True)
    monkeypatch.setattr(RecordingTrainer, "fit", lambda self, model, datamodule=None: 1 / 0)

    with pytest.raises(ZeroDivisionError):
        runner.run(bundle, _trial())


def test_cuda_context_is_dead_is_false_without_torch(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    assert cuda_context_is_dead() is False


# --- releasing a trial's resources --------------------------------------------


def test_the_allocator_is_emptied_between_trials(monkeypatch):
    """Trials range from one small block to five wide ones; freed blocks are the wrong shapes."""
    cuda = _fake_torch(monkeypatch)

    release_dataloader_workers()

    assert cuda.empty_cache_calls == 1


def test_nothing_is_emptied_when_no_context_was_built(monkeypatch):
    cuda = _fake_torch(monkeypatch, initialized=False)

    release_dataloader_workers()

    assert cuda.empty_cache_calls == 0


def test_releasing_works_without_torch_imported(monkeypatch):
    """The module lazy-imports on purpose; the collect must still run on a torch-free path."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    release_dataloader_workers()
