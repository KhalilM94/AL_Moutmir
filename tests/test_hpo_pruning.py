from types import SimpleNamespace

import optuna
import pytest

from yg_eo_soilnet.hpo.pruning import OptunaPruningCallback, metric_to_float


class FakeTrainer:
    """The three attributes the callback reads off a lightning.pytorch.Trainer."""

    def __init__(self, callback_metrics, current_epoch=0, sanity_checking=False):
        self.callback_metrics = callback_metrics
        self.current_epoch = current_epoch
        self.sanity_checking = sanity_checking


class FakeTrial:
    def __init__(self, should_prune=False, number=0):
        self.number = number
        self.reports: list[tuple[float, int]] = []
        self._should_prune = should_prune

    def report(self, value, step):
        self.reports.append((value, step))

    def should_prune(self):
        return self._should_prune


def _run_epoch(callback, trainer):
    callback.on_validation_end(trainer, SimpleNamespace())


def test_metric_to_float_unwraps_a_zero_dim_tensor():
    torch = pytest.importorskip("torch")
    assert metric_to_float(torch.tensor(0.75)) == pytest.approx(0.75)


def test_metric_to_float_rejects_nan():
    """NaN must not reach the pruner, which would rank it against real values."""
    assert metric_to_float(float("nan")) is None


def test_metric_to_float_rejects_a_non_number():
    assert metric_to_float("not a metric") is None
    assert metric_to_float(None) is None


def test_callback_reports_the_monitored_metric():
    trial = FakeTrial()
    callback = OptunaPruningCallback(trial, monitor="val_r2")
    _run_epoch(callback, FakeTrainer({"val_r2": 0.42, "val_loss": 1.0}, current_epoch=3))

    assert trial.reports == [(0.42, 3)]


def test_callback_prunes_when_the_trial_says_so():
    trial = FakeTrial(should_prune=True, number=7)
    callback = OptunaPruningCallback(trial, monitor="val_r2")

    with pytest.raises(optuna.TrialPruned, match="Trial 7 pruned at epoch 2"):
        _run_epoch(callback, FakeTrainer({"val_r2": 0.1}, current_epoch=2))


def test_callback_ignores_the_sanity_check():
    """Sanity-check metrics predate any training, so they say nothing about the trial."""
    trial = FakeTrial(should_prune=True)
    callback = OptunaPruningCallback(trial, monitor="val_r2")
    _run_epoch(callback, FakeTrainer({"val_r2": 0.0}, sanity_checking=True))

    assert trial.reports == []


def test_a_missing_metric_is_a_gap_not_a_pruning_signal():
    """_log_epoch_metrics skips val_r2 on a degenerate batch; that must not end the trial."""
    trial = FakeTrial(should_prune=True)
    callback = OptunaPruningCallback(trial, monitor="val_r2")
    _run_epoch(callback, FakeTrainer({"val_loss": 0.5}, current_epoch=1))

    assert trial.reports == []


def test_the_same_epoch_is_reported_only_once():
    """Optuna raises if a step is reported twice, and Lightning can revisit an epoch hook."""
    trial = FakeTrial()
    callback = OptunaPruningCallback(trial, monitor="val_loss")
    trainer = FakeTrainer({"val_loss": 0.5}, current_epoch=0)
    _run_epoch(callback, trainer)
    _run_epoch(callback, trainer)

    assert trial.reports == [(0.5, 0)]


def test_the_callback_hooks_on_validation_end_not_on_validation_epoch_end():
    """Lightning runs callback on_validation_epoch_end BEFORE the LightningModule's.

    `val_r2` is logged in the module's hook (_regression_base._log_epoch_metrics), so reading it
    from on_validation_epoch_end yields the previous epoch's value - None on epoch 0. Every report,
    the best value and the best epoch would silently be off by one. `val_loss` is logged in
    validation_step and is current under either hook, which is what made this easy to miss.
    """
    # Checked on the class's own __dict__: the Lightning Callback base defines both hooks as
    # no-ops, so hasattr is true either way. What matters is which one we override.
    assert "on_validation_end" in OptunaPruningCallback.__dict__
    assert "on_validation_epoch_end" not in OptunaPruningCallback.__dict__


def test_callback_drives_a_real_optuna_trial():
    """End to end against a real study: a hopeless trial is pruned by the median pruner."""
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0),
    )

    def objective(trial):
        callback = OptunaPruningCallback(trial, monitor="val_loss")
        offset = trial.suggest_float("offset", 0.0, 10.0)
        for epoch in range(5):
            _run_epoch(callback, FakeTrainer({"val_loss": offset + epoch * 0.0}, current_epoch=epoch))
        return offset

    study.optimize(lambda t: objective(t), n_trials=2)
    # The good trial completed; the study holds a usable best value either way.
    assert study.best_value is not None
