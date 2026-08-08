import logging
import optuna
import pytest
from io import StringIO
from types import SimpleNamespace

from yg_eo_soilnet.hpo.progress import (
    StudyProgress,
    TqdmLoggingHandler,
    resolve_mode,
    tqdm_safe_logging,
)
from yg_eo_soilnet.hpo.search_space import Objective
from yg_eo_soilnet.hpo.tracker import ObjectiveTracker

MAXIMIZE = Objective(metric="val_r2", direction="maximize")


class TtyStream(StringIO):
    """A StringIO that claims to be a terminal, so `auto` resolves to bars."""

    def isatty(self):
        return True


class FakeTrainer:
    def __init__(self, callback_metrics=None, current_epoch=0, sanity_checking=False, max_epochs=10):
        self.callback_metrics = callback_metrics or {}
        self.current_epoch = current_epoch
        self.sanity_checking = sanity_checking
        self.max_epochs = max_epochs


def _frozen_trial(study, value=0.5, prune=False):
    """Run one real trial through `study` so we get a genuine FrozenTrial back."""

    def objective(trial):
        trial.suggest_float("x", 0.0, 1.0)
        trial.set_user_attr("epochs_run", 22)
        if prune:
            raise optuna.TrialPruned()
        return value

    study.optimize(objective, n_trials=1)
    return study.trials[-1]


def _logger(name):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger, stream


# --- mode resolution ---------------------------------------------------------


def test_auto_becomes_bar_on_a_terminal():
    assert resolve_mode("auto", TtyStream()) == "bar"


def test_auto_becomes_plain_when_piped():
    assert resolve_mode("auto", StringIO()) == "plain"


def test_an_explicit_mode_is_respected():
    assert resolve_mode("plain", TtyStream()) == "plain"
    assert resolve_mode("none", TtyStream()) == "none"


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="progress mode must be one of"):
        resolve_mode("fancy")


# --- the logging bridge ------------------------------------------------------


def test_the_bridge_swaps_the_console_handler_and_restores_it():
    logger, _ = _logger("bridge_swap")
    original = logger.handlers[0]

    with tqdm_safe_logging(logger):
        assert isinstance(logger.handlers[0], TqdmLoggingHandler)
        assert logger.handlers[0].formatter is original.formatter

    assert logger.handlers == [original]


def test_the_bridge_leaves_a_file_handler_alone(tmp_path):
    """logging.FileHandler subclasses StreamHandler; hijacking it would corrupt the .log file."""
    logger, _ = _logger("bridge_file")
    file_handler = logging.FileHandler(tmp_path / "run.log")
    logger.addHandler(file_handler)

    with tqdm_safe_logging(logger):
        assert file_handler in logger.handlers
        assert sum(isinstance(h, TqdmLoggingHandler) for h in logger.handlers) == 1

    file_handler.close()


def test_the_bridge_tolerates_no_logger():
    with tqdm_safe_logging(None):
        pass


def test_bridged_records_still_reach_the_stream():
    logger, stream = _logger("bridge_output")
    with tqdm_safe_logging(logger):
        logger.info("hello from a trial")

    assert "hello from a trial" in stream.getvalue()


# --- none mode ---------------------------------------------------------------


def test_none_mode_writes_nothing_and_raises_nothing():
    stream = StringIO()
    logger, log_stream = _logger("none_mode")
    progress = StudyProgress(n_trials=3, objective=MAXIMIZE, mode="none", logger=logger, stream=stream)
    study = optuna.create_study(direction="maximize")

    with progress:
        progress.start_trial(10)
        progress.advance_epoch(0, 0.5)
        progress.on_trial_end(study, _frozen_trial(study))
        progress.end_trial()

    assert stream.getvalue() == ""
    assert log_stream.getvalue() == ""


# --- bar mode ----------------------------------------------------------------


def test_bar_mode_renders_both_bars():
    stream = TtyStream()
    tracker = ObjectiveTracker(MAXIMIZE)
    progress = StudyProgress(n_trials=3, objective=MAXIMIZE, tracker=tracker, mode="bar", stream=stream)
    study = optuna.create_study(direction="maximize")

    with progress:
        progress.start_trial(10)
        progress.advance_epoch(0, 0.42)
        trial = _frozen_trial(study, value=0.42)
        tracker.record(study, trial)
        progress.on_trial_end(study, trial)

    output = stream.getvalue()
    assert "Trials" in output
    assert "Trial " in output
    assert "best=0.4200" in output


def test_the_outer_bar_closes_a_leaked_inner_bar():
    """A pruned trial raises out of the Lightning callback, so on_fit_end never runs."""
    stream = TtyStream()
    progress = StudyProgress(n_trials=2, objective=MAXIMIZE, mode="bar", stream=stream)
    study = optuna.create_study(direction="maximize")

    with progress:
        progress.start_trial(10)
        assert progress._inner is not None
        progress.on_trial_end(study, _frozen_trial(study, prune=True))
        assert progress._inner is None


# --- plain mode --------------------------------------------------------------


def test_plain_mode_logs_one_line_per_trial():
    logger, log_stream = _logger("plain_trial")
    tracker = ObjectiveTracker(MAXIMIZE)
    progress = StudyProgress(
        n_trials=3, objective=MAXIMIZE, tracker=tracker, mode="plain", logger=logger, stream=StringIO()
    )
    study = optuna.create_study(direction="maximize")

    with progress:
        trial = _frozen_trial(study, value=0.42)
        tracker.record(study, trial)
        progress.on_trial_end(study, trial)

    line = log_stream.getvalue()
    assert "trial   0" in line
    assert "val_r2=0.4200" in line


def test_plain_mode_marks_a_pruned_trial():
    logger, log_stream = _logger("plain_pruned")
    progress = StudyProgress(
        n_trials=3, objective=MAXIMIZE, mode="plain", logger=logger, stream=StringIO()
    )
    study = optuna.create_study(direction="maximize")

    with progress:
        progress.on_trial_end(study, _frozen_trial(study, prune=True))

    output = log_stream.getvalue()
    assert "PRUNED" in output
    assert "@ep22" in output


def test_a_pruned_trial_that_reported_a_value_still_reads_as_pruned():
    """Optuna keeps the last intermediate value on a PRUNED trial, so `value is None` is not the
    test for completion - such a trial used to print as if it had finished."""
    logger, log_stream = _logger("plain_pruned_valued")
    progress = StudyProgress(
        n_trials=3, objective=MAXIMIZE, mode="plain", logger=logger, stream=StringIO()
    )
    study = optuna.create_study(direction="maximize")

    def objective(trial):
        trial.suggest_float("x", 0.0, 1.0)
        trial.report(0.2583, step=7)
        raise optuna.TrialPruned()

    study.optimize(objective, n_trials=1)
    trial = study.trials[-1]
    assert trial.value is not None  # the trap

    with progress:
        progress.on_trial_end(study, trial)

    output = log_stream.getvalue()
    assert "PRUNED" in output
    assert "@ep8" in output  # last reported epoch, since a pruned trial has no epochs_run
    assert "0.2583" in output  # the value it reached is still shown, in parentheses


def test_plain_mode_emits_an_epoch_heartbeat():
    logger, log_stream = _logger("plain_epochs")
    progress = StudyProgress(
        n_trials=1, objective=MAXIMIZE, mode="plain", logger=logger, stream=StringIO()
    )

    with progress:
        progress.start_trial(20)  # heartbeat every 20 // 10 == 2 epochs
        for epoch in range(4):
            progress.advance_epoch(epoch, 0.1 * epoch)

    lines = [line for line in log_stream.getvalue().splitlines() if "ep " in line]
    assert len(lines) == 2  # epochs 2 and 4, not all four


# --- the epoch callback ------------------------------------------------------


def test_the_epoch_callback_advances_on_validation_end():
    progress = StudyProgress(n_trials=1, objective=MAXIMIZE, mode="bar", stream=TtyStream())
    callback = progress.epoch_callback()

    with progress:
        callback.on_fit_start(FakeTrainer(max_epochs=5), None)
        callback.on_validation_end(FakeTrainer({"val_r2": 0.3}, current_epoch=0), None)

        assert progress._inner.n == 1
        assert progress._best_this_trial == pytest.approx(0.3)


def test_the_epoch_callback_ignores_the_sanity_check():
    progress = StudyProgress(n_trials=1, objective=MAXIMIZE, mode="bar", stream=TtyStream())
    callback = progress.epoch_callback()

    with progress:
        callback.on_fit_start(FakeTrainer(max_epochs=5), None)
        callback.on_validation_end(FakeTrainer({"val_r2": 0.9}, sanity_checking=True), None)

        assert progress._inner.n == 0
        assert progress._best_this_trial is None


def test_the_epoch_callback_tolerates_a_missing_metric():
    progress = StudyProgress(n_trials=1, objective=MAXIMIZE, mode="bar", stream=TtyStream())
    callback = progress.epoch_callback()

    with progress:
        callback.on_fit_start(FakeTrainer(max_epochs=5), None)
        callback.on_validation_end(FakeTrainer({"val_loss": 0.4}), None)

        assert progress._inner.n == 1  # still advanced
        assert progress._best_this_trial is None


def test_the_epoch_callback_tracks_the_best_in_the_objective_direction():
    progress = StudyProgress(
        n_trials=1, objective=Objective(metric="val_loss", direction="minimize"), mode="bar", stream=TtyStream()
    )
    callback = progress.epoch_callback()

    with progress:
        callback.on_fit_start(FakeTrainer(max_epochs=5), None)
        for epoch, value in enumerate([0.9, 0.2, 0.7]):
            callback.on_validation_end(FakeTrainer({"val_loss": value}, current_epoch=epoch), None)

        assert progress._best_this_trial == pytest.approx(0.2)


def test_the_epoch_callback_closes_the_bar_on_fit_end():
    progress = StudyProgress(n_trials=1, objective=MAXIMIZE, mode="bar", stream=TtyStream())
    callback = progress.epoch_callback()

    with progress:
        callback.on_fit_start(FakeTrainer(max_epochs=5), None)
        callback.on_fit_end(FakeTrainer(), None)

        assert progress._inner is None


def test_the_epoch_callback_is_accepted_by_the_trial_runner():
    """The wiring contract: TrialRunner puts extra callbacks on the Trainer it builds."""
    from yg_eo_soilnet.hpo.trial_runner import TrialRunner

    progress = StudyProgress(n_trials=1, objective=MAXIMIZE, mode="none", stream=StringIO())
    runner = TrialRunner(MAXIMIZE, extra_callbacks=[progress.epoch_callback()])

    assert len(runner.extra_callbacks) == 1


def test_a_trial_objective_forwards_progress_to_the_runner():
    from yg_eo_soilnet.hpo.objective import ObjectiveContext, TrialObjective
    from yg_eo_soilnet.hpo.search_space import SearchSpace

    registry = {
        "e": {
            "enabled": True,
            "modeltype": "dl",
            "input_kind": "sequence",
            "import_path": "x.Y",
            "datamodule_import_path": "x.Z",
        }
    }
    context = ObjectiveContext.from_config(
        "e", SimpleNamespace(LIGHTNING_MODEL_REGISTRY=registry, TARGET_COLUMNS=["t"], RANDOM_SEED=1), data={}
    )
    space = SearchSpace.from_mapping("e", {"params": {"model.dropout": {"type": "float", "low": 0.0, "high": 0.5}}})
    progress = StudyProgress(n_trials=1, objective=space.objective, mode="none", stream=StringIO())

    objective = TrialObjective(context, space, progress=progress)
    assert len(objective.runner.extra_callbacks) == 1

    assert TrialObjective(context, space).runner.extra_callbacks == []
