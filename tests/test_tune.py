"""The CLI's failure policy: a study that ran for hours must not lose its winner to a late crash.

A 266-trial sweep once died on trial 265 when the GPU driver was lost. `run_study`'s `finally` saved
the summary, trials.csv and the plots, but the exception propagated past `report_and_export`, so
`configs/lightning/tuned/` stayed empty after 174 completed trials. Every trial was already in the
storage the whole time - only the export was missing.
"""

from types import SimpleNamespace

import pytest

import tune as tune_module
from yg_eo_soilnet.hpo.trial_runner import UnrecoverableAcceleratorError


class RecordingLogger:
    def __init__(self):
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def info(self, message, *args, **kwargs):
        return None

    def warning(self, message, *args, **kwargs):
        self.warnings.append(str(message))

    def error(self, message, *args, **kwargs):
        self.errors.append(str(message))


def _patch_abort_path(monkeypatch, *, best_value, exported, study=None, load_error=None):
    """Stand in for the three collaborators handle_study_abort reaches out to."""

    def fake_create_or_load_study(space, study_name, storage):
        if load_error is not None:
            raise load_error
        return study if study is not None else SimpleNamespace(trials=[1, 2, 3])

    monkeypatch.setattr(tune_module, "create_or_load_study", fake_create_or_load_study)
    monkeypatch.setattr(tune_module, "best_value_or_none", lambda study: best_value)
    monkeypatch.setattr(
        tune_module,
        "report_and_export",
        lambda study, space, tracker, registry_entry, args, config, logger: exported.append(study),
    )


def _abort(error, logger):
    tune_module.handle_study_abort(
        error,
        object(),  # space
        object(),  # tracker
        SimpleNamespace(registry_entry={"enabled": True}),
        SimpleNamespace(storage="sqlite:///:memory:", top_n=10, export_path=None, entry="fake_entry"),
        object(),  # config
        logger,
        "fake_entry-abc123",
    )


def test_the_best_trial_is_exported_after_an_abort(monkeypatch):
    exported: list = []
    logger = RecordingLogger()
    _patch_abort_path(monkeypatch, best_value=0.63, exported=exported)

    _abort(RuntimeError("boom"), logger)

    assert len(exported) == 1
    assert any("exporting the best" in warning for warning in logger.warnings)


def test_a_keyboard_interrupt_also_exports(monkeypatch):
    """Ctrl-C on a long sweep lost the export the same way a crash did."""
    exported: list = []
    _patch_abort_path(monkeypatch, best_value=0.63, exported=exported)

    _abort(KeyboardInterrupt(), RecordingLogger())

    assert len(exported) == 1


def test_an_abort_with_nothing_completed_exports_nothing(monkeypatch):
    exported: list = []
    logger = RecordingLogger()
    _patch_abort_path(monkeypatch, best_value=None, exported=exported)

    _abort(RuntimeError("boom"), logger)

    assert exported == []
    assert any("nothing to export" in warning for warning in logger.warnings)


def test_a_failing_export_is_logged_and_never_raised(monkeypatch):
    """It runs while another exception is in flight; raising would hide why the study stopped."""
    logger = RecordingLogger()
    _patch_abort_path(monkeypatch, best_value=0.63, exported=[], load_error=RuntimeError("storage is gone"))

    _abort(RuntimeError("boom"), logger)  # must not raise

    assert any("Could not export" in error for error in logger.errors)


def test_the_reloaded_study_is_the_one_exported(monkeypatch):
    """The live study object went with the traceback, so it is reloaded rather than reused."""
    exported: list = []
    stored = SimpleNamespace(trials=[1, 2])
    _patch_abort_path(monkeypatch, best_value=0.63, exported=exported, study=stored)

    _abort(RuntimeError("boom"), RecordingLogger())

    assert exported == [stored]


def test_a_dead_accelerator_exits_with_the_message_not_a_traceback(monkeypatch):
    """The traceback names whatever CUDA call came next, not the failure - it only misleads."""
    exported: list = []
    _patch_abort_path(monkeypatch, best_value=0.63, exported=exported)

    with pytest.raises(SystemExit, match="wsl --shutdown"):
        _abort(UnrecoverableAcceleratorError("GPU context died; run `wsl --shutdown`"), RecordingLogger())

    assert len(exported) == 1  # exported first, exited second


def test_any_other_error_is_left_for_the_caller_to_re_raise(monkeypatch):
    """handle_study_abort salvages; main() re-raises, so the exit code still signals failure."""
    _patch_abort_path(monkeypatch, best_value=0.63, exported=[])

    _abort(RuntimeError("boom"), RecordingLogger())  # returns normally
