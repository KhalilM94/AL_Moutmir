"""Two-level progress display for a study: trials outside, epochs inside.

A trial-level bar alone is not enough here. With max_epochs in the low hundreds a single trial runs
for minutes, so a bar that ticks once per trial still leaves the terminal motionless for long
stretches - which is the problem this is meant to solve. The inner bar advances every validation
epoch, driven by a Lightning callback.

Falls back to compact log lines when stderr is not a terminal, so a run under nohup or piped to a
file stays readable instead of filling with carriage returns.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import Any

import optuna
from tqdm.auto import tqdm

from yg_eo_soilnet.hpo.pruning import metric_to_float
from yg_eo_soilnet.hpo.search_space import Objective
from yg_eo_soilnet.hpo.tracker import ObjectiveTracker

try:  # pragma: no cover - optional dependency, mirrors the rest of the package
    from lightning.pytorch.callbacks import Callback as LightningCallback
except ImportError:  # pragma: no cover
    LightningCallback = object  # type: ignore[assignment]

MODES = ("auto", "bar", "plain", "none")
SPARKLINE_EVERY = 10  # trials, in plain mode


def resolve_mode(mode: str, stream: Any = None) -> str:
    """Turn `auto` into a concrete mode by asking whether we are talking to a terminal."""
    if mode not in MODES:
        raise ValueError(f"progress mode must be one of {', '.join(MODES)}; got {mode!r}.")
    if mode != "auto":
        return mode
    stream = stream if stream is not None else sys.stderr
    try:
        return "bar" if stream.isatty() else "plain"
    except (AttributeError, ValueError):  # a closed or exotic stream
        return "plain"


class TqdmLoggingHandler(logging.StreamHandler):
    """A console handler that writes through `tqdm.write`, so log lines do not smear a live bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit):  # pragma: no cover
            raise
        except Exception:  # pragma: no cover - logging must never take the process down
            self.handleError(record)


@contextmanager
def tqdm_safe_logging(logger: logging.Logger | None):
    """Swap a logger's console handlers for tqdm-aware ones, and put them back afterwards.

    TrainingLogger writes to stdout while tqdm writes to stderr; different streams, same terminal,
    so an unbridged logger.info() during optimization corrupts the bars.

    The type check is exact on purpose. logging.FileHandler is a *subclass* of StreamHandler, so an
    isinstance check would also hijack the file handler and fill the .log file with terminal
    control characters.
    """
    if logger is None:
        yield
        return

    swapped: list[tuple[logging.Handler, logging.Handler]] = []
    for handler in list(logger.handlers):
        if type(handler) is not logging.StreamHandler:
            continue
        bridge = TqdmLoggingHandler(handler.stream)
        bridge.setFormatter(handler.formatter)
        bridge.setLevel(handler.level)
        logger.removeHandler(handler)
        logger.addHandler(bridge)
        swapped.append((handler, bridge))
    try:
        yield
    finally:
        for original, bridge in swapped:
            logger.removeHandler(bridge)
            logger.addHandler(original)


class _EpochProgressCallback(LightningCallback):
    """Drives the inner bar from Lightning's validation loop."""

    def __init__(self, progress: "StudyProgress"):
        self.progress = progress

    def on_fit_start(self, trainer, pl_module) -> None:
        self.progress.start_trial(getattr(trainer, "max_epochs", None))

    def on_validation_end(self, trainer, pl_module) -> None:
        # on_validation_end for the same reason as OptunaPruningCallback: callback
        # on_validation_epoch_end hooks run before the module's, where `val_r2` is logged, so the
        # reading there is a whole epoch stale.
        if getattr(trainer, "sanity_checking", False):
            return
        value = metric_to_float(trainer.callback_metrics.get(self.progress.objective.metric))
        self.progress.advance_epoch(int(getattr(trainer, "current_epoch", 0)), value)

    def on_fit_end(self, trainer, pl_module) -> None:
        self.progress.end_trial()


class StudyProgress:
    """Outer bar over trials, inner bar over epochs. `mode="none"` makes every method a no-op."""

    def __init__(
        self,
        *,
        n_trials: int,
        objective: Objective,
        tracker: ObjectiveTracker | None = None,
        mode: str = "auto",
        logger: Any = None,
        stream: Any = None,
    ):
        self.n_trials = int(n_trials)
        self.objective = objective
        self.tracker = tracker
        self.logger = logger
        self.stream = stream if stream is not None else sys.stderr
        self.mode = resolve_mode(mode, self.stream)

        self._outer: tqdm | None = None
        self._inner: tqdm | None = None
        self._logging_bridge = None
        self._trial_index = 0
        self._epoch_heartbeat = 1
        self._best_this_trial: float | None = None

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    # --- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "StudyProgress":
        if self.mode == "bar":
            self._logging_bridge = tqdm_safe_logging(self.logger)
            self._logging_bridge.__enter__()
            self._outer = tqdm(
                total=self.n_trials,
                desc="Trials",
                position=0,
                leave=True,
                file=self.stream,
                dynamic_ncols=True,
            )
            self._refresh_outer()
        return self

    def __exit__(self, *exc_info) -> None:
        self._close_inner()
        if self._outer is not None:
            self._outer.close()
            self._outer = None
        if self._logging_bridge is not None:
            self._logging_bridge.__exit__(*exc_info)
            self._logging_bridge = None

    # --- outer: trials -----------------------------------------------------

    def on_trial_end(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """An Optuna study callback. Registered after the tracker's, so the postfix is current."""
        if not self.enabled:
            return
        # A pruned trial raises out of the Lightning callback, so on_fit_end never runs and the
        # inner bar would leak. Closing here is the backstop.
        self._close_inner()
        self._trial_index += 1

        if self.mode == "bar" and self._outer is not None:
            self._outer.update(1)
            self._refresh_outer()
            return
        if self.mode == "plain":
            self._log_trial(trial)

    def _refresh_outer(self) -> None:
        if self._outer is None:
            return
        parts = []
        if self.tracker is not None:
            spark = self.tracker.sparkline()
            if spark:
                parts.append(spark)
            parts.append(self.tracker.summary_line())
        self._outer.set_postfix_str(" ".join(parts), refresh=True)

    def _log_trial(self, trial: optuna.trial.FrozenTrial) -> None:
        if self.logger is None:
            return
        duration = getattr(trial, "duration", None)
        elapsed = "" if duration is None else f" | {duration.total_seconds():5.0f}s"
        # On state, not on `value`: Optuna carries the last intermediate value onto a PRUNED trial,
        # so a pruned trial has a value and would otherwise read as if it had completed.
        if trial.state.name == "COMPLETE":
            outcome = f"{self.objective.metric}={trial.value:.4f}"
        else:
            outcome = trial.state.name
            # A pruned trial has no epochs_run - the objective sets that only on a clean return -
            # so fall back to the last epoch it reported an intermediate value for.
            epochs = trial.user_attrs.get("epochs_run")
            if epochs is None and trial.intermediate_values:
                epochs = max(trial.intermediate_values) + 1
            if epochs is not None:
                outcome += f" @ep{epochs}"
            if trial.value is not None:
                outcome += f" ({self.objective.metric}={trial.value:.4f})"
        summary = f" | {self.tracker.summary_line()}" if self.tracker is not None else ""
        self.logger.info(f"trial {trial.number:>3} | {outcome}{summary}{elapsed}")

        if self.tracker is not None and self._trial_index % SPARKLINE_EVERY == 0:
            self.logger.info(f"history  {self.tracker.sparkline()}")

    # --- inner: epochs -----------------------------------------------------

    def epoch_callback(self):
        """A lightning.pytorch Callback for TrialRunner's `extra_callbacks`."""
        return _EpochProgressCallback(self)

    def start_trial(self, max_epochs: int | None) -> None:
        if not self.enabled:
            return
        self._close_inner()
        self._best_this_trial = None
        total = int(max_epochs) if max_epochs else 0
        self._epoch_heartbeat = max(1, total // 10) if total else 1
        if self.mode == "bar":
            self._inner = tqdm(
                total=total or None,
                desc=f"Trial {self._current_trial_label()}",
                position=1,
                leave=False,
                file=self.stream,
                dynamic_ncols=True,
            )

    def advance_epoch(self, epoch: int, value: float | None) -> None:
        if not self.enabled:
            return
        if value is not None and (
            self._best_this_trial is None
            or (value > self._best_this_trial if self.objective.mode == "max" else value < self._best_this_trial)
        ):
            self._best_this_trial = value

        if self.mode == "bar" and self._inner is not None:
            self._inner.update(1)
            if value is not None:
                self._inner.set_postfix_str(
                    f"{self.objective.metric}={value:.4f} best={self._best_this_trial:.4f}", refresh=True
                )
            return
        if self.mode == "plain" and self.logger is not None and (epoch + 1) % self._epoch_heartbeat == 0:
            reading = "n/a" if value is None else f"{value:.4f}"
            self.logger.info(
                f"  trial {self._current_trial_label()} | ep {epoch + 1} | {self.objective.metric}={reading}"
            )

    def end_trial(self) -> None:
        self._close_inner()

    def _close_inner(self) -> None:
        if self._inner is not None:
            self._inner.close()
            self._inner = None

    def _current_trial_label(self) -> str:
        if self.tracker is not None and self.tracker.records:
            return str(self.tracker.records[-1].number + 1)
        return str(self._trial_index)
