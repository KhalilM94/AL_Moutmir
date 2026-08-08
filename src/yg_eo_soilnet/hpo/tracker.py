"""Following the objective across a study.

The question a running study cannot otherwise answer is not "is it alive" but "is the search
working". This accumulates the per-trial objective so the CLI can render a trend live, and turns
the finished study into a table and a CSV.

Nothing here needs new plumbing on the trial side: `study.trials` already carries the value, the
state, the parameters, the duration and the user attributes TrialObjective records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import optuna
import pandas as pd

from yg_eo_soilnet.hpo.search_space import Objective

# Eight levels, low to high. A pruned or failed trial has no value, and showing a gap where it
# happened is more informative than dropping it - a run of gaps is the pruner doing its job.
BLOCKS = "▁▂▃▄▅▆▇█"
GAP = "·"

COMPLETE = "COMPLETE"


def best_value_or_none(study: optuna.Study) -> float | None:
    """`study.best_value`, or None when no trial has completed.

    Optuna *raises* ValueError rather than returning None in that case, which makes the obvious
    `if study.best_value is not None` guard useless - it blows up before the comparison. A study
    whose first trial is pruned hits this on trial 0.
    """
    try:
        return float(study.best_value)
    except (ValueError, RuntimeError):
        return None


def best_trial_number_or_none(study: optuna.Study) -> int | None:
    try:
        return int(study.best_trial.number)
    except (ValueError, RuntimeError):
        return None


@dataclass
class TrialRecord:
    number: int
    value: float | None
    state: str
    duration_s: float | None = None
    best_epoch: int | None = None
    epochs_run: int | None = None

    @property
    def is_complete(self) -> bool:
        """Whether the trial finished.

        Not `value is not None`: Optuna carries the last intermediate value over onto a PRUNED
        trial, so a pruned trial has a value too. Only the state distinguishes them.
        """
        return self.state == COMPLETE

    @classmethod
    def from_trial(cls, trial: optuna.trial.FrozenTrial) -> "TrialRecord":
        duration = getattr(trial, "duration", None)
        return cls(
            number=int(trial.number),
            value=None if trial.value is None else float(trial.value),
            state=trial.state.name,
            duration_s=None if duration is None else duration.total_seconds(),
            best_epoch=trial.user_attrs.get("best_epoch"),
            epochs_run=trial.user_attrs.get("epochs_run"),
        )


class ObjectiveTracker:
    """Per-trial objective history, plus the renderings built from it."""

    def __init__(self, objective: Objective):
        self.objective = objective
        self.records: list[TrialRecord] = []
        self.best_value: float | None = None
        self.best_trial: int | None = None

    # --- accumulation ------------------------------------------------------

    def prime(self, study: optuna.Study) -> None:
        """Seed from a study that already holds trials, so a resumed run shows its real history."""
        self.records = [TrialRecord.from_trial(trial) for trial in study.trials]
        self._refresh_best(study)

    def record(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """An Optuna study callback: `study.optimize(..., callbacks=[tracker.record])`."""
        self.records.append(TrialRecord.from_trial(trial))
        self._refresh_best(study)

    def _refresh_best(self, study: optuna.Study) -> None:
        self.best_value = best_value_or_none(study)
        self.best_trial = best_trial_number_or_none(study)

    @property
    def counts(self) -> dict[str, int]:
        counts = {"complete": 0, "pruned": 0, "failed": 0}
        for record in self.records:
            if record.state == COMPLETE:
                counts["complete"] += 1
            elif record.state == "PRUNED":
                counts["pruned"] += 1
            elif record.state == "FAIL":
                counts["failed"] += 1
        return counts

    # --- renderings --------------------------------------------------------

    def sparkline(self, width: int = 24) -> str:
        """The last `width` trials as block characters, with `·` where a trial produced no value."""
        window = self.records[-width:] if width > 0 else []
        if not window:
            return ""

        values = [record.value for record in window if record.is_complete and record.value is not None]
        if not values:
            return GAP * len(window)

        low, high = min(values), max(values)
        span = high - low
        # A flat history divides by zero under naive min-max scaling. Every value being equal is a
        # real state early in a study (or with a degenerate space), not an error, so pin it midway.
        if span <= 0:
            middle = BLOCKS[len(BLOCKS) // 2]
            return "".join(middle if record.is_complete else GAP for record in window)

        return "".join(
            BLOCKS[min(len(BLOCKS) - 1, int((record.value - low) / span * len(BLOCKS)))]
            if record.is_complete and record.value is not None
            else GAP
            for record in window
        )

    def summary_line(self) -> str:
        """`best=0.1831 (t8) ok10 pruned2` - the postfix for a bar or a log line."""
        counts = self.counts
        if self.best_value is None:
            best = "best=n/a"
        else:
            best = f"best={self.best_value:.4f} (t{self.best_trial})"
        parts = [best, f"ok{counts['complete']}"]
        if counts["pruned"]:
            parts.append(f"pruned{counts['pruned']}")
        if counts["failed"]:
            parts.append(f"failed{counts['failed']}")
        return " ".join(parts)

    def trials_frame(self, study: optuna.Study) -> pd.DataFrame:
        """Every trial with every parameter - what lands in trials.csv."""
        frame = study.trials_dataframe()
        return frame if frame is not None else pd.DataFrame()

    def top_frame(self, study: optuna.Study, n: int = 10) -> pd.DataFrame:
        """The best `n` completed trials, in fixed narrow columns.

        Parameters are deliberately excluded: a dozen of them would make the table unreadable in a
        terminal, and the winning set is printed separately. trials.csv keeps everything.
        """
        completed = [record for record in self.records if record.is_complete and record.value is not None]
        if not completed:
            return pd.DataFrame(columns=["trial", self.objective.metric, "best_epoch", "epochs_run", "duration_s"])

        completed.sort(key=lambda record: record.value, reverse=self.objective.direction == "maximize")
        return pd.DataFrame(
            [
                {
                    "trial": record.number,
                    self.objective.metric: round(record.value, 6),
                    "best_epoch": record.best_epoch,
                    "epochs_run": record.epochs_run,
                    "duration_s": None if record.duration_s is None else round(record.duration_s, 1),
                }
                for record in completed[:n]
            ]
        )

    def best_params(self, study: optuna.Study) -> dict[str, Any]:
        try:
            return dict(study.best_trial.params)
        except (ValueError, RuntimeError):
            return {}
