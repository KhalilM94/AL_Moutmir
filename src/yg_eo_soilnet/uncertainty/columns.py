"""The uncertainty column names in an evaluation frame, and how to read them back out.

Both families write these columns and three separate readers consume them (the metric fan-out, the
per-run plot, the parent overlay). Centralising the names here is what stops the sklearn frame and
the Lightning frame from drifting apart - the same reason ArtifactLayout owns the artifact paths
rather than each logger spelling them out.

The convention mirrors the existing prediction columns exactly:

    one target      prediction, prediction_std, prediction_lower, prediction_upper
    several targets prediction_<t>, prediction_std_<t>, prediction_lower_<t>, ...

with `prediction_<t>` keeping its name and meaning - it is the ensemble MEAN - so that every reader
written before uncertainty existed keeps working unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

# The stems, in the order they are appended to a frame.
STD = "prediction_std"
EPISTEMIC_STD = "prediction_epistemic_std"
ALEATORIC_STD = "prediction_aleatoric_std"
LOWER = "prediction_lower"
UPPER = "prediction_upper"

UNCERTAINTY_STEMS: tuple[str, ...] = (STD, EPISTEMIC_STD, ALEATORIC_STD, LOWER, UPPER)

# Every prefix that starts with "prediction_" but is NOT a per-target prediction column.
#
# This exists because of a specific trap. `_resolve_prediction_column` in plot_utils falls back to
# "the first column starting with prediction_", and `prediction_std_clay_pct` sorts before
# `prediction_clay_pct` in some frames - so without this guard a plot can silently draw standard
# deviations on the predicted axis. Any new column added above must be listed here.
NON_PREDICTION_PREFIXES: tuple[str, ...] = tuple(f"{stem}_" for stem in UNCERTAINTY_STEMS)


def column_name(stem: str, target_name: Optional[str] = None, *, multi_target: bool = False) -> str:
    """``prediction_std`` for a lone target, ``prediction_std_<target>`` for one of several.

    `multi_target` is explicit rather than inferred from `target_name` being set, because a
    single-target run knows its target's name too and must still write the unsuffixed column.
    """
    if not multi_target or target_name is None:
        return stem
    return f"{stem}_{target_name}"


def is_prediction_column(column_name_value: Any) -> bool:
    """True for a per-target prediction column, False for an uncertainty column.

    The test every reader that scans for `prediction_*` columns should use.

    Both spellings have to be excluded: a single-target run writes the bare stem `prediction_std`,
    and a joint run writes `prediction_std_<target>`. Matching only the suffixed form would let the
    single-target frame - the more common one - slip a sigma through as a prediction.
    """
    name = str(column_name_value)
    if not name.startswith("prediction"):
        return False
    if name in UNCERTAINTY_STEMS:
        return False
    return not any(name.startswith(prefix) for prefix in NON_PREDICTION_PREFIXES)


def interval_columns(
    frame: pd.DataFrame,
    target_name: Optional[str] = None,
) -> Optional[tuple[pd.Series, pd.Series]]:
    """``(lower, upper)`` for this target when the frame carries them, else None.

    Returning None rather than raising is deliberate: every plot and metric call site has to work on
    frames from runs where uncertainty was off, and those are the majority.
    """
    lower = _first_present(frame, LOWER, target_name)
    upper = _first_present(frame, UPPER, target_name)
    if lower is None or upper is None:
        return None
    return frame[lower], frame[upper]


def sigma_column(frame: pd.DataFrame, target_name: Optional[str] = None) -> Optional[pd.Series]:
    """The total predictive standard deviation for this target, or None."""
    name = _first_present(frame, STD, target_name)
    return None if name is None else frame[name]


def _first_present(frame: pd.DataFrame, stem: str, target_name: Optional[str]) -> Optional[str]:
    """The suffixed name if the frame has it, else the bare stem, else None.

    Suffixed first: a per-target child frame produced by `_iter_target_eval_frames` is a COPY of the
    whole multi-target frame, so it carries every target's columns and the bare stem may be absent
    while several suffixed ones are present. Preferring the bare name there would find nothing on a
    joint run; preferring it after the suffixed one is correct in both shapes.
    """
    if target_name:
        suffixed = f"{stem}_{target_name}"
        if suffixed in frame.columns:
            return suffixed
    if stem in frame.columns:
        return stem
    return None
