"""The per-point prediction export: every model's estimate for every point, keyed on point id.

A run already records what each model predicted for its own test rows, in each child run's
``eval_results/eval_results.csv``. What it has never recorded is the other direction - given a
point, what did every model say about it - because those frames carry no point id at all and are
one file per child.

This module builds that view. Children write their own contribution keyed on the id; the parent
collects and combines them. Two shapes, because they answer different questions:

* **long** - one row per (point, target, model). Authoritative: target and model are separate
  columns, so nothing has to be parsed back out of a name.
* **wide** - one row per point, one column per ``<target>__<model>``. Readable, drops into a
  spreadsheet, and is what "a column per child run" means. Its column names are NOT reliably
  splittable back into their two parts, because both halves routinely contain underscores
  (``clay_pct``, ``soil_cnn``) - which is exactly why the long file exists beside it.

Deliberately no uncertainty columns and no observed values. Where a model was ensembled the number
here is the ensemble MEAN; its sigma and interval stay in eval_results.csv, and the observed lab
values stay in the targets file this is keyed against.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from yg_eo_soilnet.artifacts import ArtifactLayout

# Separates the two halves of a wide column name. Two underscores rather than one because every
# realistic target and model name already contains single underscores; this at least makes the
# boundary visible to a human, even though it cannot be parsed reliably.
NAME_SEPARATOR = "__"

# Long-format column names.
TARGET_COLUMN = "target"
MODEL_COLUMN = "model"
PREDICTION_COLUMN = "prediction"


def export_enabled_for(config: Any, model_name: str) -> bool:
    """Whether this registry entry should pay for a full-population prediction pass.

    Allowlist beats denylist, the same rule ``uncertainty_enabled_for`` and the SHAP seam apply -
    the cost is per model, so naming one explicitly has to override a blanket exclusion.
    """
    if not bool(getattr(config, "EXPORT_POINT_PREDICTIONS", False)):
        return False

    allowed = [str(name) for name in (getattr(config, "EXPORT_POINT_PREDICTIONS_MODELS", None) or [])]
    if allowed:
        return str(model_name) in allowed

    skipped = [
        str(name) for name in (getattr(config, "EXPORT_POINT_PREDICTIONS_SKIP_MODELS", None) or [])
    ]
    return str(model_name) not in skipped


def point_id_column(config: Any) -> str:
    """The name the id column is written under, from the run's own config."""
    return str(getattr(config, "POINT_ID_COLUMN", "point_id") or "point_id")


def point_prediction_frame(
    point_ids: Any,
    predictions: Any,
    target_names: Sequence[str],
    id_column: str = "point_id",
) -> pd.DataFrame:
    """A child's contribution: the id column plus one prediction column per target it fits.

    ``point_ids`` must already be aligned with ``predictions`` row for row. Getting that alignment
    right is the whole difficulty of this feature and it is the CALLER's job, because only the
    caller still has the object that knows it - a Series indexed like the feature frame on the
    sklearn side, the bundle's own id list on the Lightning side. Passing a bare positional range
    here would look identical and be wrong for any model whose rows were filtered.
    """
    values = np.asarray(predictions, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    ids = list(point_ids)
    if len(ids) != values.shape[0]:
        raise ValueError(
            f"{len(ids)} point ids against {values.shape[0]} prediction rows. These are paired "
            "positionally, so a mismatch means the ids describe different points than the "
            "predictions do - refusing rather than writing a plausible, wrong file."
        )
    if values.shape[1] != len(target_names):
        raise ValueError(
            f"{values.shape[1]} prediction columns against {len(target_names)} target names."
        )

    frame = pd.DataFrame({id_column: ids})
    for index, target_name in enumerate(target_names):
        frame[str(target_name)] = values[:, index]
    return frame


def to_long(frames: Iterable[tuple[str, pd.DataFrame]], id_column: str = "point_id") -> pd.DataFrame:
    """``(point_id, target, model, prediction)`` over every child's frame.

    ``frames`` is ``(model_name, child_frame)`` pairs - the model name comes from the run's tag
    rather than from inside the file, so a child never has to know what it is called.
    """
    melted = []
    for model_name, frame in frames:
        if frame is None or frame.empty or id_column not in frame.columns:
            continue
        target_columns = [column for column in frame.columns if column != id_column]
        if not target_columns:
            continue
        long_frame = frame.melt(
            id_vars=[id_column],
            value_vars=target_columns,
            var_name=TARGET_COLUMN,
            value_name=PREDICTION_COLUMN,
        )
        long_frame[MODEL_COLUMN] = str(model_name)
        melted.append(long_frame)

    if not melted:
        return pd.DataFrame(columns=[id_column, TARGET_COLUMN, MODEL_COLUMN, PREDICTION_COLUMN])

    combined = pd.concat(melted, ignore_index=True)
    return combined[[id_column, TARGET_COLUMN, MODEL_COLUMN, PREDICTION_COLUMN]]


def to_wide(long_frame: pd.DataFrame, id_column: str = "point_id") -> pd.DataFrame:
    """One row per point, one ``<target>__<model>`` column per child.

    Built from the long frame rather than from the child frames a second time, so the two files
    cannot disagree about what a model predicted.
    """
    if long_frame.empty:
        return pd.DataFrame(columns=[id_column])

    frame = long_frame.copy()
    frame["_column"] = [
        wide_column_name(target, model)
        for target, model in zip(frame[TARGET_COLUMN], frame[MODEL_COLUMN])
    ]
    # `first` rather than the default mean: a duplicated (point, target, model) means the same model
    # reported twice for one point, which is a bug upstream, and silently averaging it away would
    # hide it. The count check below is what surfaces it.
    wide = frame.pivot_table(
        index=id_column, columns="_column", values=PREDICTION_COLUMN, aggfunc="first"
    )
    wide.columns.name = None
    return wide.reset_index()


def wide_column_name(target: Any, model: Any) -> str:
    """``<target>__<model>``, both halves sanitised the way artifact paths are."""
    return f"{ArtifactLayout.safe(target)}{NAME_SEPARATOR}{ArtifactLayout.safe(model)}"


def combine(
    frames: Iterable[tuple[str, pd.DataFrame]],
    id_column: str = "point_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(wide, long)`` for every child frame collected from a parent run."""
    long_frame = to_long(frames, id_column=id_column)
    return to_wide(long_frame, id_column=id_column), long_frame


def duplicate_report(long_frame: pd.DataFrame, id_column: str = "point_id") -> Optional[str]:
    """A message when one (point, target, model) appears twice, else None.

    The wide pivot keeps the first of any duplicate, so without this check a double-counted child -
    the same model logged under two runs, say - would be invisible in the output.
    """
    if long_frame.empty:
        return None
    keys = [id_column, TARGET_COLUMN, MODEL_COLUMN]
    duplicated = int(long_frame.duplicated(subset=keys).sum())
    if duplicated == 0:
        return None
    return (
        f"{duplicated} duplicate (point, target, model) rows in the prediction export; the wide "
        "file keeps the first of each. This usually means one model was collected from two runs."
    )


def child_frame_from_predictor_output(
    frame: pd.DataFrame,
    id_column: str,
    target_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Normalise ``SoilSequencePredictor.predict_frame`` output into the child-frame shape.

    That method returns predictions indexed BY POINT ID with one column per target, which is the
    same information in a different arrangement - this moves the index into a real column so the
    Lightning and sklearn contributions are the same object.
    """
    reset = frame.reset_index()
    reset = reset.rename(columns={reset.columns[0]: id_column})
    if target_names:
        keep = [id_column] + [str(name) for name in target_names if str(name) in reset.columns]
        reset = reset[keep]
    return reset


def summarize(wide: pd.DataFrame, long_frame: pd.DataFrame, id_column: str) -> Mapping[str, Any]:
    """Log-friendly description of what was written, for the run summary."""
    return {
        "n_points": int(wide.shape[0]),
        "n_columns": int(max(wide.shape[1] - 1, 0)),
        "models": sorted(long_frame[MODEL_COLUMN].unique().tolist()) if not long_frame.empty else [],
        "targets": sorted(long_frame[TARGET_COLUMN].unique().tolist()) if not long_frame.empty else [],
        "id_column": id_column,
    }
