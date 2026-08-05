"""DataFrame cleaning shared by every datamodule builder.

These helpers are deliberately free of any graph or sequence concept: they repair and filter raw
CSV frames and nothing else, so the graph path and the sequence path clean their inputs the same
way instead of drifting apart.
"""

from __future__ import annotations

from typing import Any, Iterable, Tuple

import numpy as np
import pandas as pd


def build_finite_row_mask(
    frame: pd.DataFrame,
    *,
    required_columns: Iterable[str] = (),
    numeric_columns: Iterable[str] = (),
) -> pd.Series:
    mask = pd.Series(True, index=frame.index)

    required_columns = [column for column in required_columns if column in frame.columns]
    if required_columns:
        mask &= frame[required_columns].notna().all(axis=1)

    for column in numeric_columns:
        if column not in frame.columns:
            continue
        numeric_values = pd.to_numeric(frame[column], errors="coerce")
        mask &= np.isfinite(numeric_values.to_numpy(dtype=np.float64, copy=False))

    return mask


def drop_non_finite_rows(
    frame: pd.DataFrame,
    *,
    logger: Any,
    label: str,
    required_columns: Iterable[str] = (),
    numeric_columns: Iterable[str] = (),
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    mask = build_finite_row_mask(
        frame,
        required_columns=required_columns,
        numeric_columns=numeric_columns,
    )
    if bool(mask.all()):
        return frame.copy()

    dropped_count = int((~mask).sum())
    kept_frame = frame.loc[mask].copy()
    logger.warning(
        f"Dropped {dropped_count} row(s) with non-finite values from {label}; remaining rows: {len(kept_frame)}"
    )
    return kept_frame


def encode_categorical_features(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    logger: Any,
) -> Tuple[pd.DataFrame, list[str]]:
    """Ordinal-encode non-numeric static features so every model sees them.

    Without this the numeric-dtype filter silently dropped every categorical covariate, giving the
    network fewer predictors than the sklearn path. Codes are assigned over the whole column, so
    category identity is global; the values are standardized train-only downstream by the
    datamodule, so this is a label mapping rather than a fitted statistic.
    """
    columns = [column for column in columns if column in frame.columns]
    encoded = frame.copy()
    feature_columns: list[str] = []
    encoded_report: dict[str, int] = {}

    for column in columns:
        if pd.api.types.is_numeric_dtype(encoded[column]):
            feature_columns.append(column)
            continue
        codes, uniques = pd.factorize(encoded[column], use_na_sentinel=True)
        if len(uniques) == 0:
            logger.warning(f"Static feature '{column}' has no usable categories; dropping it")
            continue
        # factorize marks missing values as -1; NaN lets drop_non_finite_rows handle them
        # consistently with every other feature instead of inventing a category.
        encoded[column] = pd.Series(codes, index=encoded.index, dtype="float64").replace(-1.0, np.nan)
        feature_columns.append(column)
        encoded_report[column] = int(len(uniques))

    if encoded_report:
        logger.info(
            "Ordinal-encoded static categorical feature(s): "
            + ", ".join(f"{name} ({count} categories)" for name, count in encoded_report.items())
        )
    return encoded, feature_columns


def sanitize_numeric_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    logger: Any,
    return_validity: bool = False,
):
    """Make the named columns numeric and finite, without discarding rows.

    Handles three defects seen in the source data: decimal-comma strings ('0,00005') that make a
    band column object-dtype, infinities from ratio indices, and sparse columns whose NaNs would
    otherwise take the whole row down. Missing cells are median-filled per column.

    With ``return_validity`` the function also returns a boolean frame that is True where the cell
    was finite **before** the fill. Median-filling is a repair, not a measurement: without this the
    imputed value is indistinguishable from a real reading, which on this dataset silently affects
    ~11% of the soil and climate records. Consumers that can act on the difference should ask for it.
    """
    columns = [column for column in dict.fromkeys(columns) if column in frame.columns]
    if not columns:
        return (frame, pd.DataFrame(index=frame.index)) if return_validity else frame

    sanitized = frame.copy()
    validity: dict[str, np.ndarray] = {}
    repaired_text: dict[str, int] = {}
    non_finite: dict[str, int] = {}

    for column in columns:
        series = sanitized[column]
        if series.dtype == object:
            as_text = series.astype("string")
            comma_count = int(as_text.str.contains(",", na=False).sum())
            if comma_count:
                repaired_text[column] = comma_count
                as_text = as_text.str.replace(",", ".", regex=False)
            series = pd.to_numeric(as_text, errors="coerce")
        else:
            series = pd.to_numeric(series, errors="coerce")

        series = series.replace([np.inf, -np.inf], np.nan)
        # Captured before the fill below - afterwards the information is gone for good.
        validity[column] = series.notna().to_numpy(dtype=bool)
        missing = int(series.isna().sum())
        if missing:
            non_finite[column] = missing
            median = series.median()
            series = series.fillna(0.0 if pd.isna(median) else median)
        sanitized[column] = series.astype(np.float32)

    if repaired_text:
        logger.warning(
            f"Repaired decimal-comma values in time-series column(s): "
            f"{', '.join(f'{name} ({count})' for name, count in repaired_text.items())}"
        )
    if non_finite:
        total = sum(non_finite.values())
        worst = sorted(non_finite.items(), key=lambda item: item[1], reverse=True)[:3]
        logger.warning(
            f"Median-filled {total} non-finite time-series cell(s) across {len(non_finite)} column(s); "
            f"worst: {', '.join(f'{name} ({count})' for name, count in worst)}"
        )

    if return_validity:
        return sanitized, pd.DataFrame(validity, index=frame.index)
    return sanitized
