"""The cleaning helpers shared by the graph and sequence builders.

These were lifted out of SpatiotemporalGraphBuilder so both paths clean their inputs identically;
these tests pin the behaviour that both now depend on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yg_eo_soilnet.datamodules.frame_cleaning import (
    build_finite_row_mask,
    drop_non_finite_rows,
    encode_categorical_features,
    sanitize_numeric_columns,
)


def test_sanitize_repairs_decimal_comma_strings(logger) -> None:
    frame = pd.DataFrame({"S2_b2": ["0,5", "1,25", "2.0"]})
    cleaned = sanitize_numeric_columns(frame, ["S2_b2"], logger=logger)

    assert cleaned["S2_b2"].dtype == np.float32
    np.testing.assert_allclose(cleaned["S2_b2"].to_numpy(), [0.5, 1.25, 2.0])


def test_sanitize_median_fills_infinities_without_dropping_rows(logger) -> None:
    frame = pd.DataFrame({"S1_ratio": [1.0, np.inf, 3.0, -np.inf, 5.0]})
    cleaned = sanitize_numeric_columns(frame, ["S1_ratio"], logger=logger)

    assert len(cleaned) == 5, "a sparse column must not amputate whole rows"
    assert np.isfinite(cleaned["S1_ratio"]).all()
    assert cleaned["S1_ratio"].iloc[1] == pytest.approx(3.0)  # median of the finite values


def test_sanitize_ignores_columns_that_are_absent(logger) -> None:
    frame = pd.DataFrame({"present": [1.0, 2.0]})
    cleaned = sanitize_numeric_columns(frame, ["present", "missing"], logger=logger)
    assert list(cleaned.columns) == ["present"]


def test_encode_categorical_features_ordinal_encodes_and_keeps_numerics(logger) -> None:
    frame = pd.DataFrame({"texture": ["clay", "sand", "clay"], "slope": [1.0, 2.0, 3.0]})
    encoded, feature_columns = encode_categorical_features(frame, ["texture", "slope"], logger=logger)

    assert feature_columns == ["texture", "slope"]
    assert encoded["texture"].tolist() == [0.0, 1.0, 0.0]
    assert encoded["slope"].tolist() == [1.0, 2.0, 3.0]


def test_encode_categorical_features_marks_missing_as_nan(logger) -> None:
    """factorize's -1 sentinel would read as a real category; NaN defers to the row-drop pass."""
    frame = pd.DataFrame({"texture": ["clay", None, "sand"]})
    encoded, _ = encode_categorical_features(frame, ["texture"], logger=logger)

    assert encoded["texture"].isna().tolist() == [False, True, False]
    assert -1.0 not in encoded["texture"].dropna().tolist()


def test_encode_categorical_features_drops_a_column_with_no_categories(logger) -> None:
    frame = pd.DataFrame({"empty": [None, None], "slope": [1.0, 2.0]})
    _, feature_columns = encode_categorical_features(frame, ["empty", "slope"], logger=logger)
    assert feature_columns == ["slope"]


def test_build_finite_row_mask_flags_missing_and_non_finite(logger) -> None:
    frame = pd.DataFrame({"uuid": ["a", None, "c", "d"], "value": [1.0, 2.0, np.nan, 4.0]})
    mask = build_finite_row_mask(frame, required_columns=["uuid"], numeric_columns=["value"])
    assert mask.tolist() == [True, False, False, True]


def test_drop_non_finite_rows_keeps_only_clean_rows(logger) -> None:
    frame = pd.DataFrame({"uuid": ["a", "b", "c"], "value": [1.0, np.inf, 3.0]})
    kept = drop_non_finite_rows(
        frame, logger=logger, label="test frame", required_columns=["uuid"], numeric_columns=["value"]
    )
    assert kept["uuid"].tolist() == ["a", "c"]


def test_drop_non_finite_rows_returns_a_copy_when_nothing_is_dropped(logger) -> None:
    frame = pd.DataFrame({"value": [1.0, 2.0]})
    kept = drop_non_finite_rows(frame, logger=logger, label="test frame", numeric_columns=["value"])

    kept.loc[0, "value"] = 99.0
    assert frame.loc[0, "value"] == 1.0, "the caller's frame must not be mutated"
