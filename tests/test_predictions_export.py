"""The per-point prediction export: id alignment, the two file shapes, and the off-switch."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from yg_eo_soilnet.predictions_export import (
    combine,
    duplicate_report,
    export_enabled_for,
    point_id_column,
    point_prediction_frame,
    to_long,
    to_wide,
    wide_column_name,
)


def _config(**overrides) -> SimpleNamespace:
    base = dict(
        POINT_ID_COLUMN="uuid",
        EXPORT_POINT_PREDICTIONS=True,
        EXPORT_POINT_PREDICTIONS_MODELS=[],
        EXPORT_POINT_PREDICTIONS_SKIP_MODELS=[],
        EXPORT_POINT_PREDICTIONS_FAIL_ON_ERROR=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# --- the switch ------------------------------------------------------------


def test_the_master_switch_turns_the_export_off():
    assert export_enabled_for(_config(EXPORT_POINT_PREDICTIONS=False), "Ridge") is False


def test_a_skipped_model_is_not_exported():
    config = _config(EXPORT_POINT_PREDICTIONS_SKIP_MODELS=["TabICL"])
    assert export_enabled_for(config, "TabICL") is False
    assert export_enabled_for(config, "Ridge") is True


def test_naming_a_model_explicitly_beats_the_skip_list():
    config = _config(
        EXPORT_POINT_PREDICTIONS_MODELS=["TabICL"],
        EXPORT_POINT_PREDICTIONS_SKIP_MODELS=["TabICL"],
    )
    assert export_enabled_for(config, "TabICL") is True
    assert export_enabled_for(config, "Ridge") is False


def test_the_id_column_comes_from_the_runs_own_config():
    assert point_id_column(_config()) == "uuid"
    assert point_id_column(SimpleNamespace()) == "point_id"


# --- the child frame -------------------------------------------------------


def test_the_child_frame_pairs_each_id_with_its_own_prediction():
    frame = point_prediction_frame(
        ["a", "b", "c"], np.array([1.0, 2.0, 3.0]), ["clay_pct"], id_column="uuid"
    )
    assert list(frame.columns) == ["uuid", "clay_pct"]
    assert frame.loc[frame["uuid"] == "b", "clay_pct"].iloc[0] == 2.0


def test_a_joint_model_contributes_one_column_per_target():
    frame = point_prediction_frame(
        ["a", "b"], np.array([[1.0, 10.0], [2.0, 20.0]]), ["clay_pct", "sand_pct"], "uuid"
    )
    assert list(frame.columns) == ["uuid", "clay_pct", "sand_pct"]


def test_ids_and_predictions_of_different_lengths_are_refused():
    # The failure this guards is silent: a positional pairing of the wrong length still produces a
    # readable file, just one where every prediction belongs to a different point.
    with pytest.raises(ValueError, match="describe different points"):
        point_prediction_frame(["a", "b"], np.array([1.0, 2.0, 3.0]), ["clay_pct"], "uuid")


def test_a_prediction_width_that_does_not_match_the_targets_is_refused():
    with pytest.raises(ValueError, match="prediction columns against"):
        point_prediction_frame(["a"], np.array([[1.0, 2.0]]), ["clay_pct"], "uuid")


def test_ids_are_taken_by_index_not_by_position_after_row_filtering():
    """The TargetNanFilter case, which is why the ids are reindexed rather than sliced.

    TargetNanFilter drops rows per target group but PRESERVES the pandas index, so a group's frame
    has gaps in it. Pairing `point_ids` positionally against that frame would hand row 3 the id
    belonging to row 1 - a wrong file that looks entirely correct.
    """
    point_ids = pd.Series(["p0", "p1", "p2", "p3", "p4"], index=range(5), name="point_id")
    # Rows 1 and 3 dropped, index preserved.
    surviving = pd.DataFrame({"feature": [0.0, 0.0, 0.0]}, index=[0, 2, 4])

    aligned = point_ids.reindex(surviving.index).to_numpy()
    assert list(aligned) == ["p0", "p2", "p4"]

    frame = point_prediction_frame(aligned, np.array([1.0, 2.0, 3.0]), ["clay_pct"], "uuid")
    assert list(frame["uuid"]) == ["p0", "p2", "p4"]
    # The positional version would have produced p0, p1, p2 - the bug this test exists to catch.
    assert list(frame["uuid"]) != list(point_ids.iloc[: len(surviving)])


# --- combining -------------------------------------------------------------


def _children():
    return [
        ("Ridge", pd.DataFrame({"uuid": ["a", "b"], "clay_pct": [1.0, 2.0], "sand_pct": [3.0, 4.0]})),
        ("XGBoost", pd.DataFrame({"uuid": ["a", "b"], "clay_pct": [1.5, 2.5]})),
    ]


def test_the_long_file_keeps_target_and_model_as_separate_columns():
    long_frame = to_long(_children(), id_column="uuid")
    assert list(long_frame.columns) == ["uuid", "target", "model", "prediction"]
    assert len(long_frame) == 6  # 2 points x (2 Ridge targets + 1 XGBoost target)
    row = long_frame[(long_frame.uuid == "a") & (long_frame.model == "XGBoost")]
    assert row["prediction"].iloc[0] == 1.5


def test_the_wide_file_has_one_column_per_target_and_model():
    wide, _long = combine(_children(), id_column="uuid")
    assert set(wide.columns) == {"uuid", "clay_pct__Ridge", "sand_pct__Ridge", "clay_pct__XGBoost"}
    assert len(wide) == 2


def test_the_wide_file_has_one_row_per_point():
    wide, _long = combine(_children(), id_column="uuid")
    assert wide["uuid"].is_unique


def test_the_two_files_agree_on_every_value():
    wide, long_frame = combine(_children(), id_column="uuid")
    for _, row in long_frame.iterrows():
        column = wide_column_name(row["target"], row["model"])
        wide_value = wide.loc[wide["uuid"] == row["uuid"], column].iloc[0]
        assert wide_value == row["prediction"]


def test_a_model_missing_a_target_leaves_a_gap_rather_than_shifting_columns():
    # XGBoost fits only clay_pct here; its sand_pct cell must be empty, not borrowed from Ridge.
    wide, _long = combine(_children(), id_column="uuid")
    assert "sand_pct__XGBoost" not in wide.columns
    assert wide["sand_pct__Ridge"].notna().all()


def test_wide_column_names_sanitise_both_halves():
    assert wide_column_name("clay pct", "soil/cnn") == "clay_pct__soil_cnn"


def test_combining_nothing_produces_empty_frames_rather_than_raising():
    wide, long_frame = combine([], id_column="uuid")
    assert wide.empty and long_frame.empty
    assert list(long_frame.columns) == ["uuid", "target", "model", "prediction"]


def test_a_child_frame_without_the_id_column_is_skipped():
    frames = [("Ridge", pd.DataFrame({"clay_pct": [1.0]}))]
    assert to_long(frames, id_column="uuid").empty


def test_duplicate_rows_are_reported_because_the_pivot_hides_them():
    # pivot_table keeps the first of a duplicate, so without this check a model collected twice
    # would look completely normal in the output.
    duplicated = pd.concat([to_long(_children(), "uuid")] * 2, ignore_index=True)
    assert "duplicate" in duplicate_report(duplicated, id_column="uuid")
    assert duplicate_report(to_long(_children(), "uuid"), id_column="uuid") is None


def test_the_export_carries_no_uncertainty_columns():
    # The contract: estimates only. Sigma and intervals stay in eval_results.csv.
    wide, long_frame = combine(_children(), id_column="uuid")
    for frame in (wide, long_frame):
        assert not [c for c in frame.columns if c.endswith(("_std", "_lower", "_upper"))]


def test_to_wide_on_an_empty_long_frame_returns_just_the_id_column():
    assert list(to_wide(pd.DataFrame(), id_column="uuid").columns) == ["uuid"]
