from pathlib import Path
import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from yg_eo_soilnet.data_manager import DataManager


def test_load_tabular_data_reads_joint_csv(tmp_path: Path, toy_config, logger) -> None:
    csv_path = tmp_path / toy_config.DATA_FILE
    frame = pd.DataFrame({"lat": [0.0], "lon": [0.0], "target_a": [1.0], "target_b": [2.0]})
    frame.to_csv(csv_path, index=False)
    toy_config.DATA_FOLDER = str(tmp_path)

    manager = DataManager(toy_config, logger)

    loaded = manager.load_tabular_data()

    assert list(loaded.columns) == ["lat", "lon", "target_a", "target_b"]


def test_load_tabular_data_raises_when_csv_missing(tmp_path: Path, toy_config, logger) -> None:
    toy_config.DATA_FOLDER = str(tmp_path)

    manager = DataManager(toy_config, logger)

    with pytest.raises(FileNotFoundError):
        manager.load_tabular_data()


def test_load_tabular_data_joins_static_and_targets(tmp_path: Path, toy_config, logger) -> None:
    static_path = tmp_path / "static.csv"
    targets_path = tmp_path / "targets.csv"

    pd.DataFrame(
        {
            "point_id": [1, 2, 3],
            "lat": [0.0, 0.1, 0.2],
            "lon": [0.0, 0.1, 0.2],
            "feature": [10.0, 20.0, 30.0],
        }
    ).to_csv(static_path, index=False)
    pd.DataFrame(
        {
            "point_id": [1, 2, 3],
            "target_a": [1.0, 2.0, 3.0],
            "target_b": [4.0, 5.0, 6.0],
        }
    ).to_csv(targets_path, index=False)

    toy_config.DATA_FOLDER = str(tmp_path)
    toy_config.DATA_FILE = "static.csv"
    toy_config.STATIC_FEATURES_FILE = "static.csv"
    toy_config.TARGETS_FILE = "targets.csv"
    toy_config.TARGET_COLUMNS = ["target_a", "target_b"]
    toy_config.POINT_ID_COLUMN = "point_id"

    manager = DataManager(toy_config, logger)
    joined = manager.load_tabular_data()

    assert list(joined.columns) == ["point_id", "lat", "lon", "feature", "target_a", "target_b"]
    assert joined.shape == (3, 6)


def _split_sources(tmp_path: Path, toy_config) -> None:
    """Static covariates in one file, every measured lab value in another - the real layout."""
    pd.DataFrame(
        {"point_id": [1, 2, 3], "lat": [0.0, 0.1, 0.2], "lon": [0.0, 0.1, 0.2], "feature": [10.0, 20.0, 30.0]}
    ).to_csv(tmp_path / "static.csv", index=False)
    pd.DataFrame(
        {
            "point_id": [1, 2, 3],
            "target_a": [1.0, 2.0, 3.0],
            "target_b": [4.0, 5.0, 6.0],
            "lab_extra": [7.0, 8.0, 9.0],
        }
    ).to_csv(tmp_path / "targets.csv", index=False)

    toy_config.DATA_FOLDER = str(tmp_path)
    toy_config.DATA_FILE = "static.csv"
    toy_config.STATIC_FEATURES_FILE = "static.csv"
    toy_config.TARGETS_FILE = "targets.csv"
    toy_config.TARGET_COLUMNS = ["target_a"]
    toy_config.LABEL_COLUMNS = ["target_a", "target_b", "lab_extra"]
    toy_config.POINT_ID_COLUMN = "point_id"


def test_the_targets_join_carries_only_active_targets_by_default(tmp_path: Path, toy_config, logger) -> None:
    """A run that never opted in must see exactly the frame it saw before the flag existed."""
    _split_sources(tmp_path, toy_config)
    toy_config.CARRY_LABEL_COLUMNS = False

    joined = DataManager(toy_config, logger).load_tabular_data()

    assert list(joined.columns) == ["point_id", "lat", "lon", "feature", "target_a"]


def test_the_targets_join_carries_every_label_column_when_asked(tmp_path: Path, toy_config, logger) -> None:
    """Without this the split layout offers a model only the active target, while a joint file
    offers every lab value - the same declaration meaning two different things."""
    _split_sources(tmp_path, toy_config)
    toy_config.CARRY_LABEL_COLUMNS = True

    manager = DataManager(toy_config, logger)
    joined = manager.load_tabular_data()

    # target_a is both a target and a label; it must cross exactly once.
    assert list(joined.columns) == ["point_id", "lat", "lon", "feature", "target_a", "target_b", "lab_extra"]
    assert joined["target_a"].tolist() == [1.0, 2.0, 3.0]
    # Carried, but still not predictors: the schema filter drops every label by name.
    assert manager.filter_schema(joined, ["target_a"]).columns.tolist() == ["feature"]


def test_load_tabular_data_discovers_csv_and_parquet_shards_one_level_deep(tmp_path: Path, toy_config, logger) -> None:
    pytest.importorskip("pyarrow")

    static_root = tmp_path / "static_features"
    targets_root = tmp_path / "targets"
    static_nested = static_root / "nested"
    targets_nested = targets_root / "nested"
    static_nested.mkdir(parents=True)
    targets_nested.mkdir(parents=True)

    pd.DataFrame(
        {"point_id": [1, 2], "lat": [0.0, 0.1], "lon": [0.0, 0.1], "feature": [10.0, 20.0]}
    ).to_csv(static_root / "part_1.csv", index=False)
    pd.DataFrame(
        {"point_id": [3], "lat": [0.2], "lon": [0.2], "feature": [30.0]}
    ).to_parquet(static_nested / "part_2.parquet", index=False)
    pd.DataFrame({"point_id": [1, 2], "target_a": [11.0, 22.0]}).to_csv(targets_root / "targets_1.csv", index=False)
    pd.DataFrame({"point_id": [3], "target_a": [33.0]}).to_parquet(targets_nested / "targets_2.parquet", index=False)

    toy_config.DATA_FOLDER = str(tmp_path)
    toy_config.STATIC_FEATURES_FOLDER = "static_features"
    toy_config.TARGETS_FOLDER = "targets"
    toy_config.TARGET_COLUMNS = ["target_a"]
    toy_config.POINT_ID_COLUMN = "point_id"

    manager = DataManager(toy_config, logger)
    joined = manager.load_tabular_data()

    ordered = joined.sort_values("point_id").reset_index(drop=True)
    assert ordered["point_id"].tolist() == [1, 2, 3]
    assert ordered["target_a"].tolist() == [11.0, 22.0, 33.0]


def test_load_tabular_data_uses_manifest_relative_to_data_folder(tmp_path: Path, toy_config, logger) -> None:
    pytest.importorskip("pyarrow")

    static_root = tmp_path / "static_features"
    targets_root = tmp_path / "targets"
    static_nested = static_root / "nested"
    static_nested.mkdir(parents=True)
    targets_root.mkdir(parents=True)

    pd.DataFrame(
        {"point_id": [1], "lat": [0.0], "lon": [0.0], "feature": [10.0]}
    ).to_csv(static_root / "part_1.csv", index=False)
    pd.DataFrame(
        {"point_id": [2], "lat": [0.1], "lon": [0.1], "feature": [20.0]}
    ).to_parquet(static_nested / "part_2.parquet", index=False)
    pd.DataFrame({"point_id": [1], "target_a": [100.0]}).to_csv(targets_root / "targets.csv", index=False)
    pd.DataFrame({"point_id": [99], "target_a": [999.0]}).to_csv(targets_root / "ignored.csv", index=False)

    manifest_path = tmp_path / "index.json"
    manifest_path.write_text(
        json.dumps(
            {
                "static_features": ["static_features/part_1.csv", "static_features/nested/part_2.parquet"],
                "targets": ["targets/targets.csv"],
            }
        )
    )

    toy_config.DATA_FOLDER = str(tmp_path)
    toy_config.STATIC_FEATURES_FOLDER = "static_features"
    toy_config.TARGETS_FOLDER = "targets"
    toy_config.DATA_INDEX_MANIFEST_PATH = str(manifest_path)
    toy_config.TARGET_COLUMNS = ["target_a"]
    toy_config.POINT_ID_COLUMN = "point_id"

    manager = DataManager(toy_config, logger)
    joined = manager.load_tabular_data()

    assert joined["point_id"].tolist() == [1]
    assert joined["target_a"].tolist() == [100.0]


def test_load_timeseries_data_discovers_modality_folders_one_level_deep(tmp_path: Path, toy_config, logger) -> None:
    pytest.importorskip("pyarrow")

    timeseries_root = tmp_path / "timeseries"
    radar_root = timeseries_root / "radar"
    optical_root = timeseries_root / "optical"
    radar_nested = radar_root / "nested"
    radar_nested.mkdir(parents=True)
    optical_root.mkdir(parents=True)

    pd.DataFrame(
        {"point_id": [1], "date": ["2020-01-01"], "RADAR_VV": [0.1], "RADAR_VH": [0.2]}
    ).to_csv(radar_root / "radar_a.csv", index=False)
    pd.DataFrame(
        {"point_id": [2], "date": ["2020-01-02"], "RADAR_VV": [0.3], "RADAR_VH": [0.4]}
    ).to_parquet(radar_nested / "radar_b.parquet", index=False)
    pd.DataFrame(
        {"point_id": [1], "date": ["2020-01-01"], "OPT_RED": [0.3], "OPT_NIR": [0.4]}
    ).to_csv(optical_root / "optical_a.csv", index=False)

    toy_config.DATA_FOLDER = str(tmp_path)
    toy_config.TIMESERIES_FOLDER = "timeseries"
    toy_config.TEMPORAL_FEATURES = {
        "enabled": True,
        "time_column": "date",
        "modality_prefix_map": {"radar": "RADAR_", "optical": "OPT_"},
    }
    toy_config.TEMPORAL_FEATURES_ENABLED = True

    manager = DataManager(toy_config, logger)
    timeseries = manager.load_timeseries_data()

    assert timeseries is not None
    assert set(timeseries.columns) >= {"point_id", "date", "RADAR_VV", "RADAR_VH", "OPT_RED", "OPT_NIR"}
    assert timeseries.shape[0] == 2


# --- schema filtering -----------------------------------------------------


def test_filter_schema_drops_every_configured_exclusion_source(toy_config, logger) -> None:
    """All four exclusion sources must apply; a duplicated definition once silenced three of them."""
    toy_config.ELIMINATED_FEATURES = ["eliminated"]
    toy_config.EXCLUDE_CATEGORICAL = ["excluded_cat"]
    toy_config.EXISTING_HS_FEATURES = {"enabled": True, "ignore": True, "prefix": "S2_", "band_names": ["band_x"]}
    toy_config.IGNORE_BANDS = ["legacy_band"]

    frame = pd.DataFrame(
        {
            "point_id": [1, 2],
            "lat": [0.0, 1.0],
            "lon": [0.0, 1.0],
            "geometry": ["a", "b"],
            "target_a": [1.0, 2.0],
            "target_b": [3.0, 4.0],
            "eliminated": [1, 2],
            "excluded_cat": ["x", "y"],
            "S2_B2": [1, 2],
            "band_x": [1, 2],
            "legacy_band": [1, 2],
            "keep_me": [1, 2],
        }
    )

    filtered = DataManager(toy_config, logger).filter_schema(frame)

    assert list(filtered.columns) == ["keep_me"]


def test_filter_schema_returns_a_copy_when_nothing_is_dropped(toy_config, logger) -> None:
    frame = pd.DataFrame({"keep_me": [1, 2]})

    filtered = DataManager(toy_config, logger).filter_schema(frame)

    assert list(filtered.columns) == ["keep_me"]
    assert filtered is not frame


def test_metadata_columns_can_exclude_targets(toy_config, logger) -> None:
    manager = DataManager(toy_config, logger)

    with_targets = manager.metadata_columns()
    without_targets = manager.metadata_columns(include_targets=False)

    assert {"target_a", "target_b"} <= with_targets
    assert not {"target_a", "target_b"} & without_targets
    assert {"point_id", "lat", "lon", "geometry"} <= without_targets


# --- boundary: framework-specific work must live outside DataManager ------


@pytest.mark.parametrize(
    "method_name",
    [
        "preprocess_data",
        "split_data",
        "build_spatiotemporal_graph",
        "split_modalities",
        "compute_baseline_residuals",
        "build_edge_index",
        "load_static_data",
        # Retired in favour of load_dataset(): these returned frames that may lack targets,
        # which made callers re-derive joint-vs-separate for themselves.
        "load_data",
        "load_targets_data",
    ],
)
def test_data_manager_does_not_expose_framework_specific_methods(toy_config, logger, method_name) -> None:
    assert not hasattr(DataManager(toy_config, logger), method_name)


@pytest.mark.parametrize("attribute_name", ["_tabular_preprocessor", "_graph_builder", "_sklearn_splitter"])
def test_data_manager_holds_no_framework_collaborators(toy_config, logger, attribute_name) -> None:
    assert not hasattr(DataManager(toy_config, logger), attribute_name)


# --- load_dataset: the three declared input shapes -------------------------


def _write_static_and_targets(tmp_path: Path) -> None:
    pd.DataFrame(
        {"point_id": [1, 2, 3], "lat": [0.0, 0.1, 0.2], "lon": [0.0, 0.1, 0.2], "feature": [10.0, 20.0, 30.0]}
    ).to_csv(tmp_path / "static.csv", index=False)
    pd.DataFrame(
        {"point_id": [1, 2, 3], "target_a": [1.0, 2.0, 3.0], "target_b": [4.0, 5.0, 6.0]}
    ).to_csv(tmp_path / "targets.csv", index=False)


def test_load_dataset_from_joint_file_never_opens_a_targets_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, toy_config, logger
) -> None:
    """Shape 1: static and targets in one file."""
    pd.DataFrame(
        {"point_id": [1], "lat": [0.0], "lon": [0.0], "feature": [10.0], "target_a": [1.0], "target_b": [2.0]}
    ).to_csv(tmp_path / "joint.csv", index=False)
    toy_config.DATA_FOLDER = str(tmp_path)
    toy_config.DATA_FILE = "joint.csv"
    toy_config.STATIC_FEATURES_FILE = "joint.csv"

    manager = DataManager(toy_config, logger)
    reads: list[str] = []
    original = manager._read_tabular_file
    monkeypatch.setattr(manager, "_read_tabular_file", lambda p, **kw: (reads.append(p), original(p, **kw))[1])

    dataset = manager.load_dataset()

    assert {"target_a", "target_b"} <= set(dataset.tabular.columns)
    assert [Path(p).name for p in reads] == ["joint.csv"]


def test_load_dataset_joins_separate_static_and_targets_files(tmp_path: Path, toy_config, logger) -> None:
    """Shape 2a: static and targets in two files."""
    _write_static_and_targets(tmp_path)
    toy_config.DATA_FOLDER = str(tmp_path)
    toy_config.DATA_FILE = "static.csv"
    toy_config.STATIC_FEATURES_FILE = "static.csv"
    toy_config.TARGETS_FILE = "targets.csv"

    dataset = DataManager(toy_config, logger).load_dataset()

    assert dataset.tabular.shape == (3, 6)
    assert dataset.target_columns == ["target_a", "target_b"]
    assert dataset.point_id_column == "point_id"


def test_load_dataset_joins_separate_static_and_targets_folders(tmp_path: Path, toy_config, logger) -> None:
    """Shape 2b: static and targets as folders of shards."""
    static_root = tmp_path / "static_features"
    targets_root = tmp_path / "targets"
    static_root.mkdir()
    targets_root.mkdir()

    pd.DataFrame({"point_id": [1], "lat": [0.0], "lon": [0.0], "feature": [10.0]}).to_csv(
        static_root / "a.csv", index=False
    )
    pd.DataFrame({"point_id": [2], "lat": [0.1], "lon": [0.1], "feature": [20.0]}).to_csv(
        static_root / "b.csv", index=False
    )
    pd.DataFrame({"point_id": [1, 2], "target_a": [1.0, 2.0], "target_b": [3.0, 4.0]}).to_csv(
        targets_root / "t.csv", index=False
    )

    toy_config.DATA_FOLDER = str(tmp_path)
    toy_config.STATIC_FEATURES_FOLDER = str(static_root)
    toy_config.TARGETS_FOLDER = str(targets_root)

    dataset = DataManager(toy_config, logger).load_dataset()

    assert sorted(dataset.tabular["point_id"].tolist()) == [1, 2]
    assert {"target_a", "target_b"} <= set(dataset.tabular.columns)


def test_load_dataset_timeseries_is_none_when_temporal_disabled(tmp_path: Path, toy_config, logger) -> None:
    _write_static_and_targets(tmp_path)
    toy_config.DATA_FOLDER = str(tmp_path)
    toy_config.DATA_FILE = "static.csv"
    toy_config.STATIC_FEATURES_FILE = "static.csv"
    toy_config.TARGETS_FILE = "targets.csv"

    dataset = DataManager(toy_config, logger).load_dataset()

    assert dataset.has_timeseries is False
    assert dataset.timeseries is None


def test_load_dataset_does_not_read_timeseries_until_accessed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, toy_config, logger
) -> None:
    """Time-series is lazy: a sklearn-only run must never open the large file."""
    _write_static_and_targets(tmp_path)
    pd.DataFrame(
        {"point_id": [1, 1], "date": ["2020-01-01", "2020-02-01"], "RADAR_VV": [0.1, 0.2]}
    ).to_csv(tmp_path / "ts.csv", index=False)

    toy_config.DATA_FOLDER = str(tmp_path)
    toy_config.DATA_FILE = "static.csv"
    toy_config.STATIC_FEATURES_FILE = "static.csv"
    toy_config.TARGETS_FILE = "targets.csv"
    toy_config.TEMPORAL_FEATURES = {"enabled": True, "time_column": "date", "timeseries_file": "ts.csv"}
    toy_config.TEMPORAL_FEATURES_ENABLED = True

    manager = DataManager(toy_config, logger)
    reads: list[str] = []
    original = manager._read_tabular_file
    monkeypatch.setattr(manager, "_read_tabular_file", lambda p, **kw: (reads.append(Path(p).name), original(p, **kw))[1])

    dataset = manager.load_dataset()
    assert dataset.has_timeseries is True
    assert "ts.csv" not in reads

    assert dataset.timeseries is not None
    assert reads.count("ts.csv") == 1

    dataset.timeseries  # memoized on the bundle
    assert reads.count("ts.csv") == 1


def test_load_dataset_caches_sources_until_reload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, toy_config, logger
) -> None:
    """The static source used to be read twice per run - once per framework."""
    _write_static_and_targets(tmp_path)
    toy_config.DATA_FOLDER = str(tmp_path)
    toy_config.DATA_FILE = "static.csv"
    toy_config.STATIC_FEATURES_FILE = "static.csv"
    toy_config.TARGETS_FILE = "targets.csv"

    manager = DataManager(toy_config, logger)
    reads: list[str] = []
    original = manager._read_tabular_file
    monkeypatch.setattr(manager, "_read_tabular_file", lambda p, **kw: (reads.append(Path(p).name), original(p, **kw))[1])

    manager.load_dataset()
    manager.load_dataset()
    assert reads.count("static.csv") == 1

    manager.reload()
    manager.load_dataset()
    assert reads.count("static.csv") == 2


def test_manifest_is_honored_for_a_file_based_config(tmp_path: Path, toy_config, logger) -> None:
    """The manifest used to be unreachable unless the configured path was a directory."""
    pd.DataFrame(
        {"point_id": [7], "lat": [0.0], "lon": [0.0], "feature": [1.0], "target_a": [1.0], "target_b": [2.0]}
    ).to_csv(tmp_path / "from_manifest.csv", index=False)
    pd.DataFrame(
        {"point_id": [99], "lat": [9.0], "lon": [9.0], "feature": [9.0], "target_a": [9.0], "target_b": [9.0]}
    ).to_csv(tmp_path / "ignored.csv", index=False)

    manifest_path = tmp_path / "index.json"
    manifest_path.write_text(json.dumps({"static_features": ["from_manifest.csv"]}))

    toy_config.DATA_FOLDER = str(tmp_path)
    toy_config.DATA_FILE = "ignored.csv"
    toy_config.STATIC_FEATURES_FILE = "ignored.csv"
    toy_config.DATA_INDEX_MANIFEST_PATH = str(manifest_path)

    dataset = DataManager(toy_config, logger).load_dataset()

    assert dataset.tabular["point_id"].tolist() == [7]


def test_label_columns_are_excluded_even_when_not_being_fitted(toy_config, logger) -> None:
    """Commenting a target out of TARGET_COLUMNS must not promote it to a feature.

    This is the joint static+targets case: every wet-lab column is in the frame, and only the
    active target used to be dropped, so 21 co-measured labels became predictors.
    """
    toy_config.TARGET_COLUMNS = ["target_a"]
    toy_config.LABEL_COLUMNS = ["target_a", "target_b", "ph_water", "sand_pct"]

    frame = pd.DataFrame(
        {
            "point_id": [1, 2],
            "lat": [0.0, 1.0],
            "lon": [0.0, 1.0],
            "target_a": [1.0, 2.0],   # active target
            "target_b": [3.0, 4.0],   # a label we are not fitting
            "ph_water": [7.0, 8.0],   # co-measured lab value
            "sand_pct": [30.0, 40.0],
            "elevation": [100.0, 200.0],
        }
    )

    filtered = DataManager(toy_config, logger).filter_schema(frame)

    assert list(filtered.columns) == ["elevation"]


def test_label_columns_apply_even_when_targets_are_passed_explicitly(toy_config, logger) -> None:
    """The graph builder passes its own target list; LABEL_COLUMNS must still apply."""
    toy_config.LABEL_COLUMNS = ["target_b", "ph_water"]
    frame = pd.DataFrame({"target_a": [1.0], "target_b": [2.0], "ph_water": [7.0], "elevation": [10.0]})

    filtered = DataManager(toy_config, logger).filter_schema(frame, ["target_a"])

    assert list(filtered.columns) == ["elevation"]


def test_absent_label_columns_reproduces_previous_behaviour(toy_config, logger) -> None:
    """An unset LABEL_COLUMNS must leave existing configs behaving exactly as before."""
    toy_config.TARGET_COLUMNS = ["target_a"]
    toy_config.LABEL_COLUMNS = []
    frame = pd.DataFrame({"target_a": [1.0], "target_b": [2.0], "elevation": [10.0]})

    filtered = DataManager(toy_config, logger).filter_schema(frame)

    assert sorted(filtered.columns) == ["elevation", "target_b"]
