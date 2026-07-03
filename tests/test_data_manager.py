from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from yg_eo_soilnet.data_manager import DataManager
from yg_eo_soilnet.models import BaseSpatialClusterStrategy


def test_load_data_reads_csv(tmp_path: Path, toy_config, logger) -> None:
    csv_path = tmp_path / toy_config.DATA_FILE
    frame = pd.DataFrame({"lat": [0.0], "lon": [0.0], "target_a": [1.0]})
    frame.to_csv(csv_path, index=False)
    toy_config.DATA_FOLDER = str(tmp_path)

    manager = DataManager(toy_config, logger)

    loaded = manager.load_data()

    assert list(loaded.columns) == ["lat", "lon", "target_a"]


def test_load_data_raises_when_csv_missing(tmp_path: Path, toy_config, logger) -> None:
    toy_config.DATA_FOLDER = str(tmp_path)

    manager = DataManager(toy_config, logger)

    try:
        manager.load_data()
    except FileNotFoundError as exc:
        assert "CSV file not found" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_preprocess_data_returns_expected_structure(toy_config, toy_dataframe, logger, caplog) -> None:
    manager = DataManager(toy_config, logger)

    with caplog.at_level("INFO"):
        processed = manager.preprocess_data(toy_dataframe)

    assert list(processed["X"].columns) == ["lat", "lon", "cat", "feature"]
    assert list(processed["y"].columns) == ["target_a", "target_b"]
    assert list(processed["lat"].index) == [0, 1, 2, 3]
    assert "Categorical encoding will be applied" in caplog.text


def test_split_data_without_clustering(monkeypatch: pytest.MonkeyPatch, toy_config, toy_dataframe, logger) -> None:
    from sklearn.model_selection import train_test_split

    manager = DataManager(toy_config, logger)
    processed = manager.preprocess_data(toy_dataframe)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: Path(path).write_text("frame"))
    monkeypatch.setattr(pd.Series, "to_parquet", lambda self, path, index=False: Path(path).write_text("series"), raising=False)
    monkeypatch.setattr("yg_eo_soilnet.data_manager.mlflow.log_artifacts", lambda *args, **kwargs: None)

    split = manager.split_data(processed)

    expected = train_test_split(
        processed["X"].drop(["lat", "lon"], axis=1),
        processed["y"],
        processed["lat"],
        processed["lon"],
        test_size=toy_config.TEST_SIZE,
        random_state=toy_config.RANDOM_SEED,
    )

    assert split["X_train"].shape[0] == expected[0].shape[0]
    assert split["X_test"].shape[0] == expected[1].shape[0]
    assert "groups_train" not in split


def test_split_data_with_clustering(monkeypatch: pytest.MonkeyPatch, toy_config, toy_dataframe, logger) -> None:
    class FakeClusterStrategy(BaseSpatialClusterStrategy):
        def __init__(self) -> None:
            self.plot_calls = []

        def cluster(self, df: pd.DataFrame) -> pd.DataFrame:
            clustered = df.copy()
            clustered["cluster"] = [1, 1, 2, 2]
            return clustered

        def plot_train_test(self, *args, **kwargs):
            self.plot_calls.append((args, kwargs))

    class FakeFactory:
        def __init__(self, *args, **kwargs) -> None:
            self.strategy = FakeClusterStrategy()

        def load_splitter_from_config(self):
            return self.strategy

    toy_config.ENABLE_CLUSTERING = True
    toy_config.CLUSTERING_STRATEGY = {
        "enabled": True,
        "class_path": "yg_eo_soilnet.models.KMeansClusterStrategy",
        "params": {"n_clusters": 2},
    }

    manager = DataManager(toy_config, logger)
    processed = manager.preprocess_data(toy_dataframe)
    processed = {key: value.copy() if hasattr(value, "copy") else value for key, value in processed.items()}

    monkeypatch.setattr("yg_eo_soilnet.data_manager.ModelConfigFactory", FakeFactory)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: Path(path).write_text("frame"))
    monkeypatch.setattr(pd.Series, "to_parquet", lambda self, path, index=False: Path(path).write_text("series"), raising=False)
    monkeypatch.setattr("yg_eo_soilnet.data_manager.mlflow.log_artifacts", lambda *args, **kwargs: None)

    split = manager.split_data(processed)

    assert "groups_train" in split
    assert "groups_test" in split
    assert set(split["groups_train"].unique()).issubset({1, 2})