"""Preparation that used to live on DataManager now belongs to the scikit datamodule."""

from pathlib import Path

import pandas as pd
import pytest

from yg_eo_soilnet.clustering_utils import BaseSpatialClusterStrategy
from yg_eo_soilnet.data_manager import DataManager
from yg_eo_soilnet.datamodules.scikit.scikit_datamodule import ScikitDataModule


SPLITTER_MODULE = "yg_eo_soilnet.datamodules.scikit.sklearn_data_splitter"


@pytest.fixture
def datamodule(toy_config, logger) -> ScikitDataModule:
    return ScikitDataModule(toy_config, logger, DataManager(toy_config, logger))


def _stub_artifact_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: Path(path).write_text("frame"))
    monkeypatch.setattr(pd.Series, "to_parquet", lambda self, path, index=False: Path(path).write_text("series"), raising=False)
    monkeypatch.setattr(f"{SPLITTER_MODULE}.mlflow.log_artifacts", lambda *args, **kwargs: None)


def test_preprocess_returns_expected_structure(datamodule, toy_dataframe, caplog) -> None:
    with caplog.at_level("INFO"):
        processed = datamodule.preprocess(toy_dataframe)

    assert list(processed["X"].columns) == ["cat", "feature"]
    assert list(processed["y"].columns) == ["target_a", "target_b"]
    assert list(processed["lat"].index) == [0, 1, 2, 3]
    assert "Categorical encoding will be applied" in caplog.text


def test_preprocess_excludes_metadata_columns_even_if_present(datamodule, toy_dataframe) -> None:
    data = toy_dataframe.copy()
    data["point_id"] = [1, 2, 3, 4]
    data["geometry"] = ["g1", "g2", "g3", "g4"]

    processed = datamodule.preprocess(data)

    assert not {"point_id", "lat", "lon", "geometry"} & set(processed["X"].columns)


def test_preprocess_warns_when_too_many_features_are_dropped(toy_config, logger, caplog) -> None:
    toy_config.MIN_FEATURE_COUNT = 3
    toy_config.MAX_FEATURE_DROP_RATIO_WARNING = 0.2
    toy_config.EXISTING_HS_FEATURES = {"enabled": True, "ignore": True, "prefix": "S2_", "band_count": 1, "band_names": []}

    data = pd.DataFrame(
        {
            "lat": [0.0, 0.01, 0.02],
            "lon": [0.0, 0.0, 0.0],
            "target_a": [1.0, 2.0, 3.0],
            "target_b": [10.0, 11.0, 12.0],
            "cat": ["a", "b", "a"],
            "feature_keep": [100.0, 101.0, 102.0],
            "feature_drop": [None, None, None],
            "S2_B11": [0.1, 0.2, 0.3],
        }
    )

    module = ScikitDataModule(toy_config, logger, DataManager(toy_config, logger))
    with caplog.at_level("WARNING"):
        processed = module.preprocess(data)

    assert "usable features remain" in caplog.text
    assert "Dropped feature ratio" in caplog.text
    assert list(processed["X"].columns) == ["cat", "feature_keep"]


def test_preprocess_drops_existing_hyperspectral_columns_by_prefix(toy_config, logger) -> None:
    toy_config.EXISTING_HS_FEATURES = {
        "enabled": True,
        "ignore": True,
        "prefix": "S2_",
        "band_count": 2,
        "band_names": [],
    }

    data = pd.DataFrame(
        {
            "lat": [0.0, 0.01, 0.02],
            "lon": [0.0, 0.0, 0.0],
            "target_a": [1.0, 2.0, 3.0],
            "target_b": [10.0, 11.0, 12.0],
            "cat": ["a", "b", "a"],
            "feature": [100.0, 101.0, 102.0],
            "S2_B11": [0.1, 0.2, 0.3],
            "S2_B12": [0.4, 0.5, 0.6],
        }
    )

    module = ScikitDataModule(toy_config, logger, DataManager(toy_config, logger))
    processed = module.preprocess(data)

    assert list(processed["X"].columns) == ["cat", "feature"]


def test_split_selects_this_family_rows_from_the_shared_plan(
    monkeypatch: pytest.MonkeyPatch, datamodule, toy_dataframe, split_plan_for
) -> None:
    """The sklearn family no longer decides the split; it selects its rows out of the shared one."""
    processed = datamodule.preprocess(toy_dataframe)
    _stub_artifact_logging(monkeypatch)
    plan = split_plan_for(toy_dataframe)

    split = datamodule.split(processed, plan)

    point_ids = processed["point_ids"].to_numpy()
    expected_test = set(plan.point_ids_for("test"))
    assert set(point_ids[split["X_test"].index]) == expected_test
    # X_train is the FIT POOL: train ∪ val, because GridSearchCV k-folds inside it.
    assert split["X_train"].shape[0] == len(plan.point_ids_for("train")) + len(plan.point_ids_for("val"))
    assert split["X_train_only"].shape[0] == len(plan.point_ids_for("train"))
    assert split["X_val"].shape[0] == len(plan.point_ids_for("val"))
    assert "groups_train" not in split


def test_split_without_a_plan_refuses_rather_than_inventing_one(
    monkeypatch: pytest.MonkeyPatch, datamodule, toy_dataframe
) -> None:
    """A private split is exactly the bug the shared plan exists to prevent."""
    processed = datamodule.preprocess(toy_dataframe)
    _stub_artifact_logging(monkeypatch)

    with pytest.raises(ValueError, match="needs the run's split_plan"):
        datamodule.splitter.split_data(processed, split_plan=None)


def test_split_with_clustering(
    monkeypatch: pytest.MonkeyPatch, toy_config, toy_dataframe, logger, split_plan_for
) -> None:
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

    toy_config.SPLIT_HOLDOUT_STRATEGY = "spatial_group"
    toy_config.SPLIT_GROUP_STRATEGY = {
        "enabled": True,
        "class_path": "yg_eo_soilnet.models.KMeansClusterStrategy",
        "params": {"n_clusters": 2},
    }

    module = ScikitDataModule(toy_config, logger, DataManager(toy_config, logger))
    processed = module.preprocess(toy_dataframe)
    processed = {key: value.copy() if hasattr(value, "copy") else value for key, value in processed.items()}

    monkeypatch.setattr("yg_eo_soilnet.models.ModelConfigFactory", FakeFactory)
    _stub_artifact_logging(monkeypatch)

    plan = split_plan_for(toy_dataframe)
    split = module.split(processed, plan)

    # The clusters the holdout was blocked on travel with the split, so GroupKFold inside the fit
    # pool respects the same spatial structure the holdout did.
    assert "groups_train" in split
    assert "groups_test" in split
    assert set(split["groups_train"].unique()).issubset({1, 2})
    # No cluster may straddle two splits - that is the whole point of a grouped holdout.
    frame = plan.to_frame()
    assert frame.groupby("cluster")["split"].nunique().max() == 1


def test_split_sanitizes_features_with_the_full_schema_filter(monkeypatch: pytest.MonkeyPatch, toy_config, logger) -> None:
    """The splitter must receive DataManager.filter_schema, not a metadata-only variant."""
    toy_config.ELIMINATED_FEATURES = ["eliminated"]

    captured = {}

    def fake_split_data(processed_data, *, sanitize_features=None, model_config_factory=None, split_plan=None):
        captured["sanitize_features"] = sanitize_features
        return {}

    module = ScikitDataModule(toy_config, logger, DataManager(toy_config, logger))
    monkeypatch.setattr(module.splitter, "split_data", fake_split_data)

    module.split(
        {"X": pd.DataFrame(), "y": pd.DataFrame(), "lat": pd.Series(), "lon": pd.Series()},
        object(),
    )

    frame = pd.DataFrame({"eliminated": [1], "keep_me": [2]})
    assert list(captured["sanitize_features"](frame).columns) == ["keep_me"]


def test_load_frame_joins_targets_when_static_frame_lacks_them(tmp_path: Path, toy_config, logger) -> None:
    pd.DataFrame({"point_id": [1], "lat": [0.0], "lon": [0.0], "feature": [10.0]}).to_csv(
        tmp_path / "static.csv", index=False
    )
    pd.DataFrame({"point_id": [1], "target_a": [1.0], "target_b": [2.0]}).to_csv(
        tmp_path / "targets.csv", index=False
    )

    toy_config.DATA_FOLDER = str(tmp_path)
    toy_config.DATA_FILE = "static.csv"
    toy_config.TARGETS_FILE = "targets.csv"

    module = ScikitDataModule(toy_config, logger, DataManager(toy_config, logger))
    frame = module.load_frame()

    assert {"target_a", "target_b"} <= set(frame.columns)


def test_clustering_with_a_non_strategy_raises_instead_of_returning_empty_splits(
    monkeypatch: pytest.MonkeyPatch, toy_config, toy_dataframe, logger
) -> None:
    """Falling through used to yield empty frames and a KeyError('groups_train') much later.

    The guard moved with the strategy: grouping is now decided by the unified splitter, so this is
    where a mis-declared class_path has to be caught.
    """
    from yg_eo_soilnet.datamodules.splitting import UnifiedSplitter

    toy_config.SPLIT_HOLDOUT_STRATEGY = "spatial_group"
    toy_config.SPLIT_GROUP_STRATEGY = {"enabled": False, "class_path": "a.b.C", "params": {}}

    class NotAStrategy:
        def load_splitter_from_config(self):
            return None

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("yg_eo_soilnet.models.ModelConfigFactory", NotAStrategy)

    coordinates = pd.DataFrame(
        {"lat": toy_dataframe["lat"].to_numpy(), "lon": toy_dataframe["lon"].to_numpy()},
        index=pd.Index(range(len(toy_dataframe))),
    )
    with pytest.raises(TypeError, match="did not resolve to a BaseSpatialClusterStrategy"):
        UnifiedSplitter(toy_config, logger).build_plan(
            pd.Index(range(len(toy_dataframe))), coordinates=coordinates
        )
