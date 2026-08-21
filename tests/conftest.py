import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture(autouse=True)
def isolated_mlflow_tracking(tmp_path_factory):
    """Give every test its own throwaway tracking store, and let none inherit an open run.

    Two problems this removes. Tests that touch MLflow incidentally used to write into the repo's
    real ``mlruns/`` - and since the trainers open their own runs rather than the logger doing it,
    that is now most of them. And a test that left a run active made the NEXT test fail with
    "Run with UUID ... is already active", a failure that moves around as the suite is reordered
    and blames the wrong test.

    Tests that set their own tracking URI still win: this runs first and they override it.
    """
    import mlflow

    previous_uri = mlflow.get_tracking_uri()
    previous_allow = os.environ.get("MLFLOW_ALLOW_FILE_STORE")
    # MLflow 3.14 refuses a filesystem backend without this; see yg_eo_soilnet.tracking.
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(tmp_path_factory.mktemp("mlruns").as_uri())
    # A bare directory has no experiment 0, and MLflow does not create the default one lazily, so
    # the first start_run would fail with "Could not find experiment with ID 0".
    mlflow.set_experiment("pytest")
    try:
        yield
    finally:
        while mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous_uri)
        if previous_allow is None:
            os.environ.pop("MLFLOW_ALLOW_FILE_STORE", None)
        else:
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = previous_allow


@pytest.fixture
def toy_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lat": [0.0, 0.01, 0.02, 0.03],
            "lon": [0.0, 0.0, 0.0, 0.0],
            "target_a": [1.0, 2.0, 3.0, 4.0],
            "target_b": [10.0, 11.0, 12.0, 13.0],
            "cat": ["a", "b", "a", "b"],
            "feature": [100.0, 101.0, 102.0, 103.0],
            "all_null": [None, None, None, None],
        }
    )


@pytest.fixture
def toy_config() -> SimpleNamespace:
    return SimpleNamespace(
        DATA_FOLDER="/tmp",
        DATA_FILE="data.csv",
        STATIC_FEATURES_FILE="data.csv",
        TARGETS_FILE="data.csv",
        STATIC_FEATURES_FOLDER=None,
        TARGETS_FOLDER=None,
        TIMESERIES_FOLDER=None,
        TIMESERIES_CSV_PATH=None,
        STATIC_SOURCE=None,
        TARGETS_SOURCE=None,
        TIMESERIES_SOURCE=None,
        TEMPORAL_FEATURES={},
        TEMPORAL_FEATURES_ENABLED=False,
        DATA_INDEX_MANIFEST_PATH=None,
        RANDOM_SEED=42,
        TEST_SIZE=0.25,
        POINT_ID_COLUMN="point_id",
        ENABLE_CLUSTERING=False,
        CLUSTERING_STRATEGY={"enabled": False, "class_path": "yg_eo_soilnet.models.KMeansClusterStrategy", "params": {}},
        # The INNER cross-validation strategy, not the holdout.
        SPLIT_STRATEGY="kfold",
        # The shared holdout, decided once for every training family. See datamodules/splitting.py.
        SPLIT_HOLDOUT_STRATEGY="random",
        SPLIT_TEST_SIZE=0.25,
        SPLIT_VAL_SIZE=0.25,
        SPLIT_SEED=42,
        SPLIT_POPULATION_POLICY="intersect",
        SPLIT_PLAN_PATH=None,
        SPLIT_MIN_POPULATION_RATIO=0.5,
        SPLIT_GROUP_STRATEGY={},
        LAT_COLUMN="lat",
        LON_COLUMN="lon",
        IGNORE_BANDS=[],
        EXISTING_HS_FEATURES={"enabled": False, "ignore": False, "prefix": "S2_", "band_count": 6, "band_names": []},
        COLUMNS_TO_TRANSFORM=["target_a"],
        TARGET_COLUMNS=["target_a", "target_b"],
        CATEGORICAL_FEATURES=["cat"],
        EXCLUDE_CATEGORICAL=[],
        ELIMINATED_FEATURES=[],
        MODEL_REGISTRY={},
        LIGHTNING_MODEL_REGISTRY={},
    )


@pytest.fixture
def split_plan_for(toy_config, logger):
    """Build the shared split plan over a toy frame, the way the provider does for a real run.

    Tests that hand a preprocessed dict straight to the splitter cannot use SplitPlanProvider,
    which loads the dataset from disk - but they still have to supply a plan, because deciding the
    split is no longer the sklearn family's job.
    """
    from yg_eo_soilnet.datamodules.splitting import UnifiedSplitter

    def build(frame, **overrides):
        for key, value in overrides.items():
            setattr(toy_config, key, value)
        point_col = toy_config.POINT_ID_COLUMN
        ids = frame[point_col] if point_col in frame.columns else pd.Series(range(len(frame)))
        coordinates = pd.DataFrame(
            {
                toy_config.LAT_COLUMN: frame[toy_config.LAT_COLUMN].to_numpy(),
                toy_config.LON_COLUMN: frame[toy_config.LON_COLUMN].to_numpy(),
            },
            index=pd.Index(ids.to_numpy()),
        )
        return UnifiedSplitter(toy_config, logger).build_plan(
            pd.Index(ids.to_numpy()), coordinates=coordinates
        )

    return build


@pytest.fixture
def logger() -> logging.Logger:
    logger = logging.getLogger("yg-eo-soilnet-tests")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(logging.NullHandler())
    return logger