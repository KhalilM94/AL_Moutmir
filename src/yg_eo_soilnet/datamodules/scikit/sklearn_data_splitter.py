from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Callable, Dict

import mlflow
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from yg_eo_soilnet.clustering_utils import BaseSpatialClusterStrategy


class SklearnDataSplitter:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def split_data(
        self,
        processed_data: Dict[str, Any],
        *,
        sanitize_features: Callable[[pd.DataFrame, list[str]], pd.DataFrame] | None = None,
        model_config_factory: Any = None,
    ) -> Dict[str, Any]:
        X: pd.DataFrame = processed_data["X"]
        y: pd.DataFrame = processed_data["y"]
        lat = processed_data["lat"]
        lon = processed_data["lon"]
        target_columns = list(getattr(self.config, "TARGET_COLUMNS", []))

        if sanitize_features is not None:
            X = sanitize_features(X, target_columns)

        split_data = {}

        X_train = X_test = y_train = y_test = pd.DataFrame()
        lat_train = lat_test = lon_train = lon_test = pd.Series()

        if self.config.ENABLE_CLUSTERING:
            self.logger.info(
                f"Clustering dataset based on {self.config.CLUSTERING_STRATEGY.get('class_path').rsplit('.', 1)[1]}..."
            )

            if model_config_factory is None:
                from yg_eo_soilnet.models import ModelConfigFactory  # local import avoids a circular dependency

                model_config_factory = ModelConfigFactory
            factory = model_config_factory
            self.cluster_strategy = factory(
                self.config.CLUSTERING_STRATEGY,
                self.config.RANDOM_SEED,
            ).load_splitter_from_config()
            if not isinstance(self.cluster_strategy, BaseSpatialClusterStrategy):
                # Falling through here used to return empty train/test frames and log four empty
                # parquet artifacts, surfacing much later as KeyError('groups_train').
                raise TypeError(
                    "ENABLE_CLUSTERING is set but CLUSTERING_STRATEGY did not resolve to a "
                    f"BaseSpatialClusterStrategy (got {type(self.cluster_strategy).__name__}). "
                    f"Check class_path and 'enabled' in {self.config.CLUSTERING_STRATEGY!r}."
                )

            lat_col = getattr(self.config, "LAT_COLUMN", "lat")
            lon_col = getattr(self.config, "LON_COLUMN", "lon")
            clustering_frame = X.copy()
            clustering_frame[lat_col] = lat.to_numpy()
            clustering_frame[lon_col] = lon.to_numpy()
            if lat_col != "lat" and "lat" not in clustering_frame.columns:
                clustering_frame["lat"] = lat.to_numpy()
            if lon_col != "lon" and "lon" not in clustering_frame.columns:
                clustering_frame["lon"] = lon.to_numpy()

            clustered_data = self.cluster_strategy.cluster(clustering_frame)
            self.logger.info(f"Cluster value counts:\n{clustered_data['cluster'].value_counts().to_string()}")

            groups = clustered_data["cluster"]
            self.logger.info("Splitting dataset into train and test groups using GroupShuffleSplit...")
            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=self.config.TEST_SIZE,
                random_state=self.config.RANDOM_SEED,
            )
            train_idx, test_idx = next(gss.split(X, y, groups=groups))

            self.cluster_strategy.plot_train_test(
                clustered_data,
                train_idx,
                test_idx,
                title="Spatial Grid Train/Test Split",
                filename="grid_split.png",
            )

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            lat_train, lat_test = lat.iloc[train_idx], lat.iloc[test_idx]
            lon_train, lon_test = lon.iloc[train_idx], lon.iloc[test_idx]
            groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

            split_data["groups_train"] = groups_train
            split_data["groups_test"] = groups_test
            self.logger.info(f"Train group distribution:\n{groups_train.value_counts().sort_index().to_string()}")
            self.logger.info(f"Test group distribution:\n{groups_test.value_counts().sort_index().to_string()}")

        else:
            self.logger.info("Splitting dataset using simple train-test split...")
            X_train, X_test, y_train, y_test, lat_train, lat_test, lon_train, lon_test = train_test_split(
                X,
                y,
                lat,
                lon,
                test_size=self.config.TEST_SIZE,
                random_state=self.config.RANDOM_SEED,
            )
            self.logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

        with tempfile.TemporaryDirectory() as tmpdir:
            split_start = time.perf_counter()
            X_train.to_parquet(os.path.join(tmpdir, "X_train.parquet"), index=False)
            X_test.to_parquet(os.path.join(tmpdir, "X_test.parquet"), index=False)
            y_train.to_parquet(os.path.join(tmpdir, "y_train.parquet"), index=False)
            y_test.to_parquet(os.path.join(tmpdir, "y_test.parquet"), index=False)
            self.logger.info(f"split_data parquet writes completed in {time.perf_counter() - split_start:.2f}s")

            artifact_start = time.perf_counter()
            mlflow.log_artifacts(tmpdir, artifact_path="data_splits")
            self.logger.info(f"split_data mlflow.log_artifacts completed in {time.perf_counter() - artifact_start:.2f}s")

        split_data["X_train"] = X_train
        split_data["X_test"] = X_test
        split_data["y_train"] = y_train
        split_data["y_test"] = y_test
        split_data["lat_train"] = lat_train
        split_data["lat_test"] = lat_test
        split_data["lon_train"] = lon_train
        split_data["lon_test"] = lon_test

        return split_data
