from .model_config_factory import ModelConfigFactory, BaseSpatialClusterStrategy
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import pandas as pd
import mlflow
import os
import tempfile
from typing import Dict

class DataManager:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def load_data(self) -> pd.DataFrame:
        """Load and return the initial dataset from the CSV file specified in DATA_FOLDER and DATA_FILE."""
        data_path = os.path.join(self.config.DATA_FOLDER, self.config.DATA_FILE)
        self.logger.info(f"Loading data from file: {data_path}")
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"CSV file not found: {data_path}")
        return pd.read_csv(data_path)
        
    def preprocess_data(self, data: pd.DataFrame) -> Dict:
        
        # Get feature columns
        feature_columns = [col for col in data.columns 
                          if col not in self.config.TARGET_COLUMNS + 
                          self.config.ELIMINATED_FEATURES + 
                          self.config.EXCLUDE_CATEGORICAL +
                          self.config.IGNORE_BANDS]
        
        # Prepare feature matrix
        valid_feature_columns = data[feature_columns].dropna(axis=1, how='all').columns.tolist()
        X = data[valid_feature_columns]

        categorical_cols = [
            col for col in self.config.CATEGORICAL_FEATURES
            if col in X.columns and col not in self.config.EXCLUDE_CATEGORICAL
        ]
        if categorical_cols:
            self.logger.info(
                "Categorical encoding will be applied per model in the training pipeline "
                "(one-hot for linear models, ordinal encoding for tree-based models)."
            )

        data_cleaned = data.loc[X.index]

        preprocessed =  {
            'X' : X,
            'y': data_cleaned[self.config.TARGET_COLUMNS],
            'lat': data_cleaned['lat'],
            'lon': data_cleaned['lon']
        }
        return preprocessed
        
    def split_data(self, processed_data: Dict) -> Dict:
        X: pd.DataFrame = processed_data['X']
        y: pd.DataFrame = processed_data['y']
        lat = processed_data['lat']
        lon = processed_data['lon']
        splitted_data = {}

        X_train = X_test = y_train = y_test = pd.DataFrame()
        lat_train = lat_test = lon_train = lon_test = pd.Series()
        
        if self.config.ENABLE_CLUSTERING:
            """Preprocess the data and return prepared datasets."""
            self.logger.info(f"Clustering dataset based on {self.config.CLUSTERING_STRATEGY.get('class_path').rsplit('.', 1)[1]}...")

            self.cluster_strategy = ModelConfigFactory(self.config.CLUSTERING_STRATEGY, self.config.RANDOM_SEED).load_splitter_from_config()
            if isinstance(self.cluster_strategy, BaseSpatialClusterStrategy):
                self.cluster_strategy
                clustered_data = self.cluster_strategy.cluster(X)
                self.logger.info(f"Cluster value counts:\n{clustered_data['cluster'].value_counts().to_string()}")

                groups = clustered_data['cluster']
                """Split data into training and test sets."""
                self.logger.info("Splitting dataset into train and test groups using GroupShuffleSplit...")
                gss = GroupShuffleSplit(n_splits=1, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_SEED)
                train_idx, test_idx = next(gss.split(X, y, groups=groups))

                self.cluster_strategy.plot_train_test(
                    clustered_data,
                    train_idx,
                    test_idx,
                    title="Spatial Grid Train/Test Split",
                    filename="grid_split.png")

                X.drop(['lat', 'lon'], axis=1, inplace=True)

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                lat_train, lat_test = lat.iloc[train_idx], lat.iloc[test_idx]
                lon_train, lon_test = lon.iloc[train_idx], lon.iloc[test_idx]
                groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

                splitted_data['groups_train'] = groups_train
                splitted_data['groups_test'] = groups_test
                self.logger.info(f"Train group distribution:\n{groups_train.value_counts().sort_index().to_string()}")
                self.logger.info(f"Test group distribution:\n{groups_test.value_counts().sort_index().to_string()}")

        else:
            X = X.drop(['lat', 'lon'], axis=1)
            self.logger.info("Splitting dataset using simple train-test split...")
            X_train, X_test, y_train, y_test, lat_train, lat_test, lon_train, lon_test = train_test_split(
                X, y, lat, lon, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_SEED
            )
        
            self.logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

        with tempfile.TemporaryDirectory() as tmpdir:
            X_train.to_parquet(os.path.join(tmpdir, "X_train.parquet"), index=False)
            X_test.to_parquet(os.path.join(tmpdir, "X_test.parquet"), index=False)
            y_train.to_parquet(os.path.join(tmpdir, "y_train.parquet"), index=False)
            y_test.to_parquet(os.path.join(tmpdir, "y_test.parquet"), index=False)

            # Log them as artifacts
            mlflow.log_artifacts(tmpdir, artifact_path="data_splits")
        
        splitted_data['X_train'] = X_train
        splitted_data['X_test'] = X_test
        splitted_data['y_train'] = y_train
        splitted_data['y_test'] = y_test
        splitted_data['lat_train'] = lat_train
        splitted_data['lat_test'] = lat_test
        splitted_data['lon_train'] = lon_train
        splitted_data['lon_test'] = lon_test
        
        return splitted_data

