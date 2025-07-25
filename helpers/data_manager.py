from .model_config_factory import ModelConfigFactory, BaseSpatialClusterStrategy
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import os
from typing import Dict

class DataManager:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def load_data(self) -> pd.DataFrame:
        """Load and return the initial dataset from the CSV file specified in DATA_FOLDER and DATA_FILE_NAME."""
        data_path = os.path.join(self.config.DATA_FOLDER, self.config.DATA_FILE_NAME)
        self.logger.info(f"Loading data from file: {data_path}")
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"CSV file not found: {data_path}")
        return pd.read_csv(data_path)
        
    def preprocess_data(self, data: pd.DataFrame) -> Dict:
        """Preprocess the data and return prepared datasets."""
        self.logger.info(f"Clustering dataset based on {self.config.CLUSTERING_STRATEGY.get('class_path').rsplit('.', 1)[1]}...")
        
        self.cluster_strategy = ModelConfigFactory(self.config.CLUSTERING_STRATEGY, self.config.RANDOM_SEED).load_splitter_from_config()
        if isinstance(self.cluster_strategy, BaseSpatialClusterStrategy):
            self.cluster_strategy
            clustered_data = self.cluster_strategy.cluster(data)
            self.logger.info(f"Cluster value counts:\n{clustered_data['cluster'].value_counts().to_string()}")
        else:
             clustered_data = data
        
        # Get feature columns
        feature_columns = [col for col in clustered_data.columns 
                          if col not in self.config.TARGET_COLUMNS + 
                          self.config.ELIMINATED_FEATURES + 
                          self.config.EXCLUDE_CATEGORICAL +
                          self.config.IGNORE_BANDS]
        
        # Prepare feature matrix
        valid_feature_columns = clustered_data[feature_columns].dropna(axis=1, how='all').columns.tolist()
        X = clustered_data[valid_feature_columns]
        
        # One-hot encoding if needed
        categorical_cols = [col for col in self.config.CATEGORICAL_FEATURES 
                            if col in X.columns and col not in self.config.EXCLUDE_CATEGORICAL]
        if categorical_cols:
            self.logger.info(f"Applying one-hot encoding to categorical columns: {', '.join(categorical_cols)}")
            X = pd.get_dummies(X, columns=categorical_cols)
            
        # Fill remaining NaNs with column means
        X = X.apply(lambda row: row.fillna(row.mean()), axis=1)
        clustered_cleaned = clustered_data.loc[X.index]

        preprocessed =  {
            'X' : X,
            'y': clustered_cleaned[self.config.TARGET_COLUMNS],
            'Latitude_Y': clustered_data['Latitude_Y'],
            'Longitude_X': clustered_data['Longitude_X']
        }
        if self.config.ENABLE_CLUSTERING:
            preprocessed['groups'] = clustered_cleaned['cluster']
        return preprocessed
        
    def split_data(self, processed_data: Dict) -> Dict:
        X: pd.DataFrame = processed_data['X']
        y: pd.DataFrame = processed_data['y']
        lat = processed_data['Latitude_Y']
        lon = processed_data['Longitude_X']
        splitted_data = {}

        if self.config.ENABLE_CLUSTERING and not (self.config.CV_ONLY_MODE.get("enabled", True) is True):
            groups = processed_data['groups']
            """Split data into training and test sets."""
            self.logger.info("Splitting dataset into train and test groups using GroupShuffleSplit...")
            gss = GroupShuffleSplit(n_splits=1, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_SEED)
            train_idx, test_idx = next(gss.split(X, y, groups=groups))
            
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
            self.logger.info("Splitting dataset using simple train-test split...")
            X_train, X_test, y_train, y_test, lat_train, lat_test, lon_train, lon_test = train_test_split(
                X, y, lat, lon, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_SEED
            )
        
        self.logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
        
        splitted_data['X_train'] = X_train
        splitted_data['X_test'] = X_test
        splitted_data['y_train'] = y_train
        splitted_data['y_test'] = y_test
        splitted_data['lat_train'] = lat_train
        splitted_data['lat_test'] = lat_test
        splitted_data['lon_train'] = lon_train
        splitted_data['lon_test'] = lon_test
        
        return splitted_data

