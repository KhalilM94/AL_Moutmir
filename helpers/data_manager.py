from helpers.model_config_factory import ModelConfigFactory
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.cluster import KMeans
import os
from typing import Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod

class DataManager:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.cluster_strategy = ModelConfigFactory(config.CLUSTERING_STRATEGY, config.RANDOM_SEED).load_splitter_from_config()

    def load_data(self) -> pd.DataFrame:
        """Load and return the initial dataset."""
        csv_files = [f for f in os.listdir(self.config.DATA_FOLDER) if f.endswith('.csv')]
        self.logger.info(f"Found data files: {csv_files}")
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.config.DATA_FOLDER}")
            
        return pd.read_csv(os.path.join(self.config.DATA_FOLDER, csv_files[0]))
        
    def preprocess_data(self, data: pd.DataFrame) -> Dict:
        """Preprocess the data and return prepared datasets."""
        self.logger.info("Clustering dataset based on spatial location (KMeans)...")
        self.cluster_strategy = ModelConfigFactory(self.config.CLUSTERING_STRATEGY, self.config.RANDOM_SEED).load_splitter_from_config()
        clustered_data = self.cluster_strategy.cluster(data) if isinstance(self.cluster_strategy, BaseSpatialClusterStrategy) else data
        self.logger.info(f"Cluster value counts:\n{clustered_data['cluster'].value_counts().to_string()}")
        
        # Get feature columns
        feature_columns = [col for col in clustered_data.columns 
                          if col not in self.config.TARGET_COLUMNS + self.config.ELIMINATED_FEATURES]
        
        # Prepare feature matrix
        valid_feature_columns = clustered_data[feature_columns].dropna(axis=1, how='all').columns.tolist()
        X = clustered_data[valid_feature_columns]
        
        # One-hot encoding if needed
        if 'SU_WRB1_PH' in X.columns:
            self.logger.info("One-hot encoding 'SU_WRB1_PH' column...")
            X = pd.get_dummies(X, columns=['SU_WRB1_PH'])
            
        # Fill remaining NaNs with column means
        X = X.apply(lambda row: row.fillna(row.mean()), axis=1)
        clustered_cleaned = clustered_data.loc[X.index]
        
        return {
            'X': X,
            'y': clustered_cleaned[self.config.TARGET_COLUMNS],
            'groups': clustered_cleaned['cluster']
        }
        
    def split_data(self, X: pd.DataFrame, y: pd.DataFrame, groups: pd.Series) -> Dict:
        """Split data into training and test sets."""
        if self.config.ENABLE_CLUSTERING:
            self.logger.info("Splitting dataset into train and test groups using GroupShuffleSplit...")
            
            gss = GroupShuffleSplit(n_splits=1, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_SEED)
            train_idx, test_idx = next(gss.split(X, y, groups=groups))
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]
            
            self.logger.info(f"Train group distribution:\n{groups_train.value_counts().sort_index().to_string()}")
            self.logger.info(f"Test group distribution:\n{groups_test.value_counts().sort_index().to_string()}")
        else:
            self.logger.info("Splitting dataset using simple train-test split...")
            
            X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
                X, y, groups, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_SEED
            )
            
        self.logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'groups_train': groups_train,
            'groups_test': groups_test
        }

class BaseSpatialClusterStrategy(ABC):
    """
    Abstract base class for spatial clustering strategies.
    """

    @abstractmethod
    def cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clusters the input DataFrame spatially and adds a 'cluster' column.
        """
        pass

@dataclass
class KMeansClusterStrategy(BaseSpatialClusterStrategy):
    """
    KMeans-based spatial clustering.

    Parameters:
    -----------
    n_clusters : int, default=12
        Number of spatial clusters to form.
    lat_col : str, default='Latitude_Y'
        Name of the latitude column in the DataFrame.
    lon_col : str, default='Longitude_X'
        Name of the longitude column in the DataFrame.
    random_state : int, default=42
        Random seed for reproducibility of clustering.
    """
    n_clusters: int = 12
    lat_col: str = 'Latitude_Y'
    lon_col: str = 'Longitude_X'
    random_state: int = 42

    def cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        coords = df[[self.lat_col, self.lon_col]].dropna()
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        labels = kmeans.fit_predict(coords) + 1  # 1-indexed

        df = df.copy()
        df.loc[coords.index, 'cluster'] = labels.astype(int)
        return df.dropna(subset=['cluster'])

@dataclass
class GridClusterStrategy(BaseSpatialClusterStrategy):
    """
    Grid-based spatial clustering using a regular grid of fixed size (in degrees).
    """
    grid_size: float = 0.1  # e.g., 0.1 degrees
    lat_col: str = 'Latitude_Y'
    lon_col: str = 'Longitude_X'

    def cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        coords = df[[self.lat_col, self.lon_col]].dropna()

        lat_grid = (coords[self.lat_col] // self.grid_size).astype(int)
        lon_grid = (coords[self.lon_col] // self.grid_size).astype(int)

        # Create combined grid cell label and encode it as numeric cluster ID
        labels = (lat_grid.astype(str) + "_" + lon_grid.astype(str)).astype('category').cat.codes + 1

        df = df.copy()
        df.loc[coords.index, 'cluster'] = labels
        return df.dropna(subset=['cluster'])