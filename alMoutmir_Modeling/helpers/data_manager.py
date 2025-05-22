import pandas as pd
from typing import Dict
from helpers.utils import SpatialClusterSplitter
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import os

class DataManager:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.cluster_splitter = SpatialClusterSplitter(random_state=config.RANDOM_SEED)

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
        clustered_data = self.cluster_splitter.cluster(data)
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
        if self.config.USE_GROUP_SPLIT:
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