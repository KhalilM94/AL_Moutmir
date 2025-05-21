from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, GroupKFold, StratifiedShuffleSplit
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

class LogTransformer(BaseEstimator, TransformerMixin):
    def transform(self, y):
        return 10 * np.log1p(y)

    def inverse_transform(self, y):
        return np.expm1(y / 10)

class SpatialClusterSplitter:
    """
    Clusters geographic points into spatial groups using KMeans.

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

    Methods:
    --------
    cluster(df: pd.DataFrame) -> pd.DataFrame
        Adds a 'cluster' column to the DataFrame with cluster labels (1-indexed).
        Rows with missing coordinates are dropped from the result.
    """
    def __init__(self, 
                 n_clusters=12, 
                 lat_col='Latitude_Y', 
                 lon_col='Longitude_X', 
                 random_state=42):
        self.n_clusters = n_clusters
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.random_state = random_state

    def cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        coords = df[[self.lat_col, self.lon_col]].dropna()
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        labels = kmeans.fit_predict(coords) + 1

        df = df.copy()
        df.loc[coords.index, 'cluster'] = labels.astype(int)
        return df.dropna(subset=['cluster'])


class CVSplitter:
    """
    A class to create cross-validation splits compatible with ModelTrainer.
    Supports 'kfold', 'groupkfold', and 'stratifiedshuffle' strategies.
    """

    def __init__(self, cv_strategy='kfold', n_splits=5, random_state=42, shuffle=True):
        self.cv_strategy = cv_strategy.lower()
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle

    def create_splits(self, X, y=None, groups=None, target=None, model_name=None):
        """
        Generate CV splits and fold info DataFrame.

        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix.
            y (pd.Series or np.ndarray, optional): Target vector.
            groups (pd.Series or np.ndarray, optional): Group labels for GroupKFold.
            target (str, optional): Target name, for logging (not used here).
            model_name (str, optional): Model name, for logging (not used here).

        Returns:
            splits (list of (train_idx, test_idx) tuples)
            fold_info_df (pd.DataFrame): DataFrame with columns ['index', 'fold']
        """
        if self.cv_strategy == 'kfold':
            cv = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            splits = list(cv.split(X))
        elif self.cv_strategy == 'groupkfold':
            if groups is None:
                raise ValueError("Groups must be provided for groupkfold CV strategy.")
            cv = GroupKFold(n_splits=self.n_splits)
            splits = list(cv.split(X, y, groups))
        elif self.cv_strategy == 'stratifiedshuffle':
            if y is None:
                raise ValueError("Target y must be provided for stratifiedshuffle CV strategy.")
            cv = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.2, random_state=self.random_state)
            splits = list(cv.split(X, y))
        else:
            raise ValueError(f"Unsupported CV strategy: {self.cv_strategy}")

        # Prepare fold assignment series with -1 default (for samples not assigned)
        if hasattr(X, 'index'):
            indices = X.index
        else:
            indices = range(len(X))
        fold_assignments = pd.Series(data=-1, index=indices)

        for fold_number, (_, test_idx) in enumerate(splits):
            fold_assignments.iloc[test_idx] = fold_number

        fold_info_df = pd.DataFrame({'index': fold_assignments.index, 'fold': fold_assignments.values})

        return splits, fold_info_df

