from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFECV
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, GroupKFold, StratifiedShuffleSplit
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import (mean_squared_error, 
                             mean_absolute_error, 
                             r2_score, 
                             explained_variance_score)
from skopt import BayesSearchCV
import pandas as pd
import numpy as np
from dataclasses import dataclass
import os
import joblib
from typing import Union, List, Dict, Tuple
from pathlib import Path

class LogTransformer(BaseEstimator, TransformerMixin):
    def transform(self, y):
        return 10 * np.log1p(y)

    def inverse_transform(self, y):
        return np.expm1(y / 10)

@dataclass
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
    n_clusters: int = 12
    lat_col: str = 'Latitude_Y'
    lon_col: str = 'Longitude_X'
    random_state: int = 42

    def cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        coords = df[[self.lat_col, self.lon_col]].dropna()
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        labels = kmeans.fit_predict(coords) + 1

        df = df.copy()
        df.loc[coords.index, 'cluster'] = labels.astype(int)
        return df.dropna(subset=['cluster'])

@dataclass
class CVSplitter:
    """
    A class to create cross-validation splits compatible with ModelTrainer.
    Supports 'kfold', 'groupkfold', and 'stratifiedshuffle' strategies.
    """

    cv_strategy: str = 'kfold'
    n_splits: int = 5
    random_state: int = 42
    shuffle: bool = True

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
        strategy = self.cv_strategy.lower()
        if strategy == 'kfold':
            cv = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            splits = list(cv.split(X))
        elif strategy == 'groupkfold':
            if groups is None:
                raise ValueError("Groups must be provided for groupkfold CV strategy.")
            cv = GroupKFold(n_splits=self.n_splits)
            splits = list(cv.split(X, y, groups))
        elif strategy == 'stratifiedshuffle':
            if y is None:
                raise ValueError("Target y must be provided for stratifiedshuffle CV strategy.")
            cv = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.2, random_state=self.random_state)
            splits = list(cv.split(X, y))
        else:
            raise ValueError(f"Unsupported CV strategy: {self.cv_strategy}")

        # Prepare fold assignment series with -1 default (for samples not assigned)
        indices = X.index if hasattr(X, 'index') else range(len(X))

        fold_assignments = pd.Series(data=-1, index=indices)

        for fold_number, (_, test_idx) in enumerate(splits):
            fold_assignments.iloc[test_idx] = fold_number

        fold_info_df = pd.DataFrame({'index': fold_assignments.index, 'fold': fold_assignments.values})

        return splits, fold_info_df

@dataclass
class PipelineBuilder:
    enable_rfe: bool
    seed: int = 42

    def is_rfe_compatible(self, model) -> bool:
        return not (
            isinstance(model, (PLSRegression, GradientBoostingRegressor)) or
            (isinstance(model, SVR) and getattr(model, 'kernel', None) != "linear")
        )

    def build(self, model, use_rfe: bool) -> Pipeline:
        steps: List[Tuple[str, BaseEstimator]] = [('scaler', RobustScaler())]
        if self.enable_rfe and use_rfe:
            rfecv = RFECV(estimator=clone(model), step=1, cv=5,
                scoring='neg_mean_squared_error')
            steps.append(('feature_selection', rfecv))
        steps.append(('model', model))
        return Pipeline(steps)

@dataclass
class Tuner:
    use_bayes: bool
    seed: int = 42
    verbose: int = 0
    enable_tuning: bool = True  # Add this

    def tune(self, 
             pipeline, 
             model_config, 
             X_train, 
             y_train, 
             splits) -> Tuple[BaseEstimator, Union[str, Dict]]:
        # Only tune if enabled and params exist
        if not self.enable_tuning or not model_config.get("params"):
            pipeline.fit(X_train, y_train)
            return pipeline, "Default (no tuning)"
        if self.use_bayes:
            search = BayesSearchCV(
                estimator=pipeline,
                search_spaces=model_config["params"],
                n_iter=30,
                cv=splits,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                random_state=self.seed,
                verbose=self.verbose
            )
        else:
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=model_config["params"],
                cv=splits,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                verbose=self.verbose
            )

        search.fit(X_train, y_train)
        return search.best_estimator_ # type: ignore[attr-defined]

class ModelEvaluator:
    def __init__(self, logger, columns_to_transform):
        self.logger = logger
        self.columns_to_transform = columns_to_transform

    def evaluate(self, model, X_train, y_train, X_test, y_test, 
                 target, model_name, best_params, log_transformer, splits) -> Dict:
        y_pred_transformed = model.predict(X_test)
        if target in self.columns_to_transform:
            y_pred = log_transformer.inverse_transform(y_pred_transformed)
        else:
            y_pred = y_pred_transformed

        cv_scores = cross_validate(model, X_train, y_train, cv=splits,
                                   scoring='neg_mean_squared_error')

        test_metrics = {
            "target": target,
            "model": model_name,
            "Test_RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "Test_MAE": mean_absolute_error(y_test, y_pred),
            "Test_R2": r2_score(y_test, y_pred),
            "CV_RMSE_Mean": np.mean(np.sqrt(-cv_scores["test_score"])),
            "CV_RMSE_Std": np.std(np.sqrt(-cv_scores["test_score"])),
            "Test_ExplainedVar": explained_variance_score(y_test, y_pred),
            "Best_Params": str(best_params)
        }

        self.logger.info(
            f"{model_name} | {target} — CV RMSE: {test_metrics['CV_RMSE_Mean']:.4f}, "
            f"Test RMSE: {test_metrics['Test_RMSE']:.4f}, R²: {test_metrics['Test_R2']:.4f}"
        )

        return test_metrics

@dataclass
class ModelSaver:
    output_dir: str

    def _safe_filename(self, name: str) -> str:
        return str(name).replace("/", "_").replace("\\", "_")

    def save_metrics(self, metrics: Dict, model_name: str, target: str):
        safe_target = self._safe_filename(target)
        metrics_path = Path(self.output_dir) / "metrics" / f"{safe_target}_{model_name}_metrics.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        return str(metrics_path)

    def save_model(self, model: BaseEstimator, model_name: str, target: str):
        safe_target = self._safe_filename(target)
        model_path = Path(self.output_dir) / "final_models" / f"{safe_target}_{model_name}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        return str(model_path)

    def save_cv_results(self, search_obj, model_name: str, target: str):
        if hasattr(search_obj, "cv_results_"):
            safe_target = self._safe_filename(target)
            results_df = pd.DataFrame(search_obj.cv_results_)
            results_df["target"] = target
            results_df["model"] = model_name
            path = Path(self.output_dir) / "metrics" / f"{safe_target}_{model_name}_cv_results.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(path, index=False)
            return str(path)
        return None
