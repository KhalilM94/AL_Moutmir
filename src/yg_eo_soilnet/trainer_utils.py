from yg_eo_soilnet.utils import LogTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional

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

    def create_splits(self, X, y=None, groups=None):
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
                raise ValueError("Groups must be provided for GroupKFold CV strategy.")
            cv = GroupKFold(n_splits=self.n_splits)
            splits = list(cv.split(X, y, groups))
        else:
            raise ValueError(f"Unsupported CV strategy: {self.cv_strategy}")

        return splits

class TargetNanFilter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, groups=None):
        if y is None:
            return X
        mask = pd.notna(y)
        X_clean = X.loc[mask]
        y_clean = y.loc[mask]
        groups_clean = groups.loc[mask] if groups is not None else None
        return X_clean, y_clean, groups_clean

@dataclass
class PipelineBuilder:
    seed: int = 42

    def _is_tree_based_model(self, model: BaseEstimator) -> bool:
        model_name = model.__class__.__name__.lower()
        model_module = model.__class__.__module__.lower()
        tree_markers = ("tree", "forest", "boost", "xgb", "lightgbm", "catboost")
        return any(marker in model_name or marker in model_module for marker in tree_markers)

    def _build_preprocessor(self, model: BaseEstimator, categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
        is_tree_model = self._is_tree_based_model(model)

        numeric_steps: List[Tuple[str, BaseEstimator]] = [
            ('imputer', SimpleImputer(strategy='median'))
        ]
        if not is_tree_model:
            numeric_steps.append(('scaler', RobustScaler()))

        if is_tree_model:
            categorical_encoder: BaseEstimator = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1,
                encoded_missing_value=-1
            )
        else:
            categorical_encoder = OneHotEncoder(handle_unknown='ignore')

        transformers: List[Tuple[str, BaseEstimator, List[str]]] = []
        if numeric_cols:
            transformers.append(('num', Pipeline(numeric_steps), numeric_cols))
        if categorical_cols:
            transformers.append((
                'cat',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', categorical_encoder)
                ]),
                categorical_cols
            ))

        return ColumnTransformer(transformers=transformers, remainder='drop')

    def build(
        self,
        model,
        is_log_target: bool = False,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
    ) -> Pipeline:
        categorical_cols = categorical_cols or []
        numeric_cols = numeric_cols or []
        preprocessor = self._build_preprocessor(model, categorical_cols, numeric_cols)

        steps: List[Tuple[str, BaseEstimator]] = [('preprocessor', preprocessor)]
        if is_log_target:
            steps.append((
                'model', 
                TransformedTargetRegressor(
                    regressor=model, 
                    func=LogTransformer().transform, 
                    inverse_func=LogTransformer().inverse_transform,
                    check_inverse=False
                    )
                ))
        else:
            steps.append(('model', model))
        return Pipeline(steps)