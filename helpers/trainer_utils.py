from.misc_utils import LogTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, GroupKFold

import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

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

        # Prepare fold assignment series with -1 default (for samples not assigned)
        indices = X.index if hasattr(X, 'index') else range(len(X))

        fold_assignments = pd.Series(data=-1, index=indices)

        for fold_number, (_, test_idx) in enumerate(splits):
            fold_assignments.iloc[test_idx] = fold_number

        fold_info_df = pd.DataFrame({'index': fold_assignments.index, 'fold': fold_assignments.values})

        return splits, fold_info_df

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

    def build(self, model, is_log_target: bool = False) -> Pipeline:
        steps: List[Tuple[str, BaseEstimator]] = [('scaler', RobustScaler())]
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