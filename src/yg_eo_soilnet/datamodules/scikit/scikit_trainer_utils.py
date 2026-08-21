from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.model_selection import GroupKFold, KFold

from yg_eo_soilnet.utils import LogTransformer


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
        strategy = self.cv_strategy.lower()
        if strategy == 'kfold':
            cv = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            return list(cv.split(X))
        if strategy == 'groupkfold':
            if groups is None:
                raise ValueError("Groups must be provided for GroupKFold CV strategy.")
            cv = GroupKFold(n_splits=self.n_splits)
            return list(cv.split(X, y, groups))
        raise ValueError(f"Unsupported CV strategy: {self.cv_strategy}")


class TargetNanFilter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, groups=None):
        if y is None:
            return X
        # A DataFrame y is a joint fit over several targets. `pd.notna` on it is a 2-D boolean
        # frame, which is not a row selector - `X.loc[mask]` on one either raises or silently
        # misaligns. Reduced with `.all(axis=1)`: one design matrix is shared by every target in the
        # group, so a row is usable only where ALL of them were measured. That is stricter than the
        # per-target loop, which keeps each row for whichever targets it has, and it is the
        # unavoidable cost of a single fit rather than an oversight.
        if isinstance(y, pd.DataFrame):
            mask = y.notna().all(axis=1)
        else:
            mask = pd.notna(y)
        X_clean = X.loc[mask]
        y_clean = y.loc[mask]
        groups_clean = groups.loc[mask] if groups is not None else None
        return X_clean, y_clean, groups_clean


@dataclass
class PipelineBuilder:
    seed: int = 42
    # Wired from config; these keys existed in YAML but nothing read them, so setting
    # TREE_CATEGORICAL_ENCODING: onehot silently did nothing.
    tree_categorical_encoding: str = "ordinal"
    tree_onehot_max_categories: Optional[int] = None

    def _is_tree_based_model(self, model: BaseEstimator) -> bool:
        model_name = model.__class__.__name__.lower()
        model_module = model.__class__.__module__.lower()
        tree_markers = ("tree", "forest", "boost", "xgb", "lightgbm", "catboost")
        return any(marker in model_name or marker in model_module for marker in tree_markers)

    def _build_preprocessor(self, model: BaseEstimator, categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
        is_tree_model = self._is_tree_based_model(model)

        # add_indicator appends a measured-vs-filled flag, matching the validity channels the
        # Lightning datamodules carry - so a gap means the same thing to both families instead of
        # being silently indistinguishable from a real measurement on this side. Its default
        # features='missing-only' emits a flag ONLY for columns that had gaps in the fold it was
        # fitted on, which is the same "only where there is something to flag" rule. Staying inside
        # the Pipeline keeps it fitted per fold under GridSearchCV, so it cannot leak.
        numeric_steps: List[Tuple[str, BaseEstimator]] = [
            ('imputer', SimpleImputer(strategy='median', add_indicator=True))
        ]
        if not is_tree_model:
            numeric_steps.append(('scaler', RobustScaler()))

        use_ordinal = is_tree_model and str(self.tree_categorical_encoding).lower() == "ordinal"
        if use_ordinal:
            categorical_encoder: BaseEstimator = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1,
                encoded_missing_value=-1,
            )
        elif is_tree_model and self.tree_onehot_max_categories:
            categorical_encoder = OneHotEncoder(
                handle_unknown='infrequent_if_exist',
                max_categories=int(self.tree_onehot_max_categories),
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
                    # No add_indicator here: OrdinalEncoder already maps a missing category to its
                    # own reserved value, and OneHotEncoder gives it its own column, so the flag
                    # would duplicate information the encoding already carries. That matches the
                    # Lightning side, where a blank category becomes the reserved embedding index.
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', categorical_encoder),
                ]),
                categorical_cols,
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

        from sklearn.compose import TransformedTargetRegressor

        steps: List[Tuple[str, BaseEstimator]] = [('preprocessor', preprocessor)]
        if is_log_target:
            steps.append((
                'model',
                TransformedTargetRegressor(
                    regressor=model,
                    func=LogTransformer().transform,
                    inverse_func=LogTransformer().inverse_transform,
                    check_inverse=False,
                ),
            ))
        else:
            steps.append(('model', model))
        return Pipeline(steps)
