from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from yg_eo_soilnet.datamodules.frame_cleaning import assert_columns_are_dense_enough


class TabularPreprocessor:
    def __init__(self, config, logger, data_manager):
        self.config = config
        self.logger = logger
        self.data_manager = data_manager

    def preprocess_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        target_columns = list(getattr(self.config, "TARGET_COLUMNS", []))
        lat_col = getattr(self.config, "LAT_COLUMN", "lat")
        lon_col = getattr(self.config, "LON_COLUMN", "lon")

        if lat_col not in data.columns or lon_col not in data.columns:
            raise KeyError(f"Tabular data must contain coordinate columns '{lat_col}' and '{lon_col}'")

        candidate_feature_columns = self.data_manager.filter_schema(data, target_columns).columns.tolist()

        valid_feature_columns = data[candidate_feature_columns].dropna(axis=1, how="all").columns.tolist()
        original_feature_count = len(candidate_feature_columns)
        retained_feature_count = len(valid_feature_columns)
        if original_feature_count > 0 and retained_feature_count < original_feature_count:
            dropped_ratio = (original_feature_count - retained_feature_count) / float(original_feature_count)
            max_drop_ratio = float(getattr(self.config, "MAX_FEATURE_DROP_RATIO_WARNING", 0.9))
            if dropped_ratio > max_drop_ratio:
                self.logger.warning(
                    f"Dropped feature ratio for preprocessing is {dropped_ratio:.2%} "
                    f"(threshold {max_drop_ratio:.2%}); {original_feature_count - retained_feature_count} of {original_feature_count} features were removed."
                )

        min_feature_count = int(getattr(self.config, "MIN_FEATURE_COUNT", 10))
        if retained_feature_count < min_feature_count:
            self.logger.warning(
                f"Only {retained_feature_count} usable features remain after preprocessing "
                f"(minimum recommended: {min_feature_count})."
            )

        X = data[valid_feature_columns]
        categorical_cols = [
            col for col in self.config.CATEGORICAL_FEATURES if col in X.columns and col not in self.config.EXCLUDE_CATEGORICAL
        ]
        # The same gate the Lightning builders apply, so a column too empty to impute stops BOTH
        # families. This side never deleted rows over a gap - it median-filled and handed the result
        # to the model as if measured - which is why a 99.7%-blank column could reach XGBoost as a
        # feature without anything being said. Checked on the continuous block only: a missing
        # category is encoded as its own value rather than imputed.
        self.assert_covariates_are_dense_enough(
            data, [column for column in valid_feature_columns if column not in categorical_cols]
        )
        if categorical_cols:
            self.logger.info(
                "Categorical encoding will be applied per model in the training pipeline "
                "(one-hot for linear models, ordinal encoding for tree-based models)."
            )

        data_cleaned = data.loc[X.index]
        return {
            "X": X,
            "y": data_cleaned[target_columns],
            "lat": data_cleaned[lat_col],
            "lon": data_cleaned[lon_col],
            # The join key. `filter_schema` strips the id column out of X, so without carrying it
            # here the sklearn split could not be keyed on point id - and therefore could not be
            # shared with the Lightning families, whose row sets differ.
            "point_ids": self._point_ids(data_cleaned),
        }

    def assert_covariates_are_dense_enough(self, data: pd.DataFrame, columns) -> None:
        """Apply the shared sparsity gate to this family's continuous features."""
        assert_columns_are_dense_enough(
            data,
            columns,
            max_missing_ratio=float(getattr(self.config, "MAX_MISSING_COLUMN_RATIO", 0.2)),
            label="the sklearn feature matrix",
            logger=self.logger,
            allow=getattr(self.config, "ALLOW_SPARSE_COLUMNS", ()) or (),
            fail=bool(getattr(self.config, "FAIL_ON_SPARSE_COLUMNS", True)),
        )

    def usable_point_ids(self, data: pd.DataFrame) -> pd.Index:
        """Point ids this family can use. Consumed by the unified splitter.

        Every row qualifies: preprocessing drops all-null *columns*, never rows. The method exists
        so the splitter can ask each family the same question and log an honest delta.
        """
        return pd.Index(self._point_ids(data).to_numpy())

    def _point_ids(self, data: pd.DataFrame) -> pd.Series:
        point_col = getattr(self.config, "POINT_ID_COLUMN", "point_id")
        if point_col in data.columns:
            return data[point_col]
        # No id column in the source: fall back to positional ids so the plan still has a key. Both
        # families derive it from the same frame in the same order, so they agree.
        return pd.Series(np.arange(len(data)), index=data.index, name=point_col)
