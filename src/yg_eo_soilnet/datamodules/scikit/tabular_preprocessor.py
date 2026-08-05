from __future__ import annotations

from typing import Any, Dict

import pandas as pd


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
        }
