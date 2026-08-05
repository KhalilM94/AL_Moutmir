from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from yg_eo_soilnet.datamodules.scikit.sklearn_data_splitter import SklearnDataSplitter
from yg_eo_soilnet.datamodules.scikit.tabular_preprocessor import TabularPreprocessor


class ScikitDataModule:
    """Owns every sklearn-specific data preparation step.

    The DataManager only loads raw frames and filters the schema; this module decides how that
    frame becomes a feature matrix, a label frame and a train/test split.
    """

    def __init__(self, config, logger, data_manager):
        self.config = config
        self.logger = logger
        self.data_manager = data_manager
        self.preprocessor = TabularPreprocessor(config, logger, data_manager)
        self.splitter = SklearnDataSplitter(config, logger)

    def load_frame(self) -> pd.DataFrame:
        """The training frame. DataManager resolves joint-vs-separate sources; we just consume it."""
        return self.data_manager.load_dataset().tabular

    def preprocess(self, data: pd.DataFrame) -> Dict[str, Any]:
        return self.preprocessor.preprocess_data(data)

    def split(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        from yg_eo_soilnet.models import ModelConfigFactory  # local import avoids a circular dependency

        return self.splitter.split_data(
            processed_data,
            sanitize_features=self.data_manager.filter_schema,
            model_config_factory=ModelConfigFactory,
        )

    def prepare(self, data: pd.DataFrame | None = None) -> Dict[str, Any]:
        """Run the full sklearn preparation chain and return the split bundle."""
        frame = self.load_frame() if data is None else data
        return self.split(self.preprocess(frame))
