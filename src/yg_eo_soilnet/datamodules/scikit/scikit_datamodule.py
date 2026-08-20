from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from yg_eo_soilnet.datamodules.scikit.sklearn_data_splitter import SklearnDataSplitter
from yg_eo_soilnet.datamodules.scikit.tabular_preprocessor import TabularPreprocessor
from yg_eo_soilnet.datamodules.split_plan_provider import SplitPlanProvider
from yg_eo_soilnet.datamodules.splitting import SplitPlan


class ScikitDataModule:
    """Owns every sklearn-specific data preparation step.

    The DataManager only loads raw frames and filters the schema; this module decides how that
    frame becomes a feature matrix and a label frame. It no longer decides the train/test split -
    that is shared with the Lightning families and comes from :class:`SplitPlanProvider`.
    """

    def __init__(self, config, logger, data_manager, split_plan_provider=None):
        self.config = config
        self.logger = logger
        self.data_manager = data_manager
        self.preprocessor = TabularPreprocessor(config, logger, data_manager)
        self.splitter = SklearnDataSplitter(config, logger)
        # Shared with the Lightning layer when the caller passes one in, so both families resolve
        # the same object rather than two plans that merely agree by construction.
        self.split_plan_provider = split_plan_provider or SplitPlanProvider(config, logger, data_manager)

    def load_frame(self) -> pd.DataFrame:
        """The training frame. DataManager resolves joint-vs-separate sources; we just consume it."""
        return self.data_manager.load_dataset().tabular

    def preprocess(self, data: pd.DataFrame) -> Dict[str, Any]:
        return self.preprocessor.preprocess_data(data)

    def split(self, processed_data: Dict[str, Any], split_plan: SplitPlan | None = None) -> Dict[str, Any]:
        """Select this family's rows out of the run's shared split.

        `split_plan` defaults to the one the provider builds, so a caller that does not care about
        the split still gets the shared one rather than a second, private one.
        """
        from yg_eo_soilnet.models import ModelConfigFactory  # local import avoids a circular dependency

        return self.splitter.split_data(
            processed_data,
            sanitize_features=self.data_manager.filter_schema,
            model_config_factory=ModelConfigFactory,
            split_plan=split_plan if split_plan is not None else self.split_plan(),
        )

    def split_plan(self) -> SplitPlan:
        """The run's shared split plan. Built once and cached on the provider."""
        return self.split_plan_provider.plan()

    def prepare(self, data: pd.DataFrame | None = None) -> Dict[str, Any]:
        """Run the full sklearn preparation chain and return the split bundle."""
        frame = self.load_frame() if data is None else data
        return self.split(self.preprocess(frame))
