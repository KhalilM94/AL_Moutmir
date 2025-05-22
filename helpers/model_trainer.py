import pandas as pd
from typing import Optional, List, Dict, Tuple

from .logging import TrainingLogger
from .utils import (CVSplitter, 
                    LogTransformer, 
                    PipelineBuilder, 
                    Tuner,
                    ModelEvaluator,
                    ModelSaver)


class ModelTrainer:
    def __init__(
        self,
        model_pipelines: Dict[str, Dict],
        columns_to_transform: Optional[List[str]] = None,
        split_strategy: str = 'kfold',
        enable_hyperparameter_tuning: bool = False,
        use_bayes_opt: bool = False,
        enable_rfe: bool = False,
        seed: int = 42,
        logger=None,
        output_dir: str = "output",
        tuning_verbose: int = 0
    ):
        self.model_pipelines = model_pipelines
        self.columns_to_transform = columns_to_transform or []
        self.split_strategy = split_strategy
        self.enable_hyperparameter_tuning = enable_hyperparameter_tuning
        self.use_bayes_opt = use_bayes_opt
        self.enable_rfe = enable_rfe
        self.seed = seed
        self.logger = logger or TrainingLogger().get_logger()
        self.output_dir = output_dir
        self.tuning_verbose = tuning_verbose

        self._validate_model_pipelines()

        self.pipeline_builder = PipelineBuilder(self.enable_rfe, self.seed)
        self.tuner = Tuner(self.use_bayes_opt, self.seed, self.tuning_verbose)
        self.evaluator = ModelEvaluator(self.logger, self.columns_to_transform)
        self.saver = ModelSaver(self.output_dir)

    def _validate_model_pipelines(self):
        for model_name, config in self.model_pipelines.items():
            if "model" not in config or "params" not in config:
                raise ValueError(f"Model pipeline '{model_name}' must have 'model' and 'params' keys.")

    def _safe_filename(self, name: str) -> str:
        return str(name).replace("/", "_").replace("\\", "_")

    def _filter_nans(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        mask = y.notna()
        X_clean = X.loc[mask]
        y_clean = y.loc[mask]
        groups_clean = groups.loc[mask] if groups is not None else None
        return X_clean, y_clean, groups_clean

    def train(
        self,
        target: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        groups_train: Optional[pd.Series] = None
    ) -> List[Dict]:

        results = []

        X_train, y_train, groups_train = self._filter_nans(X_train, y_train, groups_train)
        X_test, y_test, _ = self._filter_nans(X_test, y_test)

        self.logger.info(f"Training for target: {target} — {len(y_train)} train samples, {len(y_test)} test samples.")

        if y_train.empty or y_test.empty:
            self.logger.warning(f"Skipping {target} — no valid data after filtering NaNs.")
            return results

        is_log_target = target in self.columns_to_transform
        log_transformer = LogTransformer()

        for model_name, config in self.model_pipelines.items():
            self.logger.info(f"Training {model_name} for {target}")
            model = config["model"]

            use_rfe = self.enable_rfe and self.pipeline_builder.is_rfe_compatible(model)
            if self.enable_rfe and not use_rfe:
                self.logger.info(f"Skipping RFE for {model_name} (not supported).")

            y_train_transformed = log_transformer.transform(y_train) if is_log_target else y_train

            pipeline = self.pipeline_builder.build(model, use_rfe)

            cv_splitter = CVSplitter(cv_strategy=self.split_strategy, random_state=self.seed)
            splits, _ = cv_splitter.create_splits(
                X_train, y_train, groups_train,
                target=target, model_name=model_name
            )

            try:
                best_model, best_params = self.tuner.tune(
                    pipeline, config, X_train, y_train_transformed, splits
                )
                self.saver.save_cv_results(best_model, model_name, target)
            except Exception as e:
                self.logger.warning(f"Training failed for {model_name} on {target}: {e}")
                continue

            try:
                test_metrics = self.evaluator.evaluate(
                    best_model, X_train, y_train_transformed,   # type: ignore[attr-defined]
                    X_test, y_test, target, model_name,
                    best_params, log_transformer, splits
                )
                model_path = self.saver.save_model(best_model, model_name, target)
                metrics_path =self.saver.save_metrics(test_metrics, model_name, target)
                self.logger.info(f"Saved model to {model_path} and metrics to {metrics_path}")
                results.append(test_metrics)
            except Exception as e:
                self.logger.warning(f"Evaluation failed for {model_name} on {target}: {e}")

        return results
