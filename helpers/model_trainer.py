import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple

from .training_logger import TrainingLogger
from .trainer_utils import (CVSplitter, PipelineBuilder, Tuner, 
                            ModelEvaluator, ModelSaver)
from .misc_utils import LogTransformer

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
        logger= None,
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
        self.pipeline_builder = PipelineBuilder(self.enable_rfe, self.seed)
        self.tuner = Tuner(
            use_bayes=self.use_bayes_opt,
            seed=self.seed,
            verbose=self.tuning_verbose,
            enable_tuning=self.enable_hyperparameter_tuning  # Pass flag here
        )
        self.evaluator = ModelEvaluator(self.logger, self.columns_to_transform)
        self.saver = ModelSaver(self.output_dir)
        self.log_transformer = LogTransformer()

        self._validate_model_pipelines()

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

        if self._should_skip_target(y_train, y_test, target):
            return results

        is_log_target = target in self.columns_to_transform

        for model_name, config in self.model_pipelines.items():
            try:
                result = self._train_single_model(
                    model_name, config, target,
                    X_train, y_train, X_test, y_test,
                    groups_train, is_log_target
                )
                if result:
                    results.append(result)
            except Exception as e:
                self.logger.warning(f"Training failed for {model_name} on {target}: {e}")

        return results

    def _should_skip_target(self, y_train: pd.Series, y_test: pd.Series, target: str) -> bool:
        self.logger.info(f"Training for target: {target} — {len(y_train)} train samples, {len(y_test)} test samples.")
        if y_train.empty or y_test.empty:
            self.logger.warning(f"Skipping {target} — no valid data after filtering NaNs.")
            return True
        return False

    def _prepare_transformed_target(self, y: pd.Series, is_log_target: bool) -> pd.Series:
        if is_log_target:
            return self.log_transformer.transform(y)
        return y

    def _train_single_model(
        self,
        model_name: str,
        config: Dict,
        target: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        groups_train: Optional[pd.Series],
        is_log_target: bool
    ) -> Optional[Dict]:
        self.logger.info(f"Training {model_name} for {target}")
        model = config["model"]
        is_rfe_compatible = self.pipeline_builder.is_rfe_compatible(model)
        if self.enable_rfe and not is_rfe_compatible:
            self.logger.info(f"Skipping RFE for {model_name} (not supported).")
        y_train_transformed = self._prepare_transformed_target(y_train, is_log_target)
        best_model, splits, best_params = self._run_training_pipeline(
            model, config, model_name, target,
            X_train, y_train_transformed, groups_train, is_rfe_compatible
        )
        
        return self._evaluate_and_save(
            best_model, model_name, target,
            X_train, y_train_transformed, X_test, y_test, splits, best_params
        )

    def _run_training_pipeline(
        self,
        model,
        config: Dict,
        model_name: str,
        target: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: Optional[pd.Series],
        use_rfe: bool
    ) -> Tuple[object, List[Tuple[np.ndarray, np.ndarray]], object]:

        pipeline = self.pipeline_builder.build(model, use_rfe)

        cv_splitter = CVSplitter(cv_strategy=self.split_strategy, random_state=self.seed)
        splits, _ = cv_splitter.create_splits(X_train, y_train, groups_train, target, model_name)

        best_model, best_params = self.tuner.tune(pipeline, config, X_train, y_train, splits)
        self.saver.save_cv_results(best_model, model_name, target)

        return best_model, splits, best_params

    def _evaluate_and_save(
        self,
        model,
        model_name: str,
        target: str,
        X_train: pd.DataFrame,
        y_train_transformed: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        best_params
    ) -> Optional[Dict]:
        try:
            test_metrics = self.evaluator.evaluate(
                model, X_train, y_train_transformed,
                X_test, y_test,
                target, model_name,
                best_params,  # <-- Use the actual best_params here!
                self.log_transformer, splits
            )
            model_path = self.saver.save_model(model, model_name, target)
            metrics_path = self.saver.save_metrics(test_metrics, model_name, target)
            self.logger.info(f"Saved model to {model_path}")
            self.logger.info(f"Saved metrics to {metrics_path}")
            return test_metrics
        except Exception as e:
            self.logger.warning(f"Evaluation failed for {model_name} on {target}: {e}")
            return None

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
        """
        Remove rows where y is NaN from X, y, and groups (if provided).
        Returns cleaned X, y, and groups.
        """
        mask = y.notna()
        X_clean = X.loc[mask]
        y_clean = y.loc[mask]
        groups_clean = groups.loc[mask] if groups is not None else None
        return X_clean, y_clean, groups_clean
