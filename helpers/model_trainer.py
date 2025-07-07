import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from sklearn.model_selection import GridSearchCV  # FIX: import added

from .training_logger import TrainingLogger
from .trainer_utils import (CVSplitter, PipelineBuilder, Tuner, 
                            ModelEvaluator, ModelSaver)
from .misc_utils import LogTransformer

class ModelTrainer:
    def __init__(
        self,
        model_pipelines: Dict[str, Dict],
        columns_to_transform: Optional[List[str]] = None,
        enable_clustering: bool = False,
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
        self.enable_clustering = enable_clustering
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
        data: Dict,
        ) -> List[Dict]:
        """
        In CV_ONLY_MODE: run n-fold CV, save model and metrics for each fold, collect predictions for plotting.
        In normal mode: train on train, test on test.
        Returns: list of metrics dicts (one per fold or one per model).
        """
        # Detect CV-only mode by key presence
        cv_only_mode = not ("X_train" in data and "y_train" in data and "X_test" in data and "y_test" in data)
        if not cv_only_mode:
            X_train = data['X_train']
            y_train = data['y_train'][target]
            X_test = data['X_test']
            y_test = data['y_test'][target]
            groups_train = data['groups_train'] if self.enable_clustering else None
        else:
            X_train = data['X']
            y_train = data['y'][target]
            X_test = None
            y_test = None
            groups_train = data['groups'] if self.enable_clustering else None

        results = []
        fold_preds = []  # For plotting: (y_val, y_pred, fold)

        X_train, y_train, groups_train = self._filter_nans(X_train, y_train, groups_train)
        if X_test is not None and y_test is not None:
            X_test, y_test, _ = self._filter_nans(X_test, y_test)
        else:
            X_test, y_test = None, None

        if self._should_skip_target(y_train, y_test, target):
            return results

        is_log_target = target in self.columns_to_transform

        for model_name, config in self.model_pipelines.items():
            try:
                model = config["model"]
                is_rfe_compatible = self.pipeline_builder.is_rfe_compatible(model)
                if self.enable_rfe and not is_rfe_compatible:
                    self.logger.info(f"Skipping RFE for {model_name} (not supported).")
                y_train_transformed = self._prepare_transformed_target(y_train, is_log_target)

                if cv_only_mode:
                    cv_splitter = CVSplitter(cv_strategy=self.split_strategy, n_splits=getattr(self, 'n_splits', 5), random_state=self.seed)
                    splits, _ = cv_splitter.create_splits(X_train, y_train, groups_train)
                    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):  # Start from 1
                        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                        y_tr, y_val = y_train_transformed.iloc[train_idx], y_train.iloc[val_idx]
                        # Log group info if available
                        if groups_train is not None:
                            train_groups = groups_train.iloc[train_idx].unique()
                            val_groups = groups_train.iloc[val_idx].unique()
                            self.logger.info(f"Fold {fold_idx}: Train groups: {train_groups}")
                            self.logger.info(f"Fold {fold_idx}: Val groups: {val_groups}")
                        else:
                            val_groups = None
                        # Build pipeline
                        pipeline = self.pipeline_builder.build(model, is_rfe_compatible)
                        # Fit
                        if self.enable_hyperparameter_tuning and config.get("params"):
                            search = GridSearchCV(
                                estimator=pipeline,
                                param_grid=config["params"],
                                cv=3,
                                scoring="neg_mean_squared_error",
                                n_jobs=-1,
                                verbose=self.tuning_verbose
                            )
                            search.fit(X_tr, y_tr)
                            best_model = search.best_estimator_
                            best_params = search.best_params_
                        else:
                            best_model = pipeline.fit(X_tr, y_tr)
                            best_params = {}
                        # Save model
                        self.saver.save_model(best_model, model_name, str(fold_idx), target)  # fold as str
                        # Predict
                        y_pred_val = best_model.predict(X_val)
                        if is_log_target:
                            y_pred_val = self.log_transformer.inverse_transform(y_pred_val)
                        # Metrics
                        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
                        metrics = {
                            "target": target,
                            "model": model_name,
                            "fold": fold_idx,
                            "Test_RMSE": np.sqrt(mean_squared_error(y_val, y_pred_val)),
                            "Test_MAE": mean_absolute_error(y_val, y_pred_val),
                            "Test_R2": r2_score(y_val, y_pred_val),
                            "Test_ExplainedVar": explained_variance_score(y_val, y_pred_val),
                            "Best_Params": str(best_params)
                        }
                        self.saver.save_metrics(metrics, model_name, str(fold_idx), target)  # fold as str
                        results.append(metrics)
                        fold_preds.append({
                            "fold": fold_idx,
                            "X_val": X_val,
                            "y_val": y_val,
                            "y_pred": y_pred_val,
                            "target": target,
                            "model": model_name,
                            "val_groups": val_groups.tolist() if val_groups is not None else None
                        })
                else:
                    result = self._train_single_model(
                        model_name, config, target,
                        X_train, y_train, X_test, y_test,
                        groups_train, is_log_target
                    )
                    if result:
                        results.extend(result)
            except Exception as e:
                self.logger.warning(f"Training failed for {model_name} on {target}: {e}")
        self.fold_preds = fold_preds if cv_only_mode else None
        return results

    def _should_skip_target(self, y_train: pd.Series, y_test: Optional[pd.Series], target: str) -> bool:
        n_train = len(y_train) if y_train is not None else 0
        n_test = len(y_test) if y_test is not None else 0
        self.logger.info(f"Training for target: {target} — {n_train} train samples, {n_test} test samples.")
        if y_train is None or y_train.empty or (y_test is not None and hasattr(y_test, 'empty') and y_test.empty):
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
        X_test: Optional[pd.DataFrame],
        y_test: Optional[pd.Series],
        groups_train: Optional[pd.Series],
        is_log_target: bool
    ) -> Optional[List[Dict]]:
        self.logger.info(f"Training {model_name} for {target}")
        model = config["model"]
        is_rfe_compatible = self.pipeline_builder.is_rfe_compatible(model)
        if self.enable_rfe and not is_rfe_compatible:
            self.logger.info(f"Skipping RFE for {model_name} (not supported).")
        y_train_transformed = self._prepare_transformed_target(y_train, is_log_target)
        best_models, splits = self._run_training_pipeline(
            model, config, model_name, target,
            X_train, y_train_transformed, groups_train, is_rfe_compatible
        )
        test_metrics = []
        for best_model in best_models:
            # Only evaluate if X_test and y_test are not None
            if X_test is not None and y_test is not None:
                test_metrics.append(self._evaluate_and_save(
                    best_model, model_name, target,
                    X_train, y_train_transformed, X_test, y_test, splits))
        return test_metrics

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
    ) -> Tuple[List, List[Tuple[np.ndarray, np.ndarray]]]:

        pipeline = self.pipeline_builder.build(model, use_rfe)

        cv_splitter = CVSplitter(cv_strategy=self.split_strategy, random_state=self.seed)
        splits, _ = cv_splitter.create_splits(X_train, y_train, groups_train)

        best_models = self.tuner.tune(pipeline, config, X_train, y_train, splits)
        for model in best_models:
            self.saver.save_cv_results(model, model_name, target)

        return best_models, splits

    def _evaluate_and_save(
        self,
        model: Dict,
        model_name: str,
        target: str,
        X_train: pd.DataFrame,
        y_train_transformed: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        splits: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        try:
            fold = model.get('fold')

            self.logger.info(f"Evaluating {model_name} for {target} on fold {fold}")
            test_metrics = self.evaluator.evaluate(
                model, X_train, y_train_transformed,
                X_test, y_test,
                target, model_name,
                self.log_transformer, splits
            )
            model_path = self.saver.save_model(model["model"], model_name, model["fold"], target)
            metrics_path = self.saver.save_metrics(test_metrics, model_name, model["fold"], target)
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
