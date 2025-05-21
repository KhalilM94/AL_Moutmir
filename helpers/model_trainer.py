import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Tuple
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFECV
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from skopt import BayesSearchCV

from .logging import TrainingLogger
from .utils import CVSplitter, LogTransformer


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

    def _is_rfe_compatible(self, model) -> bool:
        if isinstance(model, (PLSRegression, GradientBoostingRegressor)):
            return False
        if isinstance(model, SVR) and getattr(model, 'kernel', None) != "linear":
            return False
        return True

    def _build_pipeline(self, model, use_rfe: bool) -> Pipeline:
        steps: List[Tuple[str, BaseEstimator]] = [('scaler', RobustScaler())]
        if use_rfe:
            rfecv = RFECV(estimator=clone(model), step=1, cv=5, scoring='neg_mean_squared_error')
            steps.append(('feature_selection', rfecv))
        steps.append(('model', model))
        return Pipeline(steps)

    def _fit_model(
        self,
        pipeline: Pipeline,
        model_config: Dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target: str,
        model_name: str,
        splits
    ) -> Tuple[BaseEstimator, Union[str, Dict]]:
        if self.enable_hyperparameter_tuning:
            search_class = BayesSearchCV if self.use_bayes_opt else GridSearchCV
            search_kwargs = {
                "estimator": pipeline,
                "cv": splits,
                "scoring": "neg_mean_squared_error",
                "n_jobs": -1,
                "verbose": self.tuning_verbose,
                "return_train_score": True
            }

            if self.use_bayes_opt:
                search_kwargs["search_spaces"] = model_config["params"]
                search_kwargs["n_iter"] = 30
                search_kwargs["random_state"] = self.seed
            else:
                search_kwargs["param_grid"] = model_config["params"]

            search: Union[GridSearchCV, BayesSearchCV] = search_class(**search_kwargs)
            search.fit(X_train, y_train)

            # Save full CV results
            if hasattr(search, "cv_results_"):
                results_df = pd.DataFrame(search.cv_results_)   # type: ignore[attr-defined]
                results_df["target"] = target
                results_df["model"] = model_name
                safe_target = self._safe_filename(target)
                results_df.to_csv(
                    os.path.join(self.output_dir, "metrics", f"{safe_target}_{model_name}_cv_results.csv"),
                    index=False
                )

            return search.best_estimator_, search.best_params_   # type: ignore[attr-defined]
        else:
            pipeline.fit(X_train, y_train)
            return pipeline, "Default (no tuning)"

    def _evaluate_model(
        self,
        model: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        target: str,
        model_name: str,
        best_params: Union[str, Dict],
        log_transformer: LogTransformer,
        splits
    ) -> Dict:
        y_pred_transformed = model.predict(X_test)
        y_pred = log_transformer.inverse_transform(y_pred_transformed) if target in self.columns_to_transform else y_pred_transformed

        cv_scores = cross_validate(
            model, X_train, y_train,
            scoring='neg_mean_squared_error',
            cv=splits,
            return_train_score=False
        )

        test_metrics = {
            "target": target,
            "model": model_name,
            "Test_RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "Test_MAE": mean_absolute_error(y_test, y_pred),
            "Test_R2": r2_score(y_test, y_pred),
            "CV_RMSE_Mean": np.mean(np.sqrt(-cv_scores["test_score"])),
            "CV_RMSE_Std": np.std(np.sqrt(-cv_scores["test_score"])),
            "Test_ExplainedVar": explained_variance_score(y_test, y_pred),
            "Best_Params": str(best_params)
        }

        self.logger.info(
            f"🏁 {model_name} | {target} — CV RMSE: {test_metrics['CV_RMSE_Mean']:.4f}, "
            f"Test RMSE: {test_metrics['Test_RMSE']:.4f}, R²: {test_metrics['Test_R2']:.4f}"
        )

        # Save metrics
        safe_target = self._safe_filename(target)
        metrics_path = os.path.join(self.output_dir, "metrics", f"{safe_target}_{model_name}_metrics.csv")
        pd.DataFrame([test_metrics]).to_csv(metrics_path, index=False)

        # Save model
        model_path = os.path.join(self.output_dir, "final_models", f"{safe_target}_{model_name}.pkl")
        joblib.dump(model, model_path)
        self.logger.info(f"Saved model to {model_path} and metrics to {metrics_path}")

        return test_metrics

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

            use_rfe = self.enable_rfe and self._is_rfe_compatible(model)
            if self.enable_rfe and not use_rfe:
                self.logger.info(f"Skipping RFE for {model_name} (not supported).")

            y_train_transformed = log_transformer.transform(y_train) if is_log_target else y_train

            pipeline = self._build_pipeline(model, use_rfe)

            cv_splitter = CVSplitter(cv_strategy=self.split_strategy, random_state=self.seed)
            splits, _ = cv_splitter.create_splits(
                X_train, y_train, groups_train,
                target=target, model_name=model_name
            )

            try:
                best_model, best_params = self._fit_model(
                    pipeline, config, X_train, y_train_transformed, target, model_name, splits
                )
            except Exception as e:
                self.logger.warning(f"Training failed for {model_name} on {target}: {e}")
                continue

            try:
                test_metrics = self._evaluate_model(
                    best_model, X_train, y_train_transformed,   # type: ignore[attr-defined]
                    X_test, y_test, target, model_name,
                    best_params, log_transformer, splits
                )
                results.append(test_metrics)
            except Exception as e:
                self.logger.warning(f"Evaluation failed for {model_name} on {target}: {e}")

        return results
