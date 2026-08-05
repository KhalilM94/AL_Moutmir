from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
import pandas as pd
from typing import Optional, List, Dict
import traceback

from yg_eo_soilnet.logger import ChildRunLogger, TrainingLogger
from yg_eo_soilnet.datamodules.scikit.scikit_trainer_utils import (CVSplitter, PipelineBuilder, TargetNanFilter)
from yg_eo_soilnet.utils import LogTransformer

class ModelTrainer:
    def __init__(
        self,
        config,
        columns_to_transform: Optional[List[str]] = None,
        enable_clustering: bool = False,
        split_strategy: str = 'kfold',
        seed: int = 42,
        n_splits: int = 5,
        logger= None,
        tuning_verbose: int = 0
    ):
        self.config = config
        self.columns_to_transform = columns_to_transform or []
        self.enable_clustering = enable_clustering
        self.split_strategy = split_strategy
        self.seed = seed
        self.n_splits = n_splits
        if logger is not None:
            self.logger = logger
        else:
            self.logger = TrainingLogger(
                enable_file_logging=getattr(config, 'SKLEARN_FILE_LOGGING_ENABLED', True),
            ).get_logger()
        self.tuning_verbose = tuning_verbose
        self.pipeline_builder = PipelineBuilder(
            tree_categorical_encoding=getattr(config, "TREE_CATEGORICAL_ENCODING", "ordinal"),
            tree_onehot_max_categories=getattr(config, "TREE_ONEHOT_MAX_CATEGORIES", None),
        )
        
        self.log_transformer = LogTransformer()

    def train(
        self,
        target: str,
        data: Dict,
        model_pipelines: Dict[str, Dict],
        ):
        """
        Train models for a specific target variable using the provided data.
        Returns: list of metrics dicts (one per fold or one per model).
        """
        self._validate_model_pipelines(model_pipelines)
        X_train = data['X_train']
        X_train = X_train.astype({col: 'float64' for col in X_train.select_dtypes(include=['int64', 'int32']).columns})
        y_train = data['y_train'][target]
        X_test = data['X_test']
        # Must read X_test's own dtypes: X_train was already converted above, so keying off it
        # produced an empty mapping and left X_test integer-typed.
        X_test = X_test.astype({col: 'float64' for col in X_test.select_dtypes(include=['int64', 'int32']).columns})
        y_test = data['y_test'][target]
        groups_train = data['groups_train'] if self.enable_clustering else None

        X_train, y_train, groups_train = TargetNanFilter().transform(X_train, y_train, groups_train)

        if X_test is not None and y_test is not None:
            X_test, y_test, _ = TargetNanFilter().transform(X_test, y_test)
        else:
            X_test, y_test = None, None

        min_feature_count = int(getattr(self.config, "MIN_FEATURE_COUNT", 10))
        min_valid_rows = max(5, min_feature_count, int(X_train.shape[1]))
        if y_train is not None and not y_train.empty and len(y_train) < min_valid_rows:
            self.logger.warning(
                f"Target {target} has only {len(y_train)} valid training rows after NaN filtering; "
                f"minimum recommended is {min_valid_rows} for {X_train.shape[1]} features."
            )
        if y_test is not None and hasattr(y_test, 'empty') and not y_test.empty and len(y_test) < min_valid_rows:
            self.logger.warning(
                f"Target {target} has only {len(y_test)} valid test rows after NaN filtering; "
                f"minimum recommended is {min_valid_rows} for {X_train.shape[1]} features."
            )

        if not self._should_skip_target(y_train, y_test, target):

            is_log_target = target in self.columns_to_transform
            mlflow_logger = ChildRunLogger()

            trained_models = 0
            for model_name, config in model_pipelines.items():
                try:
                    self.logger.info(f"Training {model_name} for {target}")

                    model_seed = int(config.get("random_seed", self.seed))

                    cv_splitter = CVSplitter(
                        cv_strategy=self.split_strategy,
                        n_splits=self.n_splits,
                        random_state=model_seed,
                    )
                    splits = cv_splitter.create_splits(X_train, y_train, groups_train)

                    model = config["model"]
                    params = config.get("params", {})
                    modeltype = config.get("modeltype", "ml")
                    if modeltype == "ml":
                        is_tree_model = self.pipeline_builder._is_tree_based_model(model)
                        categorical_encoding = "ordinal" if is_tree_model else "onehot"
                        categorical_cols = [
                            col for col in self.config.CATEGORICAL_FEATURES
                            if col in X_train.columns
                        ]
                        numeric_cols = [
                            col for col in X_train.columns
                            if col not in categorical_cols
                        ]
                        # Build pipeline
                        pipeline = self.pipeline_builder.build(
                            model,
                            is_log_target,
                            categorical_cols=categorical_cols,
                            numeric_cols=numeric_cols,
                        )

                        # Adjust param grid if using TransformedTargetRegressor
                        if is_log_target and bool(params):
                            params = {
                                k.replace("model__", "model__regressor__") : v
                                for k, v in params.items()
                            }
                        search = GridSearchCV(
                            estimator=clone(pipeline),
                            param_grid= params if params is not None else {},
                            cv=splits, refit=False,
                            scoring= "neg_root_mean_squared_error",
                            n_jobs=-1, return_train_score=True,
                            verbose=self.tuning_verbose
                        )

                        search.fit(X_train, y_train)
                        best_params = search.best_params_ if params is not None else {}
                        cv_results = pd.DataFrame(search.cv_results_)
                        best_model = clone(pipeline)
                        if params is not None:
                            best_model.set_params(**best_params)
                        best_model.fit(X_train, y_train)

                        if not any(cv_results.get("params", [])):
                            cv_results["params"] = [best_model.get_params()]
                        #Evaluate model
                        param_names = list(params.keys()) if params else []
                        plot_func = {}
                        if len(param_names) > 1:
                            cv_plot = "yg_eo_soilnet.plot_utils.cv_parallel_coordinates"
                        elif len(param_names) == 1:
                            cv_plot = "yg_eo_soilnet.plot_utils.cv_val_curve"
                        else:
                            cv_plot = None  # No hyperparameters to plot
                            param_names = list(best_model.get_params().keys())

                        if cv_plot is not None:
                            plot_func.update({cv_plot: {"args": [cv_results]}})
                        mlflow_logger.log_child_run(
                            config=self.config,
                            search=search,
                            cv_results=cv_results,
                            best_model=best_model,
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            target=target,
                            param_names=param_names,
                            model_name=model_name,
                            plot_functions=plot_func,
                            extra_params={
                                "categorical_encoding": categorical_encoding,
                            },
                            )
                    else:
                        raise ValueError(f"Unknown modeltype: {modeltype}")
                    trained_models += 1
                except Exception as e:
                    self.logger.warning(f"Training failed for {model_name} on {target}: {e}")
                    self.logger.debug(traceback.format_exc())
                    if getattr(self.config, "FAIL_ON_MODEL_ERROR", False):
                        raise

            # Without this, a target where every model failed still exits 0 with an empty MLflow run.
            if trained_models == 0 and getattr(self.config, "FAIL_IF_ALL_MODELS_FAIL_FOR_TARGET", True):
                raise RuntimeError(
                    f"All {len(model_pipelines)} model(s) failed to train for target {target!r}; "
                    "see the warnings above. Set FAIL_IF_ALL_MODELS_FAIL_FOR_TARGET: false to continue anyway."
                )

        else:
            self.logger.warning(f"Skipping training for target {target} due to insufficient data.")

    def _should_skip_target(self, y_train: pd.Series, y_test: Optional[pd.Series], target: str) -> bool:
        n_train = len(y_train) if y_train is not None else 0
        n_test = len(y_test) if y_test is not None else 0
        self.logger.info(f"Training for target: {target} — {n_train} train samples, {n_test} test samples.")
        if y_train is None or y_train.empty or (y_test is not None and hasattr(y_test, 'empty') and y_test.empty):
            self.logger.warning(f"Skipping {target} — no valid data after filtering NaNs.")
            return True
        return False

    def _validate_model_pipelines(self, model_pipelines: Dict[str, Dict]):
        for model_name, config in model_pipelines.items():
            if "model" not in config or "params" not in config:
                raise ValueError(f"Model pipeline '{model_name}' must have 'model' and 'params' keys.")

