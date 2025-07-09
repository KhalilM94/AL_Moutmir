from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFECV
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, GroupKFold
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                             r2_score, explained_variance_score)
from skopt import BayesSearchCV
import pandas as pd
import numpy as np
from dataclasses import dataclass
import joblib
from typing import List, Dict, Tuple
from pathlib import Path

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
        """
        Generate CV splits and fold info DataFrame.

        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix.
            y (pd.Series or np.ndarray, optional): Target vector.
            groups (pd.Series or np.ndarray, optional): Group labels for GroupKFold.
            target (str, optional): Target name, for logging (not used here).
            model_name (str, optional): Model name, for logging (not used here).

        Returns:
            splits (list of (train_idx, test_idx) tuples)
            fold_info_df (pd.DataFrame): DataFrame with columns ['index', 'fold']
        """
        strategy = self.cv_strategy.lower()
        if strategy == 'kfold':
            cv = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            splits = list(cv.split(X))
        elif strategy == 'groupkfold':
            if groups is None:
                raise ValueError("Groups must be provided for GroupKFold CV strategy.")
            cv = GroupKFold(n_splits=self.n_splits)
            splits = list(cv.split(X, y, groups))
        else:
            raise ValueError(f"Unsupported CV strategy: {self.cv_strategy}")

        # Prepare fold assignment series with -1 default (for samples not assigned)
        indices = X.index if hasattr(X, 'index') else range(len(X))

        fold_assignments = pd.Series(data=-1, index=indices)

        for fold_number, (_, test_idx) in enumerate(splits):
            fold_assignments.iloc[test_idx] = fold_number

        fold_info_df = pd.DataFrame({'index': fold_assignments.index, 'fold': fold_assignments.values})

        return splits, fold_info_df

@dataclass
class PipelineBuilder:
    enable_rfe: bool
    seed: int = 42

    def is_rfe_compatible(self, model) -> bool:
        return not (
            isinstance(model, (PLSRegression, GradientBoostingRegressor)) or
            (isinstance(model, SVR) and getattr(model, 'kernel', None) != "linear")
        )

    def build(self, model, use_rfe: bool) -> Pipeline:
        steps: List[Tuple[str, BaseEstimator]] = [('scaler', RobustScaler())]
        if self.enable_rfe and use_rfe:
            rfecv = RFECV(estimator=clone(model), step=1, cv=5,
                scoring='neg_mean_squared_error')
            steps.append(('feature_selection', rfecv))
        steps.append(('model', model))
        return Pipeline(steps)

@dataclass
class Tuner:
    use_bayes: bool
    seed: int = 42
    verbose: int = 0
    enable_tuning: bool = True

    def tune(self, 
             pipeline, 
             model_config, 
             X_train, 
             y_train, 
             splits,
             fold: int = -1 # Default to -1 for no specific fold
             ):

        """
        If cv_only_mode is True, fit and return a list of models (one per fold).
        For tuning, fit a search object on each fold and save the best estimator.
        Otherwise, perform tuning or fit as usual.
        """
        if self.enable_tuning and model_config.get("params"):
            if self.use_bayes:
                search = BayesSearchCV(
                    estimator=clone(pipeline),
                    search_spaces=model_config["params"],
                    n_iter=30,
                    cv=splits,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                    random_state=self.seed,
                    verbose=self.verbose
                )
            else:
                search = GridSearchCV(
                    estimator=clone(pipeline),
                    param_grid=model_config["params"],
                    cv=splits,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                    verbose=self.verbose
                )
        else:
            search = clone(pipeline)
            
        if self.enable_tuning:
            search.fit(X_train, y_train)
            best_model = search.best_estimator_ # type: ignore[attr-defined]
            best_params = search.best_params_ # type: ignore[attr-defined]
        else:
            best_model = search.fit(X_train, y_train)
            best_params = {}
        
        return {'model': best_model,
                'best_params': best_params,
                'fold': fold if fold != -1 else None}

class ModelEvaluator:
    def __init__(self, logger, columns_to_transform, cv_only_mode=False):
        self.logger = logger
        self.columns_to_transform = columns_to_transform
        self.cv_only_mode = cv_only_mode


    def evaluate(self, model:Dict, X_train, y_train, X_test, y_test, 
                 target, model_name, log_transformer, splits, val_groups = []) -> Dict:
        fold = model.get("fold", None)
        self.logger.info(f"Evaluating {model_name} for {target} on fold {fold}")
        
        y_pred = model["model"].predict(X_test)
        if target in self.columns_to_transform:
            y_pred = log_transformer.inverse_transform(y_pred)
            if self.cv_only_mode:
                y_test = log_transformer.inverse_transform(y_test)

        test_metrics = {
            "target": target,
            "model": model_name,
            "fold": fold,
            "Test_RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "Test_MAE": mean_absolute_error(y_test, y_pred),
            "Test_R2": r2_score(y_test, y_pred),
            "Test_ExplainedVar": explained_variance_score(y_test, y_pred),
            "Best_Params": str(model["best_params"])
        }
        self.logger.info(f"{model_name} | {target} — Fold {fold}:")
        if not self.cv_only_mode:
            cv_scores = cross_validate(model["model"], X_train, y_train, cv=splits,
                                       scoring='neg_mean_squared_error')
            test_metrics.update({
                "CV_RMSE_Mean": np.mean(np.sqrt(-cv_scores["test_score"])),
                "CV_RMSE_Std": np.std(np.sqrt(-cv_scores["test_score"]))
            })
            self.logger.info(
            f"CV RMSE: {test_metrics['CV_RMSE_Mean']:.4f}, ")

        self.logger.info(f"Test RMSE: {test_metrics['Test_RMSE']:.4f}, R²: {test_metrics['Test_R2']:.4f}")
        evaluation_results = {"test_metrics" : test_metrics}
        if self.cv_only_mode:
            fold_preds = {
                "fold": fold,
                "X_val": X_test,
                "y_val": y_test,
                "y_pred": y_pred,
                "target": target,
                "model": model_name,
                "val_groups": val_groups.tolist() if val_groups is not None else None
            }
            evaluation_results["fold_preds"] = fold_preds
        return evaluation_results

@dataclass
class ModelSaver:
    output_dir: str

    def _safe_filename(self, name: str) -> str:
        return str(name).replace("/", "_").replace("\\", "_")

    def save_metrics(self, metrics: Dict, model_name: str, fold: str, target: str):
        safe_target = self._safe_filename(target)
        if fold is None or str(fold) == 'None':
            metrics_path = Path(self.output_dir) / "metrics" / f"{safe_target}_{model_name}_metrics.csv"
        else:
            metrics_path = Path(self.output_dir) / "metrics" / f"{safe_target}_{model_name}_fold_{fold}_metrics.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        return str(metrics_path)

    def save_model(self, model: BaseEstimator, model_name: str, fold: str, target: str):
        safe_target = self._safe_filename(target)
        if fold is None or str(fold) == 'None':
            model_path = Path(self.output_dir) / "final_models" / f"{safe_target}_{model_name}.pkl"
        else:
            model_path = Path(self.output_dir) / "final_models" / f"{safe_target}_{model_name}_fold_{fold}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        return str(model_path)

    def save_cv_results(self, model, model_name: str, target: str):
        search_obj = model.get('model', None)
        fold = model.get('fold', None)    
        if hasattr(search_obj, "cv_results_"):
            safe_target = self._safe_filename(target)
            results_df = pd.DataFrame(search_obj.cv_results_)
            results_df["target"] = target
            results_df["model"] = model_name
            results_df["fold"] = fold if fold is not None else -1
            path = Path(self.output_dir) / "metrics" / f"{safe_target}_{model_name}_cv_results.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(path, index=False)
            return str(path)
        return None