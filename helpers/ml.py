import pandas as pd
import numpy as np
import os
import sys
import logging
from sklearn.model_selection import (
    GridSearchCV,
    KFold, 
    GroupKFold,
    cross_validate,
    StratifiedShuffleSplit,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import RFECV
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.cluster import KMeans
from skopt import BayesSearchCV
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    explained_variance_score
)
import joblib


def get_training_logger(name='ML', log_file='metrics/model_training.log'):
    """
    Creates and returns a logger configured for model training.

    Parameters:
    - name (str): Name of the logger.
    - log_file (str): Path to the log file.

    Returns:
    - logger (logging.Logger): Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Stream (console) handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Ensure the log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# ---------------------------
# Log Transformer Definition
# ---------------------------
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, y):
        return 10 * np.log1p(y)

    def inverse_transform(self, y):
        return np.expm1(y / 10)

# ---------------------------
# KMeans Clustering for Spatial Segmentation
# ---------------------------   
""" 
def cluster_and_split(merged_df, lat_col='Latitude_Y', lon_col='Longitude_X', n_clusters=12, seed = 42):
    logger.info("Starting KMeans clustering for spatial segmentation.")
    coords = merged_df[[lat_col, lon_col]].dropna()
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    labels = kmeans.fit_predict(coords) + 1
    merged_df = merged_df.copy()
    merged_df.loc[coords.index, 'cluster'] = labels.astype(int)
    return merged_df.dropna(subset=['cluster'])
"""
def cluster_and_split(merged_df, lat_col='Latitude_Y', lon_col='Longitude_X',
    n_clusters=12, seed=42, logger=None):
    # Set up a default logger if none is provided
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    try:
        logger.info("Starting KMeans clustering for spatial segmentation.")
        
        coords = merged_df[[lat_col, lon_col]].dropna()
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
        labels = kmeans.fit_predict(coords) + 1
        
        merged_df = merged_df.copy()
        merged_df.loc[coords.index, 'cluster'] = labels.astype(int)
        
        logger.info("Clustering completed successfully.")
        return merged_df.dropna(subset=['cluster'])

    except Exception as e:
        logger.error(f"An error occurred during clustering: {e}")
        raise

# ---------------------------
# Pipeline Builder
# ---------------------------   
def build_pipeline(model, scaler=None):
    """
    model: estimator (e.g., RandomForestClassifier)
    scaler: a class (e.g., StandardScaler) or None
    """
    if scaler is None:
        scaler_step = FunctionTransformer(lambda x: x)  # Identity transformer
    elif callable(scaler):
        scaler_step = scaler()  # Instantiate the transformer
    else:
        raise ValueError("Scaler must be a callable (like StandardScaler) or None.")

    pipeline = Pipeline([
        ("scaler", scaler_step),
        ("model", model)
    ])
    
    return pipeline

# ---------------------------
# Train-test split Strategy
# ---------------------------

def create_cv_splits(X_train, y_train, groups_train, target, model_name, 
                     cv_strategy="groupkfold", n_splits=5, test_size=0.2, random_state=42):
    
    logger = logging.getLogger(__name__)
    os.makedirs("metrics", exist_ok=True)

    if cv_strategy.lower() == "groupkfold":
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(X_train, y_train, groups=groups_train))

        fold_info_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(len(splits))],
            "Validation_Groups": [
                ", ".join(map(str, groups_train.iloc[val_idx].unique())) 
                for _, val_idx in splits
            ]
        })

        fold_info_df.to_csv(f"metrics/{target.replace('/', '_')}_{model_name}_folds.csv", index=False)

        for i, (_, val_idx) in enumerate(splits):
            val_groups = sorted(groups_train.iloc[val_idx].unique())
            logger.info(f"Fold {i+1} — Validation groups: {val_groups}")

    elif cv_strategy.lower() == "stratifiedshuffle":
        try:
            n_classes = len(np.unique(y_train))
            n_test = int(test_size * len(y_train))
            if n_test < n_classes:
                raise ValueError(f"test_size too small for stratification: {n_test} < {n_classes}")

            sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
            splits = list(sss.split(X_train, y_train))

            fold_info_df = pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(len(splits))],
                "Validation_Indices_Count": [len(val_idx) for _, val_idx in splits],
                "Validation_Class_Distribution": [
                    dict(zip(*np.unique(y_train.iloc[val_idx], return_counts=True)))
                    for _, val_idx in splits
                ]
            })

            fold_info_df.to_csv(f"metrics/{target.replace('/', '_')}_{model_name}_stratified_folds.csv", index=False)

            for i, (_, val_idx) in enumerate(splits):
                val_classes, counts = np.unique(y_train.iloc[val_idx], return_counts=True)
                logger.info(f"Fold {i+1} — Validation class distribution: {dict(zip(val_classes, counts))}")

        except ValueError as e:
            logger.warning(f"StratifiedShuffleSplit failed: {e}")
            logger.warning("Falling back to StratifiedKFold.")

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = list(skf.split(X_train, y_train))

            fold_info_df = pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(len(splits))],
                "Validation_Indices_Count": [len(val_idx) for _, val_idx in splits],
                "Validation_Class_Distribution": [
                    dict(zip(*np.unique(y_train.iloc[val_idx], return_counts=True)))
                    for _, val_idx in splits
                ]
            })

            fold_info_df.to_csv(f"metrics/{target.replace('/', '_')}_{model_name}_stratkfold_fallback.csv", index=False)

            for i, (_, val_idx) in enumerate(splits):
                val_classes, counts = np.unique(y_train.iloc[val_idx], return_counts=True)
                logger.info(f"[Fallback] Fold {i+1} — Validation class distribution: {dict(zip(val_classes, counts))}")

    elif cv_strategy.lower() == "kfold":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(kf.split(X_train))

        fold_info_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(len(splits))],
            "Validation_Indices_Count": [len(val_idx) for _, val_idx in splits]
        })

        fold_info_df.to_csv(f"metrics/{target.replace('/', '_')}_{model_name}_kfold.csv", index=False)

        for i, (_, val_idx) in enumerate(splits):
            logger.info(f"Fold {i+1} — Validation size: {len(val_idx)} samples")

    else:
        raise ValueError("Invalid cv_strategy. Choose from 'groupkfold', 'stratifiedshuffle', or 'kfold'.")

    return splits, fold_info_df

# ---------------------------
# Training Models with Group Logging
# ---------------------------
def train_models_for_target(target, X_train, y_train, X_test, y_test, groups_train, model_pipelines, columns_to_transform = None, 
                            seed = 42,
                            split_strategy='kfold', 
                            enable_hyperparameter_tuning=False, 
                            use_bayes_opt=False, enable_rfe=False, logger = None):
    results = []    
    # ---------------------------
    # Filter out NaNs in y_train
    # ---------------------------
    non_nan_train_mask = y_train.notna()
    X_train = X_train.loc[non_nan_train_mask]
    y_train = y_train.loc[non_nan_train_mask]
    groups_train = groups_train.loc[non_nan_train_mask]

    # ---------------------------
    # Filter out NaNs in y_test
    # ---------------------------
    non_nan_test_mask = y_test.notna()
    X_test = X_test.loc[non_nan_test_mask]
    y_test = y_test.loc[non_nan_test_mask]

    # ---------------------------
    # Log info
    # ---------------------------
    if logger is None:
        logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    logger.info(f"Training for target: {target} — {len(y_train)} train samples, {len(y_test)} test samples.")

    if y_train.empty or y_test.empty:
        logger.warning(f"Skipping {target} — no valid data after filtering NaNs.")
        return

    
    is_log_target = target in columns_to_transform
    log_transformer = LogTransformer()

    for model_name, config in model_pipelines.items():
        logger.info(f"Training {model_name} for {target}")

        model = config["model"]

        # Skip RFE for unsupported models
        use_rfe = enable_rfe
        if isinstance(model, PLSRegression):
            logger.info(f"Skipping RFE for {model_name} (PLSRegression handles its own dimensionality reduction).")
            use_rfe = False
        elif isinstance(model, SVR) and getattr(model, 'kernel', None) != "linear":
            logger.info(f"Skipping RFE for {model_name} (SVR with non-linear kernel not supported by RFE).")
            use_rfe = False

        # Apply log transform to target if needed
        if is_log_target:
            y_train_transformed = log_transformer.transform(y_train)
        else:
            y_train_transformed = y_train

        # Standard pipeline for other models
        steps = [('scaler', RobustScaler())]

        if use_rfe:
            # Use RFECV for automatic feature selection based on cross-validation performance
            rfecv = RFECV(estimator=clone(model), step=1, cv=5, scoring='neg_mean_squared_error')
            steps.append(('feature_selection', rfecv))

        steps.append(('model', model))
        pipeline = Pipeline(steps)

        if split_strategy in ['groupkfold', 'stratifiedshuffle', 'kfold']:
            splits, fold_info_df = create_cv_splits(
                X_train, y_train, groups_train,
                target=target, model_name=model_name,
                cv_strategy=split_strategy,
                random_state=seed
            )
        else:
            raise ValueError(f"Unsupported split strategy: {split_strategy}")

        metrics_path = os.path.join("metrics", f"{target.replace('/', '_')}_{model_name}_metrics.csv")

        # -------------------------------
        # Hyperparameter Tuning
        # -------------------------------
        if enable_hyperparameter_tuning:
            search_class = BayesSearchCV if use_bayes_opt else GridSearchCV
            search_kwargs = {
                "estimator": pipeline,
                "cv": splits,
                "scoring": "neg_mean_squared_error",
                "n_jobs": -1,
                "verbose": 1,
                "return_train_score": True
            }

            if use_bayes_opt:
                search_kwargs["search_spaces"] = config["params"]
                search_kwargs["n_iter"] = 30
                search_kwargs["random_state"] = seed
            else:
                search_kwargs["param_grid"] = config["params"]

            search = search_class(**search_kwargs)

            try:
                search.fit(X_train, y_train_transformed)
            except Exception as e:
                logger.warning(f"Training failed for {model_name} on {target}: {e}")
                continue
            
            best_model = search.best_estimator_
            best_params = search.best_params_

        else:
            try:
                pipeline.fit(X_train, y_train_transformed)
            except Exception as e:
                logger.warning(f"Training failed for {model_name} on {target}: {e}")
                continue
            
            best_model = pipeline
            best_params = "Default (no tuning)"

        # -------------------------------
        # Prediction and Metrics
        # -------------------------------        
        y_pred_transformed = best_model.predict(X_test)

        if is_log_target:
            y_pred = log_transformer.inverse_transform(y_pred_transformed)
        else:
            y_pred = y_pred_transformed

        # Cross-validation on training data
        cv_scores = cross_validate(
            best_model,
            X_train,
            y_train_transformed,
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

        logger.info(
            f"🏁 {model_name} | {target} — CV RMSE: {test_metrics['CV_RMSE_Mean']:.4f}, "
            f"Test RMSE: {test_metrics['Test_RMSE']:.4f}, R²: {test_metrics['Test_R2']:.4f}"
            )


        results.append(test_metrics)

        model_path = os.path.join("final_models", f"{target.replace('/', '_')}_{model_name}.pkl")
        joblib.dump(best_model, model_path)
        logger.info(f"Saved model to {model_path} and metrics to {metrics_path}")

        if enable_hyperparameter_tuning:
            pd.DataFrame(search.cv_results_).assign(
                target=target,
                model=model_name,
                best_params=str(best_params)
            ).to_csv(metrics_path, index=False)
    
    return results