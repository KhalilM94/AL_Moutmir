from yg_eo_soilnet.utils import mlflow_rpiq_score
from yg_eo_soilnet.plot_utils import plot_leaderboard_scatter, create_pred_obs_plot, create_parent_pred_obs

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error

import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.models import infer_signature

import os
import importlib
import json
import tempfile
import re


EVAL_RESULTS_ARTIFACT_PATH = "eval_results"


def _eval_results_filename(target: str, model_name: str) -> str:
    return f"eval_results_{target}_{model_name}.csv"


def _eval_results_artifact_paths(target: str, model_name: str) -> list[str]:
    filename = _eval_results_filename(target, model_name)
    return [f"{EVAL_RESULTS_ARTIFACT_PATH}/{filename}", filename]

class ChildRunLogger:
    def __init__(self):
        pass

    @staticmethod
    def _numeric_summary(series: pd.Series | None) -> dict[str, float]:
        if series is None:
            return {}

        values = pd.to_numeric(series, errors="coerce")
        values = values[np.isfinite(values.to_numpy(dtype=float, copy=False))]
        if values.empty:
            return {}

        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
            "count": float(values.shape[0]),
        }

    def _log_split_summary(self, bundle, evaluation_df: pd.DataFrame | None, target: str) -> None:
        if bundle is None:
            return

        datamodule = getattr(bundle, "datamodule", None)
        if datamodule is None:
            return

        summary: dict[str, Any] = {}
        metric_prefixes = {
            "train": getattr(datamodule, "y_train_frame_", None),
            "val": getattr(datamodule, "y_val_frame_", None),
            "test": getattr(datamodule, "y_test_frame_", None),
        }

        for split_name, frame in metric_prefixes.items():
            if frame is None or getattr(frame, "empty", True):
                continue
            target_columns = [column for column in frame.columns if column != "target_name"]
            if not target_columns:
                continue
            column_name = target_columns[0]
            stats = self._numeric_summary(frame[column_name])
            if not stats:
                continue
            summary[split_name] = {"target_column": column_name, **stats}
            for stat_name, value in stats.items():
                mlflow.log_metric(f"{split_name}_{stat_name}", value)

        if evaluation_df is not None and target in evaluation_df.columns and "prediction" in evaluation_df.columns:
            residuals = pd.to_numeric(evaluation_df[target], errors="coerce") - pd.to_numeric(
                evaluation_df["prediction"], errors="coerce"
            )
            residuals = residuals[np.isfinite(residuals.to_numpy(dtype=float, copy=False))]
            if not residuals.empty:
                residual_summary = {
                    "mean_error": float(residuals.mean()),
                    "mae": float(residuals.abs().mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(residuals.to_numpy(dtype=float, copy=False))))),
                }
                summary["evaluation"] = residual_summary
                for stat_name, value in residual_summary.items():
                    mlflow.log_metric(f"eval_{stat_name}", value)

        if summary:
            self._write_json_artifact(summary, f"split_summary_{target}.json", artifact_path="eval_results")

    @staticmethod
    def _resolve_lightning_run_target_label(evaluation_df: pd.DataFrame | None, fallback_target: str) -> str:
        if evaluation_df is not None and "target_names" in evaluation_df.columns and not evaluation_df["target_names"].empty:
            encoded = str(evaluation_df["target_names"].iloc[0]).strip()
            if encoded:
                return encoded
        if evaluation_df is not None and "target_name" in evaluation_df.columns and not evaluation_df["target_name"].empty:
            encoded = str(evaluation_df["target_name"].iloc[0]).strip()
            if encoded:
                return encoded
        return fallback_target

    @staticmethod
    def _resolve_lightning_target_names(evaluation_df: pd.DataFrame, fallback_target: str) -> list[str]:
        if "target_names" in evaluation_df.columns and not evaluation_df["target_names"].empty:
            encoded = str(evaluation_df["target_names"].iloc[0]).strip()
            if encoded:
                names = [name for name in encoded.split("__") if name]
                if names:
                    return names

        if fallback_target:
            return [fallback_target]

        target_columns = [
            column
            for column in evaluation_df.columns
            if column not in {"prediction", "target_name", "target_names", "model_name"}
            and not column.startswith("prediction_")
        ]
        return target_columns[:1] if target_columns else []

    def _iter_lightning_target_eval_frames(self, evaluation_df: pd.DataFrame, target: str, model_name: str):
        target_names = self._resolve_lightning_target_names(evaluation_df, target)
        has_multi_prediction_columns = any(column.startswith("prediction_") for column in evaluation_df.columns)

        if "prediction" in evaluation_df.columns and not has_multi_prediction_columns:
            frame = evaluation_df.copy()
            frame["target_name"] = target_names[0] if target_names else target
            frame["model_name"] = model_name
            yield frame, frame["target_name"].iloc[0], "prediction"
            return

        for target_name in target_names:
            prediction_column = f"prediction_{target_name}"
            if prediction_column not in evaluation_df.columns:
                prediction_candidates = [
                    column for column in evaluation_df.columns if column.startswith("prediction_")
                ]
                if len(prediction_candidates) == 1:
                    prediction_column = prediction_candidates[0]
                else:
                    prediction_column = None
            if prediction_column is None or target_name not in evaluation_df.columns:
                continue

            frame = evaluation_df.copy()
            frame["prediction"] = frame[prediction_column]
            frame["target_name"] = target_name
            frame["model_name"] = model_name
            yield frame, target_name, prediction_column

    @staticmethod
    def _block_feature_dims(block):
        """(in_features, out_features) for a Linear or a Sequential wrapping several.

        Takes the FIRST linear's in_features and the LAST linear's out_features, so an MLP head
        reports the shape of the whole block rather than logging None.
        """
        if hasattr(block, "in_features") and hasattr(block, "out_features"):
            return getattr(block, "in_features", None), getattr(block, "out_features", None)

        linears = []
        if hasattr(block, "__iter__"):
            linears = [
                layer for layer in block
                if hasattr(layer, "in_features") and hasattr(layer, "out_features")
            ]
        if not linears:
            return None, None
        return getattr(linears[0], "in_features", None), getattr(linears[-1], "out_features", None)

    def _collect_lightning_architecture_params(self, model, bundle=None) -> dict[str, object]:
        params: dict[str, object] = {}

        if model is None:
            return params

        scalar_keys = [
            "static_dim",
            "target_dim",
            "hidden_dim",
            "static_hidden_dim",
            "head_num_layers",
            "head_hidden_dim",
            "use_layer_norm",
            "fusion_input_dim",
            "temporal_hidden_dim",
            "edge_attr_dim",
            "learning_rate",
            "temporal_enabled",
            "temporal_steps",
            "temporal_lstm_hidden_dim",
            "temporal_lstm_num_layers",
            "temporal_lstm_dropout",
            "temporal_lstm_bidirectional",
            "temporal_pooling",
            "spatial_graph_enabled",
        ]
        for key in scalar_keys:
            value = getattr(model, key, None)
            if value is not None:
                params[f"architecture.{key}"] = value

        graph_blocks = getattr(model, "graph_blocks", None)
        if graph_blocks is not None:
            params["architecture.num_graph_layers"] = len(graph_blocks)

        for attribute_name, param_name in (("static_encoder", "static_encoder"), ("output_head", "output_head")):
            block = getattr(model, attribute_name, None)
            if block is None:
                continue
            in_features, out_features = self._block_feature_dims(block)
            if in_features is not None or out_features is not None:
                params[f"architecture.{param_name}_in_features"] = in_features
                params[f"architecture.{param_name}_out_features"] = out_features

        temporal_encoders = getattr(model, "temporal_encoders", None)
        if temporal_encoders is not None:
            for modality_name, encoder in temporal_encoders.items():
                prefix = f"architecture.temporal_encoder.{modality_name}"
                params[f"{prefix}.input_size"] = getattr(encoder, "input_size", None)
                params[f"{prefix}.hidden_size"] = getattr(encoder, "hidden_size", None)
                params[f"{prefix}.num_layers"] = getattr(encoder, "num_layers", None)
                params[f"{prefix}.bidirectional"] = getattr(encoder, "bidirectional", None)

        modality_dims = getattr(model, "modality_dims", None)
        if isinstance(modality_dims, dict):
            for modality_name, dim in modality_dims.items():
                params[f"architecture.modality_dim.{modality_name}"] = dim

        if bundle is not None:
            datamodule = getattr(bundle, "datamodule", None)
            if datamodule is not None:
                for key in ("static_dim", "target_dim", "temporal_steps", "edge_attr_dim", "feature_dim"):
                    value = getattr(datamodule, key, None)
                    if value is not None:
                        params[f"architecture.datamodule.{key}"] = value
                datamodule_modalities = getattr(datamodule, "modality_dims", None)
                if isinstance(datamodule_modalities, dict):
                    for modality_name, dim in datamodule_modalities.items():
                        params[f"architecture.datamodule.modality_dim.{modality_name}"] = dim

            trainer_kwargs = getattr(bundle, "trainer_kwargs", None)
            if isinstance(trainer_kwargs, dict):
                params["architecture.training.max_epochs"] = trainer_kwargs.get("max_epochs")
                params["architecture.training.accelerator"] = trainer_kwargs.get("accelerator")
                params["architecture.training.devices"] = trainer_kwargs.get("devices")

        return {key: value for key, value in params.items() if value is not None}

    def _log_table_artifact(self, df: pd.DataFrame, filename: str, artifact_path: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, filename)
            df.to_csv(path, index=False)
            mlflow.log_artifact(path, artifact_path=artifact_path)

    def _write_json_artifact(self, payload: dict, filename: str, artifact_path: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, filename)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=str)
            mlflow.log_artifact(path, artifact_path=artifact_path)

    def _log_metric_dict(self, metrics: dict, prefix: str = ""):
        for metric_name, metric_value in metrics.items():
            if metric_value is None:
                continue
            try:
                mlflow.log_metric(f"{prefix}{metric_name}", float(metric_value))
            except (TypeError, ValueError):
                continue

    def _log_lightning_pred_obs_artifact(self, evaluation_df: pd.DataFrame, target: str, model_name: str) -> bool:
        if evaluation_df.empty or target not in evaluation_df.columns or "prediction" not in evaluation_df.columns:
            return False

        plot_eval_df = pd.DataFrame(
            {
                "target": evaluation_df[target],
                "prediction": evaluation_df["prediction"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = create_pred_obs_plot(plot_eval_df, builtin_metrics={}, artifacts_dir=tmpdir)
            if not artifacts:
                return False
            safe_target = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(target)).strip("_") or "target"
            safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_name)).strip("_") or "model"
            for artifact_key, artifact_path in artifacts.items():
                artifact_stem, artifact_ext = os.path.splitext(os.path.basename(artifact_path))
                renamed_filename = f"{artifact_stem}_{safe_target}_{safe_model}{artifact_ext}"
                renamed_path = os.path.join(tmpdir, renamed_filename)
                if artifact_path != renamed_path:
                    os.replace(artifact_path, renamed_path)
                mlflow.log_artifact(renamed_path, artifact_path="eval_plots")
        return True

    def _log_lightning_serialized_model(self, model, model_name: str, input_example: pd.DataFrame | None = None) -> bool:
        if model is None:
            return False

        log_kwargs = {
            "artifact_path": f"models/{model_name}",
            "pytorch_model": model,
        }
        if input_example is not None and not input_example.empty:
            log_kwargs["input_example"] = input_example.head(5)

        mlflow.pytorch.log_model(**log_kwargs)
        return True

    def _log_plots(self, plot_functions: dict, target: str, model_name: str):
        """
        Helper to log plots as MLflow artifacts.
    
        Args:
            plot_functions (dict):
                {function: {"args": [...], "kwargs": {...}}}
                - "args" is a list of positional arguments.
                - "kwargs" is a dict of keyword arguments.
        """
        if not plot_functions:
            return
    
        for func_path, call_args in plot_functions.items():
            # dynamically import function
            module_name, func_name = func_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            plot_func = getattr(module, func_name)

            args = call_args.get("args")
            kwargs = call_args.get("kwargs", {})
    
            fig = plot_func(*args, **kwargs)
    
            with tempfile.TemporaryDirectory() as tmpdir:
                plot_path = os.path.join(
                    tmpdir, f"{plot_func.__name__}_{target}_{model_name}.png"
                )
                fig.savefig(plot_path, bbox_inches="tight")
                plt.close(fig)
                mlflow.log_artifact(plot_path)

    def _log_cv_results(self, cv_results_df, target, model_name, param_names):
        """Save cv_results as artifact for later inspection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, f"cv_results_{target}_{model_name}.csv")
            cv_results_df.to_csv(path, index=False)
            mlflow.log_artifact(path, artifact_path="cv_results")
            if isinstance(param_names, str):
                for i, row in cv_results_df.iterrows():
                    train_score = -row['mean_train_score']
                    test_score = -row['mean_test_score']
                    # Log parameter as a metric step
                    mlflow.log_metric(f"train_score_{param_names.rsplit('__', 1)[-1]}", train_score, step=i)
                    mlflow.log_metric(f"test_score_{param_names.rsplit('__', 1)[-1]}", test_score, step=i)

    def log_child_run(
        self,
        config,
        search,
        cv_results,
        best_model,
        X_train,
        y_train,
        X_test,
        y_test,
        target,
        param_names,
        model_name,
        plot_functions,
        extra_params=None,
    ):
        """Log one child run with params, metrics, plots, and model."""
        run_name = f"{target}_{model_name}"
        with mlflow.start_run(run_name=run_name, nested=True):
            # --- Tags ---
            mlflow.set_tags({
                "target": target,
                "model_name": model_name
            })
            # --- Params ---
            mlflow.log_params({
                "cell_size_m": config.CLUSTERING_STRATEGY.get('params', {}).get('cell_size_m', None) if config.ENABLE_CLUSTERING else None,
                "n_clusters": config.CLUSTERING_STRATEGY.get('params', {}).get('n_clusters', None) if config.ENABLE_CLUSTERING else None,
            })
            if extra_params:
                mlflow.log_params(extra_params)
            mlflow.log_params(search.best_params_)

            # --- Model ---
            signature = infer_signature(X_test, best_model.predict(X_test))
            model_info = mlflow.sklearn.log_model(sk_model=best_model,  # type: ignore
                                         signature=signature,
                                         name = f"{target}_{model_name}",
                                         input_example=X_test[:5],
                                         skops_trusted_types=[
                                             "numpy.dtype", 
                                             "xgboost.core.Booster", 
                                             "xgboost.sklearn.XGBRegressor"
                                             ])
            # --- CV results as artifact ---
            self._log_cv_results(cv_results, target, model_name, param_names)
            # --- CV metrics (best index) ---
            r2_train = r2_score(y_train, best_model.predict(X_train))
            mlflow.log_metric("r2_train", r2_train)
            mlflow.log_metric("mean_train_score", -cv_results["mean_train_score"][search.best_index_])
            mlflow.log_metric("mean_test_score", -cv_results["mean_test_score"][search.best_index_])

            # --- Evaluation metrics and plots ---
            eval_df = pd.concat([X_test, y_test], axis=1)
            eval_df["prediction"] = best_model.predict(X_test)  # ensure predictions column exists
            
            # Save evaluation DataFrame
            with tempfile.TemporaryDirectory() as tmpdir:
                eval_path = os.path.join(tmpdir, _eval_results_filename(target, model_name))
                eval_df.to_csv(eval_path, index=False)
                mlflow.log_artifact(eval_path, artifact_path=EVAL_RESULTS_ARTIFACT_PATH)

    
            mlflow.models.evaluate(
                model_info.model_uri,
                data=pd.concat([X_test, y_test], axis=1),
                targets=target,
                model_type="regressor",
                evaluators=["regressor"],
                extra_metrics=[mlflow_rpiq_score],
                custom_artifacts=[create_pred_obs_plot]
            )

            # --- Test Plots ---
            self._log_plots(plot_functions, target, model_name)

    def log_lightning_child_run(
        self,
        config,
        target: str,
        model_name: str,
        evaluation_df: pd.DataFrame | None = None,
        validation_metrics: dict | None = None,
        test_metrics: dict | None = None,
        best_model_path: str | None = None,
        extra_params: dict | None = None,
        plot_functions: dict | None = None,
        bundle=None,
        eval_df: pd.DataFrame | None = None,
        model=None,
    ):
        """Log Lightning outputs inside an already-started child run.

        Supports both the lightweight direct form used by the trainer and the
        richer bundle-based form for future compatibility.
        """
        if evaluation_df is None:
            evaluation_df = eval_df

        run_target = self._resolve_lightning_run_target_label(evaluation_df, target)
        run_name = f"{run_target}_{model_name}"

        mlflow.set_tags({
            "mlflow.runName": run_name,
            "target": run_target,
            "model_name": model_name,
            "framework": "lightning",
        })

        params = {
            "LIGHTNING_BATCH_SIZE": getattr(config, "LIGHTNING_BATCH_SIZE", None),
            "LIGHTNING_VAL_SIZE": getattr(config, "LIGHTNING_VAL_SIZE", None),
            "LIGHTNING_MAX_EPOCHS": getattr(config, "LIGHTNING_MAX_EPOCHS", None),
            "LIGHTNING_ACCELERATOR": getattr(config, "LIGHTNING_ACCELERATOR", None),
            "LIGHTNING_DEVICES": getattr(config, "LIGHTNING_DEVICES", None),
            "LIGHTNING_PRECISION": getattr(config, "LIGHTNING_PRECISION", None),
        }

        mlflow.log_params({key: value for key, value in params.items() if value is not None})

        if bundle is not None:
            bundle_params = {
                "modeltype": bundle.registry_entry.get("modeltype"),
                "batch_size": getattr(bundle.datamodule, "batch_size", None),
                "val_size": getattr(bundle.datamodule, "val_size", None),
                "max_epochs": bundle.trainer_kwargs.get("max_epochs"),
                "accelerator": bundle.trainer_kwargs.get("accelerator"),
                "devices": bundle.trainer_kwargs.get("devices"),
                "precision": bundle.trainer_kwargs.get("precision"),
            }
            bundle_params.update(bundle.registry_entry.get("init_args", {}))
            for key, value in bundle.registry_entry.get("datamodule_init_args", {}).items():
                bundle_params[f"datamodule.{key}"] = value
            for key, value in bundle.registry_entry.get("trainer_args", {}).items():
                bundle_params[f"trainer.{key}"] = value
            mlflow.log_params({key: value for key, value in bundle_params.items() if value is not None})

        if extra_params:
            mlflow.log_params(extra_params)

        architecture_params = self._collect_lightning_architecture_params(model=model, bundle=bundle)
        if architecture_params:
            mlflow.log_params(architecture_params)

        metrics = {}
        metrics.update(validation_metrics or {})
        metrics.update(test_metrics or {})

        if evaluation_df is not None and target in evaluation_df.columns and "prediction" in evaluation_df.columns:
            try:
                y_true = evaluation_df[target]
                y_pred = evaluation_df["prediction"]
                metrics.setdefault("r2_score", r2_score(y_true, y_pred))
                metrics.setdefault("rmse_test", root_mean_squared_error(y_true, y_pred))
            except Exception:
                pass

        if "test_loss" in metrics and "mean_test_score" not in metrics:
            metrics["mean_test_score"] = -float(metrics["test_loss"])
        if "rmse_test" in metrics and "mean_test_score" not in metrics:
            metrics["mean_test_score"] = -float(metrics["rmse_test"])
        if "val_loss" in metrics and "mean_train_score" not in metrics:
            metrics["mean_train_score"] = -float(metrics["val_loss"])
        if "r2_score" in metrics and "r2_test" not in metrics:
            metrics["r2_test"] = metrics["r2_score"]

        if evaluation_df is not None:
            target_eval_frames = list(self._iter_lightning_target_eval_frames(evaluation_df, target=target, model_name=model_name))
            if target_eval_frames:
                per_target_r2_scores = []
                for target_frame, target_name, _prediction_column in target_eval_frames:
                    if target_name in target_frame.columns and "prediction" in target_frame.columns:
                        try:
                            score = r2_score(target_frame[target_name], target_frame["prediction"])
                            metrics[f"r2_score_{target_name}"] = score
                            per_target_r2_scores.append(score)
                        except Exception:
                            pass
                if per_target_r2_scores and "r2_score" not in metrics:
                    metrics["r2_score"] = float(np.mean(per_target_r2_scores))

        self._log_metric_dict(metrics)

        if best_model_path:
            mlflow.log_artifact(best_model_path, artifact_path="checkpoints")

        if evaluation_df is not None:
            self._log_table_artifact(
                evaluation_df,
                filename=_eval_results_filename(run_target, model_name),
                artifact_path=EVAL_RESULTS_ARTIFACT_PATH,
            )

        self._log_split_summary(bundle, evaluation_df, run_target)

        pred_obs_logged = False
        if evaluation_df is not None:
            try:
                pred_obs_logged = self._log_lightning_pred_obs_artifact(
                    evaluation_df,
                    target=target,
                    model_name=model_name,
                )
                target_eval_frames = list(
                    self._iter_lightning_target_eval_frames(evaluation_df, target=target, model_name=model_name)
                )
                for target_frame, target_name, _prediction_column in target_eval_frames:
                    pred_obs_logged = (
                        self._log_lightning_pred_obs_artifact(target_frame, target=target_name, model_name=model_name)
                        or pred_obs_logged
                    )
            except Exception:
                pred_obs_logged = False

        model_logged = False
        model_logging_error = None
        try:
            input_example = None
            if evaluation_df is not None:
                target_columns = self._resolve_lightning_target_names(evaluation_df, target)
                feature_columns = [
                    column
                    for column in evaluation_df.columns
                    if column not in set(target_columns) | {"prediction", "target_name", "target_names", "model_name"}
                    and not column.startswith("prediction_")
                ]
                if feature_columns:
                    input_example = evaluation_df[feature_columns]
            model_logged = self._log_lightning_serialized_model(model=model, model_name=model_name, input_example=input_example)
        except Exception as exc:
            model_logged = False
            model_logging_error = str(exc)

        summary = {
            "target": target,
            "model_name": model_name,
            "framework": "lightning",
            "metrics": metrics,
            "best_model_path": best_model_path,
            "pred_obs_artifact_logged": pred_obs_logged,
            "serialized_model_logged": model_logged,
            "serialized_model_logging_error": model_logging_error,
        }
        summary["run_name"] = run_name
        summary["resolved_target"] = run_target
        self._write_json_artifact(summary, f"lightning_run_summary_{run_target}_{model_name}.json", artifact_path="lightning_metadata")

class ParentRunLogger:
    def __init__(self):
        pass

    @staticmethod
    def _resolve_lightning_target_names(evaluation_df: pd.DataFrame, fallback_target: str) -> list[str]:
        return ChildRunLogger._resolve_lightning_target_names(evaluation_df, fallback_target)

    def _collect_leaderboard(self,parent_run_id: str):
        """
        Collects metrics from all child runs of a given parent run.
        Returns a DataFrame with leaderboard info.
        """
        client = mlflow.tracking.MlflowClient()  # type: ignore
        child_runs = client.search_runs(
            experiment_ids=[client.get_run(parent_run_id).info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'"
        )

        rows = []
        for run in child_runs:
            run_data = run.data
            target = run_data.tags.get("target")
            model_name = run_data.tags.get("model_name")
            if not target or not model_name:
                continue

            row = {
                "run_id": run.info.run_id,
                "target": target,
                "model": model_name,
            }
            # Collect all logged metrics (like rmse_test, r2_test, etc.)
            row.update(run_data.metrics)

            rows.append(row)

        return pd.DataFrame(rows)

    def _collect_eval_dfs(self,parent_run_id: str):
        client = mlflow.tracking.MlflowClient() # type: ignore
        parent_run = client.get_run(parent_run_id)
        exp_id = parent_run.info.experiment_id

        child_runs = client.search_runs(
            experiment_ids=[exp_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'"
        )

        eval_dfs = []
        for run in child_runs:
            run_id = run.info.run_id
            target = run.data.tags.get("target")
            model_name = run.data.tags.get("model_name")
            if not target or not model_name:
                continue

            # Download the optional eval-results artifact from the child run.
            try:
                local_path = None
                for artifact_path in _eval_results_artifact_paths(target, model_name):
                    try:
                        local_path = mlflow.artifacts.download_artifacts( # type: ignore
                            run_id=run_id,
                            artifact_path=artifact_path,
                        )
                        break
                    except Exception:
                        continue

                if local_path is None:
                    raise FileNotFoundError(
                        f"Optional eval CSV not found for {target}-{model_name}; tried: {', '.join(_eval_results_artifact_paths(target, model_name))}"
                    )

                df = pd.read_csv(local_path)

                target_names = self._resolve_lightning_target_names(df, target)
                if "prediction" in df.columns and not any(column.startswith("prediction_") for column in df.columns):
                    df["target_name"] = target_names[0] if target_names else target
                    df["model_name"] = model_name
                    eval_dfs.append(df)
                else:
                    for target_name in target_names:
                        prediction_column = f"prediction_{target_name}"
                        if prediction_column not in df.columns:
                            prediction_column = next(
                                (column for column in df.columns if column.startswith("prediction_")),
                                None,
                            )
                        if prediction_column is None or target_name not in df.columns:
                            continue
                        target_df = df.copy()
                        target_df["prediction"] = target_df[prediction_column]
                        target_df["target_name"] = target_name
                        target_df["model_name"] = model_name
                        eval_dfs.append(target_df)
            except Exception as e:
                print(f"⚠️ Optional eval CSV missing for {target}-{model_name}; continuing without it: {e}")

        return eval_dfs


    def log_parent_summary(self, parent_run_id: str, trainer):
        """
        Collect results from child runs and log a summary to the parent run.
        Designed to be called at the end of training in the main script.

        Args:
            experiment_name (str): Name of the MLflow experiment.
            parent_run_id (str): The active parent MLflow run ID.
        """
        mlflow.set_tags({
            "DATA_FILE": trainer.config.DATA_FILE,
            "STATIC_FEATURES_FILE": getattr(trainer.config, "STATIC_FEATURES_FILE", trainer.config.DATA_FILE),
            "TARGETS_FILE": getattr(trainer.config, "TARGETS_FILE", trainer.config.DATA_FILE),
            "ENABLE_CLUSTERING": trainer.config.ENABLE_CLUSTERING,
            "CLUSTERING_STRATEGY": trainer.config.CLUSTERING_STRATEGY if trainer.config.ENABLE_CLUSTERING else None,
            "SPLIT_STRATEGY": trainer.config.SPLIT_STRATEGY
        })
        mlflow.log_params({
            "DATA_FOLDER": trainer.config.DATA_FOLDER,
            "DATA_FILE": trainer.config.DATA_FILE,
            "STATIC_FEATURES_FILE": getattr(trainer.config, "STATIC_FEATURES_FILE", trainer.config.DATA_FILE),
            "TARGETS_FILE": getattr(trainer.config, "TARGETS_FILE", trainer.config.DATA_FILE),
            "RANDOM_SEED": trainer.config.RANDOM_SEED,
            "TARGET_COLUMNS": trainer.config.TARGET_COLUMNS,
            "COLUMNS_TO_TRANSFORM": [column for column in trainer.config.COLUMNS_TO_TRANSFORM
                                     if column in trainer.config.TARGET_COLUMNS],
            "CLUSTERING_STRATEGY": trainer.config.CLUSTERING_STRATEGY.get('class_path').rsplit('.', 1)[1] if trainer.config.ENABLE_CLUSTERING else None,
            "cell_size_m": trainer.config.CLUSTERING_STRATEGY.get('params', {}).get('cell_size_m', None) if trainer.config.ENABLE_CLUSTERING else None,
            "n_clusters": trainer.config.CLUSTERING_STRATEGY.get('params', {}).get('n_clusters', None) if trainer.config.ENABLE_CLUSTERING else None,
        })

        # 1. Fetch child runs
        leaderboard_df = self._collect_leaderboard(parent_run_id)

        # Log as artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            leaderboard_path = os.path.join(tmpdir, "leaderboard.csv")
            leaderboard_df.to_csv(leaderboard_path, index=False)
            mlflow.log_artifact(leaderboard_path)

        # --- Plots ---
        # Make leaderboard plot
        fig = plot_leaderboard_scatter(leaderboard_df, metric_x="mean_test_score", metric_y="r2_score")
        with tempfile.TemporaryDirectory() as tmpdir:
            fig_path = os.path.join(tmpdir, "leaderboard.png")
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(fig_path, artifact_path="leaderboard_plots")

        eval_dfs = self._collect_eval_dfs(parent_run_id)
        fig = create_parent_pred_obs(eval_dfs)
        if fig is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                fig_path = os.path.join(tmpdir, "pred_error_plot.png")
                fig.savefig(fig_path, bbox_inches="tight")
                mlflow.log_artifact(fig_path, artifact_path="leaderboard_plots")
                plt.close(fig)
