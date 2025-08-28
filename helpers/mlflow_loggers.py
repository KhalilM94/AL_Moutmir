from .misc_utils import mlflow_rpiq_score
from .plot_utils import plot_leaderboard_scatter, create_pred_obs_plot, create_parent_pred_obs

import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import os
import importlib
import tempfile

class ChildRunLogger:
    def __init__(self):
        pass

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
        search,
        cv_results,
        best_model,
        X_test,
        y_test,
        target,
        param_names,
        model_name,
        plot_functions,
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
            mlflow.log_params(search.best_params_)
            # --- Model ---
            signature = infer_signature(X_test, best_model.predict(X_test))
            model_info = mlflow.sklearn.log_model(sk_model=best_model,  # type: ignore
                                         signature=signature,
                                         name = f"{target}_{model_name}",
                                         input_example=X_test[:5])
            # --- CV results as artifact ---
            self._log_cv_results(cv_results, target, model_name, param_names)
            # --- CV metrics (best index) ---
            mlflow.log_metric("mean_train_score", -cv_results["mean_train_score"][search.best_index_])
            mlflow.log_metric("mean_test_score", -cv_results["mean_test_score"][search.best_index_])

            # --- Evaluation metrics and plots ---
            eval_df = pd.concat([X_test, y_test], axis=1)
            eval_df["prediction"] = best_model.predict(X_test)  # ensure predictions column exists
            
            # Save evaluation DataFrame
            with tempfile.TemporaryDirectory() as tmpdir:
                eval_path = os.path.join(tmpdir, f"eval_results_{target}_{model_name}.csv")
                eval_df.to_csv(eval_path, index=False)
                mlflow.log_artifact(eval_path, artifact_path="eval_results")

    
            mlflow.evaluate(
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

class ParentRunLogger:
    def __init__(self):
        pass
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

            row = {
                "run_id": run.info.run_id,
                "target": run_data.tags.get("target"),
                "model": run_data.tags.get("model_name"),
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

            # Download eval_df.csv from artifacts
            try:
                local_path = mlflow.artifacts.download_artifacts( # type: ignore
                    run_id=run_id,
                    artifact_path=f"eval_results/eval_results_{target}_{model_name}.csv"  # adjust if different
                )
                df = pd.read_csv(local_path)
                df["target_name"] = target
                df["model_name"] = model_name
                eval_dfs.append(df)
            except Exception as e:
                print(f"⚠️ Could not load eval_df for {target}-{model_name}: {e}")

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
            "ENABLE_CLUSTERING": trainer.config.ENABLE_CLUSTERING,
            "CLUSTERING_STRATEGY": trainer.config.CLUSTERING_STRATEGY if trainer.config.ENABLE_CLUSTERING else None,
            "SPLIT_STRATEGY": trainer.config.SPLIT_STRATEGY
        })
        mlflow.log_params({
            "DATA_FOLDER": trainer.config.DATA_FOLDER,
            "DATA_FILE": trainer.config.DATA_FILE,
            "RANDOM_SEED": trainer.config.RANDOM_SEED,
            "TARGET_COLUMNS": trainer.config.TARGET_COLUMNS,
            "COLUMNS_TO_TRANSFORM": [column for column in trainer.config.COLUMNS_TO_TRANSFORM
                                     if column in trainer.config.TARGET_COLUMNS],
            "CLUSTERING_STRATEGY": trainer.config.CLUSTERING_STRATEGY.get('class_path').rsplit('.', 1)[1] if trainer.config.ENABLE_CLUSTERING else None
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

        with tempfile.TemporaryDirectory() as tmpdir:
            fig_path = os.path.join(tmpdir, "pred_error_plot.png")
            fig.savefig(fig_path, bbox_inches="tight")
            mlflow.log_artifact(fig_path, artifact_path="leaderboard_plots")
            plt.close(fig)
