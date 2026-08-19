from yg_eo_soilnet.utils import mlflow_rpiq_score
from yg_eo_soilnet.plot_utils import plot_leaderboard_scatter, create_pred_obs_plot, create_parent_pred_obs
from yg_eo_soilnet.artifacts import (
    ArtifactLayout,
    candidate_artifact_paths,
    log_figure,
    log_json,
    log_table,
)
from yg_eo_soilnet.metrics import (
    cv_rmse_from_search,
    metric_space_for,
    regression_metrics,
)

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.models import infer_signature

import os
import importlib
import logging
import shutil
import tempfile
from typing import Any


# Kept as a module constant because callers outside this file import it. It is now sourced from
# ArtifactLayout so there is one definition of the tree.
EVAL_RESULTS_ARTIFACT_PATH = ArtifactLayout.EVAL_RESULTS


def _eval_results_filename(target: str, model_name: str) -> str:
    """LEGACY per-run name, kept only so readers can resolve pre-rename runs."""
    return ArtifactLayout.eval_results_filename(target, model_name)


def _eval_results_artifact_paths(target: str, model_name: str) -> list[str]:
    """Every path a reader should try for one run's eval CSV, current layout first.

    Three generations coexist in ``mlruns/``: the stable ``eval_results/eval_results.csv`` written
    now, the per-run ``eval_results/eval_results_<target>_<model>.csv`` written before the rename,
    and the same file at the run root from older runs still.
    """
    return candidate_artifact_paths(
        ArtifactLayout.EVAL_RESULTS,
        ArtifactLayout.EVAL_RESULTS_FILE,
        ArtifactLayout.eval_results_filename(target, model_name),
    )

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
            self._write_json_artifact(
                summary,
                ArtifactLayout.SPLIT_SUMMARY_FILE,
                artifact_path=ArtifactLayout.EVAL_RESULTS,
            )

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
            "static_hidden_dims",
            "head_hidden_dims",
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
        """Thin delegate; the implementation lives in :mod:`yg_eo_soilnet.artifacts`."""
        log_table(df, filename, artifact_path)

    def _write_json_artifact(self, payload: dict, filename: str, artifact_path: str):
        """Thin delegate; the implementation lives in :mod:`yg_eo_soilnet.artifacts`."""
        log_json(payload, filename, artifact_path)

    def _log_metric_dict(self, metrics: dict, prefix: str = ""):
        for metric_name, metric_value in metrics.items():
            if metric_value is None:
                continue
            try:
                mlflow.log_metric(f"{prefix}{metric_name}", float(metric_value))
            except (TypeError, ValueError):
                continue

    def _log_lightning_pred_obs_artifact(
        self,
        evaluation_df: pd.DataFrame,
        target: str,
        model_name: str,
        artifact_path: str | None = None,
    ) -> bool:
        if evaluation_df.empty or target not in evaluation_df.columns or "prediction" not in evaluation_df.columns:
            return False

        plot_eval_df = pd.DataFrame(
            {
                "target": evaluation_df[target],
                "prediction": evaluation_df["prediction"],
            }
        )

        # create_pred_obs_plot follows MLflow's custom-artifact contract: it SAVES into the
        # directory it is handed and returns {name: path}, rather than returning a Figure the way
        # every other plotter in plot_utils does. Hence the temp dir here instead of log_figure.
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = create_pred_obs_plot(plot_eval_df, builtin_metrics={}, artifacts_dir=tmpdir)
            if not artifacts:
                return False
            # One stable name, so this plot lines up across runs in the compare view. It used to be
            # suffixed with the target and model, which made every run's copy a different path.
            # Multi-target runs separate by DIRECTORY instead - see plots_path - so two targets
            # still cannot overwrite each other.
            destination = artifact_path or ArtifactLayout.PLOTS
            for index, (_artifact_key, source_path) in enumerate(artifacts.items()):
                _stem, source_ext = os.path.splitext(os.path.basename(source_path))
                stable_stem, stable_ext = os.path.splitext(ArtifactLayout.PRED_OBS_FILE)
                suffix = "" if index == 0 else f"_{index}"
                renamed_filename = f"{stable_stem}{suffix}{stable_ext or source_ext}"
                renamed_path = os.path.join(tmpdir, renamed_filename)
                if source_path != renamed_path:
                    os.replace(source_path, renamed_path)
                mlflow.log_artifact(renamed_path, artifact_path=destination)
        return True

    def _log_checkpoint(self, best_model_path: str) -> None:
        """Log the best checkpoint under a STABLE name.

        Lightning names its checkpoints ``epoch=NN-step=MMM.ckpt``, so two runs of the same model on
        the same target still produce different artifact paths and MLflow's compare view finds
        nothing in common. Copying to ``best.ckpt`` fixes that; the original name is not lost - it
        goes to the ``checkpoint_filename`` tag and into meta/run_summary.json, where it is still
        greppable but no longer part of the path.
        """
        original_name = os.path.basename(best_model_path)
        mlflow.set_tags({"checkpoint_filename": original_name})

        if not os.path.isfile(best_model_path):
            # The trainer reports a path that Lightning may never have written - a run with
            # checkpointing disabled, or one that stopped before the first save. Skipping keeps the
            # rest of the run's artifacts and its summary, which a raise here would discard.
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            stable_path = os.path.join(tmpdir, ArtifactLayout.CHECKPOINT_FILE)
            shutil.copyfile(best_model_path, stable_path)
            mlflow.log_artifact(stable_path, artifact_path=ArtifactLayout.CHECKPOINTS)

    def _log_lightning_serialized_model(
        self,
        model,
        model_name: str,
        target: str = "",
        bundle=None,
        best_model_path: str | None = None,
        config=None,
        input_example=None,
    ) -> bool:
        """Log the model as an mlflow.pyfunc so it can be loaded and served.

        NOT mlflow.pytorch.log_model: MLflow 3 defaults that to serialization_format="pt2", which
        traces model.forward from an example input. These models consume a dict batch of ragged,
        date-stamped sequences, so no example can trace them - the previous run failed with
        "If serialization_format is set to 'pt2', then input_example is required" and the logged
        model was left in status FAILED. A pyfunc sidesteps tracing entirely and, unlike a raw
        checkpoint, arrives with the preprocessing needed to consume raw data.
        """
        if model is None:
            return False

        import mlflow.pyfunc
        from mlflow.models import infer_signature

        from yg_eo_soilnet.serving.lightning_pyfunc import (
            SoilSequencePyfunc,
            build_input_example,
            serving_requirements,
            stage_serving_package,
        )

        # A caller may hand in a prepared example - relog.py rebuilds a model from a checkpoint and
        # has no bundle to derive one from. Everything else about the logging path is identical, so
        # a recovered model is packaged exactly like a freshly trained one.
        sequence_bundle = getattr(getattr(bundle, "datamodule", None), "sequence_bundle", None)
        if input_example is None and sequence_bundle is not None:
            input_example = build_input_example(model, sequence_bundle, n_rows=3)

        signature = None
        if input_example is not None:
            predictions = SoilSequencePyfunc(model).predict(None, input_example)
            signature = infer_signature(input_example, predictions)

        entry_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "serving", "_pyfunc_entry.py"
        )

        registered_name = (
            ArtifactLayout.logged_model_name(target, model_name)
            if bool(getattr(config, "MLFLOW_REGISTER_MODELS", True))
            else None
        )

        with tempfile.TemporaryDirectory() as staging:
            torch_model_path = self._save_nested_torch_model(model, best_model_path, staging)

            model_info = mlflow.pyfunc.log_model(
                # `name`, not the deprecated `artifact_path`. The old value was "models/{model}"
                # with no target in it, so on a multi-target run every target overwrote the slot.
                name=ArtifactLayout.logged_model_name(target, model_name),
                # A PATH, not an object. Handing mlflow the live object CloudPickles the whole
                # graph - that is what produced a 15 MB python_model.pkl whose weights could not be
                # read without unpickling it, and which mlflow warns can execute arbitrary code on
                # load. Models-from-code stores this script and loads the nested model below.
                python_model=entry_script,
                artifacts={"torch_model": torch_model_path},
                signature=signature,
                input_example=input_example,
                code_paths=[stage_serving_package(os.path.join(staging, "code"))],
                pip_requirements=serving_requirements(),
                registered_model_name=registered_name,
            )

        self._registered_version = getattr(model_info, "registered_model_version", None)
        return True

    @staticmethod
    def _save_nested_torch_model(model, best_model_path: str | None, staging: str) -> str:
        """A real ``mlflow.pytorch`` model directory, nested inside the pyfunc's artifacts.

        Nesting rather than logging a second top-level model keeps one deployable entry per run and
        stores the weights once, while still making the network recognisable: the nested MLmodel
        carries the ``pytorch`` flavor and ``mlflow.pytorch.load_model`` works against it.

        ``serialization_format="pickle"`` is forced, not chosen: ``pt2`` traces ``forward`` from a
        tensor example and this architecture consumes a dict batch of ragged sequences. So this copy
        is an object pickle. The run's ``checkpoints/best.ckpt`` remains the safe,
        ``weights_only``-loadable one.
        """
        import mlflow.pytorch

        # The checkpoint Lightning selected is the early-stopped BEST epoch; the live module may
        # hold a later, worse state. Restore before saving so the served weights are the best ones.
        publishable = model
        if best_model_path and os.path.isfile(best_model_path):
            try:
                publishable = type(model).load_from_checkpoint(best_model_path, map_location="cpu")
                publishable.eval()
            except Exception:
                # A checkpoint from a different architecture revision should not lose the model
                # entirely - the live module is still a valid, if not-best, thing to publish.
                publishable = model

        path = os.path.join(staging, "torch_model")
        mlflow.pytorch.save_model(publishable, path, serialization_format="pickle")
        return path

    @staticmethod
    def _tag_model_logging(target: str, model_name: str, logged: bool, error: str | None) -> None:
        """Make the outcome of model logging visible on the run itself.

        A failure here is not fatal - a fitted model should not be thrown away over a packaging
        problem - but it must not be silent either, because a run with no servable model looks
        exactly like a healthy one in the MLflow run list.
        """
        tags = {"model_logged": str(bool(logged)).lower()}
        if error:
            # Truncated: MLflow rejects very long tag values, and the full text is in the run
            # summary. This is the pointer, not the record.
            tags["model_logging_error"] = str(error)[:450]

        try:
            mlflow.set_tags(tags)
        except Exception:
            # Tagging is a diagnostic; failing to tag must not take the run down with it.
            pass

        if not logged:
            logging.getLogger(__name__).warning(
                "Model logging FAILED for %s_%s: the run has NO servable model and nothing was "
                "registered. %s",
                target,
                model_name,
                error or "no error recorded",
            )

    def _promote_champion(self, target: str, model_name: str, metrics: dict, version) -> dict:
        """Move the champion alias onto this version when it beats the incumbent.

        Reads rmse_test, the metric that means the same thing for both families and is in the
        target's original units - which is what makes the comparison meaningful at all. Shared by
        both families so a sklearn model and a Lightning model are promoted on identical evidence.
        """
        if version is None:
            return {"promoted": False, "reason": "the model was not registered"}

        from yg_eo_soilnet.tracking import CHAMPION_METRIC, promote_if_better

        try:
            return promote_if_better(
                ArtifactLayout.logged_model_name(target, model_name),
                version,
                metrics.get(CHAMPION_METRIC),
            )
        except Exception as exc:
            # A registry that will not take the alias must not lose a finished training run.
            return {"promoted": False, "reason": f"{type(exc).__name__}: {exc}"}

    def _log_shap_artifacts(
        self,
        config,
        target: str,
        model_name: str,
        backend: str,
        payload: dict,
    ) -> dict:
        """Build and log the SHAP artifacts for one child run, or explain why it did not.

        This method is the SHAP off-switch, and it is deliberately the ONLY place that reaches for
        the explain package. Three properties it has to keep:

        * ``EXPLAIN_ENABLED: false`` returns before importing anything - ``shap`` pulls in numba and
          is slow to import, and a run that asked for no explainability must not pay for it, nor
          risk tripping the ``filterwarnings = ["error"]`` pytest setting on a warning it emits;
        * the import is local to this function for the same reason;
        * a failure here is recorded, not raised, unless ``EXPLAIN_FAIL_ON_ERROR`` - losing a
          finished training run because an explainer choked is a bad trade.

        Returns the dict that goes into the run summary's ``explain`` key.
        """
        if not bool(getattr(config, "EXPLAIN_ENABLED", True)):
            return {"enabled": False, "reason": "EXPLAIN_ENABLED is false"}

        # Precedence: naming a model in EXPLAIN_MODELS is an explicit request, so it overrides the
        # denylist. Without that rule the two settings could contradict each other and the allowlist
        # would silently do nothing, which is the worse failure - the user asked for the expensive
        # explanation and would get no plot and no explanation of why.
        selected_models = list(getattr(config, "EXPLAIN_MODELS", None) or [])
        skipped_models = list(getattr(config, "EXPLAIN_SKIP_MODELS", None) or [])

        if selected_models and model_name not in selected_models:
            return {"enabled": False, "reason": f"{model_name} is not in EXPLAIN_MODELS"}

        if model_name in skipped_models and model_name not in selected_models:
            return {
                "enabled": True,
                "logged": False,
                "skipped": True,
                "reason": (
                    f"{model_name} is in EXPLAIN_SKIP_MODELS. It is excluded by name rather than by "
                    "EXPLAIN_MAX_EVALS because that budget counts model evaluations and cannot see "
                    "that one costs this model far more than a tree. Add it to EXPLAIN_MODELS to "
                    "explain it anyway."
                ),
            }

        try:
            from yg_eo_soilnet.explain import build_shap_results, log_shap_artifacts

            results = build_shap_results(config=config, backend=backend, **payload)
            if not results:
                return {"enabled": True, "logged": False, "reason": "explainer produced no values"}

            written = log_shap_artifacts(
                results,
                max_display=int(getattr(config, "EXPLAIN_MAX_DISPLAY", 25)),
            )
            return {"enabled": True, "logged": True, **written}
        except Exception as exc:
            # A budget skip is a decision, not a failure: the run is healthy and the explanation was
            # declined on cost. It is reported as "skipped" so it is not mistaken for a crash, and
            # it is NOT escalated by EXPLAIN_FAIL_ON_ERROR, which is there for genuine errors.
            if type(exc).__name__ == "ExplainBudgetExceeded":
                return {"enabled": True, "logged": False, "skipped": True, "reason": str(exc)}
            if bool(getattr(config, "EXPLAIN_FAIL_ON_ERROR", False)):
                raise
            return {"enabled": True, "logged": False, "error": f"{type(exc).__name__}: {exc}"}

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

            # Under PLOTS, not the run root. These used to be logged with no artifact_path at all,
            # so a CV plot landed beside leaderboard.csv while its own CSV went to cv_results/.
            log_figure(
                fig,
                f"{plot_func.__name__}.png",
                ArtifactLayout.PLOTS,
            )

    def _log_cv_results(self, cv_results_df, target, model_name, param_names):
        """Save cv_results as an artifact for later inspection.

        The per-step ``train_score_*`` / ``test_score_*`` metrics that used to be emitted here when
        ``param_names`` was a string are gone. That branch was unreachable - ModelTrainer always
        passes a list - and it carried a second, unguarded ``neg_*`` sign flip, which is exactly the
        pattern yg_eo_soilnet.metrics exists to keep in one place. The full grid is in the CSV.
        """
        log_table(
            cv_results_df,
            ArtifactLayout.CV_RESULTS_FILE,
            ArtifactLayout.CV,
        )

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
                                         name=ArtifactLayout.logged_model_name(target, model_name),
                                         # Enters the registry under the same stable name, so
                                         # versions accumulate per target+model and deployment can
                                         # reference models:/<name>/<version> rather than a
                                         # run-scoped URI.
                                         registered_model_name=(
                                             ArtifactLayout.logged_model_name(target, model_name)
                                             if bool(getattr(config, "MLFLOW_REGISTER_MODELS", True))
                                             else None
                                         ),
                                         input_example=X_test[:5],
                                         skops_trusted_types=[
                                             "numpy.dtype",
                                             "xgboost.core.Booster",
                                             "xgboost.sklearn.XGBRegressor",
                                             # TabICL ships its own preprocessing estimators inside
                                             # the fitted regressor; skops refuses to persist any of
                                             # them unless they are named here.
                                             "random.Random",
                                             "tabicl._sklearn.preprocessing.CustomStandardScaler",
                                             "tabicl._sklearn.preprocessing.EnsembleGenerator",
                                             "tabicl._sklearn.preprocessing.OutlierRemover",
                                             "tabicl._sklearn.preprocessing.PreprocessingPipeline",
                                             "tabicl._sklearn.preprocessing.TransformToNumerical",
                                             "tabicl._sklearn.preprocessing.UniqueFeatureFilter",
                                             "tabicl._sklearn.regressor.TabICLRegressor",
                                             ])
            # --- CV results as artifact ---
            self._log_cv_results(cv_results, target, model_name, param_names)

            # --- Evaluation frame, in ORIGINAL target units ---
            eval_df = pd.concat([X_test, y_test], axis=1)
            eval_df["prediction"] = best_model.predict(X_test)  # ensure predictions column exists

            # --- Metrics ---
            # The unified set, computed from the same prediction frame the Lightning path uses, so
            # the two families are directly comparable. `mean_test_score` and `mean_train_score` are
            # deliberately NOT logged any more: they meant a positive CV RMSE here and a negative
            # -test_loss on the Lightning side, under one name and on one leaderboard axis.
            metrics = regression_metrics(eval_df[target], eval_df["prediction"])
            metrics.update(cv_rmse_from_search(cv_results, search.best_index_))
            metrics["r2_train_fit"] = float(r2_score(y_train, best_model.predict(X_train)))
            self._log_metric_dict(metrics)

            self._log_table_artifact(
                eval_df,
                filename=ArtifactLayout.EVAL_RESULTS_FILE,
                artifact_path=ArtifactLayout.EVAL_RESULTS,
            )

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

            # --- SHAP ---
            explain_summary = self._log_shap_artifacts(
                config=config,
                target=target,
                model_name=model_name,
                backend="sklearn",
                payload={
                    "fitted_estimator": best_model,
                    "X_train": X_train,
                    "X_test": X_test,
                    "target": target,
                },
            )

            # --- Run summary, the same shape the Lightning path writes ---
            self._write_json_artifact(
                {
                    "target": target,
                    "model_name": model_name,
                    "framework": "sklearn",
                    "run_name": run_name,
                    "metrics": metrics,
                    "metric_space": metric_space_for(metrics),
                    "best_params": {str(key): str(value) for key, value in search.best_params_.items()},
                    "logged_model_name": ArtifactLayout.logged_model_name(target, model_name),
                    "logged_model_uri": getattr(model_info, "model_uri", None),
                    "registered_model_version": getattr(model_info, "registered_model_version", None),
                    "champion": self._promote_champion(
                        target, model_name, metrics,
                        getattr(model_info, "registered_model_version", None),
                    ),
                    "explain": explain_summary,
                },
                ArtifactLayout.RUN_SUMMARY_FILE,
                artifact_path=ArtifactLayout.META,
            )

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

        # The unified metric set, computed from the prediction frame - which predict_step has already
        # run through inverse_transform_targets, so these are in ORIGINAL target units. The
        # train/val/test_loss and *_r2 values already in `metrics` come from the LightningModule and
        # live in STANDARDIZED LOG1P space; both are kept, and metric_space_for() records which is
        # which in the run summary so nobody has to infer it from the magnitudes.
        #
        # What used to be here instead: mean_test_score = -test_loss, mean_train_score = -val_loss.
        # Those negations made the Lightning rows of the leaderboard negative while the sklearn rows
        # under the same names were positive. Nothing is negated any more.
        if evaluation_df is not None and target in evaluation_df.columns and "prediction" in evaluation_df.columns:
            metrics.update(regression_metrics(evaluation_df[target], evaluation_df["prediction"]))

        if evaluation_df is not None:
            target_eval_frames = list(
                self._iter_lightning_target_eval_frames(evaluation_df, target=target, model_name=model_name)
            )
            per_target_r2_scores = []
            for target_frame, target_name, _prediction_column in target_eval_frames:
                if target_name not in target_frame.columns or "prediction" not in target_frame.columns:
                    continue
                per_target = regression_metrics(
                    target_frame[target_name],
                    target_frame["prediction"],
                    suffix=f"_{target_name}",
                )
                metrics.update(per_target)
                r2_key = f"r2_test_{target_name}"
                if r2_key in per_target:
                    per_target_r2_scores.append(per_target[r2_key])

            # On a multi-target run the trainer hands each child an empty metric dict and a
            # single-target frame, so the loop above is what produces that child's numbers. When the
            # frame really is multi-target, the run-level r2_test is the mean across targets.
            if per_target_r2_scores and "r2_test" not in metrics:
                metrics["r2_test"] = float(np.mean(per_target_r2_scores))

        self._log_metric_dict(metrics)

        if best_model_path:
            self._log_checkpoint(best_model_path)

        if evaluation_df is not None:
            self._log_table_artifact(
                evaluation_df,
                filename=ArtifactLayout.EVAL_RESULTS_FILE,
                artifact_path=ArtifactLayout.EVAL_RESULTS,
            )

        self._log_split_summary(bundle, evaluation_df, run_target)

        pred_obs_logged = False
        if evaluation_df is not None:
            try:
                target_eval_frames = list(
                    self._iter_lightning_target_eval_frames(evaluation_df, target=target, model_name=model_name)
                )
                # One target writes plots/pred_obs.png, so it lines up with every other run in the
                # compare view. Several targets nest under plots/<target>/, because they would
                # otherwise write the same leaf and silently overwrite one another.
                nest_by_target = len(target_eval_frames) > 1

                if not target_eval_frames:
                    pred_obs_logged = self._log_lightning_pred_obs_artifact(
                        evaluation_df,
                        target=target,
                        model_name=model_name,
                        artifact_path=ArtifactLayout.plots_path(),
                    )

                for target_frame, target_name, _prediction_column in target_eval_frames:
                    pred_obs_logged = (
                        self._log_lightning_pred_obs_artifact(
                            target_frame,
                            target=target_name,
                            model_name=model_name,
                            artifact_path=ArtifactLayout.plots_path(
                                target_name if nest_by_target else None
                            ),
                        )
                        or pred_obs_logged
                    )
            except Exception:
                pred_obs_logged = False

        model_logged = False
        model_logging_error = None
        # Not a bare swallow any more: FAIL_ON_MODEL_ERROR re-raises, matching the sklearn trainer's
        # policy. Silently recording "serialized_model_logged": false in a JSON file meant a run
        # could look complete while having saved no usable model at all.
        try:
            # The bundle, not the eval frame. The eval frame is the model's OUTPUT side - test
            # features next to predictions - and feeding it as an input example is what produced
            # the pt2 tracing failure. The bundle is what the model actually consumes, so the
            # example and signature are derived from it.
            model_logged = self._log_lightning_serialized_model(
                model=model,
                model_name=model_name,
                target=run_target,
                bundle=bundle,
                best_model_path=best_model_path,
                config=config,
            )
        except Exception as exc:
            if bool(getattr(config, "FAIL_ON_MODEL_ERROR", False)):
                raise
            model_logged = False
            model_logging_error = f"{type(exc).__name__}: {exc}"

        # Tagged on the run, not just recorded in an artifact. A model-logging failure used to leave
        # the run looking clean - it finished, its metrics were there, and the only evidence sat
        # inside meta/run_summary.json - so a run that saved NO servable model was indistinguishable
        # from one that did until somebody thought to open the JSON. The success case is tagged for
        # the same reason: the absence of a tag is not evidence.
        self._tag_model_logging(run_target, model_name, model_logged, model_logging_error)

        explain_summary = self._log_shap_artifacts(
            config=config,
            target=run_target,
            model_name=model_name,
            backend="lightning",
            payload={"model": model, "bundle": bundle, "target": run_target},
        )

        summary = {
            "target": target,
            "model_name": model_name,
            "framework": "lightning",
            "metrics": metrics,
            "metric_space": metric_space_for(metrics),
            "best_model_path": best_model_path,
            # The checkpoint is logged as checkpoints/best.ckpt so runs stay comparable; this is
            # where its Lightning-assigned epoch/step name survives.
            "checkpoint_filename": os.path.basename(best_model_path) if best_model_path else None,
            "logged_model_name": ArtifactLayout.logged_model_name(run_target, model_name),
            "registered_model_version": getattr(self, "_registered_version", None),
            "champion": self._promote_champion(
                run_target, model_name, metrics, getattr(self, "_registered_version", None)
            ),
            "pred_obs_artifact_logged": pred_obs_logged,
            "serialized_model_logged": model_logged,
            "serialized_model_logging_error": model_logging_error,
            "explain": explain_summary,
        }
        summary["run_name"] = run_name
        summary["resolved_target"] = run_target
        self._write_json_artifact(
            summary,
            ArtifactLayout.RUN_SUMMARY_FILE,
            artifact_path=ArtifactLayout.META,
        )

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
                "framework": run_data.tags.get("framework", "sklearn"),
            }
            # Collect all logged metrics (rmse_test, r2_test, mae_test, ...)
            row.update(run_data.metrics)
            rows.append(self._backfill_legacy_metrics(row))

        return pd.DataFrame(rows)

    @staticmethod
    def _backfill_legacy_metrics(row: dict) -> dict:
        """Give a pre-unification run the current metric names so it still plots.

        Runs recorded before yg_eo_soilnet.metrics existed logged `mean_test_score`, which meant a
        POSITIVE cv RMSE on a sklearn run and a NEGATIVE -test_loss on a Lightning one. abs() is
        what makes those two comparable again on a single axis; it is a best effort for display
        only, which is why the row is tagged rather than silently patched.

        The Lightning value is still a loss in standardized log1p space, not an RMSE in target
        units, so a legacy Lightning row is comparable to other legacy Lightning rows and not much
        else. Re-running the model is the only way to get a real number.
        """
        if "rmse_test" in row:
            return row

        legacy_score = row.get("mean_test_score")
        if legacy_score is None:
            return row

        row["rmse_test"] = abs(float(legacy_score))
        row["legacy_metrics"] = True
        if "r2_test" not in row and "r2_score" in row:
            row["r2_test"] = row["r2_score"]
        return row

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
        # rmse_test against r2_test: both are in original target units for both families, so an
        # XGBoost point and a soil_cnn point on this axis now mean the same thing. The old pair was
        # (mean_test_score, r2_score), which put positive sklearn RMSEs and negative Lightning
        # losses on one axis in three different units.
        log_figure(
            plot_leaderboard_scatter(leaderboard_df, metric_x="rmse_test", metric_y="r2_test"),
            "leaderboard.png",
            ArtifactLayout.LEADERBOARD_PLOTS,
        )

        eval_dfs = self._collect_eval_dfs(parent_run_id)
        log_figure(
            create_parent_pred_obs(eval_dfs),
            "pred_error_plot.png",
            ArtifactLayout.LEADERBOARD_PLOTS,
        )
