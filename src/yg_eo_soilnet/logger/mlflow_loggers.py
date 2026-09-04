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
from yg_eo_soilnet.predictions_export import (
    combine as combine_point_predictions,
    duplicate_report,
    export_enabled_for,
    point_id_column,
    point_prediction_frame,
    summarize as summarize_point_predictions,
)
from yg_eo_soilnet.targets import join_target_names, split_target_names
from yg_eo_soilnet.tracking import log_params_once, run_owner_tags, start_child_run
from yg_eo_soilnet.uncertainty.intervals import describe as describe_interval
from yg_eo_soilnet.uncertainty import (
    attach_uncertainty_columns,
    interval_columns,
    is_prediction_column,
    log_uncertainty_artifacts,
    sigma_column,
    uncertainty_metrics,
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
from typing import Any, Mapping


# Kept as a module constant because callers outside this file import it. It is now sourced from
# ArtifactLayout so there is one definition of the tree.
EVAL_RESULTS_ARTIFACT_PATH = ArtifactLayout.EVAL_RESULTS

# Tag marking an ensemble member's run. Members are children of the model run, and the leaderboard
# prefers a run's grandchildren to the run itself, so without this the model's row would be replaced
# by one row per member. Defined here as well as in the trainer because this module is the reader.
MEMBER_RUN_KIND = "ensemble_member"


def _uncertainty_alpha(config) -> float:
    """The miscoverage the configured interval actually implies.

    Not simply UNCERTAINTY_ALPHA. A `sigma` band claims what k implies under a Normal - 0.683 at
    k=1 - so grading it against 0.95 would report a perfectly good ±1σ band as under-covering by
    0.27. This is the single funnel for the value, so every metric that compares against a nominal
    level gets the right one.
    """
    from yg_eo_soilnet.uncertainty.intervals import effective_alpha

    return effective_alpha(
        getattr(config, "UNCERTAINTY_INTERVAL_METHOD", "conformal"),
        alpha=float(getattr(config, "UNCERTAINTY_ALPHA", 0.05)),
        k=float(getattr(config, "UNCERTAINTY_INTERVAL_K", 1.0)),
    )


def _scoring_runs(runs):
    """The child runs that carry a score, with the ensemble members filtered out.

    Both readers below walk a model run's children and prefer them to the model run itself - which
    is right for the per-target evaluation runs a joint fit produces, and wrong for ensemble
    members. Without this filter a single-target model that gained five members would DISAPPEAR
    from the leaderboard and be replaced by five rows carrying the same target and model name and
    no test metrics at all. The `or [child]` fallback makes that silent: an empty list falls back
    to the model run, a list of five members does not.
    """
    return [run for run in runs if run.data.tags.get("run_kind") != MEMBER_RUN_KIND]


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

        self._write_split_summary(
            {
                "train": getattr(datamodule, "y_train_frame_", None),
                "val": getattr(datamodule, "y_val_frame_", None),
                "test": getattr(datamodule, "y_test_frame_", None),
            },
            evaluation_df,
            target,
        )

    @staticmethod
    def _as_target_frame(values):
        """A one-column frame for a Series, the frame itself for a DataFrame.

        The two families genuinely hold the target differently and always will: the Lightning
        datamodule carries every target column at once, while the sklearn trainer fits one target at
        a time and hands over that column alone (``data['y_train'][target]``). Normalizing here
        rather than at the call site means any future caller gets it right for free - assuming the
        DataFrame shape is exactly what broke every sklearn run with
        ``'Series' object has no attribute 'columns'``.
        """
        if isinstance(values, pd.Series):
            return values.to_frame(name=values.name if values.name is not None else "target")
        return values

    def _write_split_summary(
        self,
        split_frames: Mapping[str, "pd.DataFrame | pd.Series | None"],
        evaluation_df: pd.DataFrame | None,
        target: str,
    ) -> None:
        """One eval_results/split_summary.json shape for BOTH families.

        This used to be reachable only from the Lightning path - it early-returned without a
        `bundle.datamodule` - so a sklearn run recorded nothing about what it held out. Now that the
        two families share a split, the artifact they publish about it has to be comparable too.
        The sklearn side has no `val` frame of its own: its fit pool is train ∪ val and it validates
        by k-fold inside that pool, so what it reports under "train" is that whole pool.

        Accepts a Series or a DataFrame per split; see :meth:`_as_target_frame`.
        """
        summary: dict[str, Any] = {}

        for split_name, values in split_frames.items():
            frame = self._as_target_frame(values)
            if frame is None or getattr(frame, "empty", True):
                continue
            target_columns = [column for column in frame.columns if column != "target_name"]
            if not target_columns:
                continue
            # Prefer THIS run's target. The Lightning frames carry every target column, so taking
            # the first one would describe some other target's distribution under this run's name.
            # The sklearn side passes exactly one column, already named for this run's target, so
            # both paths land on the same column and the artifact stays comparable.
            column_name = target if target in target_columns else target_columns[0]
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
    def _resolve_run_target_label(evaluation_df: pd.DataFrame | None, fallback_target: str) -> str:
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
    def _resolve_target_names(evaluation_df: pd.DataFrame, fallback_target: str) -> list[str]:
        if "target_names" in evaluation_df.columns and not evaluation_df["target_names"].empty:
            encoded = str(evaluation_df["target_names"].iloc[0]).strip()
            if encoded:
                names = split_target_names(encoded)
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

    def _iter_target_eval_frames(self, evaluation_df: pd.DataFrame, target: str, model_name: str):
        target_names = self._resolve_target_names(evaluation_df, target)
        # is_prediction_column, not `startswith("prediction_")`. An uncertainty run's frame also
        # carries prediction_std / prediction_lower / prediction_upper, and counting those as
        # per-target prediction columns makes a SINGLE-target frame look multi-target: the branch
        # below then looks for a prediction_<target> that does not exist, finds several candidates,
        # gives up, and yields nothing - so the run logs no rmse_test and no picp_test at all.
        prediction_columns = [
            column for column in evaluation_df.columns if is_prediction_column(column)
        ]
        has_multi_prediction_columns = any(column != "prediction" for column in prediction_columns)

        if "prediction" in evaluation_df.columns and not has_multi_prediction_columns:
            frame = evaluation_df.copy()
            frame["target_name"] = target_names[0] if target_names else target
            frame["model_name"] = model_name
            yield frame, frame["target_name"].iloc[0], "prediction"
            return

        for target_name in target_names:
            prediction_column = f"prediction_{target_name}"
            if prediction_column not in evaluation_df.columns:
                prediction_candidates = list(prediction_columns)
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

    # --- the shared run tree -------------------------------------------------
    # One rule for both families: the shape of the tree follows the TARGET GROUP, not the framework.
    # A group of one is a single flat run. A group of several is a model run - holding the fitted
    # model, its params and its training curves - with one child per target holding that target's
    # evaluation. Before this, sklearn was always flat and Lightning grew a third level as soon as a
    # second target appeared, so the same experiment looked structurally different depending on
    # which family produced it.

    def _log_per_target_runs(self, evaluation_df, target: str, model_name: str, log_one):
        """Call ``log_one(frame, target_name)`` once per target, in the right run.

        A group of one logs into the CURRENT run rather than opening a child: there is nothing to
        fan out, and an extra level would make a single-target run's tree differ from every other
        single-target run's. Returns ``[(target_name, log_one(...)), ...]``.
        """
        frames = []
        if evaluation_df is not None:
            frames = list(self._iter_target_eval_frames(evaluation_df, target=target, model_name=model_name))

        if len(frames) <= 1:
            frame = frames[0][0] if frames else evaluation_df
            name = frames[0][1] if frames else target
            return [(name, log_one(frame, name))]

        results = []
        for frame, target_name, _prediction_column in frames:
            with start_child_run(
                f"{target_name}_{model_name}",
                tags={"target": target_name, "model_name": model_name},
            ):
                results.append((target_name, log_one(frame, target_name)))
        return results

    def _per_target_metrics(
        self, evaluation_df, target: str, model_name: str, alpha: float | None = None
    ) -> dict:
        """Suffixed metrics for every target in the frame, plus their means.

        The means carry the UNSUFFIXED names, which is what makes a joint run comparable with a
        per-target one and what the champion promotion reads - it looks for a scalar `rmse_test`
        and silently promotes nothing when the run has only suffixed keys.
        """
        if evaluation_df is None:
            return {}

        frames = [
            (frame, target_name)
            for frame, target_name, _column in self._iter_target_eval_frames(
                evaluation_df, target=target, model_name=model_name
            )
            if target_name in frame.columns and "prediction" in frame.columns
        ]

        metrics: dict[str, float] = {}
        collected: dict[str, list[float]] = {}
        for frame, target_name in frames:
            # One target keeps the COMPLETE unsuffixed set - mae_test, bias_test, rpd_test and the
            # rest - because there is nothing to disambiguate and every existing reader expects
            # those names. Several targets get suffixed keys plus the means below.
            if len(frames) == 1:
                metrics.update(regression_metrics(frame[target_name], frame["prediction"]))
            per_target = regression_metrics(
                frame[target_name], frame["prediction"], suffix=f"_{target_name}"
            )
            metrics.update(per_target)
            # The interval metrics follow the same suffixing, so a joint run's model run carries
            # picp_test_<target> for every target it fitted rather than only the point metrics.
            if alpha is not None:
                metrics.update(
                    self._uncertainty_metrics(frame, target_name, alpha=alpha)
                    if len(frames) == 1
                    else {
                        f"{key}_{target_name}": value
                        for key, value in self._uncertainty_metrics(
                            frame, target_name, alpha=alpha
                        ).items()
                    }
                )
            for stem in ("r2_test", "rmse_test"):
                value = per_target.get(f"{stem}_{target_name}")
                if value is not None:
                    collected.setdefault(stem, []).append(value)

        # Only r2 and rmse are averaged, and only as a FALLBACK for a run with several targets.
        # `n` would be a count and `bias` a signed quantity, so those stay per-target.
        #
        # A caveat that matters for rmse_test: a mean over targets measured in different units
        # (pH beside g/kg) is not a physical error, and it is dominated by whichever target has the
        # largest scale. It exists to RANK candidates fitted over the SAME group - which is what
        # champion promotion needs, since CHAMPION_METRIC is a scalar and a run carrying only
        # suffixed keys promotes nothing at all. Read the suffixed keys for anything else; r2_test,
        # being unitless, is the one that survives comparison across groups.
        for stem, values in collected.items():
            if values and stem not in metrics:
                metrics[stem] = float(np.mean(values))
        return metrics

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

    def _uncertainty_metrics(self, frame, target_name: str, *, alpha: float) -> dict:
        """The interval metrics for one target, empty when the frame carries no sigma.

        Empty rather than absent-with-a-check at every call site: most frames reaching here come
        from runs with uncertainty disabled, and the caller should not have to know that.
        """
        sigma = sigma_column(frame, target_name)
        if sigma is None or target_name not in frame.columns or "prediction" not in frame.columns:
            return {}

        interval = interval_columns(frame, target_name)
        lower, upper = interval if interval is not None else (None, None)
        return uncertainty_metrics(
            frame[target_name],
            frame["prediction"],
            sigma,
            lower=lower,
            upper=upper,
            alpha=alpha,
        )

    def _log_uncertainty_artifacts(
        self,
        *,
        frame,
        target: str,
        multi_target: bool,
        calibrator=None,
    ) -> dict:
        """Write the ``uncertainty/`` diagnostics, nesting per target only when there are several.

        Same failure policy as the SHAP seam: a diagnostic that cannot be drawn records why and
        leaves the training run standing, because the model and its metrics are the deliverable and
        a plot is not.
        """
        if frame is None or sigma_column(frame, target) is None:
            return {}

        try:
            return log_uncertainty_artifacts(
                frame,
                target,
                calibrator=calibrator,
                artifact_path=ArtifactLayout.uncertainty_path(target if multi_target else None),
            )
        except Exception as exc:  # pragma: no cover - defensive, mirrors _shap_failure_summary
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _log_point_predictions(
        self,
        *,
        config,
        model_name: str,
        target_names: list,
        predict,
        point_ids,
        n_expected: int | None = None,
    ) -> dict:
        """Write this child's ``predictions/point_predictions.csv``, keyed on the point id.

        ``predict`` is a zero-argument callable rather than a model plus data, so the two families
        can hand over completely different inference paths - a fitted sklearn estimator against a
        feature frame, or SoilSequencePredictor against a bundle - without this seam knowing which.
        It is only called once the switch has been checked, so a disabled run pays nothing: the
        full-population pass is the expensive part and it must not happen speculatively.

        Failure policy matches the SHAP seam - record it and keep the training run, unless the
        config asks otherwise. A prediction export is a convenience artifact; the model and its
        metrics are the deliverable.
        """
        if not export_enabled_for(config, model_name):
            return {}

        id_column = point_id_column(config)
        try:
            predictions = predict()
            if predictions is None:
                return {}
            frame = point_prediction_frame(point_ids, predictions, target_names, id_column)
            if n_expected is not None and len(frame) != n_expected:
                raise ValueError(
                    f"Exported {len(frame)} rows for {model_name} but the population has "
                    f"{n_expected}; the ids and the predictions describe different point sets."
                )
            self._log_table_artifact(
                frame,
                filename=ArtifactLayout.POINT_PREDICTIONS_FILE,
                artifact_path=ArtifactLayout.PREDICTIONS,
            )
            return {"n_points": int(len(frame)), "targets": [str(name) for name in target_names]}
        except Exception as exc:
            if bool(getattr(config, "EXPORT_POINT_PREDICTIONS_FAIL_ON_ERROR", False)):
                raise
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _log_lightning_point_predictions(
        self,
        *,
        config,
        model_name: str,
        model,
        bundle,
        full_population_predictions=None,
    ) -> dict:
        """Full-population predictions for a Lightning entry.

        Reuses SoilSequencePredictor rather than adding a second inference path: it installs the
        checkpoint's stored preprocessing state instead of re-fitting scalers on whatever rows it
        was handed, which is the property that makes a prediction over the whole population mean
        the same thing as a prediction over the test split.

        SoilGraphLightningModule's datamodule exposes no equivalent - it carries a graph, not a
        sequence bundle - so a graph entry is skipped with a reason rather than half-supported. It
        is `enabled: false` throughout the shipped registry.
        """
        if not export_enabled_for(config, model_name):
            return {}

        datamodule = getattr(bundle, "datamodule", None) if bundle is not None else None
        sequence_bundle = getattr(datamodule, "sequence_bundle", None)
        if sequence_bundle is None:
            return {
                "skipped": (
                    f"{model_name} has no sequence bundle to predict over - the graph datamodule "
                    "does not support a full-population pass."
                )
            }

        target_names = list(getattr(datamodule, "target_names", []) or [])
        point_ids = list(getattr(sequence_bundle, "point_ids", []) or [])

        def predict():
            if full_population_predictions is not None:
                # An ensemble: the trainer already ran every member and averaged them, because no
                # single model object represents the mean.
                return full_population_predictions
            from yg_eo_soilnet.serving.sequence_predictor import SoilSequencePredictor

            return SoilSequencePredictor(model).predict(sequence_bundle)

        return self._log_point_predictions(
            config=config,
            model_name=model_name,
            target_names=target_names,
            predict=predict,
            point_ids=point_ids,
            n_expected=len(point_ids),
        )

    def _log_pred_obs_artifact(
        self,
        evaluation_df: pd.DataFrame,
        target: str,
        model_name: str,
        artifact_path: str | None = None,
        interval_label: str | None = None,
    ) -> bool:
        """Write plots/pred_obs.png for ONE target, carrying its uncertainty columns.

        Used by both families now. The sklearn path also runs ``mlflow.models.evaluate``, which
        calls ``create_pred_obs_plot`` itself through its custom-artifacts hook - but the evaluator
        hands that hook a frame it builds from the `targets` and `predictions` columns alone, so the
        sigma and interval columns never reach it and the bars silently do not appear. Writing the
        plot here as well is what puts them on the picture, and it incidentally files the sklearn
        plot under the same plots/pred_obs.png path the Lightning one has always used.
        """
        if evaluation_df.empty or target not in evaluation_df.columns or "prediction" not in evaluation_df.columns:
            return False

        plot_eval_df = pd.DataFrame(
            {
                "target": evaluation_df[target],
                "prediction": evaluation_df["prediction"],
            }
        )
        # The uncertainty columns have to travel with the two above, or the plotter finds no
        # interval and silently draws the bar-less version of the picture on an ensemble run.
        # Renamed to the unsuffixed spelling because this frame holds exactly one target.
        sigma = sigma_column(evaluation_df, target)
        if sigma is not None:
            plot_eval_df["prediction_std"] = sigma.to_numpy()
        interval = interval_columns(evaluation_df, target)
        if interval is not None:
            plot_eval_df["prediction_lower"] = interval[0].to_numpy()
            plot_eval_df["prediction_upper"] = interval[1].to_numpy()
        if interval_label:
            # `attrs` rather than a column: it is one string about the whole frame, and a column
            # would end up in the CSV the evaluator writes.
            plot_eval_df.attrs["interval_label"] = interval_label

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

    # --- SHAP: explained ONCE per fitted model ------------------------------
    # An explanation is a property of the MODEL, not of a target. A joint fit therefore explains
    # once, at model-run scope, and each target's slice is written into the run that already holds
    # that target's metrics and plots. It used to be explained inside the per-target fan-out, which
    # bought the same answer N times: on the Lightning side, where the explainer returns every
    # output whatever `target` says, that also nested explain/<target>/ inside EVERY child run -
    # nine artifact sets for three targets, each child's run summary claiming to have explained the
    # other two, and the copies disagreeing with each other because GradientExplainer is stochastic.
    #
    # The seam is split in three so the two halves can happen in different runs:
    #   _shap_gate           - the off-switch, before any import;
    #   _build_shap_results  - the ONE explainer pass, on the model run;
    #   _log_shap_results    - writes explain/ into whatever run is active.
    # _log_shap_artifacts composes them and keeps its old signature: it is what the gate and budget
    # tests call directly, and it is still the only door into the explain package.

    @staticmethod
    def _shap_gate(config, model_name: str) -> dict | None:
        """The SHAP off-switch. ``None`` means go ahead; a dict is the reason not to.

        Runs BEFORE anything is imported: ``shap`` pulls in numba and is slow to import, and a run
        that asked for no explainability must not pay for it, nor risk tripping the
        ``filterwarnings = ["error"]`` pytest setting on a warning it emits.
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

        return None

    @staticmethod
    def _shap_failure_summary(exc: Exception, config) -> dict:
        """What an explainer failure puts in the run summary. Shared by the build and log halves.

        A failure here is recorded, not raised, unless ``EXPLAIN_FAIL_ON_ERROR`` - losing a finished
        training run because an explainer choked is a bad trade.
        """
        # A budget skip is a decision, not a failure: the run is healthy and the explanation was
        # declined on cost. It is reported as "skipped" so it is not mistaken for a crash, and
        # it is NOT escalated by EXPLAIN_FAIL_ON_ERROR, which is there for genuine errors.
        # Compared by NAME so this path never has to import the explain package to raise.
        if type(exc).__name__ == "ExplainBudgetExceeded":
            return {"enabled": True, "logged": False, "skipped": True, "reason": str(exc)}
        if bool(getattr(config, "EXPLAIN_FAIL_ON_ERROR", False)):
            raise exc
        return {"enabled": True, "logged": False, "error": f"{type(exc).__name__}: {exc}"}

    def _build_shap_results(
        self,
        config,
        target: str,
        model_name: str,
        backend: str,
        payload: dict,
    ) -> tuple[list | None, dict]:
        """Explain the fitted model ONCE, for every output it has. Call this on the MODEL run.

        Returns ``(results, summary)``. ``results`` is ``None`` when nothing was computed, and then
        ``summary`` says why - disabled, skipped, over budget, or errored - in the exact shape the
        run summary's ``explain`` key has always had.
        """
        gate = self._shap_gate(config, model_name)
        if gate is not None:
            return None, gate

        try:
            from yg_eo_soilnet.explain import build_shap_results

            results = build_shap_results(config=config, backend=backend, **payload)
        except Exception as exc:
            return None, self._shap_failure_summary(exc, config)

        if not results:
            return None, {"enabled": True, "logged": False, "reason": "explainer produced no values"}
        return results, {"enabled": True}

    def _log_shap_results(self, results, *, config) -> dict:
        """Write ``explain/`` for the results handed in, into the CURRENTLY ACTIVE run."""
        try:
            from yg_eo_soilnet.explain import log_shap_artifacts

            written = log_shap_artifacts(
                results,
                max_display=int(getattr(config, "EXPLAIN_MAX_DISPLAY", 25)),
            )
            return {"enabled": True, "logged": True, **written}
        except Exception as exc:
            return self._shap_failure_summary(exc, config)

    def _log_shap_artifacts(
        self,
        config,
        target: str,
        model_name: str,
        backend: str,
        payload: dict,
    ) -> dict:
        """Explain a model and log every output of it into the current run.

        The whole seam in one call. Both training families now use the two halves separately - they
        build on the model run and log each slice in its target's run - so nothing in src calls
        this; it is kept because it is the smallest thing that exercises the gate, the budget skip
        and the failure summary end to end, which is what tests/test_explain_switch.py and
        tests/test_explain_budget.py do with it. It composes the same units production uses, so
        there is no second copy of that logic to drift.
        """
        results, summary = self._build_shap_results(config, target, model_name, backend, payload)
        if results is None:
            return summary
        return self._log_shap_results(results, config=config)

    @staticmethod
    def _shap_result_for(results, target: str, target_names):
        """This target's slice of an explanation computed for the whole joint model.

        By name first, then by position. A name miss is expected rather than a bug: both explainers
        fall back to synthetic names when the model carries no target names or the wrong number of
        them, so the position in the group is the only thing left tying an output to a target.
        """
        for result in results:
            if str(result.target_name) == str(target):
                return result

        names = [str(name) for name in (target_names or [])]
        if str(target) in names and len(results) == len(names):
            return results[names.index(str(target))]

        return results[0] if len(results) == 1 else None

    def _log_shap_slice(
        self,
        results,
        summary: dict,
        *,
        target: str,
        target_names,
        config,
        is_model_run: bool,
    ) -> dict:
        """Log THIS target's slice of the joint explanation, into the run that holds this target.

        The slice goes to the FLAT explain/ path, which is where a single-target run has always
        written it: the run is now the per-target scope, so there is no second target in this run to
        nest away from - and a joint fit's explain/ then compares directly against a single-target
        fit's, which is what ArtifactLayout's flat paths exist for.

        With nothing to log, the reason belongs to the MODEL, not to this target. On the model run
        itself it is returned in full, as it always was. In a per-target child it becomes a pointer:
        the full reason is recorded once, on the model run, by _record_model_run_explain.
        """
        if not results:
            if is_model_run:
                return summary
            return {
                "enabled": summary.get("enabled", False),
                "logged": False,
                "scope": "model_run",
            }

        result = self._shap_result_for(results, target, target_names)
        if result is None:
            return {
                "enabled": True,
                "logged": False,
                "reason": f"the explainer returned no output named {target}",
            }
        return self._log_shap_results([result], config=config)

    def _record_model_run_explain(self, summary: dict, *, results, target_runs) -> None:
        """On the model run, say once why a joint fit has no explanation anywhere beneath it.

        Call AFTER the fan-out, with what it returned: the child run contexts have closed by then,
        so this lands on the model run. Only when the run actually fanned out - a single-target
        group's model run IS its target run, and _log_shap_slice has already put the full reason in
        its run summary. Tagged as well as written, for the same reason _tag_model_logging tags its
        outcome: the absence of a beeswarm is not by itself evidence of why, and a reader should not
        have to open a JSON to find out.
        """
        if results is not None or len(target_runs or []) <= 1:
            return

        status = "skipped" if summary.get("skipped") else ("error" if summary.get("error") else "not_logged")
        if not summary.get("enabled", False):
            status = "disabled"
        try:
            mlflow.set_tags({"explain_status": status})
        except Exception:
            pass
        self._write_json_artifact(
            summary,
            ArtifactLayout.EXPLAIN_SUMMARY_FILE,
            artifact_path=ArtifactLayout.META,
        )

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
        targets=None,
        export_data=None,
    ):
        """Log one fitted sklearn model inside an ALREADY-STARTED run.

        The caller opens the run - see ModelTrainer - so that the grid search runs inside it and its
        progress and system metrics attach to the right place. This used to open its own run, which
        put the search outside any child run and gave sklearn a different ownership rule from
        Lightning for no reason.

        `targets` is the group this model was fitted over. One target logs everything here; several
        log the model, its params and the CV results here and fan the evaluation out into one child
        run per target.
        """
        target_names = [str(name) for name in (targets or split_target_names(target))] or [str(target)]
        run_name = f"{target}_{model_name}"
        # --- Tags ---
        mlflow.set_tags({
            "mlflow.runName": run_name,
            "target": target,
            "model_name": model_name,
            "framework": "sklearn",
            **run_owner_tags(),
        })
        # --- Params ---
        mlflow.log_params({
            "cell_size_m": config.CLUSTERING_STRATEGY.get('params', {}).get('cell_size_m', None) if config.ENABLE_CLUSTERING else None,
            "n_clusters": config.CLUSTERING_STRATEGY.get('params', {}).get('n_clusters', None) if config.ENABLE_CLUSTERING else None,
        })
        if extra_params:
            mlflow.log_params(extra_params)
        mlflow.log_params(search.best_params_)

        # The ONE prediction pass over the test set. It used to happen twice - once here for the
        # signature and once again when the evaluation frame was built - which is invisible for a
        # tree and another full inference pass for an in-context model.
        #
        # An ensemble is asked for its decomposition rather than its mean, and the mean is read off
        # the result. Calling predict() and then predict_uncertainty() would run every member twice,
        # which is the same cost this comment exists to prevent, multiplied by n_members.
        ensemble = best_model if hasattr(best_model, "predict_uncertainty") else None
        if ensemble is None:
            ensemble_prediction = None
            test_predictions = np.asarray(best_model.predict(X_test))
        else:
            ensemble_prediction = ensemble.predict_uncertainty(X_test)
            test_predictions = ensemble_prediction.mean
            if len(target_names) <= 1:
                test_predictions = test_predictions.reshape(-1)

        # --- Model ---
        # Logged ONCE, under the group's label. A joint model predicts every target in the group,
        # so registering it once per target would put the same multi-output estimator in the
        # registry under several names, each claiming to be about one target.
        logged_model_name = ArtifactLayout.logged_model_name(target, model_name)
        signature = infer_signature(X_test, test_predictions)
        model_info = mlflow.sklearn.log_model(sk_model=best_model,  # type: ignore
                                     signature=signature,
                                     name=logged_model_name,
                                     # Deliberately NOT registered here. The artifact is written
                                     # now because the signature needs test_predictions, but the
                                     # registry entry is created after the metrics exist - see
                                     # _register_sklearn_model below.
                                     registered_model_name=None,
                                     input_example=X_test[:5],
                                     skops_trusted_types=[
                                         "numpy.dtype",
                                         "xgboost.core.Booster",
                                         "xgboost.sklearn.XGBRegressor",
                                         # An uncertainty run logs the ENSEMBLE, not a bare
                                         # pipeline, and skops refuses any type it was not told
                                         # about - including ours. Without these two the model
                                         # logging step raises and the whole run is lost, having
                                         # already paid for n_members fits.
                                         "yg_eo_soilnet.uncertainty.predictors.EnsembleRegressor",
                                         # One entry per interval estimator the ensemble may carry.
                                         # skops refuses any type it was not told about, so a method
                                         # missing from this list fails the model logging step -
                                         # after the run has already paid for every fit.
                                         "yg_eo_soilnet.uncertainty.conformal.ConformalCalibrator",
                                         "yg_eo_soilnet.uncertainty.intervals.GaussianInterval",
                                         "yg_eo_soilnet.uncertainty.intervals.SigmaInterval",
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
        eval_df = self._build_sklearn_evaluation_frame(
            test_predictions, X_test, y_test, target_names, model_name
        )
        if ensemble_prediction is not None:
            # The sigma and interval columns land beside the predictions the builder just wrote, so
            # eval_results.csv carries the estimate and its uncertainty in one row per point.
            attach_uncertainty_columns(
                eval_df, ensemble_prediction, target_names, ensemble.calibrators
            )
            mlflow.log_params(ensemble.describe())

        # --- Metrics ---
        # The unified set, computed from the same prediction frame the Lightning path uses, so
        # the two families are directly comparable. `mean_test_score` and `mean_train_score` are
        # deliberately NOT logged any more: they meant a positive CV RMSE here and a negative
        # -test_loss on the Lightning side, under one name and on one leaderboard axis.
        model_metrics = dict(cv_rmse_from_search(cv_results, search.best_index_))
        # r2_train_fit is NOT computed here any more. It costs a full pass over the TRAINING set,
        # which is the single largest allocation in this function, and it used to sit ahead of every
        # artifact - so an OOM kill during it lost the metrics, the plots and the run summary that
        # were all already computable. It now runs last, after everything durable is written. See
        # the block below _log_per_target_runs.
        model_metrics.update(
            self._per_target_metrics(eval_df, target, model_name, alpha=_uncertainty_alpha(config))
        )
        self._log_metric_dict(model_metrics)

        self._log_table_artifact(
            eval_df,
            filename=ArtifactLayout.EVAL_RESULTS_FILE,
            artifact_path=ArtifactLayout.EVAL_RESULTS,
        )

        # Registration happens HERE, not at log_model above, and the ordering is the point: the
        # model is in the registry only once this run has produced the metrics that justify it.
        # Registering at log_model time meant every run killed partway through minted a READY
        # version backed by a RUNNING run with no rmse_test - four of them accumulated that way
        # before this was noticed, and promote_if_better would have refused all of them anyway.
        registered_version = self._register_sklearn_model(config, model_info, logged_model_name)

        champion = self._promote_champion(
            target, model_name, model_metrics, registered_version,
        )

        # Predictions for EVERY point, not just the holdout. Written on the model run because that
        # is where the fitted model lives; a joint model contributes one column per target from a
        # single pass. For an ensemble `best_model` is the EnsembleRegressor, whose predict()
        # returns the mean - so the exported number is the ensemble's estimate and carries no
        # uncertainty, which is what this file is meant to hold.
        export_summary = {}
        if export_data is not None:
            X_all = export_data["X"]
            export_summary = self._log_point_predictions(
                config=config,
                model_name=model_name,
                target_names=target_names,
                predict=lambda: best_model.predict(X_all),
                # reindex, not a positional slice: X_all and point_ids share the frame's index, and
                # that is the only thing tying a prediction to the point it belongs to.
                point_ids=export_data["point_ids"].reindex(X_all.index).to_numpy(),
                n_expected=len(X_all),
            )

        # --- SHAP, computed ONCE for the fitted model ---
        # On the MODEL run, before the fan-out: best_model is the joint estimator that predicts
        # every target in the group, so explaining it once per target bought N copies of one answer
        # and, for a permutation explainer, N times the EXPLAIN_MAX_EVALS budget. log_one writes
        # each target's slice into that target's own run.
        shap_results, shap_summary = self._build_shap_results(
            config,
            target,
            model_name,
            "sklearn",
            {
                "fitted_estimator": best_model,
                "X_train": X_train,
                "X_test": X_test,
                "target": target,
                "target_names": target_names,
            },
        )

        def log_one(frame, target_name):
            """Everything that is about ONE target, in whichever run holds that target."""
            metrics = regression_metrics(frame[target_name], frame["prediction"])
            metrics.update(
                self._uncertainty_metrics(frame, target_name, alpha=_uncertainty_alpha(config))
            )
            if target_name == target:
                # Single-target run: the CV and train-fit numbers belong here too, since there is
                # no separate model run holding them.
                metrics = {**model_metrics, **metrics}
            else:
                metrics["n_test"] = float(len(frame))
            self._log_metric_dict(metrics)

            if target_name != target:
                self._log_table_artifact(
                    frame,
                    filename=ArtifactLayout.EVAL_RESULTS_FILE,
                    artifact_path=ArtifactLayout.EVAL_RESULTS,
                )

            self._evaluate_sklearn_target(frame, target_name)

            # Our own copy of the pred-vs-obs plot, with the uncertainty bars on it. The evaluator
            # above draws one too, but from a frame it rebuilds out of the target and prediction
            # columns alone - so its copy has no bars however many uncertainty columns the run
            # produced. See _log_pred_obs_artifact.
            # Guarded for the same reason the Lightning path guards its copy: a plotting backend
            # failure is not a reason to lose the split summary and run_summary.json written below
            # it. Unguarded, one bad figure discarded everything after this line.
            try:
                self._log_pred_obs_artifact(
                    frame,
                    target=target_name,
                    model_name=model_name,
                    artifact_path=ArtifactLayout.plots_path(),
                    interval_label=describe_interval(
                        ensemble.calibrators.get(target_name) if ensemble else None
                    ),
                )

                # --- Test Plots ---
                self._log_plots(plot_functions, target_name, model_name)
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    f"Plots failed for {target_name}/{model_name}: {type(exc).__name__}: {exc}"
                )

            # --- Uncertainty diagnostics ---
            uncertainty_summary = self._log_uncertainty_artifacts(
                frame=frame,
                target=target_name,
                multi_target=len(target_names) > 1,
                calibrator=(ensemble.calibrators.get(target_name) if ensemble else None),
            )

            # --- SHAP: this target's slice of the explanation computed above ---
            explain_summary = self._log_shap_slice(
                shap_results,
                shap_summary,
                target=target_name,
                target_names=target_names,
                config=config,
                is_model_run=(target_name == target),
            )

            # The same split_summary.json the Lightning path writes, so the two families' holdouts
            # can be compared directly - which is the point of sharing one split plan.
            self._write_split_summary({"train": y_train, "test": y_test}, None, target_name)

            # --- Run summary, the same shape the Lightning path writes ---
            self._write_json_artifact(
                {
                    "target": target_name,
                    "model_name": model_name,
                    "framework": "sklearn",
                    "run_name": f"{target_name}_{model_name}",
                    "metrics": metrics,
                    "metric_space": metric_space_for(metrics),
                    "best_params": {str(key): str(value) for key, value in search.best_params_.items()},
                    "logged_model_name": logged_model_name,
                    "logged_model_uri": getattr(model_info, "model_uri", None),
                    "registered_model_version": registered_version,
                    "champion": champion,
                    "explain": explain_summary,
                    "uncertainty": uncertainty_summary,
                    "point_predictions": export_summary,
                },
                ArtifactLayout.RUN_SUMMARY_FILE,
                artifact_path=ArtifactLayout.META,
            )
            return explain_summary

        target_runs = self._log_per_target_runs(eval_df, target, model_name, log_one)
        self._record_model_run_explain(shap_summary, results=shap_results, target_runs=target_runs)

        # --- The train-fit diagnostic, LAST ---
        # One number - R2 of the fitted model against its own training split - that costs a full
        # prediction pass over the LARGER of the two splits. Free for a tree; for an in-context
        # model like TabICL at 213 features it is the biggest allocation the run makes, and on
        # 2026-08-29 the kernel's OOM killer took the process here, discarding the metrics, plots
        # and summaries that were all already computable. Two rules follow from that:
        #   * it runs after everything durable is written, so a kill here costs only this number;
        #   * it is guarded, so a MemoryError does not fail a run that is otherwise complete.
        # It is logged as an MLflow metric but is deliberately absent from meta/run_summary.json,
        # which is written above - nothing reads it from there, and computing it earlier just to
        # populate that field would undo the ordering this block exists for.
        self._log_train_fit_metric(config, model_name, best_model, X_train, y_train)

    def _log_train_fit_metric(self, config, model_name, best_model, X_train, y_train) -> None:
        """Log r2_train_fit, unless this model is too expensive to ask. Never raises."""
        if not bool(getattr(config, "LOG_TRAIN_FIT_METRIC", True)):
            return
        # Same shape as EXPLAIN_SKIP_MODELS, and for the same reason: the cost of this pass is a
        # property of the estimator, not of a row or evaluation budget, so it can only be declined
        # by name. An in-context model pays for it in full checkpoint inference.
        if model_name in list(getattr(config, "LOG_TRAIN_FIT_METRIC_SKIP_MODELS", None) or []):
            return
        try:
            # 2-D under a joint fit; r2_score averages the outputs, which is the same convention
            # the per-target r2_test mean uses.
            value = float(r2_score(y_train, best_model.predict(X_train)))
        except Exception as exc:
            logging.getLogger(__name__).warning(
                f"r2_train_fit failed for {model_name}: {type(exc).__name__}: {exc}. "
                "The run is otherwise complete; this is a diagnostic, not a result."
            )
            return
        self._log_metric_dict({"r2_train_fit": value})

    def _register_sklearn_model(self, config, model_info, logged_model_name):
        """Enter an already-logged model into the registry. Returns the version, or None.

        Split out of ``log_model`` so registration happens only after this run has produced its
        metrics - see the caller. The stable name is what lets deployment reference
        ``models:/<name>/<version>`` rather than a run-scoped URI, and is also why versions
        accumulate one per run for a given target+model.

        Follows the same rule the Lightning path has had since b9368a1 - record the outcome on the
        run, escalate only under ``FAIL_ON_MODEL_ERROR`` - but tags it separately from
        :meth:`_tag_model_logging`. That helper means "this run has no servable model", which is a
        different and worse thing than what can go wrong here: by this point log_model has already
        succeeded, so the model IS servable and only its registry entry is missing.
        """
        if not bool(getattr(config, "MLFLOW_REGISTER_MODELS", True)):
            return None
        model_uri = getattr(model_info, "model_uri", None)
        if model_uri is None:
            return None
        try:
            version = mlflow.register_model(model_uri, logged_model_name).version
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            try:
                # Truncated for the same reason _tag_model_logging truncates: MLflow rejects very
                # long tag values. Tagging is a diagnostic and must not take the run down with it.
                mlflow.set_tags({
                    "model_registered": "false",
                    "model_registration_error": error[:450],
                })
            except Exception:
                pass
            if bool(getattr(config, "FAIL_ON_MODEL_ERROR", False)):
                raise
            logging.getLogger(__name__).warning(
                "Registration FAILED for %s: the model is logged and servable at %s, but is NOT "
                "in the registry, so models:/%s/<version> will not resolve to it. %s",
                logged_model_name, model_uri, logged_model_name, error,
            )
            return None
        try:
            mlflow.set_tags({"model_registered": "true"})
        except Exception:
            pass
        return version

    @staticmethod
    def _build_sklearn_evaluation_frame(predictions, X_test, y_test, target_names, model_name):
        """Test features, observed targets and predictions, in ORIGINAL units.

        Takes the predictions rather than the model on purpose: the caller has already computed
        them for the model signature, and asking an in-context estimator for them twice is a real
        cost.

        Deliberately the same column convention the Lightning trainer emits - one
        ``prediction_<target>`` per output when there are several, a plain ``prediction`` when there
        is one - so a single reader can fan either family's frame out per target.
        """
        eval_df = pd.concat([X_test, y_test], axis=1)
        predictions = np.asarray(predictions)
        if predictions.ndim == 1 or predictions.shape[1] == 1:
            eval_df["prediction"] = predictions.reshape(-1)
        else:
            for index in range(predictions.shape[1]):
                name = target_names[index] if index < len(target_names) else str(index)
                eval_df[f"prediction_{name}"] = predictions[:, index]
        encoded = join_target_names(target_names)
        eval_df["target_name"] = encoded
        eval_df["target_names"] = encoded
        eval_df["model_name"] = model_name
        return eval_df

    def _evaluate_sklearn_target(self, frame, target_name: str) -> None:
        """MLflow's own regressor evaluation, over the predictions ALREADY computed.

        Static-dataset form - no `model` argument, `predictions` naming a column that is already
        in the frame. Passing a model URI instead made MLflow reload the model and predict the test
        set a second time, duplicating work `_build_sklearn_evaluation_frame` had just done. For a
        tree that is invisible; for an in-context model it is another full inference pass.

        The frame carries exactly one prediction column here because the caller has already fanned
        it out per target, so the ambiguity that made this multi-output-unsafe is gone.
        """
        if frame is None or target_name not in frame.columns or "prediction" not in frame.columns:
            return
        mlflow.models.evaluate(
            data=frame,
            targets=target_name,
            predictions="prediction",
            model_type="regressor",
            evaluators=["regressor"],
            extra_metrics=[mlflow_rpiq_score],
            custom_artifacts=[create_pred_obs_plot]
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
        calibrators: dict | None = None,
        full_population_predictions=None,
    ):
        """Log Lightning outputs inside an already-started child run.

        Supports both the lightweight direct form used by the trainer and the
        richer bundle-based form for future compatibility.

        `calibrators` is passed only on an ensemble run; the evaluation frame already carries the
        interval columns by then, and these are here so the reliability plot can be drawn against
        the same q the interval was built with.

        `full_population_predictions` is likewise ensemble-only: the mean over every point, which
        only the trainer can compute because no single model object represents a Lightning
        ensemble. A single-model run leaves it None and this method runs the pass itself.
        """
        calibrators = calibrators or {}
        if evaluation_df is None:
            evaluation_df = eval_df

        run_target = self._resolve_run_target_label(evaluation_df, target)
        run_name = f"{run_target}_{model_name}"

        mlflow.set_tags({
            "mlflow.runName": run_name,
            "target": run_target,
            "model_name": model_name,
            "framework": "lightning",
            **run_owner_tags(),
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

        # The model-run metric set. The train/val/test_loss and *_r2 values come from the
        # LightningModule and live in STANDARDIZED LOG1P space; the unified set below is computed
        # from the prediction frame - which predict_step has already run through
        # inverse_transform_targets - and is in ORIGINAL target units. Both are kept, and
        # metric_space_for() records which is which in the run summary so nobody has to infer it
        # from the magnitudes.
        #
        # What used to be here instead: mean_test_score = -test_loss, mean_train_score = -val_loss.
        # Those negations made the Lightning rows of the leaderboard negative while the sklearn rows
        # under the same names were positive. Nothing is negated any more.
        model_metrics = {}
        model_metrics.update(validation_metrics or {})
        model_metrics.update(test_metrics or {})
        model_metrics.update(
            self._per_target_metrics(
                evaluation_df, target, model_name, alpha=_uncertainty_alpha(config)
            )
        )
        self._log_metric_dict(model_metrics)

        for target_name, calibrator in calibrators.items():
            mlflow.log_params(
                {
                    f"{key}{'' if len(calibrators) <= 1 else f'_{target_name}'}": value
                    for key, value in calibrator.to_dict().items()
                }
            )

        # Checkpoint and serialized model live in the model run, ONCE. A joint model predicts every
        # target in the group, so logging it inside the per-target fan-out put the same multi-output
        # checkpoint in the registry under one name per target, each claiming a single target.
        if best_model_path:
            self._log_checkpoint(best_model_path)

        if evaluation_df is not None:
            self._log_table_artifact(
                evaluation_df,
                filename=ArtifactLayout.EVAL_RESULTS_FILE,
                artifact_path=ArtifactLayout.EVAL_RESULTS,
            )

        self._log_split_summary(bundle, evaluation_df, run_target)

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

        registered_version = getattr(self, "_registered_version", None)
        champion = self._promote_champion(run_target, model_name, model_metrics, registered_version)

        export_summary = self._log_lightning_point_predictions(
            config=config,
            model_name=model_name,
            model=model,
            bundle=bundle,
            full_population_predictions=full_population_predictions,
        )

        # --- SHAP, computed ONCE for the fitted model ---
        # The same rule the sklearn path follows, and the one this family needed most: the Lightning
        # explainer returns EVERY output whatever `target` says, so running it inside the fan-out
        # wrote explain/<target>/ for all N targets inside each of the N child runs - N^2 artifact
        # sets, each run's summary claiming to have explained the others, and the copies disagreeing
        # because GradientExplainer is stochastic. _resolve_target_names is the same function the
        # fan-out uses to decide which children exist, so the ordering here matches theirs.
        group_target_names = (
            self._resolve_target_names(evaluation_df, run_target)
            if evaluation_df is not None
            else [run_target]
        )
        shap_results, shap_summary = self._build_shap_results(
            config,
            run_target,
            model_name,
            "lightning",
            {"model": model, "bundle": bundle, "target": run_target},
        )

        def log_one(frame, target_name):
            """Everything that is about ONE target, in whichever run holds that target."""
            is_model_run = target_name == run_target
            metrics = dict(model_metrics) if is_model_run else {}
            if frame is not None and target_name in frame.columns and "prediction" in frame.columns:
                metrics.update(regression_metrics(frame[target_name], frame["prediction"]))
                metrics.update(
                    self._uncertainty_metrics(
                        frame, target_name, alpha=_uncertainty_alpha(config)
                    )
                )

            if not is_model_run:
                self._log_metric_dict(metrics)
                self._log_table_artifact(
                    frame,
                    filename=ArtifactLayout.EVAL_RESULTS_FILE,
                    artifact_path=ArtifactLayout.EVAL_RESULTS,
                )
                self._log_split_summary(bundle, frame, target_name)

            # Flat plots/pred_obs.png in every case: the RUN is now the per-target scope, so there
            # is no longer a second target writing the same leaf to nest away from.
            pred_obs_logged = False
            try:
                pred_obs_logged = self._log_pred_obs_artifact(
                    frame if frame is not None else evaluation_df,
                    target=target_name,
                    model_name=model_name,
                    artifact_path=ArtifactLayout.plots_path(),
                    interval_label=describe_interval(calibrators.get(target_name)),
                )
            except Exception:
                pred_obs_logged = False

            uncertainty_summary = self._log_uncertainty_artifacts(
                frame=frame if frame is not None else evaluation_df,
                target=target_name,
                multi_target=False,
                calibrator=calibrators.get(target_name),
            )

            explain_summary = self._log_shap_slice(
                shap_results,
                shap_summary,
                target=target_name,
                target_names=group_target_names,
                config=config,
                is_model_run=is_model_run,
            )

            summary = {
                "target": target_name,
                "model_name": model_name,
                "framework": "lightning",
                "metrics": metrics,
                "metric_space": metric_space_for(metrics),
                "best_model_path": best_model_path,
                # The checkpoint is logged as checkpoints/best.ckpt so runs stay comparable; this is
                # where its Lightning-assigned epoch/step name survives.
                "checkpoint_filename": os.path.basename(best_model_path) if best_model_path else None,
                "logged_model_name": ArtifactLayout.logged_model_name(run_target, model_name),
                "registered_model_version": registered_version,
                "champion": champion,
                "pred_obs_artifact_logged": pred_obs_logged,
                "serialized_model_logged": model_logged,
                "serialized_model_logging_error": model_logging_error,
                "explain": explain_summary,
                "uncertainty": uncertainty_summary,
                "point_predictions": export_summary,
                "run_name": f"{target_name}_{model_name}",
                "resolved_target": target_name,
            }
            self._write_json_artifact(
                summary,
                ArtifactLayout.RUN_SUMMARY_FILE,
                artifact_path=ArtifactLayout.META,
            )
            return explain_summary

        target_runs = self._log_per_target_runs(evaluation_df, run_target, model_name, log_one)
        self._record_model_run_explain(shap_summary, results=shap_results, target_runs=target_runs)


class ParentRunLogger:
    def __init__(self):
        pass

    @staticmethod
    def _resolve_target_names(evaluation_df: pd.DataFrame, fallback_target: str) -> list[str]:
        return ChildRunLogger._resolve_target_names(evaluation_df, fallback_target)

    def _collect_leaderboard(self,parent_run_id: str):
        """
        Collects metrics from all child runs of a given parent run.
        Returns a DataFrame with leaderboard info.
        """
        client = mlflow.tracking.MlflowClient()  # type: ignore
        experiment_id = client.get_run(parent_run_id).info.experiment_id

        def children_of(run_id: str):
            return client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=f"tags.mlflow.parentRunId = '{run_id}'",
            )

        # A joint run is a model run with one child per target, so the rows worth comparing sit a
        # level deeper than they used to. Searching only direct children left a joint run out of
        # the leaderboard entirely: its model run carried no target tag and its per-target runs
        # were grandchildren.
        leaf_runs = []
        for child in children_of(parent_run_id):
            grandchildren = _scoring_runs(children_of(child.info.run_id))
            # The model run's own metrics are means over its children, so listing it beside them
            # would put the same model on the board twice, once under a label ("a__b") that names
            # no measurable target.
            leaf_runs.extend(grandchildren or [child])

        rows = []
        for run in leaf_runs:
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

        def children_of(run_id: str):
            return client.search_runs(
                experiment_ids=[exp_id],
                filter_string=f"tags.mlflow.parentRunId = '{run_id}'",
            )

        # Same reason as _collect_leaderboard: a joint run keeps its per-target evaluation frames
        # one level deeper, in the child runs, and its model run holds the joint frame those were
        # split from.
        child_runs = []
        for child in children_of(parent_run_id):
            grandchildren = _scoring_runs(children_of(child.info.run_id))
            child_runs.extend(grandchildren or [child])

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

                target_names = self._resolve_target_names(df, target)
                # is_prediction_column, not `startswith("prediction_")`. An uncertainty run's frame
                # also carries prediction_std / prediction_lower / prediction_upper; counting those
                # sent a SINGLE-target frame down the multi-target branch, where the fallback below
                # picked `prediction_std` and assigned it to `prediction` - so the parent's
                # pred_error_plot.png plotted standard deviations on the predicted axis. Same bug,
                # and same fix, as in _iter_target_eval_frames.
                prediction_columns = [
                    column for column in df.columns if is_prediction_column(column)
                ]
                if "prediction" in df.columns and not any(
                    column != "prediction" for column in prediction_columns
                ):
                    df["target_name"] = target_names[0] if target_names else target
                    df["model_name"] = model_name
                    eval_dfs.append(df)
                else:
                    for target_name in target_names:
                        prediction_column = f"prediction_{target_name}"
                        if prediction_column not in df.columns:
                            prediction_column = next(iter(prediction_columns), None)
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


    def _collect_point_predictions(self, parent_run_id: str, id_column: str):
        """``(model_name, frame)`` for every child that exported per-point predictions.

        DIRECT children only, unlike the leaderboard and eval-frame collectors. Those two want the
        per-target evaluation runs, which sit a level deeper on a joint fit; this wants the run that
        holds the MODEL, because that is the one that ran the full-population pass and a joint
        model writes all of its targets into one file. Ensemble members are children of the model
        run rather than of the parent, so they never appear at this level - but the filter is
        applied anyway, so this does not silently depend on that.
        """
        client = mlflow.tracking.MlflowClient()  # type: ignore
        experiment_id = client.get_run(parent_run_id).info.experiment_id
        children = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        )

        collected = []
        for run in _scoring_runs(children):
            model_name = run.data.tags.get("model_name")
            if not model_name:
                continue
            try:
                local_path = mlflow.artifacts.download_artifacts(  # type: ignore
                    run_id=run.info.run_id,
                    artifact_path=(
                        f"{ArtifactLayout.PREDICTIONS}/{ArtifactLayout.POINT_PREDICTIONS_FILE}"
                    ),
                )
            except Exception:
                # A child that did not export - skipped by name, a graph entry, or a failure it
                # already recorded in its own run summary. Not this collector's business.
                continue
            frame = pd.read_csv(local_path)
            if id_column in frame.columns:
                collected.append((model_name, frame))

        return collected

    def _log_point_prediction_export(self, parent_run_id: str, config) -> None:
        """Combine the children's per-point predictions into the two parent-level files."""
        if not bool(getattr(config, "EXPORT_POINT_PREDICTIONS", False)):
            return

        id_column = point_id_column(config)
        frames = self._collect_point_predictions(parent_run_id, id_column)
        if not frames:
            return

        wide, long_frame = combine_point_predictions(frames, id_column=id_column)
        warning = duplicate_report(long_frame, id_column=id_column)
        if warning:
            mlflow.set_tags({"point_predictions_warning": warning})

        log_table(wide, ArtifactLayout.POINT_PREDICTIONS_WIDE_FILE, ArtifactLayout.PREDICTIONS)
        log_table(long_frame, ArtifactLayout.POINT_PREDICTIONS_LONG_FILE, ArtifactLayout.PREDICTIONS)
        log_json(
            dict(summarize_point_predictions(wide, long_frame, id_column)),
            "point_predictions_summary.json",
            ArtifactLayout.PREDICTIONS,
        )

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
            # The INNER cross-validation strategy. SPLIT_HOLDOUT_STRATEGY below is the holdout.
            "SPLIT_STRATEGY": trainer.config.SPLIT_STRATEGY,
            "SPLIT_HOLDOUT_STRATEGY": getattr(trainer.config, "SPLIT_HOLDOUT_STRATEGY", None),
            "SPLIT_POPULATION_POLICY": getattr(trainer.config, "SPLIT_POPULATION_POLICY", None),
        })
        # log_params_once, not log_params: these land at the END of a run, so an immutable-param
        # clash here destroys the summary of work that has already fully succeeded.
        log_params_once({
            "DATA_FOLDER": trainer.config.DATA_FOLDER,
            "DATA_FILE": trainer.config.DATA_FILE,
            "STATIC_FEATURES_FILE": getattr(trainer.config, "STATIC_FEATURES_FILE", trainer.config.DATA_FILE),
            "TARGETS_FILE": getattr(trainer.config, "TARGETS_FILE", trainer.config.DATA_FILE),
            "RANDOM_SEED": trainer.config.RANDOM_SEED,
            # TARGET_COLUMNS is deliberately NOT logged here. SoilModelTraining._log_target_plan
            # writes it at the START of training, beside the resolved target groups, so a run that
            # is killed partway still records what it set out to fit. Writing it again here - in a
            # different format - is what made a finished run fail on MLflow's immutable params.
            "COLUMNS_TO_TRANSFORM": [column for column in trainer.config.COLUMNS_TO_TRANSFORM
                                     if column in trainer.config.TARGET_COLUMNS],
            "CLUSTERING_STRATEGY": trainer.config.CLUSTERING_STRATEGY.get('class_path').rsplit('.', 1)[1] if trainer.config.ENABLE_CLUSTERING else None,
            "cell_size_m": trainer.config.CLUSTERING_STRATEGY.get('params', {}).get('cell_size_m', None) if trainer.config.ENABLE_CLUSTERING else None,
            "n_clusters": trainer.config.CLUSTERING_STRATEGY.get('params', {}).get('n_clusters', None) if trainer.config.ENABLE_CLUSTERING else None,
            # The shared holdout. TEST_SIZE was previously logged nowhere at all, so a finished run
            # did not record how much data it held out, let alone which points.
            "SPLIT_HOLDOUT_STRATEGY": getattr(trainer.config, "SPLIT_HOLDOUT_STRATEGY", None),
            "SPLIT_TEST_SIZE": getattr(trainer.config, "SPLIT_TEST_SIZE", None),
            "SPLIT_VAL_SIZE": getattr(trainer.config, "SPLIT_VAL_SIZE", None),
            "SPLIT_SEED": getattr(trainer.config, "SPLIT_SEED", None),
            "SPLIT_POPULATION_POLICY": getattr(trainer.config, "SPLIT_POPULATION_POLICY", None),
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

        self._log_point_prediction_export(parent_run_id, trainer.config)
