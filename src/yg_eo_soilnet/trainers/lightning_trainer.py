from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Mapping

import mlflow
import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from lightning.pytorch.callbacks import Callback as LightningCallback
except ImportError:  # pragma: no cover
    LightningCallback = object  # type: ignore[assignment]

from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger
from yg_eo_soilnet.models.config_fatories.lightning_config_factory import LightningModelBundle
from yg_eo_soilnet.targets import join_target_names
from yg_eo_soilnet.tracking import start_child_run
from yg_eo_soilnet.predictions_export import export_enabled_for
from yg_eo_soilnet.uncertainty.intervals import (
    build_interval_estimators,
    needs_calibration_set,
    normalize_method,
)
from yg_eo_soilnet.uncertainty import (
    aggregate,
    attach_uncertainty_columns,
    fit_calibrators,
    member_seeds,
    uncertainty_enabled_for,
)

# Mirrors the sklearn trainer's constant. See ParentRunLogger._collect_leaderboard for why a member
# run has to be distinguishable from a per-target evaluation run.
MEMBER_RUN_KIND = "ensemble_member"


class _LightningMlflowEpochMetricCallback(LightningCallback):
    def __init__(self):
        self._last_logged_epoch: dict[str, int] = {}

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().item() if getattr(value, "ndim", 0) == 0 else value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.size != 1:
                return None
            value = float(value.reshape(-1)[0])
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _log_metrics(self, trainer, metric_names: tuple[str, ...]) -> None:
        if getattr(trainer, "sanity_checking", False):
            return

        current_epoch = int(getattr(trainer, "current_epoch", 0))
        for metric_name in metric_names:
            if self._last_logged_epoch.get(metric_name) == current_epoch:
                continue
            metric_value = trainer.callback_metrics.get(metric_name)
            metric_float = self._to_float(metric_value)
            if metric_float is None:
                continue
            mlflow.log_metric(metric_name, metric_float, step=current_epoch)
            self._last_logged_epoch[metric_name] = current_epoch

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._log_metrics(trainer, ("train_loss",))

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._log_metrics(trainer, ("val_loss",))

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        self._log_metrics(trainer, ("test_loss",))


@dataclass
class LightningRunResult:
    model_name: str
    target: str
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    best_model_path: str | None


class LightningTrainer:
    def __init__(self, config, logger=None, mlflow_logger=None):
        self.config = config
        self.logger = logger
        self.mlflow_logger = mlflow_logger or ChildRunLogger()

    def train(
        self,
        target: str,
        data: Mapping[str, Any],
        model_bundles: Mapping[str, LightningModelBundle],
        bundle_builder: Any = None,
    ):
        """Fit every bundle for one target group.

        ``bundle_builder(seed) -> {model_name: bundle}`` is what makes an ensemble possible here.
        A Lightning model's weights are constructed by the factory, seeded immediately beforehand,
        so a second member cannot be made from an existing bundle - it needs the factory to build a
        new one at a new seed. Absent (or with uncertainty off) every entry takes the single-fit
        path below, unchanged.
        """
        results: dict[str, LightningRunResult] = {}

        for model_name, bundle in model_bundles.items():
            if bundle_builder is not None and uncertainty_enabled_for(self.config, model_name):
                results[model_name] = self._train_ensemble(
                    target=target,
                    model_name=model_name,
                    bundle_builder=bundle_builder,
                )
                continue
            # Deliberately no seeding here. Seeding at this point is too late to reach the weights -
            # the factory built them already - and resetting the stream now would start fit() from a
            # different place than the HPO trial that chose these hyperparameters, so a tuned config
            # could never reproduce its score. LightningConfigFactory.build_lightning_configs seeds
            # per entry instead, immediately before it constructs each model.
            # Named for the target GROUP, not bare. A joint run used to be called just "soil_cnn"
            # while every other run in the experiment carried its target, which made the one run
            # spanning several targets the hardest to identify.
            run_name = f"{target}_{model_name}"
            with start_child_run(run_name):
                trainer = self._build_trainer(bundle)
                bundle.datamodule.setup("fit")
                # After setup (the scalers and vocabulary are fitted there) and before fit, so the
                # state is inside every checkpoint the run writes. Without it a restored model has
                # its weights but no way to standardize raw input, which makes it unservable.
                self._attach_preprocessing_state(bundle)
                trainer.fit(bundle.model, datamodule=bundle.datamodule)

                best_model_path = self._resolve_best_checkpoint(trainer)

                validation_metrics = self._normalize_metrics(
                    self._call_trainer_method(
                        trainer,
                        "validate",
                        bundle.model,
                        bundle.datamodule,
                        ckpt_path=best_model_path,
                    )
                )
                test_metrics = self._normalize_metrics(
                    self._call_trainer_method(
                        trainer,
                        "test",
                        bundle.model,
                        bundle.datamodule,
                        ckpt_path=best_model_path,
                    )
                )

                evaluation_df = self._build_evaluation_frame(bundle, trainer, target, ckpt_path=best_model_path)

                # One call, whatever the target count. The logger owns the run tree now: it keeps
                # the model, the curves and the aggregate metrics here and opens one child per
                # target when there is more than one. This used to fan out here and hand each
                # child empty metric dicts, which is why val_loss and test_loss never reached
                # MLflow at all on a multi-target run.
                self.mlflow_logger.log_lightning_child_run(
                    config=self.config,
                    target=target,
                    model_name=model_name,
                    evaluation_df=evaluation_df,
                    validation_metrics=validation_metrics,
                    test_metrics=test_metrics,
                    best_model_path=best_model_path,
                    extra_params=self._serialize_params(bundle),
                    plot_functions={},
                    bundle=bundle,
                    model=bundle.model,
                )

                results[model_name] = LightningRunResult(
                    model_name=model_name,
                    target=target,
                    validation_metrics=validation_metrics,
                    test_metrics=test_metrics,
                    best_model_path=best_model_path,
                )

        return results

    # --- uncertainty --------------------------------------------------------

    def _train_ensemble(
        self,
        *,
        target: str,
        model_name: str,
        bundle_builder: Any,
    ) -> LightningRunResult:
        """Fit n_members independently seeded models and log them as one ensemble.

        Each member is a fresh bundle from the factory at its own seed, which is the only way to get
        a different weight initialization - the factory seeds immediately before it constructs the
        model, precisely so that a run is reproducible from its seed.

        The members do NOT resample their training rows. Bootstrapping is what gives a deterministic
        estimator its diversity; a neural network trained from a different initialization on a
        non-convex loss surface already lands somewhere else, and taking 36.8% of its rows away as
        well would cost accuracy to buy spread it already has.
        """
        n_members = int(getattr(self.config, "UNCERTAINTY_N_MEMBERS", 5))
        stride = int(getattr(self.config, "UNCERTAINTY_SEED_STRIDE", 1000))
        seeds = member_seeds(int(getattr(self.config, "RANDOM_SEED", 42)), n_members, stride)

        run_name = f"{target}_{model_name}"
        with start_child_run(run_name):
            mlflow.log_params(
                {
                    "uncertainty_n_members": n_members,
                    "uncertainty_member_seeds": ",".join(str(seed) for seed in seeds),
                    "uncertainty_bootstrapped": False,
                    "uncertainty_calibration_source": getattr(
                        self.config, "UNCERTAINTY_CALIBRATION_SOURCE", "val"
                    ),
                }
            )

            members = []
            for index, seed in enumerate(seeds):
                with start_child_run(
                    f"{run_name}_member{index}",
                    tags={
                        "run_kind": MEMBER_RUN_KIND,
                        "target": target,
                        "model_name": model_name,
                        "ensemble_member": str(index),
                        "ensemble_seed": str(seed),
                    },
                ):
                    members.append(
                        self._fit_member(
                            bundle=bundle_builder(seed)[model_name],
                            index=index,
                            seed=seed,
                        )
                    )

            return self._log_ensemble(
                target=target, model_name=model_name, members=members, seeds=seeds
            )

    def _fit_member(self, *, bundle: LightningModelBundle, index: int, seed: int) -> dict:
        """Fit one member and collect everything the ensemble needs from it.

        Predictions are taken on BOTH the test split and the val split. The val predictions are the
        calibration set: unlike the sklearn side, where val had to be carved out of the fit pool,
        Lightning has always early-stopped on val and never fitted on it, so it is available here
        at no cost in training rows. It is not perfectly held out either - early stopping read it -
        so the calibration is mildly optimistic, which is worth knowing and not worth a third split.
        """
        trainer = self._build_trainer(bundle)
        bundle.datamodule.setup("fit")
        self._attach_preprocessing_state(bundle)
        trainer.fit(bundle.model, datamodule=bundle.datamodule)

        best_model_path = self._resolve_best_checkpoint(trainer)
        validation_metrics = self._normalize_metrics(
            self._call_trainer_method(
                trainer, "validate", bundle.model, bundle.datamodule, ckpt_path=best_model_path
            )
        )
        test_metrics = self._normalize_metrics(
            self._call_trainer_method(
                trainer, "test", bundle.model, bundle.datamodule, ckpt_path=best_model_path
            )
        )

        mlflow.log_params(
            {
                "ensemble_member": index,
                "ensemble_seed": seed,
                # The local path Lightning wrote to. Only the model run used to record one, and only
                # for the reference member, so a member's weights could be found afterwards solely
                # by scavenging lightning_logs and matching on val_loss.
                "best_model_path": best_model_path,
            }
        )
        # Each member keeps its OWN checkpoint. Without this the run logs one checkpoint for the
        # whole ensemble - the reference member's - and the other n-1 exist only as local files
        # under lightning_logs, which nothing records and any cleanup removes. That is what made
        # export_predictions.py unable to reconstruct a Lightning ensemble's mean from MLflow alone.
        # Costs n_members checkpoints per entry instead of one; see uncertainty.n_members.
        if best_model_path:
            self.mlflow_logger._log_checkpoint(best_model_path)

        test_predictions, test_sigmas = self._predict_split(
            bundle, trainer, best_model_path, "predict"
        )
        calibration_predictions, calibration_sigmas = self._predict_split(
            bundle, trainer, best_model_path, "val"
        )

        return {
            "full_predictions": self._predict_full_population(bundle),
            "bundle": bundle,
            "trainer": trainer,
            "best_model_path": best_model_path,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "test_predictions": test_predictions,
            "test_sigmas": test_sigmas,
            "calibration_predictions": calibration_predictions,
            "calibration_sigmas": calibration_sigmas,
        }

    def _predict_full_population(self, bundle) -> np.ndarray | None:
        """This member's prediction for EVERY point, or None when the export is off.

        Only an ensemble needs this at member level: the exported number is the mean across
        members, and there is no single model object that computes it. Gated on the switch because
        it is a whole extra inference pass over the full dataset, per member.
        """
        if not export_enabled_for(self.config, bundle.name):
            return None

        datamodule = getattr(bundle, "datamodule", None)
        sequence_bundle = getattr(datamodule, "sequence_bundle", None)
        if sequence_bundle is None:
            return None

        try:
            from yg_eo_soilnet.serving.sequence_predictor import SoilSequencePredictor

            return SoilSequencePredictor(bundle.model).predict(sequence_bundle)
        except Exception as exc:  # pragma: no cover - the logger reports and keeps the run
            if self.logger is not None:
                self.logger.warning(
                    f"Full-population prediction failed for one member of {bundle.name}: "
                    f"{type(exc).__name__}: {exc}. The ensemble export will be skipped."
                )
            return None

    def _predict_split(self, bundle, trainer, ckpt_path, split: str):
        """``(means, sigmas)`` over one split, in original target units; sigmas None without a head.

        ``predict`` uses the datamodule's own predict_dataloader, which is the test split and is
        never shuffled - the frame is aligned positionally with y_test_frame_. The val loader is
        passed explicitly because there is no predict-style hook for it.
        """
        predict_method = getattr(trainer, "predict", None)
        if predict_method is None:
            return None, None

        try:
            if split == "val":
                dataloader = bundle.datamodule.val_dataloader()
                if dataloader is None:
                    return None, None
                predictions = predict_method(
                    bundle.model, dataloaders=dataloader, ckpt_path=ckpt_path
                )
            else:
                predictions = predict_method(
                    bundle.model, datamodule=bundle.datamodule, ckpt_path=ckpt_path
                )
        except (TypeError, RuntimeError, ValueError):
            return None, None

        return self._flatten_predictions_with_sigma(predictions)

    def _log_ensemble(self, *, target, model_name, members, seeds) -> LightningRunResult:
        """Aggregate the members, calibrate, and hand one frame to the logger."""
        reference = members[0]
        bundle = reference["bundle"]
        datamodule = bundle.datamodule
        target_names = list(getattr(datamodule, "target_names", []) or [])

        test_stack, test_sigmas = self._stack_member_outputs(members, "test")
        prediction = aggregate(test_stack, test_sigmas) if test_stack else None

        calibrators = self._calibrate_ensemble(members, datamodule, target_names)

        evaluation_df = self._build_evaluation_frame(
            bundle, reference["trainer"], target, ckpt_path=reference["best_model_path"]
        )
        if evaluation_df is not None and prediction is not None and target_names:
            # The ensemble MEAN replaces the reference member's predictions: the frame builder wrote
            # one member's numbers, and what the run reports must be what the ensemble predicts.
            self._overwrite_predictions(evaluation_df, prediction, target_names)
            attach_uncertainty_columns(evaluation_df, prediction, target_names, calibrators)

        # Metrics are averaged across members so val_loss and test_loss describe the ensemble rather
        # than whichever member happened to be built first.
        validation_metrics = self._average_metrics([m["validation_metrics"] for m in members])
        test_metrics = self._average_metrics([m["test_metrics"] for m in members])

        self.mlflow_logger.log_lightning_child_run(
            config=self.config,
            target=target,
            model_name=model_name,
            evaluation_df=evaluation_df,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            best_model_path=reference["best_model_path"],
            extra_params=self._serialize_params(bundle),
            plot_functions={},
            bundle=bundle,
            model=bundle.model,
            calibrators=calibrators,
            full_population_predictions=self._ensemble_full_population(members),
        )

        return LightningRunResult(
            model_name=model_name,
            target=target,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            best_model_path=reference["best_model_path"],
        )

    def _calibrate_ensemble(self, members, datamodule, target_names) -> dict:
        """One interval estimator per target, of whichever kind the config asked for.

        Only conformal reaches the val split below; gaussian and sigma are arithmetic on the sigma
        the ensemble already produced, so they need no held-out predictions at all.
        """
        method = normalize_method(getattr(self.config, "UNCERTAINTY_INTERVAL_METHOD", "conformal"))
        if not needs_calibration_set(method):
            return build_interval_estimators(
                method,
                target_names,
                alpha=float(getattr(self.config, "UNCERTAINTY_ALPHA", 0.05)),
                k=float(getattr(self.config, "UNCERTAINTY_INTERVAL_K", 1.0)),
            )

        y_val = getattr(datamodule, "y_val_frame_", None)
        stack, sigmas = self._stack_member_outputs(members, "calibration")
        if y_val is None or not len(y_val) or not stack or not target_names:
            if self.logger is not None:
                self.logger.warning(
                    "No usable validation split for conformal calibration; the ensemble reports a "
                    "standard deviation but no calibrated interval."
                )
            return {}

        calibration = aggregate(stack, sigmas)
        if calibration.mean.shape[0] != len(y_val):
            if self.logger is not None:
                self.logger.warning(
                    f"Validation predictions ({calibration.mean.shape[0]} rows) do not line up with "
                    f"y_val_frame_ ({len(y_val)} rows); skipping calibration rather than pairing "
                    "residuals with the wrong observations."
                )
            return {}

        return fit_calibrators(
            calibration,
            y_val,
            target_names,
            alpha=float(getattr(self.config, "UNCERTAINTY_ALPHA", 0.05)),
            logger=self.logger,
        )

    @staticmethod
    def _ensemble_full_population(members) -> np.ndarray | None:
        """The ensemble MEAN over every point, or None when any member could not produce one.

        All-or-nothing: averaging over the subset of members that happened to succeed would export
        a number that is neither one member's prediction nor the ensemble's, under a column name
        claiming to be the ensemble's. The sigma is discarded here - the export carries estimates
        only.
        """
        full = [
            member["full_predictions"]
            for member in members
            if member.get("full_predictions") is not None
        ]
        if not full or len(full) != len(members):
            return None
        return aggregate(full).mean

    @staticmethod
    def _stack_member_outputs(members, split: str):
        """``(means, sigmas)`` across members for one split, sigmas None unless EVERY member has one.

        All-or-nothing on the sigmas because ``aggregate`` averages variances across members: a
        partial list would average the aleatoric term over the members that reported one and
        silently treat the rest as noiseless, understating it by exactly the fraction missing.
        """
        means = [
            member[f"{split}_predictions"]
            for member in members
            if member.get(f"{split}_predictions") is not None
        ]
        sigmas = [
            member[f"{split}_sigmas"]
            for member in members
            if member.get(f"{split}_sigmas") is not None
        ]
        return means, (sigmas if means and len(sigmas) == len(means) else None)

    @staticmethod
    def _overwrite_predictions(evaluation_df, prediction, target_names) -> None:
        """Replace the frame's prediction columns with the ensemble mean, in place."""
        multi_target = len(target_names) > 1
        for index, target_name in enumerate(target_names):
            column = f"prediction_{target_name}" if multi_target else "prediction"
            if column in evaluation_df.columns:
                evaluation_df[column] = prediction.mean[:, index]

    @staticmethod
    def _average_metrics(metric_dicts: list[dict]) -> dict[str, float]:
        """Mean of each metric across the members, over the keys they all report."""
        if not metric_dicts:
            return {}
        shared = set(metric_dicts[0])
        for metrics in metric_dicts[1:]:
            shared &= set(metrics)
        return {
            key: float(np.mean([metrics[key] for metrics in metric_dicts])) for key in sorted(shared)
        }

    @staticmethod
    def _attach_preprocessing_state(bundle: LightningModelBundle) -> None:
        """Copy the datamodule's fitted input statistics onto the model, when both support it.

        Both sides are optional on purpose: the graph datamodule and SoilGraphLightningModule do not
        implement this pair, and a model that cannot carry the state should train exactly as before
        rather than fail.
        """
        state_source = getattr(bundle.datamodule, "preprocessing_state", None)
        attach = getattr(bundle.model, "attach_preprocessing_state", None)
        if not callable(state_source) or not callable(attach):
            return
        attach(state_source())

    def _build_trainer(self, bundle: LightningModelBundle):
        lightning = self._get_lightning_module()
        callbacks = self._build_callbacks(bundle.callback_specs)

        trainer_kwargs = dict(bundle.trainer_kwargs)
        trainer_kwargs["callbacks"] = callbacks
        if not getattr(self.config, "LIGHTNING_ENABLE_DEFAULT_LOGGER", True):
            trainer_kwargs.setdefault("logger", False)
        return lightning.Trainer(**trainer_kwargs)

    def _build_callbacks(self, callback_specs: Mapping[str, Any]):
        lightning = self._get_lightning_module()
        callbacks = [_LightningMlflowEpochMetricCallback()]

        early_stopping = callback_specs.get("early_stopping")
        if early_stopping:
            callbacks.append(lightning.callbacks.EarlyStopping(**early_stopping))

        checkpoint = callback_specs.get("checkpoint")
        if checkpoint:
            # dirpath comes from the registry's checkpoint block if set; Lightning defaults it
            # otherwise. (The old LIGHTNING_CHECKPOINT_DIR lookup was never defined anywhere.)
            callbacks.append(lightning.callbacks.ModelCheckpoint(**dict(checkpoint)))

        return callbacks

    def _get_lightning_module(self):
        try:
            return importlib.import_module("lightning.pytorch")
        except ImportError as exc:  # pragma: no cover - exercised only when lightning is absent
            raise ImportError(
                "lightning.pytorch is required to execute LightningTrainer.train()."
            ) from exc

    def _call_trainer_method(self, trainer, method_name: str, model, datamodule, ckpt_path: str | None = None):
        method = getattr(trainer, method_name, None)
        if method is None:
            return []
        try:
            if ckpt_path is not None:
                return method(model, datamodule=datamodule, verbose=False, ckpt_path=ckpt_path)
            return method(model, datamodule=datamodule, verbose=False)
        except TypeError:
            if ckpt_path is not None:
                return method(model, datamodule=datamodule, ckpt_path=ckpt_path)
            return method(model, datamodule=datamodule)

    def _resolve_best_checkpoint(self, trainer) -> str | None:
        checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
        best_model_path = getattr(checkpoint_callback, "best_model_path", None)
        if best_model_path:
            return best_model_path

        for callback in getattr(trainer, "callbacks", []):
            best_model_path = getattr(callback, "best_model_path", None)
            if best_model_path:
                return best_model_path

        return None

    def _build_evaluation_frame(
        self,
        bundle: LightningModelBundle,
        trainer,
        target: str,
        ckpt_path: str | None = None,
    ) -> pd.DataFrame | None:
        datamodule = bundle.datamodule
        if getattr(datamodule, "X_test_frame_", None) is None or getattr(datamodule, "y_test_frame_", None) is None:
            return None

        predict_method = getattr(trainer, "predict", None)
        if predict_method is None:
            return None

        try:
            if ckpt_path is not None:
                predictions = predict_method(bundle.model, datamodule=datamodule, ckpt_path=ckpt_path)
            else:
                predictions = predict_method(bundle.model, datamodule=datamodule)
        except TypeError:
            if ckpt_path is not None:
                predictions = predict_method(bundle.model, ckpt_path=ckpt_path)
            else:
                predictions = predict_method(bundle.model)

        if predictions is None:
            return None

        prediction_values = self._flatten_predictions(predictions)
        if prediction_values is None:
            return None

        eval_df = datamodule.X_test_frame_.copy()
        target_frame = datamodule.y_test_frame_.copy()
        target_names = list(getattr(datamodule, "target_names", []) or target_frame.columns.tolist())

        for column in target_frame.columns:
            eval_df[column] = target_frame[column].to_numpy()

        is_single_output = prediction_values.ndim == 1 or (
            prediction_values.ndim == 2 and prediction_values.shape[1] == 1
        )

        if is_single_output:
            eval_df["prediction"] = prediction_values.reshape(-1)
        else:
            target_columns = list(target_names)
            for column_index in range(prediction_values.shape[1]):
                column_name = target_columns[column_index] if column_index < len(target_columns) else str(column_index)
                eval_df[f"prediction_{column_name}"] = prediction_values[:, column_index]
            # No plain `prediction` column here. It used to alias target 0, so anyone reading
            # eval_results.csv from a joint run got the first target's predictions under a name
            # that claims to be the run's. Readers fan out over prediction_<target> instead.

        if len(target_names) > 1:
            encoded = join_target_names(target_names)
            eval_df["target_name"] = encoded
            eval_df["target_names"] = encoded
        else:
            eval_df["target_name"] = target
        return eval_df

    def _flatten_predictions(self, predictions) -> np.ndarray | None:
        """Just the means, for every caller that only wants predictions."""
        return self._flatten_predictions_with_sigma(predictions)[0]

    def _flatten_predictions_with_sigma(
        self, predictions
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """``(means, sigmas)`` from a list of predict_step outputs, sigmas None on a point head.

        A heteroscedastic ``predict_step`` returns a ``(mean, sigma)`` tuple per batch and a point
        head returns a bare tensor, so both shapes have to be unpacked here rather than at each
        call site.
        """
        means, sigmas = [], []
        for batch in predictions:
            if batch is None:
                continue
            if isinstance(batch, tuple):
                mean_batch, sigma_batch = batch[0], batch[1] if len(batch) > 1 else None
            else:
                mean_batch, sigma_batch = batch, None

            means.append(self._as_2d_array(mean_batch))
            if sigma_batch is not None:
                sigmas.append(self._as_2d_array(sigma_batch))

        if not means:
            return None, None

        stacked_means = np.concatenate(means, axis=0)
        # All-or-nothing: a partial sigma would silently pair some rows' uncertainty with other
        # rows' predictions once the arrays were concatenated to different lengths.
        stacked_sigmas = (
            np.concatenate(sigmas, axis=0) if len(sigmas) == len(means) else None
        )
        return stacked_means, stacked_sigmas

    @staticmethod
    def _as_2d_array(batch) -> np.ndarray:
        if hasattr(batch, "detach"):
            batch = batch.detach().cpu().numpy()
        else:
            batch = np.asarray(batch)

        if batch.ndim == 0:
            return batch.reshape(1, 1)
        if batch.ndim == 1:
            return batch.reshape(-1, 1)
        return batch

    def _normalize_metrics(self, metrics) -> dict[str, float]:
        if not metrics:
            return {}

        if isinstance(metrics, list):
            metrics = metrics[0] if metrics else {}

        normalized = {}
        for key, value in dict(metrics).items():
            try:
                normalized[key] = float(value)
            except (TypeError, ValueError):
                continue

        return normalized

    def _serialize_params(self, bundle: LightningModelBundle) -> dict[str, Any]:
        params = {
            "model_name": bundle.name,
            "target": bundle.target,
            "modeltype": bundle.registry_entry.get("modeltype"),
        }
        params.update(bundle.trainer_kwargs)

        init_args = bundle.registry_entry.get("init_args", {})
        datamodule_args = bundle.registry_entry.get("datamodule_init_args", {})
        for prefix, values in (("model", init_args), ("datamodule", datamodule_args)):
            for key, value in values.items():
                params[f"{prefix}.{key}"] = value

        return params