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

    def train(self, target: str, data: Mapping[str, Any], model_bundles: Mapping[str, LightningModelBundle]):
        results: dict[str, LightningRunResult] = {}

        for model_name, bundle in model_bundles.items():
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
        flattened = []
        for batch in predictions:
            if batch is None:
                continue
            if hasattr(batch, "detach"):
                batch = batch.detach().cpu().numpy()
            else:
                batch = np.asarray(batch)

            if batch.ndim == 0:
                batch = batch.reshape(1, 1)
            elif batch.ndim == 1:
                batch = batch.reshape(-1, 1)

            flattened.append(batch)

        if not flattened:
            return None

        return np.concatenate(flattened, axis=0)

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