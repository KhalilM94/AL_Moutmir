from __future__ import annotations

import importlib
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping

import mlflow
import numpy as np
import pandas as pd

from yg_eo_soilnet.models.lightning_config_factory import LightningModelBundle


@dataclass
class LightningRunResult:
    model_name: str
    target: str
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    best_model_path: str | None


class LightningTrainer:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger

    def train(self, target: str, data: Mapping[str, Any], model_bundles: Mapping[str, LightningModelBundle]):
        results: dict[str, LightningRunResult] = {}

        for model_name, bundle in model_bundles.items():
            run_name = f"{target}_{model_name}"
            with mlflow.start_run(run_name=run_name, nested=True):
                mlflow.set_tags({"target": target, "model_name": model_name, "framework": "lightning"})
                mlflow.log_params(self._serialize_params(bundle))

                trainer = self._build_trainer(bundle)
                bundle.datamodule.setup("fit")
                trainer.fit(bundle.model, datamodule=bundle.datamodule)

                validation_metrics = self._normalize_metrics(
                    self._call_trainer_method(trainer, "validate", bundle.model, bundle.datamodule)
                )
                test_metrics = self._normalize_metrics(
                    self._call_trainer_method(trainer, "test", bundle.model, bundle.datamodule)
                )

                best_model_path = self._resolve_best_checkpoint(trainer)
                if best_model_path:
                    mlflow.log_artifact(best_model_path, artifact_path="checkpoints")

                for metric_name, metric_value in {**validation_metrics, **test_metrics}.items():
                    mlflow.log_metric(metric_name, metric_value)

                evaluation_df = self._build_evaluation_frame(bundle, trainer, target)
                if evaluation_df is not None:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        eval_path = os.path.join(tmpdir, f"eval_results_{target}_{model_name}.csv")
                        evaluation_df.to_csv(eval_path, index=False)
                        mlflow.log_artifact(eval_path, artifact_path="eval_results")

                results[model_name] = LightningRunResult(
                    model_name=model_name,
                    target=target,
                    validation_metrics=validation_metrics,
                    test_metrics=test_metrics,
                    best_model_path=best_model_path,
                )

        return results

    def _build_trainer(self, bundle: LightningModelBundle):
        lightning = self._get_lightning_module()
        callbacks = self._build_callbacks(bundle.callback_specs)

        trainer_kwargs = dict(bundle.trainer_kwargs)
        trainer_kwargs["callbacks"] = callbacks
        return lightning.Trainer(**trainer_kwargs)

    def _build_callbacks(self, callback_specs: Mapping[str, Any]):
        lightning = self._get_lightning_module()
        callbacks = []

        early_stopping = callback_specs.get("early_stopping")
        if early_stopping:
            callbacks.append(lightning.callbacks.EarlyStopping(**early_stopping))

        checkpoint = callback_specs.get("checkpoint")
        if checkpoint:
            checkpoint_config = dict(checkpoint)
            checkpoint_config.setdefault("dirpath", getattr(self.config, "LIGHTNING_CHECKPOINT_DIR", None))
            callbacks.append(lightning.callbacks.ModelCheckpoint(**checkpoint_config))

        return callbacks

    def _get_lightning_module(self):
        try:
            return importlib.import_module("lightning.pytorch")
        except ImportError as exc:  # pragma: no cover - exercised only when lightning is absent
            raise ImportError(
                "lightning.pytorch is required to execute LightningTrainer.train()."
            ) from exc

    def _call_trainer_method(self, trainer, method_name: str, model, datamodule):
        method = getattr(trainer, method_name, None)
        if method is None:
            return []
        try:
            return method(model, datamodule=datamodule, verbose=False)
        except TypeError:
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

    def _build_evaluation_frame(self, bundle: LightningModelBundle, trainer, target: str) -> pd.DataFrame | None:
        datamodule = bundle.datamodule
        if getattr(datamodule, "X_test_frame_", None) is None or getattr(datamodule, "y_test_frame_", None) is None:
            return None

        predict_method = getattr(trainer, "predict", None)
        if predict_method is None:
            return None

        try:
            predictions = predict_method(bundle.model, datamodule=datamodule)
        except TypeError:
            predictions = predict_method(bundle.model)

        if predictions is None:
            return None

        prediction_values = self._flatten_predictions(predictions)
        if prediction_values is None:
            return None

        eval_df = datamodule.X_test_frame_.copy()
        target_frame = datamodule.y_test_frame_.copy()

        if prediction_values.ndim == 1 or prediction_values.shape[1] == 1:
            eval_df["prediction"] = prediction_values.reshape(-1)
        else:
            for column_index in range(prediction_values.shape[1]):
                eval_df[f"prediction_{column_index}"] = prediction_values[:, column_index]

        for column in target_frame.columns:
            eval_df[column] = target_frame[column].to_numpy()

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