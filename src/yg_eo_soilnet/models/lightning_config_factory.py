from __future__ import annotations

import importlib
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from yg_eo_soilnet.datamodules.lightning_datamodule import LightningTabularDataModule


@dataclass
class LightningModelBundle:
    name: str
    target: str
    model: Any
    datamodule: LightningTabularDataModule
    trainer_kwargs: dict[str, Any]
    callback_specs: dict[str, Any]
    registry_entry: dict[str, Any]


class LightningConfigFactory:
    def __init__(self, registry: Mapping[str, dict], config: Any):
        self.registry = registry
        self.config = config

    @staticmethod
    def _dynamic_import(import_path: str):
        module_path, attr_name = import_path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(f"Failed to import module '{module_path}': {exc}") from exc
        return getattr(module, attr_name)

    def build_lightning_configs(self, target: str, data: Mapping[str, Any]) -> dict[str, LightningModelBundle]:
        bundles: dict[str, LightningModelBundle] = {}
        for name, spec in self.registry.items():
            if not spec.get("enabled", False):
                continue

            self._validate_entry(name, spec)
            datamodule = self._build_datamodule(target=target, spec=spec, data=data)
            model = self._build_model(spec, datamodule)
            bundles[name] = LightningModelBundle(
                name=name,
                target=target,
                model=model,
                datamodule=datamodule,
                trainer_kwargs=self._build_trainer_kwargs(spec),
                callback_specs=self._build_callback_specs(spec),
                registry_entry=deepcopy(spec),
            )

        return bundles

    def _validate_entry(self, name: str, spec: Mapping[str, Any]) -> None:
        required_keys = {"enabled", "modeltype", "import_path", "datamodule_import_path"}
        missing = sorted(required_keys - set(spec))
        if missing:
            raise ValueError(f"Lightning registry entry '{name}' is missing required keys: {', '.join(missing)}")
        if spec.get("modeltype") != "dl":
            raise ValueError(f"Lightning registry entry '{name}' must use modeltype 'dl'.")

    def _build_datamodule(self, target: str, spec: Mapping[str, Any], data: Mapping[str, Any]) -> LightningTabularDataModule:
        datamodule_cls = self._dynamic_import(spec["datamodule_import_path"])
        y_train = data["y_train"][target] if hasattr(data["y_train"], "__getitem__") else data["y_train"]
        y_test = None
        if data.get("y_test") is not None:
            y_test = data["y_test"][target] if hasattr(data["y_test"], "__getitem__") else data["y_test"]

        datamodule_kwargs = {
            "X_train": data["X_train"],
            "y_train": y_train,
            "X_test": data.get("X_test"),
            "y_test": y_test,
            "batch_size": getattr(self.config, "LIGHTNING_BATCH_SIZE", 32),
            "val_size": getattr(self.config, "LIGHTNING_VAL_SIZE", 0.2),
            "num_workers": getattr(self.config, "LIGHTNING_NUM_WORKERS", 0),
            "pin_memory": getattr(self.config, "LIGHTNING_PIN_MEMORY", False),
            "persistent_workers": getattr(self.config, "LIGHTNING_PERSISTENT_WORKERS", False),
            "seed": getattr(self.config, "LIGHTNING_SEED", getattr(self.config, "RANDOM_SEED", 42)),
            "target_columns": [target],
        }
        datamodule_kwargs.update(deepcopy(spec.get("datamodule_init_args", {})))
        datamodule_kwargs.setdefault("shuffle", True)
        datamodule_kwargs.setdefault("drop_last", False)
        datamodule_kwargs.setdefault("feature_columns", None)
        datamodule_kwargs.setdefault("target_columns", [target])

        datamodule = datamodule_cls(**datamodule_kwargs)
        datamodule.setup("fit")
        return datamodule

    def _build_model(self, spec: Mapping[str, Any], datamodule: LightningTabularDataModule):
        model_cls = self._dynamic_import(spec["import_path"])
        init_args = deepcopy(spec.get("init_args", {}))

        if "input_dim" in init_args and init_args["input_dim"] in (None, 0, "auto"):
            init_args["input_dim"] = datamodule.feature_dim
        if "num_features" in init_args and init_args["num_features"] in (None, 0, "auto"):
            init_args["num_features"] = datamodule.feature_dim
        if "output_dim" in init_args and init_args["output_dim"] in (None, 0, "auto"):
            init_args["output_dim"] = datamodule.target_dim

        return model_cls(**init_args)

    def _build_trainer_kwargs(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        trainer_kwargs = {
            "max_epochs": getattr(self.config, "LIGHTNING_MAX_EPOCHS", 50),
            "accelerator": getattr(self.config, "LIGHTNING_ACCELERATOR", "auto"),
            "devices": getattr(self.config, "LIGHTNING_DEVICES", "auto"),
            "precision": getattr(self.config, "LIGHTNING_PRECISION", "32-true"),
            "accumulate_grad_batches": getattr(self.config, "LIGHTNING_ACCUMULATE_GRAD_BATCHES", 1),
            "gradient_clip_val": getattr(self.config, "LIGHTNING_GRADIENT_CLIP_VAL", 0.0),
            "log_every_n_steps": getattr(self.config, "LIGHTNING_LOG_EVERY_N_STEPS", 1),
            "enable_checkpointing": True,
            "deterministic": True,
        }
        trainer_kwargs.update(deepcopy(spec.get("trainer_args", {})))
        return trainer_kwargs

    def _build_callback_specs(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        callbacks = {
            "early_stopping": {
                "monitor": getattr(self.config, "LIGHTNING_EARLY_STOPPING_MONITOR", "val_loss"),
                "mode": getattr(self.config, "LIGHTNING_EARLY_STOPPING_MODE", "min"),
                "patience": getattr(self.config, "LIGHTNING_EARLY_STOPPING_PATIENCE", 5),
            },
            "checkpoint": {
                "monitor": getattr(self.config, "LIGHTNING_CHECKPOINT_MONITOR", "val_loss"),
                "mode": getattr(self.config, "LIGHTNING_CHECKPOINT_MODE", "min"),
                "save_top_k": getattr(self.config, "LIGHTNING_SAVE_TOP_K", 1),
            },
        }

        if spec.get("callbacks"):
            callbacks.update(deepcopy(spec["callbacks"]))

        return callbacks