from __future__ import annotations

import importlib
import inspect
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

import numpy as np

from yg_eo_soilnet.datamodules.lightning.spatiotemporal_graph_builder import SpatiotemporalGraphBuilder
from yg_eo_soilnet.seeding import seed_everything


@dataclass
class LightningModelBundle:
    name: str
    target: str
    model: Any
    datamodule: Any
    trainer_kwargs: dict[str, Any]
    callback_specs: dict[str, Any]
    registry_entry: dict[str, Any]


class LightningConfigFactory:
    def __init__(
        self,
        registry: Mapping[str, dict],
        config: Any,
        logger: Any = None,
        data_manager: Any = None,
        datamodule_cache: MutableMapping[Any, Any] | None = None,
    ):
        self.registry = registry
        self.config = config
        self.logger = logger
        self.data_manager = data_manager
        # Opt-in reuse of datamodules across factories, keyed by the arguments that built them.
        # Off by default, so every existing call site behaves exactly as before. The hyperparameter
        # search turns it on: SoilSequenceDataModule deep-copies the whole bundle in __init__ and
        # re-fits normalization and vocabularies in setup(), which dominates the cost of a short
        # trial. Sharing is safe because setup() is idempotent and a datamodule holds no model state.
        self.datamodule_cache = datamodule_cache

    @staticmethod
    def _dynamic_import(import_path: str):
        module_path, attr_name = import_path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(f"Failed to import module '{module_path}': {exc}") from exc
        return getattr(module, attr_name)

    def build_lightning_configs(
        self, target: str, data: Mapping[str, Any], seed: int | None = None
    ) -> dict[str, LightningModelBundle]:
        """Build one bundle per enabled entry.

        `seed`, when given, is applied immediately before each model is constructed. Weight
        initialization draws from the global torch generator, so this is the only point at which
        seeding reaches the weights - seeding later leaves them at whatever state the preceding work
        happened to leave behind, and a tuned configuration cannot then reproduce the trial that
        selected it. Per entry rather than once for the whole loop, so a second enabled entry does
        not inherit the stream the first one consumed.
        """
        bundles: dict[str, LightningModelBundle] = {}
        for name, spec in self.registry.items():
            if not spec.get("enabled", False):
                continue

            self._validate_entry(name, spec)
            datamodule = self._build_datamodule(target=target, spec=spec, data=data)
            if seed is not None:
                seed_everything(int(spec.get("random_seed", seed)))
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

    @staticmethod
    def _input_kind(spec: Mapping[str, Any]) -> str:
        """The datamodule shape an entry declares. An explicit `input_kind` wins."""
        return spec.get("input_kind", spec.get("datamodule_type", "tabular"))

    def graph_spec(self) -> dict[str, Any] | None:
        """The enabled registry entry that needs a spatiotemporal graph, if any."""
        for spec in self.registry.values():
            if spec.get("enabled", False) and self._input_kind(spec) == "graph":
                return spec
        return None

    def has_graph_input(self) -> bool:
        return self.graph_spec() is not None

    def sequence_spec(self) -> dict[str, Any] | None:
        """The enabled registry entry that needs a sequence bundle, if any."""
        for spec in self.registry.values():
            if spec.get("enabled", False) and self._input_kind(spec) == "sequence":
                return spec
        return None

    def has_sequence_input(self) -> bool:
        return self.sequence_spec() is not None

    def covers_all_targets_in_one_run(self) -> bool:
        """True when an enabled datamodule spans every point, and so every target, in one pass.

        Both the graph and the sequence datamodules are built from the whole dataset at once, so a
        multi-target run is one combined run rather than one run per target.
        """
        return self.has_graph_input() or self.has_sequence_input()

    def _validate_entry(self, name: str, spec: Mapping[str, Any]) -> None:
        required_keys = {"enabled", "modeltype", "import_path", "datamodule_import_path"}
        missing = sorted(required_keys - set(spec))
        if missing:
            raise ValueError(f"Lightning registry entry '{name}' is missing required keys: {', '.join(missing)}")
        if spec.get("modeltype") != "dl":
            raise ValueError(f"Lightning registry entry '{name}' must use modeltype 'dl'.")

    def _build_datamodule(self, target: str, spec: Mapping[str, Any], data: Mapping[str, Any]) -> Any:
        input_kind = self._input_kind(spec)
        if input_kind not in {"graph", "sequence"}:
            raise ValueError(
                f"Unsupported input_kind {input_kind!r} in the Lightning registry; "
                "expected 'graph' or 'sequence'"
            )

        datamodule_cls = self._dynamic_import(spec["datamodule_import_path"])
        fallback_seed = int(
            spec.get(
                "random_seed",
                getattr(self.config, "RANDOM_SEED", 42),
            )
        )

        datamodule_kwargs = deepcopy(spec.get("datamodule_init_args", {}))
        datamodule_kwargs.setdefault("batch_size", getattr(self.config, "LIGHTNING_BATCH_SIZE", 32))
        datamodule_kwargs.setdefault("val_size", getattr(self.config, "LIGHTNING_VAL_SIZE", 0.2))
        datamodule_kwargs.setdefault("test_size", getattr(self.config, "TEST_SIZE", 0.2))
        datamodule_kwargs.setdefault("num_workers", getattr(self.config, "LIGHTNING_NUM_WORKERS", 0))
        datamodule_kwargs.setdefault("pin_memory", getattr(self.config, "LIGHTNING_PIN_MEMORY", False))
        datamodule_kwargs.setdefault(
            "persistent_workers", getattr(self.config, "LIGHTNING_PERSISTENT_WORKERS", False)
        )
        datamodule_kwargs.setdefault("seed", fallback_seed)

        # Built before the payload is attached: the payload is a whole dataset, so it is identified
        # by object identity rather than by value.
        cache_key = (spec["datamodule_import_path"], repr(sorted(datamodule_kwargs.items())))

        if input_kind == "graph":
            payload = data.get("spatiotemporal_graph")
            if payload is None:
                payload = self._build_spatiotemporal_graph(spec)
            datamodule_kwargs["spatiotemporal_graph"] = payload
        else:
            payload = data.get("sequence_bundle")
            if payload is None:
                payload = self._build_sequence_bundle(spec)
            datamodule_kwargs["sequence_bundle"] = payload

        if self.datamodule_cache is not None:
            cache_key = (*cache_key, id(payload))
            cached = self.datamodule_cache.get(cache_key)
            if cached is not None:
                return cached

        datamodule = datamodule_cls(**datamodule_kwargs)
        datamodule.setup("fit")
        if self.datamodule_cache is not None:
            self.datamodule_cache[cache_key] = datamodule
        return datamodule

    def _build_sequence_bundle(self, spec: Mapping[str, Any]) -> Any:
        """Build the sequence bundle on demand, mirroring the graph path."""
        if self.data_manager is None:
            raise KeyError(
                "Sequence lightning registry entries require either a 'sequence_bundle' payload "
                "or a data_manager on LightningConfigFactory"
            )
        # Imported here rather than at module scope so the graph path does not pay for it, and vice
        # versa - neither branch should drag the other's dependencies in.
        from yg_eo_soilnet.datamodules.sequence.sequence_builder import SoilSequenceBuilder

        builder = SoilSequenceBuilder(self.config, self.logger, self.data_manager)
        return builder.build(sequence_data_args=dict(spec.get("sequence_data_args", {}) or {}))

    def _build_spatiotemporal_graph(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        """Build the graph bundle on demand; graph construction belongs to the Lightning layer."""
        if self.data_manager is None:
            raise KeyError(
                "Graph lightning registry entries require either a 'spatiotemporal_graph' payload "
                "or a data_manager on LightningConfigFactory"
            )
        builder = SpatiotemporalGraphBuilder(self.config, self.logger, self.data_manager)
        return builder.build(graph_data_args=self._graph_data_args(spec))

    @staticmethod
    def _graph_data_args(spec: Mapping[str, Any]) -> dict[str, Any]:
        graph_data_args = dict(spec.get("graph_data_args", {}) or {})
        init_args = dict(spec.get("init_args", {}) or {})
        if "spatial_graph_enabled" in init_args and "spatial_graph_enabled" not in graph_data_args:
            graph_data_args["spatial_graph_enabled"] = init_args["spatial_graph_enabled"]
        return graph_data_args

    @staticmethod
    def _accepted_init_args(model_cls: Any) -> set[str] | None:
        """Keyword names ``model_cls.__init__`` accepts, or None when it takes ``**kwargs``."""
        try:
            parameters = inspect.signature(model_cls.__init__).parameters
        except (TypeError, ValueError):  # pragma: no cover - builtins and C extensions
            return None
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return None
        return {
            name
            for name, parameter in parameters.items()
            if name != "self"
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }

    def _build_model(self, spec: Mapping[str, Any], datamodule: Any):
        model_cls = self._dynamic_import(spec["import_path"])
        init_args = deepcopy(spec.get("init_args", {}))
        # Several models share one datamodule, so what the datamodule can offer is not what a given
        # model wants: `grid_years` is meaningful to the calendar-grid CNN and meaningless to the
        # sequence encoders, yet both read the same SoilSequenceDataModule. Injections the factory
        # makes on its own initiative are therefore filtered to what the model actually accepts.
        # Anything the registry names explicitly is still passed through untouched, so a typo in
        # init_args fails loudly instead of being silently dropped.
        accepted = self._accepted_init_args(model_cls)

        def offer(key: str, value: Any) -> None:
            if accepted is None or key in accepted:
                init_args[key] = value

        shape_args = {
            "static_dim": getattr(datamodule, "static_dim", None),
            "target_dim": getattr(datamodule, "target_dim", None),
            "modality_dims": getattr(datamodule, "modality_dims", None),
            "temporal_steps": getattr(datamodule, "temporal_steps", None),
            "edge_attr_dim": getattr(datamodule, "edge_attr_dim", None),
            # Calendar-grid span, inferred from the data by the sequence datamodule. Only the CNN
            # rasterises, so only it declares this argument.
            "grid_years": getattr(datamodule, "grid_years", None),
            # Entity-embedding contract, fitted train-only by the datamodule's setup(). The
            # vocabularies travel into the model's hyper_parameters so the checkpoint carries its
            # own label->index mapping instead of re-deriving one from whatever frame it is given.
            "categorical_cardinalities": getattr(datamodule, "categorical_cardinalities", None),
            "categorical_vocabularies": getattr(datamodule, "categorical_vocabularies", None),
            "categorical_feature_names": getattr(datamodule, "categorical_feature_names", None),
        }
        # An empty list is "this dataset has no categoricals", not data worth offering - without it
        # in the sentinel set a model would be handed [] as though it were a real shape.
        unset = (None, 0, "auto", {}, [])
        for key, value in shape_args.items():
            if key in init_args and init_args[key] in (None, 0, "auto", {}):
                init_args[key] = value
            elif key not in init_args and value not in unset:
                offer(key, value)

        if "temporal_enabled" in init_args and init_args["temporal_enabled"] in (None, "auto"):
            init_args["temporal_enabled"] = getattr(datamodule, "temporal_enabled", False)
        elif "temporal_enabled" not in init_args:
            offer("temporal_enabled", getattr(datamodule, "temporal_enabled", False))
        if "output_dim" in init_args and init_args["output_dim"] in (None, 0, "auto"):
            init_args["output_dim"] = getattr(datamodule, "target_dim", 1)
        # Target standardization stats, so the model can invert them for prediction. Passed as
        # plain floats: numpy values here end up in the checkpoint's hyper_parameters and make
        # it unloadable under torch.load's weights_only=True default.
        for key, attribute in (("target_mean", "target_mean_"), ("target_scale", "target_scale_")):
            value = getattr(datamodule, attribute, None)
            if value is not None and init_args.get(key) in (None, "auto"):
                offer(key, [float(item) for item in np.asarray(value).ravel()])
        # The datamodule owns the choice of target transform; the model only needs to know so it can
        # invert it. Propagating it here keeps the two from ever disagreeing.
        if init_args.get("target_transform") in (None, "auto"):
            offer("target_transform", getattr(datamodule, "target_transform", None))

        return model_cls(**init_args)

    def _build_trainer_kwargs(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        trainer_kwargs = deepcopy(spec.get("trainer_args", {}))
        trainer_kwargs.setdefault("max_epochs", getattr(self.config, "LIGHTNING_MAX_EPOCHS", 50))
        trainer_kwargs.setdefault("accelerator", getattr(self.config, "LIGHTNING_ACCELERATOR", "auto"))
        trainer_kwargs.setdefault("devices", getattr(self.config, "LIGHTNING_DEVICES", "auto"))
        trainer_kwargs.setdefault("precision", getattr(self.config, "LIGHTNING_PRECISION", "32-true"))
        trainer_kwargs.setdefault(
            "accumulate_grad_batches", getattr(self.config, "LIGHTNING_ACCUMULATE_GRAD_BATCHES", 1)
        )
        trainer_kwargs.setdefault("gradient_clip_val", getattr(self.config, "LIGHTNING_GRADIENT_CLIP_VAL", 0.0))
        trainer_kwargs.setdefault("log_every_n_steps", getattr(self.config, "LIGHTNING_LOG_EVERY_N_STEPS", 1))
        trainer_kwargs.setdefault("enable_checkpointing", True)
        trainer_kwargs.setdefault("deterministic", True)
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

        # Merged group by group, not `callbacks.update(...)`. Replacing a whole group meant an entry
        # that named only `early_stopping: {patience: 3}` silently lost monitor and mode, i.e. lost
        # the config.LIGHTNING_EARLY_STOPPING_* defaults this dict was just built from.
        for group, overrides in deepcopy(spec.get("callbacks") or {}).items():
            if isinstance(overrides, Mapping) and isinstance(callbacks.get(group), MutableMapping):
                callbacks[group].update(overrides)
            else:
                callbacks[group] = overrides

        return callbacks