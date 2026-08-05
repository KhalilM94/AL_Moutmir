from yg_eo_soilnet.clustering_utils import BaseSpatialClusterStrategy
import importlib
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfigFactory:
    """Factory class to build model configurations dynamically based on a registry."""
    registry:dict
    random_state:int = 42
    
    @staticmethod
    def _dynamic_import(import_path):
        module_path, class_name = import_path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(f"Failed to import module '{module_path}': {e}")
        return getattr(module, class_name)
    

    def build_model_configs(self, num_features, default_seed: int | None = None):
        """Build dynamic model configurations from self.config.MODEL_REGISTRY."""
        model_configs = {}

        for name, spec in self.registry.items():
            if not spec.get("enabled", False):
                continue
            try:
                ModelClass = self._dynamic_import(spec["import_path"])
            except Exception as e:
                print(f"[Warning] Failed to import {name}: {e}")
                continue

            init_args = spec.get("init_args", {})
            custom_model_builder = spec.get("custom_model_builder", None)

            # Handle Keras or other wrappers with custom model builders
            if custom_model_builder:
                builder_func = self._dynamic_import(custom_model_builder)
                model_instance = ModelClass(build_fn=lambda: builder_func(num_features))
            else:
                if 'input_dim' in init_args:
                    init_args['input_dim'] = num_features
                model_instance = ModelClass(**init_args)

            model_configs[name] = {
                "model": model_instance,
                "params": spec.get("params", {}),
                "modeltype": spec.get("modeltype", "ml"),
                "random_seed": spec.get("random_seed", default_seed),
            }

        return model_configs
    
    def load_splitter_from_config(self) -> Optional[BaseSpatialClusterStrategy]:
        """Load and return the splitter configuration from the registry."""
        if self.registry.get("enabled", True) is True:
            class_path = self.registry["class_path"]
            params = self.registry.get("params", {})
            params.setdefault("random_state", self.random_state)

            SplitterClass = self._dynamic_import(class_path)
            return SplitterClass(**params)
        else:
            print("[Warning] Splitter configuration not found or disabled in the registry.")
            return None