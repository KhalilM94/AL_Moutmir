import importlib

class ModelConfigFactory:
    """Factory class to build model configurations dynamically based on a registry."""
    def __init__(self, model_registry):
        self.model_registry = model_registry
    
    @staticmethod
    def _dynamic_import(import_path):
        module_path, class_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    

    def build_model_configs(self, num_features):
        """Build dynamic model configurations from self.config.MODEL_REGISTRY."""
        model_configs = {}

        for name, spec in self.model_registry.items():
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
                model_instance = ModelClass(**init_args)

            model_configs[name] = {
                "model": model_instance,
                "params": spec.get("params", {})
            }

        return model_configs