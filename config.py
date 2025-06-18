import os
import random
import yaml
import json
import numpy as np
from typing import Any

class Config:
    def __init__(self):
        config_path = os.getenv('CONFIG_PATH', 'config.yml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

        # General settings
        self.DATA_FOLDER = self._get_config('DATA_FOLDER', 'doukkala_ssl_datasets')
        self.OUTPUT_FOLDER = self._get_config('OUTPUT_FOLDER', 'outputs')
        self.RANDOM_SEED = self._get_config('RANDOM_SEED', random.randint(0, 1000000))
        self.TEST_SIZE = self._get_config('TEST_SIZE', 0.2)
        self.ENABLE_CLUSTERING = self._get_config('CLUSTERING_STRATEGY', None).get('enabled', False)
        self.CLUSTERING_STRATEGY = self._get_config('CLUSTERING_STRATEGY', None)
        self.SPLIT_STRATEGY = self._get_config('SPLIT_STRATEGY', 'kfold')

        # Model training settings
        self.ENABLE_TUNING = self._get_config('ENABLE_TUNING', False)
        self.USE_BAYES_OPT = self._get_config('USE_BAYES_OPT', True)
        self.ENABLE_RFE = self._get_config('ENABLE_RFE', False)

        # Target and feature configuration
        self.COLUMNS_TO_TRANSFORM = self._get_config('COLUMNS_TO_TRANSFORM', [...])
        self.TARGET_COLUMNS = self._get_config('TARGET_COLUMNS', [...])
        self.ELIMINATED_FEATURES = self._get_config('ELIMINATED_FEATURES', [...])

        # Model registry loaded here
        self.MODEL_REGISTRY = self._load_model_registry()

    def _get_config(self, key: str, default: Any) -> Any:
        val = os.environ.get(key)
        if val is not None:
            if isinstance(default, bool):
                return val.lower() in ('true', '1', 't', 'y', 'yes')
            elif isinstance(default, int):
                return int(val)
            elif isinstance(default, float):
                return float(val)
            elif isinstance(default, list):
                return [x.strip() for x in val.split(',')]
            elif isinstance(default, dict):
                try:
                    return json.loads(val)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON format for environment variable {key}: {val}")
            return val
        return self.config.get(key, default)

    def _load_model_registry(self):
        registry_path = os.getenv("MODEL_REGISTRY_PATH", "model_registry.yml")
        try:
            with open(registry_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model registry YAML not found at {registry_path}. Stopping execution.")