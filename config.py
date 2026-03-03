import os
import random
import yaml
import json
from typing import Any, Optional

class Config:
    def __init__(self, config_path: Optional[str] = None, registry_path: Optional[str] = None):
        self.config_path = config_path or os.getenv('CONFIG_PATH', 'configs/config.yml')
        self.registry_path = registry_path or os.getenv('MODEL_REGISTRY_PATH', 'configs/model_registry.yml')

        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

        # General settings
        self.DATA_FOLDER = self._get_config('DATA_FOLDER', 'doukkala_ssl_datasets')
        self.DATA_FILE = self._get_config('DATA_FILE', 'data.csv')
        self.SOIL_GROUPS_FILE_PATH = self._get_config('SOIL_GROUPS_FILE_PATH', 'soil_groups.txt')
        self.RANDOM_SEED = self._get_config('RANDOM_SEED', random.randint(0, 1000000))
        self.TEST_SIZE = self._get_config('TEST_SIZE', 0.2)
        self.ENABLE_CLUSTERING = self._get_config('CLUSTERING_STRATEGY', None).get('enabled', False)
        self.CLUSTERING_STRATEGY = self._get_config('CLUSTERING_STRATEGY', None)
        self.SPLIT_STRATEGY = self._get_config('SPLIT_STRATEGY', 'kfold')

        # Target and feature configuration
        self.IGNORE_BANDS = self._get_ignore_bands([])
        self.COLUMNS_TO_TRANSFORM = self._get_config('COLUMNS_TO_TRANSFORM', [])
        self.TARGET_COLUMNS = self._get_config('TARGET_COLUMNS', [])
        self.CATEGORICAL_FEATURES = self._get_config('CATEGORICAL_FEATURES', [])
        self.EXCLUDE_CATEGORICAL = self._get_config('EXCLUDE_CATEGORICAL', [])
        self.ELIMINATED_FEATURES = self._get_config('ELIMINATED_FEATURES', [])

        # Model registry loaded here
        self.MODEL_REGISTRY = self._load_model_registry()
        self.BANDS_CSV_PATH = self._get_config('BANDS_CSV_PATH', None)
        self.WORLDCLIM_CSV_PATH = self._get_config('WORLDCLIM_CSV_PATH', None)

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
        config_val = self.config.get(key, default)
        if isinstance(default, list) and config_val is None:
            return []
        return config_val

    def _load_model_registry(self):
        try:
            with open(self.registry_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model registry YAML not found at {self.registry_path}. Stopping execution.")
    
    def _get_ignore_bands(self, default: Any) -> list:
        if self._get_config('IGNORE_BANDS', False):
            return [f"Band_{i}" for i in range(1, self._get_config('N_BANDS', 234) + 1)]
        else:
            return default if isinstance(default, list) else []
