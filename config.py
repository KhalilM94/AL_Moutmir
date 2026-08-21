import os
import random
import yaml
import json
from copy import deepcopy
from typing import Any, Mapping, Optional

# Reserved top-level key in the Lightning registry holding settings shared by every entry.
LIGHTNING_REGISTRY_DEFAULTS_KEY = 'defaults'


def deep_merge(base: Mapping, override: Mapping) -> dict:
    """`override` on top of `base`, recursing into nested mappings.

    A mapping on both sides merges key by key; anything else in `override` - a scalar, a list -
    replaces. So an entry naming `trainer_args.max_epochs` keeps the rest of the shared trainer args,
    while `head_hidden_dims: [64]` replaces the default list outright rather than merging into it.
    """
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


class Config:
    def __init__(
        self,
        config_path: Optional[str] = None,
        registry_path: Optional[str] = None,
        lightning_registry_path: Optional[str] = None,
    ):
        self.config_path = config_path or os.getenv('CONFIG_PATH', 'configs/main_config.yml')
        self._config_dir = os.path.dirname(os.path.abspath(self.config_path))

        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

        self.DATA_SPEC_CONFIG = {}
        self.SKLEARN_CONFIG = {}
        self.LIGHTNING_CONFIG = {}

        self.COMMON_CONFIG = self._normalize_mapping(self.config.get('common', {}))
        self.DATA_CONFIG = self._normalize_mapping(self.config.get('data', {}))
        common_data = self._normalize_mapping(self.COMMON_CONFIG.get('data', {}))
        if common_data:
            self.DATA_CONFIG = {**self.DATA_CONFIG, **common_data}
        self.data_spec_path = self._resolve_config_path(self._get_config('DATA_SPEC_PATH', self._get_config('data_spec_path', None)))
        self.MAIN_TEMPORAL_FEATURES = self._normalize_mapping(self.config.get('temporal', {}))
        common_temporal = self._normalize_mapping(self.COMMON_CONFIG.get('temporal', {}))
        if common_temporal:
            self.MAIN_TEMPORAL_FEATURES = {**self.MAIN_TEMPORAL_FEATURES, **common_temporal}
        self.sklearn_config_path = self._resolve_config_path(
            self._get_config('SKLEARN_CONFIG_PATH', self._get_config('sklearn_config_path', None))
        )
        self.lightning_config_path = self._resolve_config_path(
            self._get_config('LIGHTNING_CONFIG_PATH', self._get_config('lightning_config_path', None))
        )

        self.DATA_SPEC_CONFIG = self._load_yaml_mapping(self.data_spec_path)
        self.SKLEARN_CONFIG = self._load_yaml_mapping(self.sklearn_config_path)
        self.SKLEARN_CATEGORICAL_CONFIG = self._normalize_mapping(self.SKLEARN_CONFIG.get('categorical', {}))
        self.LIGHTNING_CONFIG = self._load_yaml_mapping(self.lightning_config_path)
        self.EXISTING_HS_FEATURES = self._normalize_mapping(self._get_config('existing_hs_features', {}))

        registry_path_value = (
            registry_path
            or os.getenv('MODEL_REGISTRY_PATH')
            or self._get_config('SKLEARN_REGISTRY_PATH', 'configs/sklearn/model_registry.yml')
        )
        lightning_registry_path_value = (
            lightning_registry_path
            or os.getenv('LIGHTNING_MODEL_REGISTRY_PATH')
            or self._get_config('LIGHTNING_REGISTRY_PATH', 'configs/lightning/lightning_registry.yml')
        )

        self.registry_path = self._resolve_config_path(registry_path_value)
        self.lightning_registry_path = self._resolve_config_path(lightning_registry_path_value)

        self.TEMPORAL_FEATURES = {
            **self._normalize_mapping(self.LIGHTNING_CONFIG.get('temporal', {})),
            **self.MAIN_TEMPORAL_FEATURES,
        }
        if not self.TEMPORAL_FEATURES:
            self.TEMPORAL_FEATURES = self._normalize_mapping(self._get_config('TEMPORAL_FEATURES', {}))

        # General settings
        self.DATA_FOLDER = self._get_data_config('root', 'DATA_FOLDER', 'doukkala_ssl_datasets')
        self.DATA_ROOT = self.DATA_FOLDER
        self.DATA_INDEX_MANIFEST = self._get_data_config('manifest', 'DATA_INDEX_MANIFEST', None)
        self.DATA_INDEX_MANIFEST_PATH = self._resolve_data_path(self.DATA_INDEX_MANIFEST)
        self.DATA_MANIFEST_PATH = self.DATA_INDEX_MANIFEST_PATH
        self.DATA_FILE = self._get_data_config('static', 'DATA_FILE', 'data.csv')
        self.STATIC_FEATURES_FILE = self._get_config('STATIC_FEATURES_FILE', self.DATA_FILE)
        self.TARGETS_FILE = self._get_data_config('targets', 'TARGETS_FILE', self.DATA_FILE)
        self.STATIC_FEATURES_FOLDER = self._resolve_data_path(self._get_config('STATIC_FEATURES_FOLDER', None))
        self.TARGETS_FOLDER = self._resolve_data_path(self._get_config('TARGETS_FOLDER', None))
        self.STATIC_CSV_PATH = self._resolve_data_path(self.STATIC_FEATURES_FILE)
        self.TARGETS_CSV_PATH = self._resolve_data_path(self.TARGETS_FILE)
        self.TIMESERIES_FOLDER = self._resolve_data_path(
            self._get_temporal_config('timeseries_folder', 'TIMESERIES_FOLDER', None)
        )
        self.TIMESERIES_CSV_PATH = self._resolve_data_path(
            self._get_data_config('timeseries', 'TIMESERIES_CSV_PATH', None)
            or self._get_temporal_config('timeseries_file', 'TIMESERIES_CSV_PATH', None)
            or self._get_temporal_config('timeseries_csv_path', 'TIMESERIES_CSV_PATH', None)
        )

        # Unified source resolution: one path per source, each a file or a folder.
        self.STATIC_SOURCE = self.STATIC_FEATURES_FOLDER or self.STATIC_CSV_PATH
        self.TARGETS_SOURCE = self.TARGETS_FOLDER or self._explicit_targets_path()
        self.TIMESERIES_SOURCE = self.TIMESERIES_FOLDER or self.TIMESERIES_CSV_PATH
        self.POINT_ID_COLUMN = self._get_config('POINT_ID_COLUMN', 'point_id')
        self.LAT_COLUMN = self._get_config('LAT_COLUMN', 'lat')
        self.LON_COLUMN = self._get_config('LON_COLUMN', 'lon')
        self.TIME_COLUMN = self._get_temporal_config('time_column', 'TIME_COLUMN', 'date')
        self.TEMPORAL_FEATURES_ENABLED = self._get_temporal_config('enabled', 'TEMPORAL_FEATURES_ENABLED', False)
        self.MODALITY_PREFIX_MAP = self._normalize_mapping(
            self._get_temporal_config('modality_prefix_map', 'MODALITY_PREFIX_MAP', {})
        )
        self.S1_COLUMNS = self._get_temporal_config('s1_columns', 'S1_COLUMNS', [])
        self.S2_COLUMNS = self._get_temporal_config('s2_columns', 'S2_COLUMNS', [])
        self.MODIS_COLUMNS = self._get_temporal_config('modis_columns', 'MODIS_COLUMNS', [])
        self.SPATIAL_RADIUS = self._get_config('SPATIAL_RADIUS', 50000)
        self.BASELINE_METHOD = self._get_config('BASELINE_METHOD', 'knn')
        self.BASELINE_K_NEIGHBORS = self._get_config('BASELINE_K_NEIGHBORS', 5)
        self.LIGHTNING_BATCH_SIZE = self._get_config('LIGHTNING_BATCH_SIZE', 32)
        self.LIGHTNING_VAL_SIZE = self._get_config('LIGHTNING_VAL_SIZE', 0.2)
        self.LIGHTNING_NUM_WORKERS = self._get_config('LIGHTNING_NUM_WORKERS', 0)
        self.LIGHTNING_PIN_MEMORY = self._get_config('LIGHTNING_PIN_MEMORY', False)
        self.LIGHTNING_PERSISTENT_WORKERS = self._get_config('LIGHTNING_PERSISTENT_WORKERS', False)
        self.LIGHTNING_MAX_EPOCHS = self._get_config('LIGHTNING_MAX_EPOCHS', 50)
        self.LIGHTNING_ACCELERATOR = self._get_config('LIGHTNING_ACCELERATOR', 'auto')
        self.LIGHTNING_DEVICES = self._get_config('LIGHTNING_DEVICES', 'auto')
        self.LIGHTNING_PRECISION = self._get_config('LIGHTNING_PRECISION', '32-true')
        self.LIGHTNING_ENABLE_DEFAULT_LOGGER = self._get_config('LIGHTNING_ENABLE_DEFAULT_LOGGER', True)
        self.LIGHTNING_ACCUMULATE_GRAD_BATCHES = self._get_config('LIGHTNING_ACCUMULATE_GRAD_BATCHES', 1)
        self.LIGHTNING_GRADIENT_CLIP_VAL = self._get_config('LIGHTNING_GRADIENT_CLIP_VAL', 0.0)
        self.LIGHTNING_LOG_EVERY_N_STEPS = self._get_config('LIGHTNING_LOG_EVERY_N_STEPS', 1)
        self.MAIN_FILE_LOGGING_ENABLED = self._get_config('MAIN_FILE_LOGGING_ENABLED', True)
        # Failure policy for the sklearn training loop.
        self.FAIL_ON_MODEL_ERROR = self._get_config('FAIL_ON_MODEL_ERROR', False)
        self.FAIL_IF_ALL_MODELS_FAIL_FOR_TARGET = self._get_config('FAIL_IF_ALL_MODELS_FAIL_FOR_TARGET', True)
        self.SKLEARN_FILE_LOGGING_ENABLED = self._get_config('SKLEARN_FILE_LOGGING_ENABLED', True)
        self.MLFLOW_EXPERIMENT_EXPORT_ENABLED = self._get_config('MLFLOW_EXPERIMENT_EXPORT_ENABLED', False)
        self.MLFLOW_EXPERIMENT_EXPORT_PATH = self._get_config('MLFLOW_EXPERIMENT_EXPORT_PATH', 'mlflow_exports')
        # Where runs are recorded, and under which experiment. Both are configurable because an
        # experiment's artifact_location is an ABSOLUTE path baked in at creation time: the original
        # experiment was created in a different checkout, so its metadata and its artifacts have
        # been landing in two different directories ever since. A new experiment created under the
        # current tracking root gets a correct artifact_location from MLflow automatically.
        self.MLFLOW_TRACKING_URI = self._get_config('MLFLOW_TRACKING_URI', '')
        self.MLFLOW_EXPERIMENT_NAME = self._get_config('MLFLOW_EXPERIMENT_NAME', 'Soil_Model_Training_v2')
        # Enter each fitted model into the MLflow Model Registry as a new version of
        # <target>_<model>, so deployment can reference models:/<name>/<version> or the champion
        # alias instead of a run-scoped URI. false for throwaway experiments that should not
        # accumulate versions.
        self.MLFLOW_REGISTER_MODELS = self._get_config('MLFLOW_REGISTER_MODELS', True)
        # SHAP explainability. EXPLAIN_ENABLED is a real off-switch, not just a plot suppressor:
        # when it is false the logger returns before importing shap at all, so a run that does not
        # want explanations does not pay for numba's import either.
        self.EXPLAIN_ENABLED = self._get_config('EXPLAIN_ENABLED', True)
        self.EXPLAIN_MAX_SAMPLES = self._get_config('EXPLAIN_MAX_SAMPLES', 500)
        self.EXPLAIN_BACKGROUND_SAMPLES = self._get_config('EXPLAIN_BACKGROUND_SAMPLES', 100)
        self.EXPLAIN_MAX_DISPLAY = self._get_config('EXPLAIN_MAX_DISPLAY', 25)
        # Ceiling on model evaluations for the model-agnostic explainer, which is what any model
        # that is neither a tree nor linear falls back to. Without it, TabICL cost ~1.08M forward
        # passes and the run never terminated.
        self.EXPLAIN_MAX_EVALS = self._get_config('EXPLAIN_MAX_EVALS', 200000)
        # Empty means every model. Named entries restrict it, because SHAP on a large TabICL
        # regressor costs far more than on XGBoost and you want that choice per model, not global.
        self.EXPLAIN_MODELS = self._get_config('EXPLAIN_MODELS', [])
        # Models never explained unless EXPLAIN_MODELS names them explicitly. TabICL is an
        # in-context learner: one prediction re-processes the training set, measured at ~13 ms per
        # row against microseconds for a tree. EXPLAIN_MAX_EVALS counts evaluations and cannot see
        # that difference, so the exclusion is by name rather than by budget.
        self.EXPLAIN_SKIP_MODELS = self._get_config('EXPLAIN_SKIP_MODELS', ['TabICL'])
        self.EXPLAIN_FAIL_ON_ERROR = self._get_config('EXPLAIN_FAIL_ON_ERROR', False)
        self.LIGHTNING_EARLY_STOPPING_MONITOR = self._get_config('LIGHTNING_EARLY_STOPPING_MONITOR', 'val_loss')
        self.LIGHTNING_EARLY_STOPPING_MODE = self._get_config('LIGHTNING_EARLY_STOPPING_MODE', 'min')
        self.LIGHTNING_EARLY_STOPPING_PATIENCE = self._get_config('LIGHTNING_EARLY_STOPPING_PATIENCE', 5)
        self.LIGHTNING_CHECKPOINT_MONITOR = self._get_config('LIGHTNING_CHECKPOINT_MONITOR', 'val_loss')
        self.LIGHTNING_CHECKPOINT_MODE = self._get_config('LIGHTNING_CHECKPOINT_MODE', 'min')
        self.LIGHTNING_SAVE_TOP_K = self._get_config('LIGHTNING_SAVE_TOP_K', 1)
        self.RANDOM_SEED = self._get_config('RANDOM_SEED', random.randint(0, 1000000))
        self.TEST_SIZE = self._get_config('TEST_SIZE', 0.2)
        self.CLUSTERING_STRATEGY = self._get_config('CLUSTERING_STRATEGY', None)
        self.CLUSTERING_STRATEGY = self._normalize_mapping(self.CLUSTERING_STRATEGY)
        self.ENABLE_CLUSTERING = self.CLUSTERING_STRATEGY.get('enabled', False)
        # The INNER cross-validation strategy for sklearn's GridSearchCV ('kfold'/'groupkfold').
        # Not the holdout: that is SPLIT_HOLDOUT_STRATEGY below. The two are different decisions and
        # the names are kept distinct on purpose.
        self.SPLIT_STRATEGY = self._get_config('SPLIT_STRATEGY', 'kfold')

        # --- the unified train/val/test holdout, shared by every training family ---------------
        # One split, decided once over POINT_ID_COLUMN before the family fork, so sklearn and
        # Lightning score on the same test points. See datamodules/splitting.py.
        self.SPLIT_CONFIG = {
            **self._normalize_mapping(self.config.get('split', {})),
            **self._normalize_mapping(self.COMMON_CONFIG.get('split', {})),
        }
        # ENABLE_CLUSTERING is the legacy spelling of split.strategy: spatial_group. A config that
        # only ever turned clustering on keeps the grouped holdout it had - and now the Lightning
        # families get it too, which they never did before.
        legacy_grouped = 'spatial_group' if self.ENABLE_CLUSTERING else 'random'
        self.SPLIT_HOLDOUT_STRATEGY = self._get_split_config(
            'strategy', 'SPLIT_HOLDOUT_STRATEGY', legacy_grouped
        )
        # TEST_SIZE is honoured as the fallback so a config predating `split:` keeps working.
        self.SPLIT_TEST_SIZE = self._get_split_config('test_size', 'SPLIT_TEST_SIZE', self.TEST_SIZE)
        self.SPLIT_VAL_SIZE = self._get_split_config(
            'val_size', 'SPLIT_VAL_SIZE', self._get_config('LIGHTNING_VAL_SIZE', 0.2)
        )
        self.SPLIT_SEED = self._get_split_config('seed', 'SPLIT_SEED', self.RANDOM_SEED)
        self.SPLIT_POPULATION_POLICY = self._get_split_config(
            'population_policy', 'SPLIT_POPULATION_POLICY', 'intersect'
        )
        self.SPLIT_PLAN_PATH = self._get_split_config('plan_path', 'SPLIT_PLAN_PATH', None)
        # Floor on what `intersect` may leave behind. Intersecting is only sound while the families
        # roughly agree on which rows are usable; below this the narrowest family is dictating the
        # whole run's population, which is a data problem to fix rather than a split to accept.
        self.SPLIT_MIN_POPULATION_RATIO = self._get_split_config(
            'min_population_ratio', 'SPLIT_MIN_POPULATION_RATIO', 0.5
        )
        self.SPLIT_GROUP_STRATEGY = self._normalize_mapping(
            self._get_split_config('group', 'SPLIT_GROUP_STRATEGY', None)
        ) or dict(self.CLUSTERING_STRATEGY)
        if 'test_size' in self.SPLIT_CONFIG and 'TEST_SIZE' in getattr(self, 'COMMON_CONFIG', {}):
            print(
                "[Warning] Both TEST_SIZE and split.test_size are set; split.test_size wins for the "
                "unified holdout. Remove TEST_SIZE to avoid the ambiguity."
            )

        # --- data quality: one missingness rule for every training family ----------------------
        # A covariate blank on more than MAX_MISSING_COLUMN_RATIO of rows stops the run; anything
        # under it is median-filled and flagged rather than costing the whole row. See
        # datamodules/frame_cleaning.assert_columns_are_dense_enough.
        self.DATA_QUALITY_CONFIG = {
            **self._normalize_mapping(self.config.get('data_quality', {})),
            **self._normalize_mapping(self.COMMON_CONFIG.get('data_quality', {})),
        }
        self.MAX_MISSING_COLUMN_RATIO = self._get_data_quality_config(
            'max_missing_column_ratio', 'MAX_MISSING_COLUMN_RATIO', 0.2
        )
        self.ALLOW_SPARSE_COLUMNS = self._get_data_quality_config(
            'allow_sparse_columns', 'ALLOW_SPARSE_COLUMNS', []
        )
        self.FAIL_ON_SPARSE_COLUMNS = self._get_data_quality_config(
            'fail_on_sparse_columns', 'FAIL_ON_SPARSE_COLUMNS', True
        )

        # Target and feature configuration
        self.IGNORE_BANDS = self._get_ignore_bands([])
        self.COLUMNS_TO_TRANSFORM = self._get_config('COLUMNS_TO_TRANSFORM', [])
        self.TARGET_COLUMNS = self._get_config('TARGET_COLUMNS', self._get_config('target_columns', []))
        # How several targets are grouped into models. 'joint' fits ONE model with a target_dim-wide
        # head over all of them; 'per_target' fits an independent model each. Both families obey
        # this: before it existed Lightning was always joint and sklearn always per-target, and
        # neither was switchable. A registry entry may override it with its own `multi_target:` key.
        # Irrelevant when a single target is configured. See yg_eo_soilnet.targets.
        self.MULTI_TARGET_MODE = self._get_config('MULTI_TARGET_MODE', 'joint')
        # Every measured label, whether or not a model is fitted for it. Defaults to empty so a
        # config that has not adopted the key behaves exactly as before.
        self.LABEL_COLUMNS = self._get_config('LABEL_COLUMNS', self._get_config('label_columns', []))
        # Whether measured lab values travel with the data as AUXILIARY INPUTS. Off by default, so
        # a config that has not opted in behaves exactly as before. This governs availability only -
        # which columns reach the frame and the bundle - never whether they are predictors, which
        # metadata_columns still refuses for every LABEL_COLUMNS entry. A model then names the subset
        # it wants; see soil_cnn's auxiliary_label_columns.
        self.CARRY_LABEL_COLUMNS = self._get_config('CARRY_LABEL_COLUMNS', False)
        self.PREDICTOR_COLUMNS = self._get_config('PREDICTOR_COLUMNS', self._get_config('predictor_columns', []))
        self.IGNORED_COLUMNS = self._get_config('IGNORED_COLUMNS', self._get_config('ignored_columns', []))
        self.TREE_CATEGORICAL_ENCODING = self._get_sklearn_categorical_config('TREE_CATEGORICAL_ENCODING', 'onehot')
        self.TREE_ONEHOT_MAX_CATEGORIES = self._get_sklearn_categorical_config('TREE_ONEHOT_MAX_CATEGORIES', 30)
        self.MIN_FEATURE_COUNT = self._get_sklearn_categorical_config('MIN_FEATURE_COUNT', 10)
        # Whether to spend a full prediction pass over the TRAINING split on the r2_train_fit
        # overfitting diagnostic. See configs/sklearn/config.yml.
        self.LOG_TRAIN_FIT_METRIC = self._get_config('LOG_TRAIN_FIT_METRIC', True)
        self.MAX_FEATURE_DROP_RATIO_WARNING = self._get_sklearn_categorical_config(
            'MAX_FEATURE_DROP_RATIO_WARNING', 0.9
        )
        self.CATEGORICAL_FEATURES = self._get_config(
            'CATEGORICAL_FEATURES', self._get_sklearn_categorical_config('CATEGORICAL_FEATURES', [])
        )
        self.EXCLUDE_CATEGORICAL = self._get_sklearn_categorical_config('EXCLUDE_CATEGORICAL', [])
        self.ELIMINATED_FEATURES = self._get_config('ELIMINATED_FEATURES', self.IGNORED_COLUMNS)

        # Model registries loaded here
        self.MODEL_REGISTRY = self._load_model_registry()
        self.LIGHTNING_MODEL_REGISTRY = self._load_lightning_model_registry()

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
        for section in (
            getattr(self, 'COMMON_CONFIG', {}),
            getattr(self, 'DATA_SPEC_CONFIG', {}),
            getattr(self, 'SKLEARN_CONFIG', {}),
            getattr(self, 'LIGHTNING_CONFIG', {}),
        ):
            if key in section:
                config_val = section[key]
                if isinstance(default, list) and config_val is None:
                    return []
                return config_val
        config_val = self.config.get(key, default)
        if isinstance(default, list) and config_val is None:
            return []
        return config_val

    def _load_yaml_mapping(self, path_value: Optional[str]) -> dict:
        if not path_value:
            raise FileNotFoundError("Expected a config file path, but none was provided.")
        resolved_path = self._resolve_config_path(path_value)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Config YAML not found at {resolved_path}.")
        with open(resolved_path, 'r') as f:
            return self._normalize_mapping(yaml.safe_load(f) or {})

    def _load_model_registry(self):
        try:
            with open(self.registry_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Model registry YAML not found at {self.registry_path}. Stopping execution.")

    def _load_lightning_model_registry(self):
        try:
            with open(self.lightning_registry_path, 'r') as f:
                document = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Lightning model registry YAML not found at {self.lightning_registry_path}. Stopping execution."
            )

        # The trainer args and the callbacks were byte-identical on every entry, so they live once
        # under `defaults:` and are merged in here. Merging at load time rather than in the factory
        # means every consumer - LightningConfigFactory, tune.py, the HPO exporter, the tests - keeps
        # seeing one fully materialized entry and needs to know nothing about this. A tuned file from
        # configs/lightning/tuned/ carries no `defaults:` key, so for it this is a no-op.
        defaults = document.pop(LIGHTNING_REGISTRY_DEFAULTS_KEY, None) or {}
        return {name: deep_merge(defaults, spec or {}) for name, spec in document.items()}

    def _get_data_config(self, data_key: str, flat_key: str, default: Any) -> Any:
        """Read from the unified `data:` block, falling back to the legacy flat key."""
        if data_key in self.DATA_CONFIG and self.DATA_CONFIG[data_key] is not None:
            return self.DATA_CONFIG[data_key]
        return self._get_config(flat_key, default)

    def _explicit_targets_path(self) -> Optional[str]:
        """Targets path only when separately configured.

        TARGETS_FILE defaults to DATA_FILE, so a joint dataset would otherwise look like it has a
        separate targets source. Returning None here keeps 'joint file' detectable.
        """
        if self.DATA_CONFIG.get('targets'):
            return self._resolve_data_path(self.DATA_CONFIG['targets'])
        for key in ('TARGETS_FILE', 'TARGETS_CSV_PATH'):
            value = self._get_config(key, None)
            if value:
                return self._resolve_data_path(value)
        return None

    def _get_temporal_config(self, temporal_key: str, flat_key: str, default: Any) -> Any:
        if temporal_key in self.TEMPORAL_FEATURES and self.TEMPORAL_FEATURES[temporal_key] is not None:
            return self.TEMPORAL_FEATURES[temporal_key]
        return self._get_config(flat_key, default)

    def _get_split_config(self, split_key: str, flat_key: str, default: Any) -> Any:
        """Read `split.<split_key>`, falling back to a flat key and then the default.

        Mirrors _get_temporal_config, so the nested `split:` block behaves like `temporal:` - an
        env var of the flat name still overrides everything, which is what makes an A/B of
        SPLIT_POPULATION_POLICY a one-liner.
        """
        env_value = os.environ.get(flat_key)
        if env_value is None and split_key in self.SPLIT_CONFIG and self.SPLIT_CONFIG[split_key] is not None:
            return self.SPLIT_CONFIG[split_key]
        return self._get_config(flat_key, default)

    def _get_data_quality_config(self, quality_key: str, flat_key: str, default: Any) -> Any:
        """Read `data_quality.<quality_key>`, falling back to a flat key and then the default.

        Same shape as _get_split_config, so an env var of the flat name still overrides the YAML -
        which is what makes MAX_MISSING_COLUMN_RATIO=0.9 a one-liner when triaging a new dataset.
        """
        env_value = os.environ.get(flat_key)
        if (
            env_value is None
            and quality_key in self.DATA_QUALITY_CONFIG
            and self.DATA_QUALITY_CONFIG[quality_key] is not None
        ):
            return self.DATA_QUALITY_CONFIG[quality_key]
        return self._get_config(flat_key, default)

    def _get_sklearn_categorical_config(self, key: str, default: Any) -> Any:
        if key in self.SKLEARN_CATEGORICAL_CONFIG and self.SKLEARN_CATEGORICAL_CONFIG[key] is not None:
            return self.SKLEARN_CATEGORICAL_CONFIG[key]
        return self._get_config(key, default)
    
    def _get_ignore_bands(self, default: Any) -> list:
        hs_config = self._normalize_mapping(getattr(self, 'EXISTING_HS_FEATURES', {}))
        if hs_config.get('enabled', False) and hs_config.get('ignore', False):
            band_names = hs_config.get('band_names', [])
            if isinstance(band_names, list) and band_names:
                return [str(name) for name in band_names if name]

            band_count = hs_config.get('band_count', None)
            prefix = hs_config.get('prefix', '')
            if band_count and prefix:
                try:
                    return [f"{prefix}{i}" for i in range(1, int(band_count) + 1)]
                except (TypeError, ValueError):
                    return default if isinstance(default, list) else []

        if self._get_config('IGNORE_BANDS', False):
            return [f"Band_{i}" for i in range(1, self._get_config('N_BANDS', 234) + 1)]
        else:
            return default if isinstance(default, list) else []

    def _resolve_data_path(self, path_value: Optional[str]) -> Optional[str]:
        if not path_value:
            return None
        if os.path.isabs(path_value):
            return path_value
        return os.path.join(self.DATA_FOLDER, path_value)

    def _resolve_config_path(self, path_value: Optional[str]) -> Optional[str]:
        if not path_value:
            return None
        if os.path.isabs(path_value):
            return path_value

        candidate_from_config_dir = os.path.normpath(os.path.join(self._config_dir, path_value))
        if os.path.exists(candidate_from_config_dir):
            return candidate_from_config_dir

        candidate_from_cwd = os.path.normpath(os.path.abspath(path_value))
        if os.path.exists(candidate_from_cwd):
            return candidate_from_cwd

        return candidate_from_config_dir

    @staticmethod
    def _normalize_mapping(value: Any) -> dict:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}