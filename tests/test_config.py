from pathlib import Path
import pytest

from config import Config

BASE_CONFIG_CONTENT = """
common:
    DATA_FOLDER: base_folder
    STATIC_FEATURES_FILE: base_static.csv
    TARGETS_FILE: base_targets.csv
    RANDOM_SEED: 1
    TEST_SIZE: 0.3
    DATA_SPEC_PATH: data_spec.yml
    SKLEARN_CONFIG_PATH: sklearn.yml
    LIGHTNING_CONFIG_PATH: lightning.yml
    SKLEARN_REGISTRY_PATH: registry.yml
    LIGHTNING_REGISTRY_PATH: lightning_registry.yml
    MAIN_FILE_LOGGING_ENABLED: false
    MLFLOW_EXPERIMENT_EXPORT_ENABLED: false
    MLFLOW_EXPERIMENT_EXPORT_PATH: export_dir
    CLUSTERING_STRATEGY:
        enabled: false
        class_path: yg_eo_soilnet.clustering_utils.KMeansClusterStrategy
        params: {}

temporal:
    enabled: true
    time_column: month
    modality_prefix_map:
        radar: R_
        optical: O_
"""

BASE_DATA_SPEC_CONTENT = """
TARGET_COLUMNS: [base_target]
PREDICTOR_COLUMNS: []
IGNORED_COLUMNS: []
CATEGORICAL_FEATURES: [cat]
existing_hs_features:
    enabled: false
    ignore: false
    prefix: S2_
    band_count: 6
    band_names: []
"""

BASE_SKLEARN_CONFIG_CONTENT = """
SKLEARN_FILE_LOGGING_ENABLED: false
MIN_FEATURE_COUNT: 10
MAX_FEATURE_DROP_RATIO_WARNING: 0.9
ELIMINATED_FEATURES: []
COLUMNS_TO_TRANSFORM: []
categorical:
    TREE_CATEGORICAL_ENCODING: ordinal
    TREE_ONEHOT_MAX_CATEGORIES: 7
    CATEGORICAL_FEATURES: [cat]
    EXCLUDE_CATEGORICAL: []
"""

BASE_LIGHTNING_CONFIG_CONTENT = """
LIGHTNING_ENABLE_DEFAULT_LOGGER: true
"""

MOCK_LIGHTNING_REGISTRY = "toy_lightning:\n  enabled: true\n  modeltype: dl\n"


# This fixture builds your base filesystem setup inside the isolated sandbox
@pytest.fixture
def base_config_paths(tmp_path: Path):
    config_path = tmp_path / "config.yml"
    registry_path = tmp_path / "registry.yml"
    lightning_registry_path = tmp_path / "lightning_registry.yml"
    data_spec_path = tmp_path / "data_spec.yml"
    sklearn_config_path = tmp_path / "sklearn.yml"
    lightning_config_path = tmp_path / "lightning.yml"

    config_path.write_text(BASE_CONFIG_CONTENT.strip())
    data_spec_path.write_text(BASE_DATA_SPEC_CONTENT.strip())
    sklearn_config_path.write_text(BASE_SKLEARN_CONFIG_CONTENT.strip())
    lightning_config_path.write_text(BASE_LIGHTNING_CONFIG_CONTENT.strip())
    registry_path.write_text("enabled: true\n")
    lightning_registry_path.write_text(MOCK_LIGHTNING_REGISTRY)

    return {
        "config_path": str(config_path),
        "registry_path": str(registry_path),
        "lightning_registry_path": str(lightning_registry_path),
    }


def test_config_reads_env_overrides(
    monkeypatch: pytest.MonkeyPatch, base_config_paths: dict
) -> None:
    monkeypatch.setenv("DATA_FOLDER", "override_folder")
    monkeypatch.setenv("RANDOM_SEED", "7")
    monkeypatch.setenv("TEST_SIZE", "0.25")
    monkeypatch.setenv("IGNORE_BANDS", "true")
    monkeypatch.setenv("N_BANDS", "3")
    monkeypatch.setenv(
        "CLUSTERING_STRATEGY",
        '{"enabled": true, "class_path": "yg_eo_soilnet.clustering_utils.KMeansClusterStrategy", "params": {"n_clusters": 2}}',
    )

    config = Config(**base_config_paths)

    assert config.DATA_FOLDER == "override_folder"
    assert config.RANDOM_SEED == 7
    assert config.TEST_SIZE == 0.25
    assert config.IGNORE_BANDS == ["Band_1", "Band_2", "Band_3"]
    assert config.ENABLE_CLUSTERING is True
    assert config.CLUSTERING_STRATEGY["params"]["n_clusters"] == 2
    assert config.MAIN_FILE_LOGGING_ENABLED is False
    assert config.SKLEARN_FILE_LOGGING_ENABLED is False
    assert config.MLFLOW_EXPERIMENT_EXPORT_ENABLED is False
    assert config.STATIC_CSV_PATH == "override_folder/base_static.csv"
    assert config.TARGETS_CSV_PATH == "override_folder/base_targets.csv"


def test_config_reads_file_logging_toggles(base_config_paths: dict) -> None:
    config = Config(**base_config_paths)

    assert config.MAIN_FILE_LOGGING_ENABLED is False
    assert config.SKLEARN_FILE_LOGGING_ENABLED is False
    assert config.MLFLOW_EXPERIMENT_EXPORT_PATH.endswith("export_dir")
    assert config.MIN_FEATURE_COUNT == 10
    assert config.MAX_FEATURE_DROP_RATIO_WARNING == 0.9


def test_config_raises_for_missing_registry(base_config_paths: dict, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Model registry YAML not found"):
        Config(
            config_path=base_config_paths["config_path"],
            registry_path=str(tmp_path / "missing.yml"),
        )


def test_config_loads_lightning_registry(base_config_paths: dict) -> None:
    config = Config(**base_config_paths)
    assert config.LIGHTNING_MODEL_REGISTRY["toy_lightning"]["modeltype"] == "dl"


def test_config_reads_nested_sklearn_categorical_features(base_config_paths: dict) -> None:
    config = Config(**base_config_paths)

    assert config.CATEGORICAL_FEATURES == ["cat"]
    assert config.EXCLUDE_CATEGORICAL == []
    assert config.TREE_CATEGORICAL_ENCODING == "ordinal"
    assert config.TREE_ONEHOT_MAX_CATEGORIES == 7


def test_config_reads_existing_hyperspectral_features(base_config_paths: dict) -> None:
    config = Config(**base_config_paths)

    assert config.EXISTING_HS_FEATURES["enabled"] is False
    assert config.EXISTING_HS_FEATURES["prefix"] == "S2_"
    assert config.EXISTING_HS_FEATURES["band_count"] == 6


def test_config_reads_nested_temporal_features(base_config_paths: dict) -> None:
    config = Config(**base_config_paths)

    assert config.TEMPORAL_FEATURES_ENABLED is True
    assert config.TIME_COLUMN == "month"
    assert config.MODALITY_PREFIX_MAP == {"radar": "R_", "optical": "O_"}


def test_config_reads_temporal_features_from_common_section(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    data_spec_path = tmp_path / "data_spec.yml"
    sklearn_config_path = tmp_path / "sklearn.yml"
    lightning_config_path = tmp_path / "lightning.yml"
    registry_path = tmp_path / "registry.yml"
    lightning_registry_path = tmp_path / "lightning_registry.yml"

    config_path.write_text(
        f"""
common:
  DATA_FOLDER: {tmp_path}
  DATA_SPEC_PATH: data_spec.yml
  SKLEARN_CONFIG_PATH: sklearn.yml
  LIGHTNING_CONFIG_PATH: lightning.yml
  SKLEARN_REGISTRY_PATH: registry.yml
  LIGHTNING_REGISTRY_PATH: lightning_registry.yml
  temporal:
    enabled: true
    time_column: month
    modality_prefix_map:
      radar: R_
      optical: O_
""".strip()
    )
    data_spec_path.write_text("TARGET_COLUMNS: [target]\n")
    sklearn_config_path.write_text("{}\n")
    lightning_config_path.write_text("{}\n")
    registry_path.write_text("enabled: true\n")
    lightning_registry_path.write_text("toy_lightning:\n  enabled: true\n  modeltype: dl\n")

    config = Config(
        config_path=str(config_path),
        registry_path=str(registry_path),
        lightning_registry_path=str(lightning_registry_path),
    )

    assert config.TEMPORAL_FEATURES_ENABLED is True
    assert config.TIME_COLUMN == "month"
    assert config.MODALITY_PREFIX_MAP == {"radar": "R_", "optical": "O_"}


def test_config_resolves_manifest_and_folder_paths_relative_to_data_folder(tmp_path: Path) -> None:
    data_root = tmp_path / "data_root"
    config_path = tmp_path / "config.yml"
    data_spec_path = tmp_path / "data_spec.yml"
    sklearn_config_path = tmp_path / "sklearn.yml"
    lightning_config_path = tmp_path / "lightning.yml"
    registry_path = tmp_path / "registry.yml"
    lightning_registry_path = tmp_path / "lightning_registry.yml"

    config_path.write_text(
        f"""
common:
  DATA_FOLDER: {data_root}
  DATA_INDEX_MANIFEST: index.json
  STATIC_FEATURES_FOLDER: static_features
  TARGETS_FOLDER: targets
  DATA_SPEC_PATH: data_spec.yml
  SKLEARN_CONFIG_PATH: sklearn.yml
  LIGHTNING_CONFIG_PATH: lightning.yml
  SKLEARN_REGISTRY_PATH: registry.yml
  LIGHTNING_REGISTRY_PATH: lightning_registry.yml
""".strip()
    )
    data_spec_path.write_text("TARGET_COLUMNS: [target]\n")
    sklearn_config_path.write_text("{}\n")
    lightning_config_path.write_text("temporal: {}\n")
    registry_path.write_text("enabled: true\n")
    lightning_registry_path.write_text("toy_lightning:\n  enabled: true\n  modeltype: dl\n")

    config = Config(
        config_path=str(config_path),
        registry_path=str(registry_path),
        lightning_registry_path=str(lightning_registry_path),
    )

    assert config.DATA_INDEX_MANIFEST_PATH == str(data_root / "index.json")
    assert config.STATIC_FEATURES_FOLDER == str(data_root / "static_features")
    assert config.TARGETS_FOLDER == str(data_root / "targets")


def test_config_loads_layered_data_spec_and_family_files(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    data_spec_path = tmp_path / "data_spec.yml"
    sklearn_config_path = tmp_path / "sklearn.yml"
    lightning_config_path = tmp_path / "lightning.yml"
    registry_path = tmp_path / "registry.yml"
    lightning_registry_path = tmp_path / "lightning_registry.yml"

    config_path.write_text(
        f"""
common:
  DATA_FOLDER: {tmp_path}
  DATA_SPEC_PATH: data_spec.yml
  SKLEARN_CONFIG_PATH: sklearn.yml
  LIGHTNING_CONFIG_PATH: lightning.yml
  SKLEARN_REGISTRY_PATH: registry.yml
  LIGHTNING_REGISTRY_PATH: lightning_registry.yml
""".strip()
    )
    data_spec_path.write_text(
        """
TARGET_COLUMNS: [yield]
PREDICTOR_COLUMNS: [feature_a, feature_b]
IGNORED_COLUMNS: [id]
CATEGORICAL_FEATURES: [category]
""".strip()
    )
    sklearn_config_path.write_text(
        """
SPLIT_STRATEGY: groupkfold
COLUMNS_TO_TRANSFORM: [yield]
""".strip()
    )
    lightning_config_path.write_text(
        """
temporal:
  enabled: true
  time_column: timestamp
""".strip()
    )
    registry_path.write_text("enabled: true\n")
    lightning_registry_path.write_text("toy_lightning:\n  enabled: true\n  modeltype: dl\n")

    config = Config(
        config_path=str(config_path),
        registry_path=str(registry_path),
        lightning_registry_path=str(lightning_registry_path),
    )

    assert config.TARGET_COLUMNS == ["yield"]
    assert config.PREDICTOR_COLUMNS == ["feature_a", "feature_b"]
    assert config.IGNORED_COLUMNS == ["id"]
    assert config.CATEGORICAL_FEATURES == ["category"]
    assert config.SPLIT_STRATEGY == "groupkfold"
    assert config.TEMPORAL_FEATURES_ENABLED is True
    assert config.TIME_COLUMN == "timestamp"
    assert config.registry_path == str(registry_path)
    assert config.lightning_registry_path == str(lightning_registry_path)


def test_config_fails_when_family_config_path_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    registry_path = tmp_path / "registry.yml"
    lightning_registry_path = tmp_path / "lightning_registry.yml"
    data_spec_path = tmp_path / "data_spec.yml"
    sklearn_config_path = tmp_path / "sklearn.yml"

    config_path.write_text(
        """
common:
  DATA_SPEC_PATH: data_spec.yml
  SKLEARN_CONFIG_PATH: sklearn.yml
  LIGHTNING_CONFIG_PATH: missing_lightning.yml
  SKLEARN_REGISTRY_PATH: registry.yml
  LIGHTNING_REGISTRY_PATH: lightning_registry.yml
""".strip()
    )
    data_spec_path.write_text("TARGET_COLUMNS: [target]\n")
    sklearn_config_path.write_text("{}\n")
    registry_path.write_text("enabled: true\n")
    lightning_registry_path.write_text("toy_lightning:\n  enabled: true\n  modeltype: dl\n")

    with pytest.raises(FileNotFoundError, match="Config YAML not found"):
        Config(
            config_path=str(config_path),
            registry_path=str(registry_path),
            lightning_registry_path=str(lightning_registry_path),
        )

# --- unified data: block ---------------------------------------------------


UNIFIED_DATA_CONFIG_CONTENT = """
common:
    data:
        root: unified_root
        static: static_dir
        targets: targets_dir
        timeseries: ts_dir
        manifest: index.json
    RANDOM_SEED: 1
    DATA_SPEC_PATH: data_spec.yml
    SKLEARN_CONFIG_PATH: sklearn.yml
    LIGHTNING_CONFIG_PATH: lightning.yml
    SKLEARN_REGISTRY_PATH: registry.yml
    LIGHTNING_REGISTRY_PATH: lightning_registry.yml
    temporal:
        enabled: true
        time_column: month
"""


def _write_config(base_config_paths: dict, content: str) -> dict:
    Path(base_config_paths["config_path"]).write_text(content.strip())
    return base_config_paths


def test_data_block_resolves_every_source(base_config_paths: dict) -> None:
    config = Config(**_write_config(base_config_paths, UNIFIED_DATA_CONFIG_CONTENT))

    assert config.DATA_FOLDER == "unified_root"
    assert config.DATA_ROOT == "unified_root"
    assert config.STATIC_SOURCE == "unified_root/static_dir"
    assert config.TARGETS_SOURCE == "unified_root/targets_dir"
    assert config.TIMESERIES_SOURCE == "unified_root/ts_dir"
    assert config.DATA_MANIFEST_PATH == "unified_root/index.json"


def test_data_block_omitting_targets_means_joint_file(base_config_paths: dict) -> None:
    """A joint static file must leave TARGETS_SOURCE unset, not aliased to the static path."""
    content = UNIFIED_DATA_CONFIG_CONTENT.replace("        targets: targets_dir\n", "")
    config = Config(**_write_config(base_config_paths, content))

    assert config.STATIC_SOURCE == "unified_root/static_dir"
    assert config.TARGETS_SOURCE is None


def test_legacy_flat_keys_still_resolve_without_a_data_block(base_config_paths: dict) -> None:
    config = Config(**base_config_paths)

    assert config.DATA_FOLDER == "base_folder"
    assert config.STATIC_SOURCE == "base_folder/base_static.csv"
    assert config.TARGETS_SOURCE == "base_folder/base_targets.csv"


def test_data_file_is_not_rebound_after_static_path_is_derived(base_config_paths: dict) -> None:
    """config.py used to reassign DATA_FILE = STATIC_FEATURES_FILE after deriving STATIC_CSV_PATH."""
    content = BASE_CONFIG_CONTENT.replace(
        "    STATIC_FEATURES_FILE: base_static.csv", "    DATA_FILE: base_data.csv\n    STATIC_FEATURES_FILE: base_static.csv"
    )
    config = Config(**_write_config(base_config_paths, content))

    assert config.DATA_FILE == "base_data.csv"
    assert config.STATIC_FEATURES_FILE == "base_static.csv"
    assert config.STATIC_CSV_PATH == "base_folder/base_static.csv"


@pytest.mark.parametrize("key", ["timeseries_file", "timeseries_csv_path"])
def test_both_timeseries_key_spellings_resolve(base_config_paths: dict, key: str) -> None:
    """config.py read `timeseries_file` while data_manager read `timeseries_csv_path`."""
    content = BASE_CONFIG_CONTENT.replace(
        "temporal:\n    enabled: true", f"temporal:\n    enabled: true\n    {key}: ts.csv"
    )
    config = Config(**_write_config(base_config_paths, content))

    assert config.TIMESERIES_SOURCE == "base_folder/ts.csv"


# --- lightning registry `defaults:` -------------------------------------------------------------

REGISTRY_WITH_DEFAULTS = """
defaults:
  modeltype: dl
  trainer_args:
    max_epochs: 500
    accelerator: cuda
    deterministic: true
  callbacks:
    early_stopping: {monitor: val_loss, mode: min, patience: 30}
    checkpoint: {monitor: val_loss, mode: min, save_top_k: 1}
  datamodule_init_args:
    batch_size: 32
    num_workers: 11

inheritor:
  enabled: true

overrider:
  enabled: false
  trainer_args:
    max_epochs: 150
  callbacks:
    early_stopping: {patience: 3}
  datamodule_init_args:
    batch_size: 64
    max_sequence_length: null
"""


def _registry(base_config_paths: dict, content: str) -> Config:
    Path(base_config_paths["lightning_registry_path"]).write_text(content.strip())
    return Config(**base_config_paths)


def test_registry_defaults_are_merged_into_every_entry(base_config_paths: dict) -> None:
    registry = _registry(base_config_paths, REGISTRY_WITH_DEFAULTS).LIGHTNING_MODEL_REGISTRY

    assert registry["inheritor"]["modeltype"] == "dl"
    assert registry["inheritor"]["trainer_args"] == {
        "max_epochs": 500, "accelerator": "cuda", "deterministic": True
    }
    assert registry["inheritor"]["callbacks"]["checkpoint"]["save_top_k"] == 1


def test_an_entry_overriding_one_key_keeps_the_rest_of_the_block(base_config_paths: dict) -> None:
    """The whole point of merging per key: `max_epochs: 150` must not drop the accelerator."""
    registry = _registry(base_config_paths, REGISTRY_WITH_DEFAULTS).LIGHTNING_MODEL_REGISTRY

    assert registry["overrider"]["trainer_args"] == {
        "max_epochs": 150, "accelerator": "cuda", "deterministic": True
    }
    assert registry["overrider"]["datamodule_init_args"] == {
        "batch_size": 64, "num_workers": 11, "max_sequence_length": None
    }


def test_merging_recurses_into_a_callback_group(base_config_paths: dict) -> None:
    """`early_stopping: {patience: 3}` keeps monitor and mode rather than replacing the group."""
    registry = _registry(base_config_paths, REGISTRY_WITH_DEFAULTS).LIGHTNING_MODEL_REGISTRY

    assert registry["overrider"]["callbacks"]["early_stopping"] == {
        "monitor": "val_loss", "mode": "min", "patience": 3
    }


def test_defaults_is_not_itself_a_registry_entry(base_config_paths: dict) -> None:
    registry = _registry(base_config_paths, REGISTRY_WITH_DEFAULTS).LIGHTNING_MODEL_REGISTRY

    assert set(registry) == {"inheritor", "overrider"}


def test_a_registry_without_defaults_is_unchanged(base_config_paths: dict) -> None:
    """Tuned files exported by tune.py carry no `defaults:` key, so for them this is a no-op."""
    registry = _registry(
        base_config_paths, "solo:\n  enabled: true\n  modeltype: dl\n  trainer_args: {max_epochs: 7}\n"
    ).LIGHTNING_MODEL_REGISTRY

    assert registry == {"solo": {"enabled": True, "modeltype": "dl", "trainer_args": {"max_epochs": 7}}}


def test_an_entry_cannot_mutate_the_defaults_for_the_next_one(base_config_paths: dict) -> None:
    """Deep copies, not shared references: mutating one entry's block must not reach another's."""
    registry = _registry(base_config_paths, REGISTRY_WITH_DEFAULTS).LIGHTNING_MODEL_REGISTRY
    registry["overrider"]["callbacks"]["checkpoint"]["save_top_k"] = 99

    assert registry["inheritor"]["callbacks"]["checkpoint"]["save_top_k"] == 1
