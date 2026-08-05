"""Every dotted path shipped in configs/ must resolve.

Module paths inside YAML are resolved at runtime, so a file move that misses them fails only
once training starts. These tests catch that at collection time instead.
"""

import importlib
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_ROOT = PROJECT_ROOT / "configs"


def _resolve(dotted_path: str):
    module_path, attr_name = dotted_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), attr_name)


def _enabled_lightning_entries() -> list[tuple[str, dict]]:
    registry = yaml.safe_load((CONFIGS_ROOT / "lightning" / "lightning_registry.yml").read_text())
    return [(name, spec) for name, spec in registry.items() if spec.get("enabled", False)]


def _sklearn_model_entries() -> list[tuple[str, dict]]:
    """All entries, enabled or not - these are third-party paths that must stay valid."""
    registry = yaml.safe_load((CONFIGS_ROOT / "sklearn" / "model_registry.yml").read_text())
    return list(registry.items())


@pytest.mark.parametrize("name,spec", _enabled_lightning_entries())
@pytest.mark.parametrize("key", ["import_path", "datamodule_import_path"])
def test_enabled_lightning_registry_paths_resolve(name: str, spec: dict, key: str) -> None:
    assert _resolve(spec[key]) is not None, f"{name}.{key} does not resolve"


@pytest.mark.parametrize("name,spec", _sklearn_model_entries())
def test_sklearn_model_registry_paths_resolve(name: str, spec: dict) -> None:
    assert _resolve(spec["import_path"]) is not None, f"{name}.import_path does not resolve"


def test_clustering_strategy_class_path_resolves() -> None:
    sklearn_config = yaml.safe_load((CONFIGS_ROOT / "sklearn" / "config.yml").read_text())
    class_path = sklearn_config["CLUSTERING_STRATEGY"]["class_path"]

    from yg_eo_soilnet.clustering_utils import BaseSpatialClusterStrategy

    assert issubclass(_resolve(class_path), BaseSpatialClusterStrategy)


# --- the shipped configs must actually load ---------------------------------
# A dangling YAML anchor in data_spec.yml once broke `python main.py` at startup while the suite
# stayed green, because every test builds its own config fixture instead of reading these files.


def test_shipped_configs_load() -> None:
    from config import Config

    config = Config(config_path=str(CONFIGS_ROOT / "main_config.yml"))

    assert config.TARGET_COLUMNS, "TARGET_COLUMNS must not be empty"
    assert config.DATA_FOLDER


def test_active_targets_are_declared_as_labels() -> None:
    """TARGET_COLUMNS should be a subset of LABEL_COLUMNS; both are excluded from features anyway."""
    from config import Config

    config = Config(config_path=str(CONFIGS_ROOT / "main_config.yml"))

    undeclared = sorted(set(config.TARGET_COLUMNS) - set(config.LABEL_COLUMNS))
    assert not undeclared, f"targets missing from LABEL_COLUMNS: {undeclared}"
