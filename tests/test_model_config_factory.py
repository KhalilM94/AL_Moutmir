from typing import Any

import pytest

from yg_eo_soilnet.models import ModelConfigFactory
from yg_eo_soilnet.clustering_utils import BaseSpatialClusterStrategy


def test_dynamic_import_loads_known_class() -> None:
    loaded = ModelConfigFactory._dynamic_import("sklearn.linear_model.LinearRegression")

    assert loaded.__name__ == "LinearRegression"


def test_dynamic_import_raises_for_missing_module() -> None:
    with pytest.raises(ImportError, match="Failed to import module"):
        ModelConfigFactory._dynamic_import("does_not_exist.SomeClass")


def test_build_model_configs_expands_enabled_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    def fake_builder(num_features: int) -> dict[str, int]:
        return {"num_features": num_features}

    registry = {
        "enabled_model": {
            "enabled": True,
            "import_path": "fake.module.FakeModel",
            "init_args": {"input_dim": 4, "alpha": 0.5},
            "params": {"grid": [1, 2]},
            "modeltype": "ml",
            "random_seed": 123,
        },
        "custom_builder_model": {
            "enabled": True,
            "import_path": "fake.module.FakeModel",
            "custom_model_builder": "fake.module.fake_builder",
        },
        "disabled_model": {
            "enabled": False,
            "import_path": "fake.module.FakeModel",
        },
    }

    factory = ModelConfigFactory(registry=registry)

    def fake_dynamic_import(path: str):
        if path == "fake.module.FakeModel":
            return FakeModel
        if path == "fake.module.fake_builder":
            return fake_builder
        raise AssertionError(f"Unexpected import path: {path}")

    monkeypatch.setattr(factory, "_dynamic_import", fake_dynamic_import)

    configs = factory.build_model_configs(num_features=8)

    assert set(configs) == {"enabled_model", "custom_builder_model"}
    assert configs["enabled_model"]["model"].kwargs["input_dim"] == 8
    assert configs["enabled_model"]["model"].kwargs["alpha"] == 0.5
    assert configs["enabled_model"]["params"] == {"grid": [1, 2]}
    assert configs["enabled_model"]["random_seed"] == 123
    assert configs["custom_builder_model"]["model"].kwargs["build_fn"]() == {"num_features": 8}


def test_build_model_configs_passes_through_search_n_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entries opt out of GridSearchCV fan-out; everything else keeps the -1 default."""

    class FakeModel:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    registry = {
        "capped": {
            "enabled": True,
            "import_path": "fake.module.FakeModel",
            "search_n_jobs": 1,
        },
        "default_parallelism": {
            "enabled": True,
            "import_path": "fake.module.FakeModel",
        },
    }

    factory = ModelConfigFactory(registry=registry)
    monkeypatch.setattr(factory, "_dynamic_import", lambda path: FakeModel)

    configs = factory.build_model_configs(num_features=8)

    assert configs["capped"]["search_n_jobs"] == 1
    assert configs["default_parallelism"]["search_n_jobs"] == -1


def test_estimators_inherit_the_run_seed_instead_of_a_hardcoded_one() -> None:
    """RANDOM_SEED must reach the estimator, not just the CV splitter.

    Entries used to carry `random_state: 42` in init_args, so changing the main seed moved the
    folds but left every model on 42.
    """
    registry = {
        "inherits": {"enabled": True, "import_path": "sklearn.linear_model.Ridge"},
        "per_entry_override": {
            "enabled": True,
            "import_path": "sklearn.linear_model.Ridge",
            "random_seed": 5,
        },
        "pinned_in_init_args": {
            "enabled": True,
            "import_path": "sklearn.linear_model.Ridge",
            "init_args": {"random_state": 99},
        },
        "has_no_seed": {"enabled": True, "import_path": "sklearn.cross_decomposition.PLSRegression"},
    }

    configs = ModelConfigFactory(registry=registry).build_model_configs(
        num_features=4, default_seed=7
    )

    assert configs["inherits"]["model"].get_params()["random_state"] == 7
    assert configs["per_entry_override"]["model"].get_params()["random_state"] == 5
    # An explicit init_args entry is still an explicit override.
    assert configs["pinned_in_init_args"]["model"].get_params()["random_state"] == 99
    # Estimators without a random_state are left alone rather than erroring.
    assert "random_state" not in configs["has_no_seed"]["model"].get_params()


def test_xgboost_inherits_the_run_seed_despite_kwargs_signature() -> None:
    """XGBRegressor keeps random_state in **kwargs, so signature inspection would miss it."""
    registry = {"XGBoost": {"enabled": True, "import_path": "xgboost.XGBRegressor"}}

    configs = ModelConfigFactory(registry=registry).build_model_configs(
        num_features=4, default_seed=7
    )

    assert configs["XGBoost"]["model"].get_params()["random_state"] == 7


def test_build_model_configs_does_not_mutate_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """main.py rebuilds configs once per target off the same registry dict."""

    class FakeModel:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    registry = {
        "model": {
            "enabled": True,
            "import_path": "fake.module.FakeModel",
            "init_args": {"input_dim": 4},
        }
    }

    factory = ModelConfigFactory(registry=registry)
    monkeypatch.setattr(factory, "_dynamic_import", lambda path: FakeModel)

    factory.build_model_configs(num_features=8)

    assert registry["model"]["init_args"]["input_dim"] == 4


def test_load_splitter_from_config_returns_enabled_splitter() -> None:
    registry = {
        "enabled": True,
        "class_path": "yg_eo_soilnet.clustering_utils.KMeansClusterStrategy",
        "params": {"n_clusters": 3},
    }

    splitter = ModelConfigFactory(registry=registry, random_state=11).load_splitter_from_config()

    assert isinstance(splitter, BaseSpatialClusterStrategy)
    assert splitter.random_state == 11
    assert splitter.n_clusters == 3


def test_load_splitter_from_config_returns_none_when_disabled() -> None:
    registry = {"enabled": False}

    assert ModelConfigFactory(registry=registry).load_splitter_from_config() is None