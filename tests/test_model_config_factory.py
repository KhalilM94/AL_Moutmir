from typing import Any

import pytest

from yg_eo_soilnet.models import BaseSpatialClusterStrategy, ModelConfigFactory


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
    assert configs["custom_builder_model"]["model"].kwargs["build_fn"]() == {"num_features": 8}


def test_load_splitter_from_config_returns_enabled_splitter() -> None:
    registry = {
        "enabled": True,
        "class_path": "yg_eo_soilnet.models.KMeansClusterStrategy",
        "params": {"n_clusters": 3},
    }

    splitter = ModelConfigFactory(registry=registry, random_state=11).load_splitter_from_config()

    assert isinstance(splitter, BaseSpatialClusterStrategy)
    assert splitter.random_state == 11
    assert splitter.n_clusters == 3


def test_load_splitter_from_config_returns_none_when_disabled() -> None:
    registry = {"enabled": False}

    assert ModelConfigFactory(registry=registry).load_splitter_from_config() is None