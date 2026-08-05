from types import SimpleNamespace

import pytest
import numpy as np

from yg_eo_soilnet.models.config_fatories.lightning_config_factory import LightningConfigFactory


class FakeModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeGraphDataModule:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.spatiotemporal_graph = kwargs["spatiotemporal_graph"]
        self.static_dim = 3
        self.target_dim = 2
        self.modality_dims = {"radar": 4, "optical": 5, "thermal": 6}
        self.temporal_steps = 7
        self.edge_attr_dim = 1

    def setup(self, stage=None):
        return None


class FakeGraphModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_lightning_factory_rejects_non_dl_entries() -> None:
    factory = LightningConfigFactory(registry={}, config=SimpleNamespace())

    try:
        factory._validate_entry(
            "bad",
            {
                "enabled": True,
                "modeltype": "ml",
                "import_path": "fake.module.FakeModel",
                "datamodule_import_path": "fake.module.FakeGraphDataModule",
            },
        )
    except ValueError as exc:
        assert "must use modeltype 'dl'" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_lightning_factory_builds_spatiotemporal_graph_and_infers_graph_dimensions() -> None:
    registry = {
        "graph_lightning": {
            "enabled": True,
            "modeltype": "dl",
            "input_kind": "graph",
            "random_seed": 13,
            "import_path": "fake.module.FakeGraphModel",
            "datamodule_import_path": "fake.module.FakeGraphDataModule",
            "init_args": {
                "static_dim": "auto",
                "target_dim": "auto",
                "temporal_steps": "auto",
                "edge_attr_dim": "auto",
                "temporal_lstm_hidden_dim": 16,
                "temporal_lstm_num_layers": 2,
                "temporal_lstm_dropout": 0.1,
                "temporal_lstm_bidirectional": False,
                "temporal_pooling": "last",
                "spatial_graph_enabled": False,
            },
            "datamodule_init_args": {"batch_size": 1},
        }
    }
    config = SimpleNamespace(
        LIGHTNING_BATCH_SIZE=1,
        LIGHTNING_VAL_SIZE=0.2,
        LIGHTNING_NUM_WORKERS=0,
        LIGHTNING_PIN_MEMORY=False,
        LIGHTNING_PERSISTENT_WORKERS=False,
        RANDOM_SEED=42,
    )

    factory = LightningConfigFactory(registry=registry, config=config)

    def fake_dynamic_import(path: str):
        if path == "fake.module.FakeGraphModel":
            return FakeGraphModel
        if path == "fake.module.FakeGraphDataModule":
            return FakeGraphDataModule
        raise AssertionError(f"Unexpected import path: {path}")

    factory._dynamic_import = fake_dynamic_import  # type: ignore[method-assign]

    spatiotemporal_graph = {
        "static_features": np.asarray([[1.0, 2.0, 3.0]]),
        "targets": np.asarray([[4.0, 5.0]]),
        "temporal_features": {},
        "temporal_enabled": False,
        "edge_index": np.zeros((2, 0), dtype=np.int64),
        "edge_attr": np.zeros((0, 1), dtype=np.float32),
        "train_idx": np.asarray([0]),
        "val_idx": np.asarray([], dtype=np.int64),
        "test_idx": np.asarray([], dtype=np.int64),
    }

    bundles = factory.build_lightning_configs(target="target_a", data={"spatiotemporal_graph": spatiotemporal_graph})

    bundle = bundles["graph_lightning"]
    assert isinstance(bundle.model, FakeGraphModel)
    assert bundle.datamodule.kwargs["seed"] == 13
    assert bundle.datamodule.spatiotemporal_graph is spatiotemporal_graph
    assert bundle.model.kwargs["static_dim"] == 3
    assert bundle.model.kwargs["target_dim"] == 2
    assert bundle.model.kwargs["modality_dims"] == {"radar": 4, "optical": 5, "thermal": 6}
    assert bundle.model.kwargs["temporal_steps"] == 7
    assert bundle.model.kwargs["edge_attr_dim"] == 1
    assert bundle.model.kwargs["temporal_lstm_hidden_dim"] == 16
    assert bundle.model.kwargs["temporal_lstm_num_layers"] == 2
    assert bundle.model.kwargs["temporal_lstm_dropout"] == 0.1
    assert bundle.model.kwargs["temporal_lstm_bidirectional"] is False
    assert bundle.model.kwargs["temporal_pooling"] == "last"
    assert bundle.model.kwargs["spatial_graph_enabled"] is False


def _factory(registry: dict) -> LightningConfigFactory:
    return LightningConfigFactory(registry, SimpleNamespace())


def test_graph_data_args_carries_spatial_graph_flag_from_init_args() -> None:
    spec = {
        "enabled": True,
        "input_kind": "graph",
        "graph_data_args": {"spatial_radius": 50000},
        "init_args": {"spatial_graph_enabled": False},
    }

    graph_data_args = LightningConfigFactory._graph_data_args(spec)

    assert graph_data_args["spatial_graph_enabled"] is False
    assert graph_data_args["spatial_radius"] == 50000


def test_graph_data_args_does_not_override_an_explicit_flag() -> None:
    spec = {
        "graph_data_args": {"spatial_graph_enabled": True},
        "init_args": {"spatial_graph_enabled": False},
    }

    assert LightningConfigFactory._graph_data_args(spec)["spatial_graph_enabled"] is True


def test_has_graph_input_finds_an_enabled_graph_entry() -> None:
    factory = _factory({"soil_graph": {"enabled": True, "input_kind": "graph"}})

    assert factory.has_graph_input() is True
    assert factory.graph_spec()["input_kind"] == "graph"


def test_has_graph_input_ignores_a_disabled_graph_entry() -> None:
    factory = _factory({"soil_graph": {"enabled": False, "input_kind": "graph"}})

    assert factory.has_graph_input() is False
    assert factory.graph_spec() is None


def test_has_graph_input_is_false_for_a_tabular_only_registry() -> None:
    factory = _factory({"ts_soilnet": {"enabled": True, "input_kind": "tabular"}})

    assert factory.has_graph_input() is False


def test_has_graph_input_accepts_the_legacy_datamodule_type_spelling() -> None:
    factory = _factory({"soil_graph": {"enabled": True, "datamodule_type": "graph"}})

    assert factory.has_graph_input() is True


def test_explicit_input_kind_wins_over_datamodule_type() -> None:
    """The factory builds on input_kind, so the predicate must agree with what gets built."""
    factory = _factory({"soil_graph": {"enabled": True, "input_kind": "tabular", "datamodule_type": "graph"}})

    assert factory.has_graph_input() is False


def test_factory_rejects_a_non_graph_input_kind() -> None:
    """The tabular datamodule is gone; a registry entry asking for it must fail loudly."""
    factory = _factory({})
    spec = {
        "enabled": True,
        "modeltype": "dl",
        "input_kind": "tabular",
        "import_path": "fake.module.FakeModel",
        "datamodule_import_path": "fake.module.Whatever",
    }

    with pytest.raises(ValueError, match="Unsupported input_kind 'tabular'"):
        factory._build_datamodule(target="target_a", spec=spec, data={})


class FakeSequenceDataModule:
    """Mirrors the real sequence datamodule's contract: no temporal_steps, no edge_attr_dim."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.sequence_bundle = kwargs["sequence_bundle"]
        self.static_dim = 3
        self.target_dim = 1
        self.modality_dims = {"s1": 4, "s2": 5}
        self.temporal_enabled = True
        self.target_mean_ = np.array([2.0])
        self.target_scale_ = np.array([0.5])
        self.target_transform = "log1p"

    def setup(self, stage=None):
        self.did_setup = stage


def _sequence_spec(**overrides) -> dict:
    spec = {
        "enabled": True,
        "modeltype": "dl",
        "input_kind": "sequence",
        "import_path": f"{__name__}.FakeModel",
        "datamodule_import_path": f"{__name__}.FakeSequenceDataModule",
        "init_args": {"static_dim": "auto", "target_dim": "auto", "modality_dims": "auto"},
        "datamodule_init_args": {"batch_size": 8},
    }
    spec.update(overrides)
    return spec


def test_factory_builds_a_sequence_datamodule_from_a_supplied_bundle() -> None:
    factory = _factory({})
    datamodule = factory._build_datamodule(
        target="target_a", spec=_sequence_spec(), data={"sequence_bundle": {"marker": 1}}
    )

    assert isinstance(datamodule, FakeSequenceDataModule)
    assert datamodule.sequence_bundle == {"marker": 1}
    assert datamodule.kwargs["batch_size"] == 8
    assert datamodule.did_setup == "fit"
    assert "spatiotemporal_graph" not in datamodule.kwargs


def test_factory_does_not_inject_graph_or_step_shapes_into_a_sequence_model() -> None:
    """A length-agnostic, graph-free model must never receive temporal_steps or edge_attr_dim."""
    factory = _factory({})
    datamodule = factory._build_datamodule(
        target="target_a", spec=_sequence_spec(), data={"sequence_bundle": {}}
    )
    model = factory._build_model(_sequence_spec(), datamodule)

    assert model.kwargs["static_dim"] == 3
    assert model.kwargs["modality_dims"] == {"s1": 4, "s2": 5}
    assert "temporal_steps" not in model.kwargs
    assert "edge_attr_dim" not in model.kwargs
    # Target stats still arrive as plain floats so the checkpoint stays weights_only-loadable.
    assert model.kwargs["target_mean"] == [2.0]
    assert model.kwargs["target_transform"] == "log1p"


def test_sequence_bundle_requires_a_data_manager_when_none_is_supplied() -> None:
    factory = _factory({})

    with pytest.raises(KeyError, match="data_manager"):
        factory._build_datamodule(target="target_a", spec=_sequence_spec(), data={})


def test_has_sequence_input_and_combined_target_predicate() -> None:
    sequence_factory = _factory({"soil_sequence": {"enabled": True, "input_kind": "sequence"}})
    assert sequence_factory.has_sequence_input() is True
    assert sequence_factory.has_graph_input() is False
    assert sequence_factory.covers_all_targets_in_one_run() is True

    graph_factory = _factory({"soil_graph": {"enabled": True, "input_kind": "graph"}})
    assert graph_factory.covers_all_targets_in_one_run() is True

    tabular_factory = _factory({"ts": {"enabled": True, "input_kind": "tabular"}})
    assert tabular_factory.covers_all_targets_in_one_run() is False


def test_has_sequence_input_ignores_a_disabled_entry() -> None:
    factory = _factory({"soil_sequence": {"enabled": False, "input_kind": "sequence"}})
    assert factory.has_sequence_input() is False
    assert factory.sequence_spec() is None


class FakeGridModel:
    """Accepts grid_years; stands in for the calendar-grid CNN."""

    def __init__(self, static_dim=None, target_dim=None, modality_dims=None, grid_years=None, **rest):
        self.kwargs = {
            "static_dim": static_dim,
            "target_dim": target_dim,
            "modality_dims": modality_dims,
            "grid_years": grid_years,
            **rest,
        }


class FakeGridFreeModel:
    """Does NOT accept grid_years; stands in for the sequence encoders."""

    def __init__(
        self,
        static_dim=None,
        target_dim=None,
        modality_dims=None,
        temporal_enabled=None,
        target_mean=None,
        target_scale=None,
        target_transform=None,
    ):
        self.kwargs = {
            "static_dim": static_dim,
            "target_dim": target_dim,
            "modality_dims": modality_dims,
            "temporal_enabled": temporal_enabled,
            "target_mean": target_mean,
            "target_scale": target_scale,
            "target_transform": target_transform,
        }


class FakeGridDataModule(FakeSequenceDataModule):
    """The shared sequence datamodule, which exposes grid_years for whoever wants it."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid_years = 9


def test_grid_years_is_injected_only_into_models_that_accept_it() -> None:
    """Two entries share one datamodule; what it can offer is not what each model wants.

    Regression test: the factory used to inject every datamodule attribute unconditionally, so
    enabling the CNN and the sequence model together crashed the sequence model with an unexpected
    'grid_years' keyword.
    """
    factory = _factory({})
    datamodule = FakeGridDataModule(sequence_bundle={})

    grid_model = factory._build_model(
        {"import_path": f"{__name__}.FakeGridModel", "init_args": {}}, datamodule
    )
    assert grid_model.kwargs["grid_years"] == 9

    grid_free_model = factory._build_model(
        {"import_path": f"{__name__}.FakeGridFreeModel", "init_args": {}}, datamodule
    )
    assert "grid_years" not in grid_free_model.kwargs
    # The shapes it does accept must still arrive.
    assert grid_free_model.kwargs["static_dim"] == 3
    assert grid_free_model.kwargs["target_mean"] == [2.0]
    assert grid_free_model.kwargs["target_transform"] == "log1p"


def test_a_model_taking_kwargs_still_receives_everything() -> None:
    factory = _factory({})
    model = factory._build_model(
        {"import_path": f"{__name__}.FakeModel", "init_args": {}}, FakeGridDataModule(sequence_bundle={})
    )
    assert model.kwargs["grid_years"] == 9


def test_an_unknown_key_written_in_init_args_still_fails_loudly() -> None:
    """Filtering must not swallow a typo the user actually wrote in the registry."""
    factory = _factory({})
    with pytest.raises(TypeError, match="nonsense_arg"):
        factory._build_model(
            {"import_path": f"{__name__}.FakeGridFreeModel", "init_args": {"nonsense_arg": 1}},
            FakeGridDataModule(sequence_bundle={}),
        )
