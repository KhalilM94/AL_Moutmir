import numpy as np
import pytest

from yg_eo_soilnet.hpo.overrides import apply_overrides, split_dotted, to_builtin, validate_override_keys


def _spec() -> dict:
    return {
        "enabled": True,
        "modeltype": "dl",
        "init_args": {"static_dim": "auto", "learning_rate": 0.001},
        "datamodule_init_args": {"batch_size": 32},
        "trainer_args": {"max_epochs": 500},
        "callbacks": {"early_stopping": {"monitor": "val_loss", "patience": 30}},
    }


# --- routing -----------------------------------------------------------------


def test_each_prefix_lands_in_its_registry_section():
    spec = apply_overrides(
        _spec(),
        {
            "model.learning_rate": 0.01,
            "datamodule.batch_size": 64,
            "trainer.max_epochs": 150,
            "callbacks.early_stopping.patience": 20,
        },
    )

    assert spec["init_args"]["learning_rate"] == 0.01
    assert spec["datamodule_init_args"]["batch_size"] == 64
    assert spec["trainer_args"]["max_epochs"] == 150
    assert spec["callbacks"]["early_stopping"]["patience"] == 20
    # Untouched neighbours survive.
    assert spec["init_args"]["static_dim"] == "auto"
    assert spec["callbacks"]["early_stopping"]["monitor"] == "val_loss"


def test_overrides_create_missing_sections():
    spec = apply_overrides({"enabled": True}, {"model.dropout": 0.2, "callbacks.checkpoint.save_top_k": 3})

    assert spec["init_args"] == {"dropout": 0.2}
    assert spec["callbacks"]["checkpoint"] == {"save_top_k": 3}


@pytest.mark.parametrize(
    "dotted",
    ["learning_rate", "optimizer.lr", "model.a.b", "callbacks.early_stopping", "callbacks.a.b.c"],
)
def test_malformed_keys_are_rejected(dotted):
    with pytest.raises(ValueError):
        split_dotted(dotted)


# --- guards ------------------------------------------------------------------


def test_factory_resolved_model_keys_are_refused():
    """These are filled from the datamodule; a tuned value corrupts the shape contract."""
    with pytest.raises(ValueError, match="resolved from the datamodule"):
        validate_override_keys(["model.static_dim"], searched=True)


def test_factory_resolved_keys_are_refused_even_when_pinned():
    with pytest.raises(ValueError, match="resolved from the datamodule"):
        validate_override_keys(["model.categorical_cardinalities"], searched=False)


def test_embedding_dims_is_not_treated_as_factory_resolved():
    """The factory never touches embedding_dims; 'auto' is resolved inside the model."""
    validate_override_keys(["model.embedding_dims"], searched=True)


def test_split_defining_keys_cannot_be_searched():
    with pytest.raises(ValueError, match="would not be comparable"):
        validate_override_keys(["datamodule.val_size"], searched=True)


def test_split_defining_keys_may_be_pinned():
    """Fixing the split for the whole study is fine - only varying it breaks comparability."""
    validate_override_keys(["datamodule.val_size", "datamodule.seed"], searched=False)


def test_every_offending_key_is_reported_at_once():
    with pytest.raises(ValueError) as excinfo:
        validate_override_keys(["model.static_dim", "model.grid_years", "datamodule.test_size"], searched=True)

    message = str(excinfo.value)
    assert "model.static_dim" in message
    assert "model.grid_years" in message
    assert "datamodule.test_size" in message


# --- builtin coercion --------------------------------------------------------


def test_numpy_scalars_become_builtins():
    """save_hyperparameters() puts these in the checkpoint; numpy there breaks weights_only=True."""
    assert type(to_builtin(np.float32(0.5))) is float
    assert type(to_builtin(np.int64(8))) is int
    assert type(to_builtin(np.bool_(True))) is bool


def test_a_bool_does_not_degrade_to_an_int():
    assert to_builtin(True) is True
    assert type(to_builtin(True)) is bool


def test_sequences_and_mappings_are_converted_elementwise():
    assert to_builtin(np.array([64, 32])) == [64, 32]
    assert all(type(item) is int for item in to_builtin((np.int64(64), np.int64(32))))
    assert to_builtin({"s1": np.int64(4)}) == {"s1": 4}


def test_apply_overrides_coerces_on_the_way_in():
    spec = apply_overrides(_spec(), {"model.static_hidden_dim": np.int64(128)})

    assert type(spec["init_args"]["static_hidden_dim"]) is int
