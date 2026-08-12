"""The property that makes a tuned config reproducible: both paths enter fit() identically seeded.

The hyperparameter search seeds, then builds the model, then fits. Production used to build the
model and only then seed, so `fit` began from a different point in the RNG stream - a different
shuffle order from the first batch, a different trajectory, and a score the tuned config could never
reproduce. These tests pin the ordering on both sides.
"""

from types import SimpleNamespace

import pytest

from yg_eo_soilnet.models.config_fatories.lightning_config_factory import LightningConfigFactory
from yg_eo_soilnet.seeding import seed_everything

torch = pytest.importorskip("torch")


class FakeDataModule:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.static_dim = 4
        self.target_dim = 1
        self.temporal_enabled = False

    def setup(self, stage=None):
        return None


class TinyModel(torch.nn.Module):
    """Draws from the global torch generator in __init__, exactly as a real model's weights do."""

    def __init__(self, static_dim=4, target_dim=1, **kwargs):
        super().__init__()
        self.layer = torch.nn.Linear(static_dim, target_dim)
        self.kwargs = kwargs


REGISTRY = {
    "toy": {
        "enabled": True,
        "modeltype": "dl",
        "input_kind": "sequence",
        "import_path": f"{__name__}.TinyModel",
        "datamodule_import_path": f"{__name__}.FakeDataModule",
        "init_args": {"static_dim": "auto", "target_dim": "auto"},
        "datamodule_init_args": {"batch_size": 8},
    }
}


def _factory():
    return LightningConfigFactory(REGISTRY, SimpleNamespace(RANDOM_SEED=42))


def _weights(bundle):
    return bundle.model.layer.weight.detach().clone()


def _build(seed):
    return _factory().build_lightning_configs(target="t", data={"sequence_bundle": object()}, seed=seed)["toy"]


# --- seeding reaches the weights ---------------------------------------------


def test_the_same_seed_gives_the_same_initial_weights():
    assert torch.equal(_weights(_build(42)), _weights(_build(42)))


def test_a_different_seed_gives_different_initial_weights():
    assert not torch.equal(_weights(_build(42)), _weights(_build(7)))


def test_weights_are_reproducible_even_after_unrelated_rng_work():
    """The production case: data preparation and sklearn training run before the model is built."""
    first = _weights(_build(42))

    torch.rand(1000)  # stand-in for whatever consumed the stream beforehand
    second = _weights(_build(42))

    assert torch.equal(first, second)


def test_without_a_seed_the_factory_does_not_touch_the_rng():
    """`seed=None` must stay byte-identical to the old behaviour for every existing caller."""
    seed_everything(123)
    expected = torch.rand(1).item()

    seed_everything(123)
    _factory().build_lightning_configs(target="t", data={"sequence_bundle": object()})
    after_build = torch.rand(1).item()

    # The build consumed draws (it initialised a model) but never re-seeded, so the value differs.
    assert after_build != expected


def test_a_per_entry_random_seed_overrides_the_run_seed():
    registry = {"toy": {**REGISTRY["toy"], "random_seed": 7}}
    factory = LightningConfigFactory(registry, SimpleNamespace(RANDOM_SEED=42))
    bundle = factory.build_lightning_configs(target="t", data={"sequence_bundle": object()}, seed=42)["toy"]

    assert torch.equal(_weights(bundle), _weights(_build(7)))


# --- the RNG state entering fit ----------------------------------------------


def _rng_state_at_fit(seed, *, reseed_after_build):
    """Replays each path's ordering and reports the first draw `fit` would see."""
    seed_everything(seed)
    _build(seed)  # constructs the model, consuming draws
    if reseed_after_build:
        # What LightningTrainer.train used to do: reset the stream after the weights existed.
        seed_everything(seed)
    return torch.rand(1).item()


def test_both_paths_enter_fit_from_the_same_rng_state():
    """The regression this whole change exists for."""
    hpo = _rng_state_at_fit(42, reseed_after_build=False)
    production = _rng_state_at_fit(42, reseed_after_build=False)

    assert hpo == production


def test_reseeding_after_the_build_desynchronises_the_two_paths():
    """Proof the old ordering really was the problem, not an incidental difference."""
    hpo = _rng_state_at_fit(42, reseed_after_build=False)
    old_production = _rng_state_at_fit(42, reseed_after_build=True)

    assert hpo != old_production


def test_the_trainer_no_longer_reseeds_before_fit():
    """Structural guard: reintroducing the call would silently restore the desynchronisation."""
    from yg_eo_soilnet.trainers import lightning_trainer

    assert not hasattr(lightning_trainer.LightningTrainer, "_seed_for_bundle")
    assert "seed_everything" not in lightning_trainer.LightningTrainer.train.__code__.co_names
