"""A loader's serving settings must not change the order a run trains in or the dropout it draws.

The failure this pins: a study ran with persistent_workers true, and the exported config restored the
registry's false. The retrain matched its trial at epoch 0, then diverged at epoch 1 and finished at
val_loss 0.5566 against the trial's 0.5405. A default DataLoader draws from the global torch stream
once per new iterator, and only a non-persistent loader builds a new iterator every epoch.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler

from yg_eo_soilnet.datamodules.lightning.lightning_graph import SingleNodeGraphDataModule
from yg_eo_soilnet.datamodules.loaders import build_loader
from yg_eo_soilnet.datamodules.sequence.sequence_datamodule import SoilSequenceDataModule

EPOCHS = 3


def _run(make_loader, seed=42, **serving):
    """Epochs as a fit loop runs them: train, then validation.

    Returns each epoch's order, plus one global draw per epoch standing in for that epoch's dropout.
    """
    torch.manual_seed(seed)
    train = make_loader(list(range(96)), batch_size=16, shuffle=True, **serving)
    val = make_loader(list(range(32)), batch_size=16, **serving)
    orders, draws = [], []
    for _ in range(EPOCHS):
        orders.append([int(index) for batch in train for index in batch])
        draws.append(torch.rand(1).item())
        for _ in val:
            pass
    return orders, draws


def _bare_loader(dataset, **kwargs):
    return DataLoader(dataset, **kwargs)


def test_persistent_workers_do_not_change_the_run():
    """The regression this module exists for."""
    persistent = _run(build_loader, num_workers=2, persistent_workers=True)
    fresh_each_epoch = _run(build_loader, num_workers=2, persistent_workers=False)

    assert persistent == fresh_each_epoch


def test_num_workers_do_not_change_the_run():
    assert _run(build_loader, num_workers=0) == _run(build_loader, num_workers=2)


def test_a_bare_dataloader_does_change_the_run():
    """Proof that the loader's own draws were the problem, and not an incidental difference."""
    persistent_orders, persistent_draws = _run(_bare_loader, num_workers=2, persistent_workers=True)
    fresh_orders, fresh_draws = _run(_bare_loader, num_workers=2, persistent_workers=False)

    assert persistent_orders[0] == fresh_orders[0]  # epoch 0 agrees, as it did in the study
    assert persistent_orders[1:] != fresh_orders[1:]
    assert persistent_draws != fresh_draws


def test_the_run_seed_still_reaches_the_shuffle():
    """The private generators are seeded from the global stream, so seeding still matters."""
    assert _run(build_loader, seed=42)[0] != _run(build_loader, seed=7)[0]


def test_shuffle_off_keeps_the_dataset_order():
    orders, _ = _run(lambda dataset, shuffle=False, **kwargs: build_loader(dataset, **kwargs))

    assert all(order == list(range(96)) for order in orders)


@pytest.mark.parametrize(
    "datamodule_cls, collate_attr",
    [(SoilSequenceDataModule, "_collate_points"), (SingleNodeGraphDataModule, "_collate_nodes")],
)
def test_every_datamodule_builds_through_build_loader(datamodule_cls, collate_attr):
    """Structural guard: a bare DataLoader in either datamodule would silently bring the bug back."""
    fake = SimpleNamespace(batch_size=4, num_workers=0, pin_memory=False, persistent_workers=False)
    setattr(fake, collate_attr, lambda batch: batch)

    loader = datamodule_cls._make_loader(fake, np.arange(8), shuffle=True)

    assert loader.generator is not None
    assert isinstance(loader.sampler, RandomSampler)
    assert loader.sampler.generator is not None
