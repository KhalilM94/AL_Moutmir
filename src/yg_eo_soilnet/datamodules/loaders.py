"""DataLoaders whose results do not depend on how they are served.

A DataLoader left to its defaults draws from the GLOBAL torch generator - the same stream dropout
draws from - and how often it draws depends on its serving settings. Every new iterator draws a
worker base seed. A non-persistent loader builds a new iterator every epoch, while a persistent one
builds it once and then resets it. So flipping `persistent_workers` shifted the global stream from
epoch 1 onward, giving a different shuffle order and different dropout masks. That is how a tuned
config trained to val_loss 0.5566 against its trial's 0.5405 from an identical epoch 0: the study
ran with `persistent_workers: true`, and the export restored the registry's `false`.

Here the loader and its sampler each get a private generator, seeded from the global stream once,
when the loader is built. After that, iterating the loader never touches the global stream, so
num_workers and persistent_workers change only how fast batches arrive. The run seed still reaches
the shuffle, because the private seeds are drawn from the seeded global stream, so a different run
seed or ensemble member still sees a different order.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler


def _private_generator() -> torch.Generator:
    """A generator seeded by exactly one draw from the global stream."""
    seed = int(torch.empty((), dtype=torch.int64).random_().item())
    return torch.Generator().manual_seed(seed)


def build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    collate_fn: Callable[[Any], Any] | None = None,
) -> DataLoader:
    """A DataLoader that draws from the global torch stream here, and never while it is iterated.

    Two generators rather than one. The sampler draws a permutation per epoch, and the iterator draws
    a worker base seed per new iterator. If they shared a generator, the base-seed draws - whose
    count depends on persistent_workers - would shift every later permutation.
    """
    # Both drawn whether or not this loader shuffles, so the number of global draws a build makes
    # does not depend on its arguments either.
    sampler_generator = _private_generator()
    loader_generator = _private_generator()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=RandomSampler(dataset, generator=sampler_generator) if shuffle else None,
        drop_last=bool(drop_last),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
        generator=loader_generator,
    )
