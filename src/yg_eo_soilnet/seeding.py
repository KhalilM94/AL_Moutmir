"""One seeding function, shared by every path that trains a Lightning model.

Its own module because both the config factory and the trainer need it, and the trainer already
imports from the factory - putting it in either would make that circular.
"""

from __future__ import annotations

import importlib
import random

import numpy as np

try:  # pragma: no cover - optional dependency, mirrors the rest of the package
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def seed_everything(seed: int) -> None:
    """Seed every RNG a Lightning run touches, dataloader workers included.

    Call this immediately BEFORE the model is constructed. Weight initialization draws from the
    global torch generator, so seeding afterwards leaves the weights at whatever state the preceding
    work happened to leave behind, and starts `fit` from a different point in the stream than a run
    that seeded first. That asymmetry is what stopped a tuned configuration from reproducing the
    hyperparameter trial that selected it.

    `workers=True` is the part a hand-rolled seeder misses: it sets PL_SEED_WORKERS so each
    DataLoader worker derives its stream from this seed rather than an arbitrary one.
    """
    seed_value = int(seed)
    try:
        lightning = importlib.import_module("lightning.pytorch")
    except ImportError:  # pragma: no cover - exercised only when lightning is absent
        lightning = None
    lightning_seed = getattr(lightning, "seed_everything", None)
    if callable(lightning_seed):
        lightning_seed(seed_value, workers=True)
        return

    random.seed(seed_value)
    np.random.seed(seed_value)
    if torch is not None:
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():  # pragma: no cover - hardware dependent
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
