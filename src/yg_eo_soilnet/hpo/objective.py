"""The Optuna objective: a trial's draw becomes a registry entry, a bundle, and a number.

This is where the loose coupling lives. Nothing below names a model or a datamodule class - a trial
mutates one registry entry and hands it to an unmodified LightningConfigFactory, so any model the
registry can already build is tunable, including ones added after this file was written.
"""

from __future__ import annotations

import importlib
from copy import deepcopy
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Mapping

import optuna

from yg_eo_soilnet.hpo.overrides import apply_overrides
from yg_eo_soilnet.hpo.search_space import SearchSpace
from yg_eo_soilnet.hpo.trial_runner import (
    TrialRunner,
    UnrecoverableAcceleratorError,
    raise_if_accelerator_is_dead,
    release_dataloader_workers,
)
from yg_eo_soilnet.models.config_fatories.lightning_config_factory import LightningConfigFactory

# The key an objective stores its resolved overrides under, so export does not have to replay the
# search space to reconstruct what a trial actually ran.
OVERRIDES_ATTR = "overrides"


def seed_everything(seed: int) -> None:
    lightning = importlib.import_module("lightning.pytorch")
    lightning.seed_everything(seed, workers=True)


@dataclass
class ObjectiveContext:
    """Everything a study needs that is the same for every trial, built once up front."""

    entry: str
    registry_entry: dict[str, Any]
    config: Any
    target: str
    data: Mapping[str, Any]
    logger: Any = None
    data_manager: Any = None
    datamodule_cache: dict[Any, Any] | None = field(default=None)

    @classmethod
    def from_config(cls, entry: str, config: Any, data: Mapping[str, Any], **kwargs) -> "ObjectiveContext":
        registry = config.LIGHTNING_MODEL_REGISTRY
        if entry not in registry:
            available = ", ".join(sorted(registry)) or "(none)"
            raise KeyError(f"No entry {entry!r} in the Lightning registry. Available: {available}.")
        targets = list(config.TARGET_COLUMNS)
        # Mirrors main.py: the sequence and graph datamodules span every point at once, so a
        # multi-target run is one combined run rather than one run per target.
        target = "__".join(targets) if len(targets) > 1 else (targets[0] if targets else "target")
        return cls(entry=entry, registry_entry=deepcopy(registry[entry]), config=config, target=target, data=data, **kwargs)


class TrialObjective:
    """Callable passed to ``study.optimize``."""

    def __init__(
        self,
        context: ObjectiveContext,
        space: SearchSpace,
        *,
        seed: int = 42,
        seed_repeats: int = 1,
        fail_fast: bool = False,
        progress: Any = None,
    ):
        self.context = context
        self.space = space
        self.seed = int(seed)
        self.seed_repeats = max(1, int(seed_repeats))
        self.runner = TrialRunner(
            space.objective,
            logger=context.logger,
            fail_fast=fail_fast,
            extra_callbacks=[progress.epoch_callback()] if progress is not None else None,
        )

    def build_bundle(self, overrides: Mapping[str, Any]):
        """A ready-to-train bundle for one set of overrides.

        Also the export path's verification hook: the same function that runs a trial builds the
        bundle from the exported configuration, so the two cannot drift.
        """
        spec = apply_overrides(deepcopy(self.context.registry_entry), overrides)
        # A single-entry registry: build_lightning_configs skips anything not enabled, and the
        # study tunes one entry at a time regardless of what the file-level registry has switched on.
        spec["enabled"] = True
        factory = LightningConfigFactory(
            {self.context.entry: spec},
            self.context.config,
            logger=self.context.logger,
            data_manager=self.context.data_manager,
            datamodule_cache=self.context.datamodule_cache,
        )
        return factory.build_lightning_configs(target=self.context.target, data=self.context.data)[self.context.entry]

    def __call__(self, trial: optuna.Trial) -> float:
        overrides = self.space.suggest(trial)
        # Stored now rather than derived later: replaying a search space outside a live trial cannot
        # reproduce a conditional draw, and export must emit exactly what ran.
        trial.set_user_attr(OVERRIDES_ATTR, overrides)

        values: list[float] = []
        for repeat in range(self.seed_repeats):
            bundle = None
            try:
                # Before the factory builds the model, not after. LightningTrainer._seed_for_bundle
                # seeds inside train(), by which point the weights already exist - so on the factory
                # path initialization is unseeded. Seeding here is what makes two trials differ by
                # their hyperparameters and nothing else.
                #
                # Inside the try, because seeding touches the accelerator: seed_everything reaches
                # torch.cuda.manual_seed_all, which is where a study died on the trial after the GPU
                # was lost - outside any guard, so it killed the process instead of stopping.
                seed_everything(self.seed + repeat)
                bundle = self.build_bundle(overrides)
                # Only the first repeat reports intermediates: Optuna keeps one value per step, so a
                # second curve would overwrite the first.
                result = self.runner.run(bundle, trial, report=repeat == 0)
                values.append(result.value)
                if repeat == 0:
                    trial.set_user_attr("best_epoch", result.best_epoch)
                    trial.set_user_attr("epochs_run", result.epochs_run)
            except (optuna.TrialPruned, UnrecoverableAcceleratorError):
                raise
            except Exception as exc:
                # Seeding and model construction sit outside TrialRunner's guard, so this is the
                # only place a lost device can be caught before it escapes the study.
                raise_if_accelerator_is_dead(exc, trial.number)
                raise
            finally:
                # In a finally because a pruned trial raises out of run(), and the traceback keeps
                # this frame - and so the bundle, the model and the Trainer behind it - alive.
                # Dropping the bundle closes the Trainer cycle; the collect then shuts the trial's
                # DataLoader workers down here rather than leaving them for the next fork to inherit.
                bundle = None  # noqa: F841
                release_dataloader_workers()

        if self.seed_repeats > 1:
            trial.set_user_attr("seed_values", values)
        return fmean(values)
