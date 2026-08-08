"""Turning a winning trial into a registry file you can train from.

The export goes through the same `apply_overrides` a trial used, so the emitted YAML is provably the
configuration that produced the recorded value rather than a re-derivation of it.
"""

from __future__ import annotations

import datetime
from copy import deepcopy
from pathlib import Path
from typing import Any

import optuna
import yaml

from yg_eo_soilnet.hpo.overrides import apply_overrides
from yg_eo_soilnet.hpo.search_space import Objective

OVERRIDES_ATTR = "overrides"

# Dataloader throughput knobs, not hyperparameters: they change no arithmetic and no result, only
# how fast batches arrive. A study pins them to values that suit hundreds of short trials, and
# without this the exported production config would silently inherit those instead of the
# registry's - which is how `num_workers: 4` and `persistent_workers: true` ended up in a tuned file
# whose registry said 11 and false.
INFRASTRUCTURE_DATAMODULE_KEYS = ("num_workers", "pin_memory", "persistent_workers")


def best_overrides(study: optuna.Study) -> dict[str, Any]:
    """The overrides the best trial actually ran with.

    Read back from the trial rather than replayed through the search space: a conditional draw
    cannot be reproduced outside a live trial, and `derive` hooks draw parameters of their own.
    """
    trial = study.best_trial
    overrides = trial.user_attrs.get(OVERRIDES_ATTR)
    if overrides is None:
        raise KeyError(
            f"Best trial {trial.number} of study {study.study_name!r} carries no {OVERRIDES_ATTR!r} "
            f"attribute. It predates this exporter, or was not run by TrialObjective."
        )
    return dict(overrides)


def build_tuned_spec(registry_entry: dict[str, Any], overrides: dict[str, Any], objective: Objective) -> dict[str, Any]:
    """The pristine registry entry with the winning overrides and production settings restored."""
    spec = apply_overrides(deepcopy(registry_entry), overrides)
    spec["enabled"] = True
    spec.setdefault("trainer_args", {})["enable_checkpointing"] = True

    # The study early-stopped and scored on the objective metric. If the production run reverted to
    # the registry's val_loss it would select a different epoch than the one the trial was ranked
    # on, and the retrained model would not reproduce the study's number.
    callbacks = spec.setdefault("callbacks", {})
    for group in ("early_stopping", "checkpoint"):
        callbacks.setdefault(group, {}).update(monitor=objective.metric, mode=objective.mode)

    # Throughput settings come back from the registry, not from whatever the trial ran with. A key
    # the registry does not declare is removed rather than kept, so the entry falls back to the
    # factory's config.LIGHTNING_* default exactly as the registry itself would.
    source_datamodule_args = registry_entry.get("datamodule_init_args", {}) or {}
    datamodule_args = spec.setdefault("datamodule_init_args", {})
    for key in INFRASTRUCTURE_DATAMODULE_KEYS:
        if key in source_datamodule_args:
            datamodule_args[key] = deepcopy(source_datamodule_args[key])
        else:
            datamodule_args.pop(key, None)
    return spec


def _header(
    study: optuna.Study, entry: str, objective: Objective, registry_path: str | None, path: Path
) -> str:
    trial = study.best_trial
    lines = [
        "# Tuned Lightning registry entry, exported from an Optuna study.",
        "#",
        f"#   study      : {study.study_name}",
        f"#   entry      : {entry}",
        f"#   best trial : #{trial.number} of {len(study.trials)}",
        f"#   objective  : {objective.metric} = {study.best_value:.6f} ({objective.direction})",
        f"#   exported   : {datetime.datetime.now().isoformat(timespec='seconds')}",
    ]
    if registry_path:
        lines.append(f"#   source     : {registry_path}")
    lines += [
        "#",
        "# Values pinned under `fixed:` in the search space are baked in here too - notably",
        "# trainer.max_epochs, which is the tuning budget. Raise it for a final production run if you",
        "# want early stopping, rather than the epoch budget, to decide when to stop.",
        "#",
        f"# Exceptions: {', '.join(INFRASTRUCTURE_DATAMODULE_KEYS)} are restored from the source",
        "# registry. They are throughput knobs that change no result, so production keeps its own.",
        "#",
        "# Train it with:",
        f"#   LIGHTNING_MODEL_REGISTRY_PATH={path} python main.py",
        "",
    ]
    return "\n".join(lines)


def export_best_config(
    study: optuna.Study,
    entry: str,
    registry_entry: dict[str, Any],
    objective: Objective,
    path: str | Path,
    *,
    registry_path: str | None = None,
) -> Path:
    """Write `{entry: tuned_spec}` to `path` and return it."""
    spec = build_tuned_spec(registry_entry, best_overrides(study), objective)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(_header(study, entry, objective, registry_path, path))
        yaml.safe_dump({entry: spec}, handle, sort_keys=False, default_flow_style=False)
    return path
