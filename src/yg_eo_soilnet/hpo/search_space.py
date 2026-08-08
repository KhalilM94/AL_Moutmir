"""Declarative search spaces: YAML in, Optuna suggestions out.

A search space is data, not code, and it lives beside the registry it tunes rather than inside any
model class. Model modules stay completely unaware that HPO exists.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import optuna
import yaml

from yg_eo_soilnet.hpo.constraints import ConstraintHook, resolve_constraints, split_derive_entry
from yg_eo_soilnet.hpo.overrides import to_builtin, validate_override_keys

DIRECTIONS = {"minimize": "min", "maximize": "max"}

DEFAULT_OBJECTIVE_METRIC = "val_r2"
DEFAULT_OBJECTIVE_DIRECTION = "maximize"

# Exactly what SoilRegressionLightningBase logs: `{stage}_loss` in _shared_step, `{stage}_r2` and
# `{stage}_pred_std_ratio` in _log_epoch_metrics. Every Lightning model in the registry inherits it.
#
# Checked at load time on purpose. An objective naming a metric nothing logs is not detectable at
# runtime until a trial finishes and finds nothing to score - at which point every trial is pruned
# and a study of any length yields nothing. Failing here costs a second instead of a night.
KNOWN_METRICS = frozenset(
    f"{stage}_{name}"
    for stage in ("train", "val", "test")
    for name in ("loss", "r2", "pred_std_ratio")
)

_SPACE_KEYS = {"objective", "sampler", "pruner", "fixed", "params", "derive"}

# How many hex characters of the search-space digest go into a study name. Six is short enough to
# type and read in a log line, and collisions do not silently corrupt anything: the full digest is
# stored on the study and compared on resume.
FINGERPRINT_CHARS = 6


def describe_derive(entry: Any) -> str:
    """A stable one-line rendering of a `derive:` entry, options included."""
    name, options = split_derive_entry(entry)
    if not options:
        return name
    rendered = ", ".join(f"{key}={value}" for key, value in sorted(options.items()))
    return f"{name}({rendered})"


@dataclass(frozen=True)
class Objective:
    """What a study optimizes, and the Lightning monitor/mode that must agree with it."""

    metric: str = DEFAULT_OBJECTIVE_METRIC
    direction: str = DEFAULT_OBJECTIVE_DIRECTION

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(
                f"objective.direction must be one of {', '.join(sorted(DIRECTIONS))}; got {self.direction!r}."
            )
        if self.metric not in KNOWN_METRICS:
            hint = ""
            if "mse" in self.metric.lower() or "loss" in self.metric.lower():
                hint = (
                    " For mean squared error use 'val_loss' with direction 'minimize' - the models "
                    "log the loss under that name, and loss_name: mse makes it the MSE."
                )
            raise ValueError(
                f"objective.metric {self.metric!r} is not logged by any model in this repo. "
                f"Expected one of: {', '.join(sorted(KNOWN_METRICS))}.{hint} "
                f"(If you added a new logged metric, extend KNOWN_METRICS in hpo/search_space.py.)"
            )

    @property
    def mode(self) -> str:
        """`"min"` or `"max"`, as EarlyStopping and ModelCheckpoint spell it."""
        return DIRECTIONS[self.direction]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "Objective":
        mapping = mapping or {}
        return cls(
            metric=str(mapping.get("metric", DEFAULT_OBJECTIVE_METRIC)),
            direction=str(mapping.get("direction", DEFAULT_OBJECTIVE_DIRECTION)),
        )


@dataclass(frozen=True)
class Distribution:
    """One searched parameter, addressed by its dotted registry path."""

    dotted: str
    kind: str
    spec: dict[str, Any] = field(default_factory=dict)
    when: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, dotted: str, mapping: Mapping[str, Any]) -> "Distribution":
        if not isinstance(mapping, Mapping):
            raise ValueError(f"Search space entry {dotted!r} must be a mapping, e.g. {{type: float, low: 0, high: 1}}.")
        spec = dict(mapping)
        when = dict(spec.pop("when", {}) or {})
        kind = str(spec.pop("type", "")).lower()

        if kind == "categorical":
            choices = spec.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"{dotted!r}: a categorical needs a non-empty 'choices' list.")
            # Optuna stores categorical choices in the study database, so they must be scalars.
            # A list-valued hyperparameter (head_hidden_dims) belongs in a constraint hook.
            for choice in choices:
                if not isinstance(choice, (bool, int, float, str)) and choice is not None:
                    raise ValueError(
                        f"{dotted!r}: categorical choices must be null, bool, int, float or str; "
                        f"got {type(choice).__name__}. Use a 'derive' hook for structured values."
                    )
        elif kind in {"float", "int"}:
            missing = [bound for bound in ("low", "high") if bound not in spec]
            if missing:
                raise ValueError(f"{dotted!r}: a {kind} distribution needs {' and '.join(missing)}.")
            if spec.get("log") and spec.get("step") is not None:
                raise ValueError(f"{dotted!r}: Optuna rejects 'log' and 'step' together.")
            if float(spec["low"]) > float(spec["high"]):
                raise ValueError(f"{dotted!r}: low ({spec['low']}) is above high ({spec['high']}).")
        else:
            raise ValueError(f"{dotted!r}: unknown type {kind!r}; expected float, int or categorical.")

        return cls(dotted=dotted, kind=kind, spec=spec, when=when)

    def applies(self, chosen: Mapping[str, Any]) -> bool:
        """True when every `when:` guard matches what has already been drawn.

        A guard on an undrawn parameter is False, not an error at this point: the referenced
        parameter may itself have been guarded out this trial.
        """
        return all(dotted in chosen and chosen[dotted] == expected for dotted, expected in self.when.items())

    def suggest(self, trial: optuna.Trial) -> Any:
        if self.kind == "categorical":
            return trial.suggest_categorical(self.dotted, self.spec["choices"])
        low, high, step = self.spec["low"], self.spec["high"], self.spec.get("step")
        log = bool(self.spec.get("log", False))
        if self.kind == "int":
            return trial.suggest_int(self.dotted, int(low), int(high), step=int(step or 1), log=log)
        if step is None:
            return trial.suggest_float(self.dotted, float(low), float(high), log=log)
        return trial.suggest_float(self.dotted, float(low), float(high), step=float(step))

    def describe(self) -> str:
        if self.kind == "categorical":
            body = "|".join(str(choice) for choice in self.spec["choices"])
        else:
            body = f"{self.spec['low']}..{self.spec['high']}"
            if self.spec.get("log"):
                body += " log"
            if self.spec.get("step") is not None:
                body += f" step={self.spec['step']}"
        guard = f" when {self.when}" if self.when else ""
        return f"{self.kind}({body}){guard}"


@dataclass
class SearchSpace:
    """A whole study's declaration: what to optimize, what to pin, and what to search."""

    entry: str
    objective: Objective
    fixed: dict[str, Any] = field(default_factory=dict)
    distributions: list[Distribution] = field(default_factory=list)
    # Each entry is a hook name, or a single-key mapping of a hook name to its options.
    derive: list[Any] = field(default_factory=list)
    sampler_spec: dict[str, Any] = field(default_factory=dict)
    pruner_spec: dict[str, Any] = field(default_factory=dict)
    _hooks: list[ConstraintHook] = field(default_factory=list, repr=False)

    @classmethod
    def from_mapping(cls, entry: str, mapping: Mapping[str, Any]) -> "SearchSpace":
        unknown = sorted(set(mapping) - _SPACE_KEYS)
        if unknown:
            raise ValueError(
                f"Search space {entry!r} has unknown key(s): {', '.join(unknown)}. "
                f"Expected any of: {', '.join(sorted(_SPACE_KEYS))}."
            )

        fixed = {key: to_builtin(value) for key, value in (mapping.get("fixed") or {}).items()}
        validate_override_keys(fixed, searched=False)

        params = mapping.get("params") or {}
        if not params and not mapping.get("derive"):
            raise ValueError(f"Search space {entry!r} declares no 'params' and no 'derive' hooks.")
        validate_override_keys(params, searched=True)

        distributions = [Distribution.from_mapping(dotted, spec) for dotted, spec in params.items()]

        # A `when:` guard may only reference a parameter drawn earlier - dict order is the draw
        # order, so a forward reference would silently never match.
        drawn: set[str] = set()
        for distribution in distributions:
            for guarded in distribution.when:
                if guarded not in drawn:
                    raise ValueError(
                        f"{distribution.dotted!r}: 'when' references {guarded!r}, which is not "
                        f"declared before it. Move {guarded!r} earlier in the 'params' block."
                    )
            drawn.add(distribution.dotted)

        derive = list(mapping.get("derive") or [])
        return cls(
            entry=entry,
            objective=Objective.from_mapping(mapping.get("objective")),
            fixed=fixed,
            distributions=distributions,
            derive=derive,
            sampler_spec=dict(mapping.get("sampler") or {}),
            pruner_spec=dict(mapping.get("pruner") or {}),
            _hooks=resolve_constraints(derive),
        )

    @classmethod
    def from_yaml(cls, path: str, entry: str) -> "SearchSpace":
        with open(path, "r") as handle:
            document = yaml.safe_load(handle) or {}
        if entry not in document:
            available = ", ".join(sorted(document)) or "(none)"
            raise KeyError(f"No search space for registry entry {entry!r} in {path}. Available: {available}.")
        return cls.from_mapping(entry, document[entry] or {})

    def suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Draw one full set of overrides: the pinned values, then the searched ones."""
        chosen: dict[str, Any] = dict(self.fixed)
        for distribution in self.distributions:
            if distribution.applies(chosen):
                chosen[distribution.dotted] = to_builtin(distribution.suggest(trial))
        for hook in self._hooks:
            hook(trial, chosen)
        return {key: to_builtin(value) for key, value in chosen.items()}

    def make_sampler(self) -> optuna.samplers.BaseSampler:
        spec = dict(self.sampler_spec)
        name = str(spec.pop("name", "tpe")).lower()
        if name == "tpe":
            return optuna.samplers.TPESampler(**spec)
        if name == "random":
            return optuna.samplers.RandomSampler(**spec)
        raise ValueError(f"Unknown sampler {name!r}; expected 'tpe' or 'random'.")

    def make_pruner(self) -> optuna.pruners.BasePruner:
        spec = dict(self.pruner_spec)
        name = str(spec.pop("name", "median")).lower()
        if name == "median":
            return optuna.pruners.MedianPruner(**spec)
        if name == "hyperband":
            return optuna.pruners.HyperbandPruner(**spec)
        if name == "none":
            return optuna.pruners.NopPruner()
        raise ValueError(f"Unknown pruner {name!r}; expected 'median', 'hyperband' or 'none'.")

    def describe(self) -> dict[str, str]:
        """A flat, loggable summary of the space - used as MLflow params on the study run."""
        described = {f"space.{distribution.dotted}": distribution.describe() for distribution in self.distributions}
        described.update({f"fixed.{key}": str(value) for key, value in self.fixed.items()})
        if self.derive:
            described["space.derive"] = ", ".join(describe_derive(entry) for entry in self.derive)
        return described

    def fingerprint(self) -> str:
        """A digest of everything that makes two trials comparable.

        Optuna resumes a study by name, and the default name is derived from this, so editing a
        bound, a metric, a pinned value or a derive hook starts a clean study instead of appending
        incomparable trials to the old leaderboard - which is what `best_trial`, the exported YAML
        and every plot are read off.

        Sampler and pruner are deliberately excluded. They change HOW the space is searched, not what
        a recorded value means, so swapping TPE for random or retuning the pruner still resumes.
        """
        payload = {
            "metric": self.objective.metric,
            "direction": self.objective.direction,
            **self.describe(),
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        return digest[:FINGERPRINT_CHARS]
