"""Writing suggested hyperparameters back into a Lightning registry entry.

This module is the entire coupling between Optuna and the rest of the codebase. A search space names
its parameters with a dotted prefix, and `apply_overrides` writes each one into the matching section
of a deep-copied registry entry. The mutated entry then goes to an unmodified LightningConfigFactory,
so `auto` shape resolution, signature filtering and loud failure on typos all keep working.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

# Dotted prefix -> the registry section it writes to. `callbacks` is handled separately because it
# nests one level deeper (callbacks.<group>.<key>).
PREFIX_SECTIONS = {
    "model": "init_args",
    "datamodule": "datamodule_init_args",
    "trainer": "trainer_args",
}

# Model init_args that LightningConfigFactory resolves from the datamodule via its
# auto/None/0 sentinels (see _build_model). A tuned value here is either silently overwritten or
# corrupts the shape contract between the datamodule and the model, so both are worth refusing.
# NOTE: `embedding_dims` is deliberately absent - the factory never touches it, and
# resolve_embedding_dims accepts "auto" or a scalar, which makes it a genuine search dimension.
FACTORY_RESOLVED_MODEL_KEYS = frozenset(
    {
        "static_dim",
        "target_dim",
        "modality_dims",
        "temporal_steps",
        "edge_attr_dim",
        "grid_years",
        "categorical_cardinalities",
        "categorical_vocabularies",
        "categorical_feature_names",
        "temporal_enabled",
        "output_dim",
        "target_mean",
        "target_scale",
        "target_transform",
    }
)

# Datamodule args that define the train/val/test split. SoilSequenceDataModule.setup fits the
# scaler, the categorical vocabulary and target_mean_/target_scale_ on the train split, so varying
# any of these changes what val_loss and val_r2 are even measuring. Pinning them in `fixed` is fine;
# searching over them is not.
SPLIT_DEFINING_DATAMODULE_KEYS = frozenset({"val_size", "test_size", "seed"})


def to_builtin(value: Any) -> Any:
    """A plain-Python copy of `value`.

    Every model calls save_hyperparameters(), so anything landing in init_args ends up in the
    checkpoint's hyper_parameters. A numpy scalar there makes the checkpoint unloadable under
    torch.load's weights_only=True default - the same reason the model constructors coerce their
    categorical and target-statistic arguments by hand.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (str, bytes)) or value is None:
        return value.decode() if isinstance(value, bytes) else value
    if isinstance(value, Mapping):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [to_builtin(item) for item in value]
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if hasattr(value, "item"):  # any remaining 0-d numpy/torch scalar
        return to_builtin(value.item())
    return value


def split_dotted(dotted: str) -> tuple[str, str]:
    """`("model", "learning_rate")` for `"model.learning_rate"`, validating the prefix."""
    prefix, _, remainder = dotted.partition(".")
    if not remainder:
        raise ValueError(
            f"Override key {dotted!r} needs a section prefix, e.g. 'model.learning_rate' or "
            f"'datamodule.batch_size'."
        )
    if prefix == "callbacks":
        group, _, key = remainder.partition(".")
        if not key or "." in key:
            raise ValueError(
                f"Override key {dotted!r} must be 'callbacks.<group>.<key>', e.g. "
                f"'callbacks.early_stopping.patience'."
            )
        return prefix, remainder
    if prefix not in PREFIX_SECTIONS:
        allowed = ", ".join(sorted([*PREFIX_SECTIONS, "callbacks"]))
        raise ValueError(f"Unknown override prefix {prefix!r} in {dotted!r}; expected one of: {allowed}.")
    if "." in remainder:
        raise ValueError(f"Override key {dotted!r} has too many dots for a '{prefix}' override.")
    return prefix, remainder


def validate_override_keys(dotted_keys: Iterable[str], *, searched: bool) -> None:
    """Reject keys that must not be overridden. Raises with every offender, not just the first.

    `searched` distinguishes a tuned parameter from one pinned for the whole study: a fixed
    `datamodule.val_size` is harmless, a searched one silently invalidates the objective.
    """
    problems: list[str] = []
    for dotted in dotted_keys:
        prefix, key = split_dotted(dotted)
        if prefix == "model" and key in FACTORY_RESOLVED_MODEL_KEYS:
            problems.append(
                f"  {dotted}: resolved from the datamodule by LightningConfigFactory._build_model; "
                f"overriding it breaks the model/datamodule shape contract."
            )
        elif searched and prefix == "datamodule" and key in SPLIT_DEFINING_DATAMODULE_KEYS:
            problems.append(
                f"  {dotted}: changes the train/val/test split, and with it the fitted scaler, the "
                f"categorical vocabulary and target_mean_/target_scale_. Trials would not be "
                f"comparable. Pin it under 'fixed:' instead of searching it."
            )
    if problems:
        section = "params" if searched else "fixed"
        raise ValueError(f"Invalid keys in the '{section}' block of the search space:\n" + "\n".join(problems))


def apply_overrides(spec: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Write `overrides` into `spec` in place and return it.

    `spec` is expected to be a deep copy of a registry entry - this mutates it.
    """
    for dotted, value in overrides.items():
        prefix, remainder = split_dotted(dotted)
        if prefix == "callbacks":
            group, _, key = remainder.partition(".")
            destination = spec.setdefault("callbacks", {}).setdefault(group, {})
        else:
            destination, key = spec.setdefault(PREFIX_SECTIONS[prefix], {}), remainder
        destination[key] = to_builtin(value)
    return spec
