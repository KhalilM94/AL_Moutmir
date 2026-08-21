"""How a run's targets are grouped, and how a group is named.

One question, asked the same way by both families: given the configured TARGET_COLUMNS and a
registry entry, which targets does a single fitted model cover? The answer is a list of GROUPS.

  joint       -> [[a, b, c]]        one model with a 3-wide head
  per_target  -> [[a], [b], [c]]    three independent models

Before this module the answer was implicit and family-specific. Lightning was ALWAYS joint - the
sequence and graph builders read config.TARGET_COLUMNS wholesale and nothing ever narrowed `y` to
the run's target - while sklearn was always per-target. The `run_lightning_once` predicate in
main.py looked like the switch but only controlled how many times that same multi-output model was
trained. Grouping is now an explicit choice, and both families obey it.

The "__" encoding of a joint group's name also lives here. It was copy-pasted in four places and a
target name containing "__" broke the round trip silently; join_target_names refuses that name
instead.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

JOINT = "joint"
PER_TARGET = "per_target"
VALID_MODES = (JOINT, PER_TARGET)

# sklearn entries declare capability with the same key: "native" means the estimator fits a 2-D y
# itself. Anything that is not a mode and not "native" is a configuration error.
NATIVE = "native"

TARGET_NAME_SEPARATOR = "__"


def join_target_names(names: Sequence[str]) -> str:
    """The run label for a group of targets.

    A single target is its own name, so single-target runs keep the labels they have always had.
    """
    names = [str(name) for name in names]
    if not names:
        raise ValueError("Cannot build a run label from an empty target group.")
    if len(names) == 1:
        return names[0]
    # The separator is also the decoder, so a name containing it would split into pieces that name
    # no column. Better to refuse the config than to emit a label that cannot be read back.
    offenders = [name for name in names if TARGET_NAME_SEPARATOR in name]
    if offenders:
        raise ValueError(
            f"Target names may not contain {TARGET_NAME_SEPARATOR!r}, which separates the names of "
            f"a joint run: {sorted(offenders)}. Rename the column or fit these targets separately."
        )
    return TARGET_NAME_SEPARATOR.join(names)


def split_target_names(encoded: Any) -> list[str]:
    """The targets behind a run label. Inverse of :func:`join_target_names`."""
    if encoded is None:
        return []
    return [name for name in str(encoded).split(TARGET_NAME_SEPARATOR) if name]


def resolve_mode(config: Any, spec: Optional[Mapping[str, Any]] = None) -> str:
    """The grouping mode in force, entry override first, then the global default.

    A registry entry declaring ``multi_target: native`` is stating an sklearn capability rather than
    asking for a mode, so it falls through to the global default.
    """
    if spec is not None:
        declared = spec.get("multi_target")
        if declared is not None and str(declared).lower() != NATIVE:
            mode = str(declared).lower()
            if mode not in VALID_MODES:
                raise ValueError(
                    f"multi_target must be one of {VALID_MODES} or {NATIVE!r}, got {declared!r}."
                )
            return mode

    mode = str(getattr(config, "MULTI_TARGET_MODE", JOINT) or JOINT).lower()
    if mode not in VALID_MODES:
        raise ValueError(f"MULTI_TARGET_MODE must be one of {VALID_MODES}, got {mode!r}.")
    return mode


def supports_joint(spec: Optional[Mapping[str, Any]] = None) -> bool:
    """Whether an sklearn registry entry can fit a 2-D y itself.

    Lightning entries do not use this: every Lightning head is a Linear(..., target_dim) and is
    joint-capable by construction. sklearn estimators are not - GradientBoostingRegressor and
    TabICLRegressor are single-output - so an entry has to opt in.
    """
    return spec is not None and str(spec.get("multi_target", "")).lower() == NATIVE


def resolve_target_groups(
    config: Any,
    spec: Optional[Mapping[str, Any]] = None,
    *,
    require_joint_support: bool = False,
    logger: Any = None,
    entry_name: str = "",
) -> list[list[str]]:
    """The target groups one registry entry is fitted over.

    ``require_joint_support`` is for the sklearn side: an entry that has not declared itself
    multi-output falls back to per-target with a warning rather than failing. One unsupported
    estimator must not take the whole run down.
    """
    targets = [str(name) for name in (getattr(config, "TARGET_COLUMNS", []) or [])]
    if not targets:
        return []
    if len(targets) == 1:
        return [targets]

    mode = resolve_mode(config, spec)
    if mode == JOINT and require_joint_support and not supports_joint(spec):
        if logger is not None:
            label = entry_name or "this entry"
            logger.warning(
                f"MULTI_TARGET_MODE is 'joint' but {label} does not declare 'multi_target: native', "
                f"so it cannot fit a 2-D target. Falling back to one model per target."
            )
        mode = PER_TARGET

    return [list(targets)] if mode == JOINT else [[name] for name in targets]


def group_label(group: Iterable[str]) -> str:
    """Convenience: the run label for a group, in the shape callers actually hold it."""
    return join_target_names(list(group))


def select_target_columns(
    targets: Any,
    target_names: Sequence[str],
    active_targets: Optional[Sequence[str]],
) -> tuple[Any, list[str], Optional[list[int]]]:
    """Narrow an ``(n_points, n_targets)`` block to the targets this run actually fits.

    Returns the narrowed array, its names, and the column indices taken (None when nothing was
    narrowed, so callers can skip work). The bundle is built once over every configured target and
    cached across registry entries, so per-target runs slice here rather than rebuilding it - the
    build is the expensive half and would otherwise be repaid once per target.
    """
    import numpy as np

    names = [str(name) for name in target_names]
    if not active_targets:
        return targets, names, None

    wanted = [str(name) for name in active_targets]
    if wanted == names:
        return targets, names, None

    missing = [name for name in wanted if name not in names]
    if missing:
        raise ValueError(
            f"active_targets names columns the bundle does not carry: {sorted(missing)}. "
            f"Available: {names}."
        )

    indices = [names.index(name) for name in wanted]
    array = np.asarray(targets)
    # An empty block (no targets built at all) has nothing to take columns from; leaving it alone
    # keeps the (0, 0) shape the dataclasses default to rather than raising on the index.
    if array.ndim != 2 or array.shape[1] == 0:
        return targets, wanted, indices
    return array[:, indices], wanted, indices
