"""Named hooks for the couplings a declarative search space cannot express.

Almost every hyperparameter is an independent draw and belongs in YAML. The exceptions are
parameters constrained by *each other* - a width that must divide by a head count, or a
variable-length list of layer widths. Those live here as named callables a search space opts into
with `derive: [<name>]`, so the YAML stays declarative and the escape hatch stays small and tested.

A hook receives the live Optuna trial and the parameters chosen so far, and mutates that dict.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Mapping, MutableMapping

import optuna

ConstraintHook = Callable[[optuna.Trial, MutableMapping[str, Any]], None]

CONSTRAINTS: dict[str, ConstraintHook] = {}


def constraint(name: str) -> Callable[[ConstraintHook], ConstraintHook]:
    def register(hook: ConstraintHook) -> ConstraintHook:
        if name in CONSTRAINTS:
            raise ValueError(f"A constraint hook named {name!r} is already registered.")
        CONSTRAINTS[name] = hook
        return hook

    return register


def split_derive_entry(entry: Any) -> tuple[str, dict[str, Any]]:
    """`"name"` or `{"name": {...options}}` -> `(name, options)`.

    The mapping form lets one hook serve several search spaces on its own terms - the head pyramid
    wants a different floor per model - without turning every option into a separate hook.
    """
    if isinstance(entry, str):
        return entry, {}
    if isinstance(entry, Mapping) and len(entry) == 1:
        name, options = next(iter(entry.items()))
        if isinstance(name, str) and isinstance(options or {}, Mapping):
            return name, dict(options or {})
    raise ValueError(
        f"Malformed 'derive' entry {entry!r}; expected a hook name, or a single-key mapping of "
        f"a hook name to its options, e.g. {{head_hidden_dims_pyramid: {{floor: 16}}}}."
    )


def resolve_constraints(entries: list[Any]) -> list[ConstraintHook]:
    """Look up hooks by name, failing at search-space load time rather than mid-study.

    Options given in the mapping form are bound here, so a resolved hook is always callable as
    `(trial, chosen)` and nothing downstream has to carry them.
    """
    parsed = [split_derive_entry(entry) for entry in entries]
    unknown = [name for name, _ in parsed if name not in CONSTRAINTS]
    if unknown:
        available = ", ".join(sorted(CONSTRAINTS)) or "(none)"
        raise ValueError(f"Unknown constraint hook(s): {', '.join(unknown)}. Available: {available}.")
    return [
        partial(CONSTRAINTS[name], **options) if options else CONSTRAINTS[name] for name, options in parsed
    ]


@constraint("d_model_divisible_by_nhead")
def d_model_divisible_by_nhead(trial: optuna.Trial, chosen: MutableMapping[str, Any]) -> None:
    """Round `model.d_model` up to a multiple of `model.nhead`.

    TimeAwareTransformerEncoder raises unless d_model % nhead == 0. Repairing the draw instead of
    rejecting it keeps every trial valid by construction, and costs no extra search dimension - the
    two parameters stay independent draws in the YAML.
    """
    d_model = chosen.get("model.d_model")
    nhead = chosen.get("model.nhead")
    if d_model is None or nhead is None:
        return

    d_model, nhead = int(d_model), int(nhead)
    remainder = d_model % nhead
    if remainder:
        chosen["model.d_model"] = d_model + (nhead - remainder)


DEFAULT_PYRAMID_WIDTHS = [32, 64, 128, 256]


@constraint("dims_pyramid")
def dims_pyramid(
    trial: optuna.Trial,
    chosen: MutableMapping[str, Any],
    *,
    key: str = "model.head_hidden_dims",
    min_depth: int = 1,
    max_depth: int = 3,
    widths: list[int] | None = None,
    floor: int = 8,
    taper: float = 0.5,
) -> None:
    """Draw a list of per-layer widths into `key`: `depth` layers scaled by `taper` from a base width.

    Every list-valued width argument in the models - `head_hidden_dims`, `static_hidden_dims`,
    `cnn_hidden_dims` - needs this, because no single Optuna distribution produces a list: a
    categorical's choices are stored in the study database and so must be scalars. Depth and base
    width are the two dimensions that matter, and one `taper` covers the shapes worth searching
    without spending a parameter per layer:

        taper 0.5  halves      128 -> [128, 64, 32]   the classic head
        taper 1.0  constant    128 -> [128, 128, 128] what a shared-width conv stack was
        taper 2.0  widens       32 -> [32, 64, 128]   the conventional conv shape

    `floor` clamps the taper so a deep pyramid does not shrink to a handful of dimensions and undo
    the depth it is adding. Parameter names are derived from `key`, so several pyramids can coexist
    in one search space without colliding.
    """
    label = key.rsplit(".", 1)[-1]
    depth = trial.suggest_int(f"{label}_depth", int(min_depth), int(max_depth))
    if depth <= 0:
        # No width to draw: a zero-layer stack is a bare projection, and suggesting one anyway
        # would leave the sampler a dimension that changes nothing about the trial.
        chosen[key] = []
        return

    base = trial.suggest_categorical(
        f"{label}_width", [int(w) for w in (widths or DEFAULT_PYRAMID_WIDTHS)]
    )
    chosen[key] = [max(int(floor), int(base * float(taper) ** step)) for step in range(depth)]
