"""The MLP stack every branch in every model is built from.

The output head, the static encoder and any future block share one shape - `Linear` per width, each
optionally followed by a norm, an activation and dropout - so it lives here once. Callers differ only
in how the block feeding whatever comes next is treated, and in whether a final projection is
appended, both of which are arguments.

Widths are always an explicit list. An earlier generation took a layer *count* plus one width and
halved it internally, which meant the built network could not be read off the config: every registry
comment describing such a block had drifted from what it actually built. A halving pyramid is still a
sensible default shape, so the hyperparameter search draws one - see ``hpo/constraints.py`` - but it
draws it into an explicit list.
"""

from __future__ import annotations

from typing import Optional, Sequence

from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}


def build_mlp_stack(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: Optional[int] = None,
    *,
    dropout: float,
    use_layer_norm: bool = True,
    activation: str = "relu",
    norm_final: bool = False,
    dropout_final: bool = False,
) -> nn.Module:
    """``input_dim -> hidden_dims -> output_dim``, one Linear/norm/activation/dropout block per width.

    ``output_dim=None`` omits the final projection, leaving a stack that is ``hidden_dims[-1]`` wide.
    An empty ``hidden_dims`` is then either a single ``Linear(input_dim, output_dim)`` or, with no
    output_dim, an ``Identity``.

    ``norm_final`` and ``dropout_final`` control the block feeding whatever comes next. A **head**
    leaves both off: normalizing there forces the penultimate vector to unit variance, leaving the
    final Linear only its direction - and magnitude is what a regressor needs to reach the tails -
    while dropout on that same vector is minimized under MSE by shrinking the readout toward its
    bias, i.e. toward the target mean. A block feeding a *fusion* rather than a readout turns both on,
    because there is no readout for either to degrade.
    """
    try:
        activation_cls = ACTIVATIONS[str(activation).lower()]
    except KeyError:
        raise ValueError(
            f"Unknown activation {activation!r}; expected one of: {', '.join(sorted(ACTIVATIONS))}."
        ) from None

    hidden_dims = [int(width) for width in hidden_dims]
    if not hidden_dims:
        return nn.Identity() if output_dim is None else nn.Linear(input_dim, int(output_dim))

    layers: list[nn.Module] = []
    dim = input_dim
    for index, width in enumerate(hidden_dims):
        layers.append(nn.Linear(dim, width))
        is_last_block = index == len(hidden_dims) - 1
        if use_layer_norm and (not is_last_block or norm_final):
            layers.append(nn.LayerNorm(width))
        layers.append(activation_cls())
        if not is_last_block or dropout_final:
            layers.append(nn.Dropout(dropout))
        dim = width
    if output_dim is not None:
        layers.append(nn.Linear(dim, int(output_dim)))
    return nn.Sequential(*layers)
