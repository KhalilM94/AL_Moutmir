"""The shared regression head, which replaced four near-identical copies.

Each model used to build its own, and two of them took a `head_num_layers` + halving `head_hidden_dim`
+ `head_min_hidden_dim` triple rather than a width list - so the built network could not be read off
the config, and every registry comment describing one had drifted from what it actually built.
"""

from __future__ import annotations

import pytest
from torch import nn

from yg_eo_soilnet.models.lightningmodules.mlp import build_mlp_stack


def _widths(head: nn.Module) -> list[int]:
    return [layer.out_features for layer in head.modules() if isinstance(layer, nn.Linear)]


def test_an_empty_width_list_is_a_single_linear_readout() -> None:
    head = build_mlp_stack(16, [], 1, dropout=0.1)

    assert isinstance(head, nn.Linear)
    assert (head.in_features, head.out_features) == (16, 1)


def test_widths_are_built_exactly_as_listed() -> None:
    head = build_mlp_stack(16, [64, 32, 32], 2, dropout=0.1)

    assert _widths(head) == [64, 32, 32, 2]


def test_the_block_feeding_the_readout_carries_no_norm_and_no_dropout() -> None:
    """Normalizing there leaves the final Linear only a direction; magnitude reaches the tails."""
    modules = list(build_mlp_stack(16, [64, 32], 1, dropout=0.1))
    last_hidden_linear = max(i for i, m in enumerate(modules[:-1]) if isinstance(m, nn.Linear))
    tail = modules[last_hidden_linear + 1 : -1]

    assert not any(isinstance(m, (nn.LayerNorm, nn.Dropout)) for m in tail)
    assert any(isinstance(m, nn.LayerNorm) for m in modules[:last_hidden_linear])
    assert any(isinstance(m, nn.Dropout) for m in modules[:last_hidden_linear])


def test_norm_final_restores_the_norm_on_that_block_but_not_the_dropout() -> None:
    modules = list(build_mlp_stack(16, [64, 32], 1, dropout=0.1, norm_final=True))
    last_hidden_linear = max(i for i, m in enumerate(modules[:-1]) if isinstance(m, nn.Linear))
    tail = modules[last_hidden_linear + 1 : -1]

    assert any(isinstance(m, nn.LayerNorm) for m in tail)
    assert not any(isinstance(m, nn.Dropout) for m in tail)


def test_use_layer_norm_false_removes_every_norm() -> None:
    head = build_mlp_stack(16, [64, 32], 1, dropout=0.1, use_layer_norm=False, norm_final=True)

    assert not any(isinstance(m, nn.LayerNorm) for m in head.modules())


@pytest.mark.parametrize("activation,expected", [("relu", nn.ReLU), ("gelu", nn.GELU), ("GELU", nn.GELU)])
def test_the_activation_is_selectable(activation: str, expected: type) -> None:
    head = build_mlp_stack(16, [64], 1, dropout=0.1, activation=activation)

    assert any(isinstance(m, expected) for m in head.modules())


def test_an_unknown_activation_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="Unknown activation 'swish'"):
        build_mlp_stack(16, [64], 1, dropout=0.1, activation="swish")


def test_the_cnn_head_layout_is_reproduced() -> None:
    """LayerNorm + GELU per block, dropout on every block but the last, then the readout."""
    head = build_mlp_stack(10, [64], 1, dropout=0.2, activation="gelu", norm_final=False)

    assert [type(module) for module in head] == [nn.Linear, nn.GELU, nn.Linear]


def test_the_sequence_and_graph_head_layout_is_reproduced() -> None:
    head = build_mlp_stack(10, [64, 32], 1, dropout=0.1)

    assert [type(module) for module in head] == [
        nn.Linear, nn.LayerNorm, nn.ReLU, nn.Dropout, nn.Linear, nn.ReLU, nn.Linear
    ]


def test_widths_given_as_strings_or_floats_are_coerced() -> None:
    """Values arrive from YAML and from Optuna, which is not always as much of an int as it looks."""
    assert _widths(build_mlp_stack(16, ["64", 32.0], 1, dropout=0.1)) == [64, 32, 1]


# --- build_mlp_stack beyond a head -------------------------------------------


def test_no_output_dim_leaves_the_stack_at_its_last_width() -> None:
    """A branch feeding a fusion has no readout to project to."""
    stack = build_mlp_stack(16, [64, 32], None, dropout=0.1)

    assert _widths(stack) == [64, 32]
    assert isinstance(list(stack)[-1], nn.ReLU)


def test_an_empty_stack_with_no_output_dim_is_an_identity() -> None:
    assert isinstance(build_mlp_stack(16, [], None, dropout=0.1), nn.Identity)


def test_dropout_final_makes_every_block_a_full_block() -> None:
    modules = list(
        build_mlp_stack(16, [64, 32], None, dropout=0.1, norm_final=True, dropout_final=True)
    )

    assert [type(m) for m in modules] == [
        nn.Linear, nn.LayerNorm, nn.ReLU, nn.Dropout,
        nn.Linear, nn.LayerNorm, nn.ReLU, nn.Dropout,
    ]


def test_the_static_encoder_layout_is_reproduced() -> None:
    """What TabularStaticEncoder and the graph module used to hand-build, projection included."""
    modules = list(
        build_mlp_stack(16, [64], 32, dropout=0.1, norm_final=True, dropout_final=True)
    )

    assert [type(m) for m in modules] == [nn.Linear, nn.LayerNorm, nn.ReLU, nn.Dropout, nn.Linear]
