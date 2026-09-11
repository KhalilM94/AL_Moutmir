"""Residual calendar-grid CNN with attention fusion: the fusion, and that nothing else moved.

Everything after the fusion - the base offset, the base and auxiliary blocks, the variance head - is
inherited from SoilResidualCNNLightningModule and pinned by tests/test_residual_cnn.py. This file
pins what changed: the attention fusion itself, the two ways static covariates become tokens, and
the contracts the swap could silently break - the head's width, the attribution seam's exact
equality, and the checkpoint round trip of settings saved by a third save_hyperparameters() call.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from yg_eo_soilnet.datamodules.sequence.sequence_datamodule import SoilSequenceDataModule
from yg_eo_soilnet.models.lightningmodules.soil_residual_attention_cnn_lightning_module import (
    SoilResidualAttentionCNNLightningModule,
)
from yg_eo_soilnet.models.lightningmodules.temporal_cnn_encoders import AttentionFusion, ConcatGatedFusion

from tests.test_cnn_pipeline import LABEL_NAMES, _batch, _bundle
from tests.test_residual_cnn import LABEL_MEAN, LABEL_SCALE, _expected_base, _zero_head

STATIC_MODES = ["summary", "per_feature"]
READOUTS = ["cls", "mean", "flatten"]
CPU = torch.device("cpu")


def _module(**kwargs) -> SoilResidualAttentionCNNLightningModule:
    defaults = dict(
        static_dim=5,
        target_dim=1,
        modality_dims={"m": 3},
        temporal_encoder="dilated_tempcnn",
        grid_years=3,
        target_names=["target_a"],
        auxiliary_available_names=LABEL_NAMES,
        auxiliary_label_mean=LABEL_MEAN,
        auxiliary_label_scale=LABEL_SCALE,
        residual_base_columns={"target_a": "lab_b"},
        cnn_hidden_dims=[8],
        modality_embed_dim=8,
        static_hidden_dims=[8],
        head_hidden_dims=[8],
        attention_d_model=16,
        attention_nhead=4,
        attention_num_layers=1,
    )
    defaults.update(kwargs)
    torch.manual_seed(0)
    return SoilResidualAttentionCNNLightningModule(**defaults).eval()


def _first_linear(block: nn.Module) -> nn.Linear:
    return next(child for child in block.modules() if isinstance(child, nn.Linear))


# --- AttentionFusion on its own ----------------------------------------------


def _fusion(**kwargs) -> AttentionFusion:
    defaults = dict(
        static_dims=[5], temporal_dims=[4, 3], coordinate_dim=2, d_model=8, nhead=2, num_layers=1, dropout=0.0
    )
    defaults.update(kwargs)
    torch.manual_seed(0)
    return AttentionFusion(**defaults).eval()


def _inputs(batch_size=3, static=5, temporal=7, coords=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn(batch_size, static, generator=generator),
        torch.randn(batch_size, temporal, generator=generator),
        torch.randn(batch_size, coords, generator=generator) if coords else None,
    ]


@pytest.mark.parametrize("readout, width", [("cls", 8), ("mean", 8), ("flatten", 4 * 8)])
def test_the_output_width_follows_the_readout(readout, width) -> None:
    fusion = _fusion(readout=readout)
    with torch.no_grad():
        out = fusion(*_inputs())

    assert fusion.num_tokens == 4  # static, two modalities, location
    assert fusion.output_dim == width
    assert out.shape == (3, width)


def test_per_feature_static_dims_give_one_token_per_column() -> None:
    """[1]*5 is FT-Transformer's tokenizer: a width-1 chunk through a Linear is x*w + b."""
    fusion = _fusion(static_dims=[1] * 5)

    assert fusion.num_tokens == 5 + 2 + 1
    assert [tokenizer.in_features for tokenizer in fusion.tokenizers] == [1] * 5 + [4, 3, 2]
    with torch.no_grad():
        assert torch.isfinite(fusion(*_inputs())).all()


def test_no_coordinate_token_without_coordinates() -> None:
    fusion = _fusion(coordinate_dim=0)
    static, temporal, _ = _inputs(coords=0)

    assert [tokenizer.in_features for tokenizer in fusion.tokenizers] == [5, 4, 3]
    with torch.no_grad():
        assert fusion(static, temporal, None).shape == (3, 8)


def test_an_empty_static_dims_reads_no_static_tokens_whatever_arrives() -> None:
    """per_feature mode relies on this when a dataset has no static features at all."""
    fusion = _fusion(static_dims=[])
    static, temporal, coords = _inputs()

    with torch.no_grad():
        assert torch.equal(fusion(static, temporal, coords), fusion(torch.zeros(3, 11), temporal, coords))


@pytest.mark.parametrize(
    "position, columns",
    [(0, slice(0, 5)), (1, slice(0, 4)), (1, slice(4, 7)), (2, slice(0, 2))],
    ids=["static", "first_modality", "second_modality", "location"],
)
def test_every_chunk_reaches_the_cls_readout(position, columns) -> None:
    fusion = _fusion(readout="cls")
    inputs = _inputs()

    with torch.no_grad():
        before = fusion(*inputs)
        inputs[position] = inputs[position].clone()
        inputs[position][:, columns] += 1.0
        after = fusion(*inputs)

    assert not torch.allclose(before, after)


def test_branches_are_rebuilt_from_each_other_before_the_head() -> None:
    """What attention adds over the gate: a modality token's OUTPUT depends on the static input.

    Under flatten every token keeps its own slot, so this reads token 1 - the first modality - alone.
    """
    fusion = _fusion(readout="flatten")
    static, temporal, coords = _inputs()

    with torch.no_grad():
        before = fusion(static, temporal, coords)[:, 8:16]
        after = fusion(static + 1.0, temporal, coords)[:, 8:16]

    assert not torch.allclose(before, after)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(d_model=10, nhead=4), "divisible by nhead"),
        (dict(readout="max"), "readout must be one of"),
        (dict(static_dims=[], temporal_dims=[], coordinate_dim=0), "at least one input token"),
        (dict(static_dims=[0]), "must all be positive"),
        (dict(ff_multiplier=0), "ff_multiplier"),
    ],
)
def test_bad_settings_are_refused_at_construction(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        _fusion(**kwargs)


def test_a_chunk_width_mismatch_is_refused_rather_than_resliced() -> None:
    fusion = _fusion()
    static, _, coords = _inputs()

    with pytest.raises(ValueError, match="expected 7 temporal"):
        fusion(static, torch.randn(3, 6), coords)


# --- the module: what the swap changed -----------------------------------------


@pytest.mark.parametrize("static_tokens", STATIC_MODES)
def test_the_gate_is_replaced_and_the_head_is_sized_to_the_new_fusion(static_tokens) -> None:
    module = _module(attention_static_tokens=static_tokens, auxiliary_label_columns=["lab_a"])

    assert isinstance(module.fusion, AttentionFusion)
    assert not any(isinstance(child, ConcatGatedFusion) for child in module.modules())
    # fused vector + auxiliary block (value + flag) + base block (value + flag)
    assert _first_linear(module.output_head).in_features == module.fusion.output_dim + 2 + 2


def test_summary_mode_reads_the_static_branch_as_one_token() -> None:
    module = _module(attention_static_tokens="summary")

    assert module.fusion.static_dims == [8]  # static_hidden_dims[-1]
    assert module.fusion.num_tokens == 2  # static + the one modality
    assert any(isinstance(child, nn.Linear) for child in module.static_encoder.modules())


def test_per_feature_mode_gives_every_column_a_token_and_drops_the_static_mlp() -> None:
    module = _module(attention_static_tokens="per_feature")

    assert module.fusion.static_dims == [1] * 5
    assert module.fusion.num_tokens == 5 + 1
    # Dead weight in this mode; with it gone the encoder returns the raw per-column block.
    assert not any(isinstance(child, nn.Linear) for child in module.static_encoder.modules())
    assert module.static_encoder.output_dim == 5


def test_per_feature_mode_gives_each_categorical_embedding_its_own_token() -> None:
    module = _module(
        attention_static_tokens="per_feature",
        categorical_cardinalities=[4],
        categorical_vocabularies=[["a", "b", "c"]],
        categorical_feature_names=["texture"],
    )

    assert len(module.static_encoder.embedding_dims) == 1
    assert module.fusion.static_dims == [1] * 5 + module.static_encoder.embedding_dims


@pytest.mark.parametrize("static_tokens", STATIC_MODES)
def test_a_zeroed_head_still_reproduces_the_base(static_tokens) -> None:
    """The residual contract survives the swap: prediction = base + head, whatever the fusion."""
    module = _module(attention_static_tokens=static_tokens)
    _zero_head(module)
    batch = _batch(labels=3)

    with torch.no_grad():
        assert torch.allclose(module(batch).reshape(-1).double(), _expected_base(batch), atol=1e-5)


def test_per_feature_mode_lets_a_single_covariate_move_the_prediction() -> None:
    module = _module(attention_static_tokens="per_feature")
    batch = _batch(labels=3)

    with torch.no_grad():
        before = module(batch)
        batch["x_static"] = batch["x_static"].clone()
        batch["x_static"][:, 3] += 1.0
        after = module(batch)

    assert not torch.allclose(before, after)


def _full_module(**kwargs) -> SoilResidualAttentionCNNLightningModule:
    """Every optional branch on, so every part of the attribution seam is exercised."""
    return _module(
        coord_dim=2,
        categorical_cardinalities=[4],
        categorical_vocabularies=[["a", "b", "c"]],
        categorical_feature_names=["texture"],
        auxiliary_label_columns=["lab_a"],
        residual_base_hidden_dims=[8],
        **kwargs,
    )


def _full_batch() -> dict:
    batch = _batch(labels=3)
    generator = torch.Generator().manual_seed(1)
    batch["x_categorical"] = torch.randint(0, 4, (4, 1), generator=generator)
    batch["x_coords"] = torch.rand(4, 2, generator=generator) * 2 - 1
    return batch


@pytest.mark.parametrize("static_tokens", STATIC_MODES)
@pytest.mark.parametrize("readout", READOUTS)
def test_forward_from_parts_reproduces_forward_exactly(static_tokens, readout) -> None:
    """The equality every explainer depends on, in both static modes and every readout."""
    module = _full_module(attention_static_tokens=static_tokens, attention_readout=readout)
    batch = _full_batch()

    with torch.no_grad():
        parts, groups = module.explanation_parts(batch)
        assert torch.allclose(module.forward_from_parts(parts), module(batch), atol=1e-6)

    # Static covariates are attributed per column in both modes; only the tokenization differs.
    assert len([group for group in groups if group["kind"] == "static"]) == 5
    assert [group["name"] for group in groups if group["kind"] == "residual_base"] == ["lab_b"]


def test_the_attention_settings_survive_a_weights_only_checkpoint_round_trip(tmp_path: Path) -> None:
    """Saved by a THIRD save_hyperparameters() call; losing them would rebuild the default fusion."""
    module = _module(attention_static_tokens="per_feature", attention_readout="mean", attention_num_layers=2)
    hparams = dict(module.hparams)
    assert hparams["attention_static_tokens"] == "per_feature"
    assert hparams["attention_readout"] == "mean"
    assert hparams["attention_num_layers"] == 2
    # The two earlier calls still merged in, not replaced.
    assert hparams["residual_base_columns"] == {"target_a": "lab_b"}
    assert hparams["static_dim"] == 5

    path = tmp_path / "attention.ckpt"
    torch.save(
        {"state_dict": module.state_dict(), "hyper_parameters": hparams, "pytorch-lightning_version": "2.0.0"},
        path,
    )
    torch.load(path, weights_only=True)  # the constraint that forbids numpy in hyper_parameters

    restored = SoilResidualAttentionCNNLightningModule.load_from_checkpoint(path, map_location="cpu").eval()
    batch = _batch(labels=3)
    with torch.no_grad():
        assert torch.allclose(restored(batch), module(batch), atol=1e-6)


def test_the_offset_lands_on_the_mean_half_only_on_a_variance_head() -> None:
    module = _module(predict_variance=True)
    batch = _batch(labels=3)

    with torch.no_grad():
        fused = module._fuse(batch, device=CPU, dtype=torch.float32)
        block = module._residual_base(batch, device=CPU, dtype=torch.float32)
        head = module.output_head(torch.cat([fused, module.base_encoder(block)], dim=-1))
        out = module(batch)

    assert out.shape[-1] == 2
    assert torch.allclose(out[:, 1:], head[:, 1:].clamp(-10.0, 10.0), atol=1e-6)
    assert torch.allclose(out[:, 0] - head[:, 0], _expected_base(batch).float(), atol=1e-5)


@pytest.mark.parametrize("encoder", ["dilated_tempcnn", "annual_grid2d"])
@pytest.mark.parametrize("static_tokens", STATIC_MODES)
def test_every_fusion_parameter_receives_a_gradient(encoder, static_tokens) -> None:
    module = _module(temporal_encoder=encoder, attention_static_tokens=static_tokens, attention_num_layers=2)
    module.train()
    batch = _batch(labels=3)

    (module(batch) - batch["y"]).pow(2).mean().backward()

    for name, parameter in module.fusion.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_without_temporal_data_the_fusion_reads_static_tokens_only() -> None:
    module = _module(temporal_enabled=False)

    assert module.fusion.temporal_dims == []
    with torch.no_grad():
        assert torch.isfinite(module(_batch(labels=3))).all()


def test_modalities_of_different_widths_each_get_their_own_projection() -> None:
    module = _module(modality_dims={"m": 3, "n": 2}, modality_embed_dim={"m": 8, "n": 4})
    batch = _batch(labels=3)
    for key in ("sequences", "sequence_validity"):
        batch[key]["n"] = batch[key]["m"][..., :2]
    for key in ("sequence_mask", "sequence_time"):
        batch[key]["n"] = batch[key]["m"]

    assert module.fusion.temporal_dims == [8, 4]
    with torch.no_grad():
        assert torch.isfinite(module(batch)).all()


def test_the_serving_signature_still_asks_for_the_base_column() -> None:
    from yg_eo_soilnet.serving.lightning_pyfunc import required_label_columns

    assert required_label_columns(_module()) == ["lab_b"]


# --- data path ---------------------------------------------------------------


def _datamodule(tmp_path: Path, logger) -> SoilSequenceDataModule:
    dates = [f"20{year:02d}-{month:02d}-01" for year in range(19, 23) for month in range(1, 13)]
    bundle = _bundle(tmp_path, logger, {point: dates[: 20 + 4 * point] for point in range(1, 9)})
    datamodule = SoilSequenceDataModule(bundle, batch_size=2, val_size=0.4, test_size=0.25, seed=5)
    datamodule.setup("fit")
    return datamodule


@pytest.mark.parametrize("static_tokens", STATIC_MODES)
def test_builder_to_attention_module_end_to_end(tmp_path: Path, logger, static_tokens) -> None:
    """A real bundle, a real datamodule, two epochs, and a finite prediction out the other side."""
    from lightning.pytorch import Trainer

    datamodule = _datamodule(tmp_path, logger)
    module = SoilResidualAttentionCNNLightningModule(
        static_dim=datamodule.static_dim,
        target_dim=datamodule.target_dim,
        target_names=datamodule.target_names,
        modality_dims=datamodule.modality_dims,
        grid_years=datamodule.grid_years,
        temporal_encoder="annual_grid2d",
        cnn_norm="group",  # the fixture's batches are too small for BatchNorm
        auxiliary_available_names=datamodule.label_feature_names,
        auxiliary_label_mean=datamodule.label_mean_,
        auxiliary_label_scale=datamodule.label_scale_,
        residual_base_columns={"target_a": "lab_dense"},
        target_mean=datamodule.target_mean_,
        target_scale=datamodule.target_scale_,
        target_transform=datamodule.target_transform,
        attention_static_tokens=static_tokens,
        attention_d_model=16,
        attention_nhead=4,
    )

    trainer = Trainer(
        max_epochs=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, datamodule=datamodule)

    assert torch.isfinite(torch.as_tensor(trainer.callback_metrics["train_loss"]))
    predicted = torch.cat(trainer.predict(module, datamodule=datamodule)).reshape(-1)
    assert torch.isfinite(predicted).all()


def test_the_shipped_registry_entry_builds_through_the_factory(tmp_path: Path, logger) -> None:
    """Every `auto` in the entry must be a keyword the module declares - see the residual twin."""
    from config import load_lightning_registry
    from yg_eo_soilnet.models.config_fatories.lightning_config_factory import LightningConfigFactory

    registry = load_lightning_registry("configs/lightning/models/defaults.yml")
    spec = registry["soil_residual_attention_cnn"]
    # The shipped entry names the real dataset's columns; the fixture carries its own.
    spec["init_args"]["residual_base_columns"] = {"target_a": "lab_dense"}

    datamodule = _datamodule(tmp_path, logger)
    factory = LightningConfigFactory(registry, SimpleNamespace(TARGET_COLUMNS=["target_a"]))
    module = factory._build_model(spec, datamodule)

    assert isinstance(module, SoilResidualAttentionCNNLightningModule)
    assert isinstance(module.fusion, AttentionFusion)
    assert module.hparams["attention_static_tokens"] == spec["init_args"]["attention_static_tokens"]
    assert module.serving_label_columns == ["lab_dense"]
