"""SHAP over both families.

The Lightning explainer cuts the model open at the rasterizer output and rebuilds the forward pass
from the pieces. Three things have to hold or the resulting plot is confidently wrong:

* ``forward_from_parts(explanation_parts(batch))`` must equal ``forward(batch)`` EXACTLY - otherwise
  the attribution describes a model that is not the one being scored, and a mis-sliced channel
  produces a plausible-looking beeswarm rather than an error;
* the grouped values must satisfy SHAP additivity;
* every incoming feature - static, categorical, each band of each modality, each auxiliary lab
  column - must get exactly one row.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from yg_eo_soilnet.explain import build_shap_results
from yg_eo_soilnet.explain.plots import shap_bar, shap_beeswarm, shap_block_bar
from yg_eo_soilnet.explain.result import ShapResult
from yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module import SoilCNNLightningModule

STATIC_NAMES = ["clay_pct", "sand_pct", "ph"]
S2_BANDS = ["S2_B02", "S2_B08"]
CLIM_BANDS = ["CLIM_precip"]
LAB_ROSTER = ["ph_lab", "clay_lab", "sand_lab"]
AUXILIARY = ["ph_lab", "clay_lab"]


def _model(*, categorical: bool = True, auxiliary: bool = True) -> SoilCNNLightningModule:
    model = SoilCNNLightningModule(
        static_dim=len(STATIC_NAMES),
        target_dim=1,
        target_names=["organic_matter_pct"],
        categorical_cardinalities=[4] if categorical else [],
        categorical_vocabularies=[["a", "b", "c", "unk"]] if categorical else [],
        categorical_feature_names=["texture"] if categorical else [],
        modality_dims={"s2": len(S2_BANDS), "clim": len(CLIM_BANDS)},
        temporal_enabled=True,
        grid_years=2,
        auxiliary_label_columns=AUXILIARY if auxiliary else None,
        auxiliary_available_names=LAB_ROSTER if auxiliary else None,
        static_hidden_dims=[6],
        head_hidden_dims=[6],
        cnn_hidden_dims=[4],
        modality_embed_dim=4,
        dropout=0.0,
    )
    model.attach_preprocessing_state(
        {
            "static_feature_names": STATIC_NAMES,
            "static_mean": [10.0, 20.0, 7.0],
            "static_scale": [2.0, 3.0, 0.5],
            "modality_column_names": {"s2": S2_BANDS, "clim": CLIM_BANDS},
            "sequence_mean": {"s2": [0.1, 0.2], "clim": [50.0]},
            "sequence_scale": {"s2": [0.05, 0.06], "clim": [10.0]},
            "label_mean": [7.0, 30.0, 40.0],
            "label_scale": [0.5, 5.0, 6.0],
        }
    )
    model.eval()
    return model


def _batch(n: int, *, categorical: bool = True, auxiliary: bool = True, seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    length = 5
    batch = {
        "x_static": torch.randn(n, len(STATIC_NAMES), generator=generator),
        "sequences": {
            "s2": torch.randn(n, length, len(S2_BANDS), generator=generator),
            "clim": torch.randn(n, length, len(CLIM_BANDS), generator=generator),
        },
        "sequence_mask": {
            "s2": torch.ones(n, length, dtype=torch.bool),
            "clim": torch.ones(n, length, dtype=torch.bool),
        },
        "sequence_time": {
            "s2": 2020.0 + torch.rand(n, length, generator=generator),
            "clim": 2020.0 + torch.rand(n, length, generator=generator),
        },
        "sequence_validity": {
            "s2": torch.ones(n, length, len(S2_BANDS), dtype=torch.bool),
            "clim": torch.ones(n, length, len(CLIM_BANDS), dtype=torch.bool),
        },
        "y": torch.randn(n, 1, generator=generator),
    }
    if categorical:
        batch["x_categorical"] = torch.randint(0, 4, (n, 1), generator=generator)
    if auxiliary:
        batch["x_labels"] = torch.randn(n, len(LAB_ROSTER), generator=generator)
        batch["x_label_validity"] = torch.ones(n, len(LAB_ROSTER))
    return batch


# --- the attribution seam ---------------------------------------------------


@pytest.mark.parametrize("categorical", [True, False])
@pytest.mark.parametrize("auxiliary", [True, False])
def test_forward_from_parts_reproduces_forward_exactly(categorical, auxiliary) -> None:
    model = _model(categorical=categorical, auxiliary=auxiliary)
    batch = _batch(6, categorical=categorical, auxiliary=auxiliary)

    with torch.no_grad():
        direct = model(batch)
        parts, _groups = model.explanation_parts(batch)
        rebuilt = model.forward_from_parts(parts)

    assert torch.equal(direct, rebuilt), (
        "the attribution seam drifted from forward(); a mis-sliced channel here would attribute "
        "importance to the wrong band without raising"
    )


def test_every_incoming_feature_gets_exactly_one_group() -> None:
    model = _model()
    _parts, groups = model.explanation_parts(_batch(4))
    names = [group["name"] for group in groups]

    for expected in STATIC_NAMES + ["texture"] + S2_BANDS + CLIM_BANDS + AUXILIARY:
        assert expected in names, f"{expected} has no SHAP row"

    assert len(names) == len(set(names)), f"duplicate feature rows: {names}"
    # The positional pair is a coordinate, not a measurement, so it gets one row per modality.
    assert "s2_month_positional" in names
    assert "clim_month_positional" in names


def test_group_columns_stay_inside_their_part() -> None:
    model = _model()
    parts, groups = model.explanation_parts(_batch(4))

    for group in groups:
        part = parts[group["part"]]
        assert max(group["columns"]) < part.shape[1]


def test_a_band_group_folds_in_its_own_validity_channel() -> None:
    """Value and validity describe the same band, so they belong to one bar."""
    model = _model()
    _parts, groups = model.explanation_parts(_batch(4))

    band = next(group for group in groups if group["name"] == "S2_B02")
    layout = model.rasterizers["s2"].channel_layout()

    assert band["columns"] == [layout["values"][0], layout["validity"][0]]


def test_positional_channels_are_not_folded_into_a_band() -> None:
    model = _model()
    _parts, groups = model.explanation_parts(_batch(4))

    positional = next(group for group in groups if group["name"] == "s2_month_positional")
    layout = model.rasterizers["s2"].channel_layout()

    assert positional["columns"] == layout["month_positional"]


def test_channel_layout_accounts_for_every_output_channel() -> None:
    model = _model()
    rasterizer = model.rasterizers["s2"]
    layout = rasterizer.channel_layout()

    covered = sorted(
        layout["values"] + layout["validity"] + layout["cell_observed"] + layout["month_positional"]
    )
    assert covered == list(range(rasterizer.output_channels))


def test_cell_mask_is_recoverable_from_the_grid() -> None:
    """forward_from_parts derives cell_mask from the grid's own cell_observed channel, which is
    what lets the explainer treat the forward pass as a pure function of the parts."""
    model = _model()
    batch = _batch(5)

    with torch.no_grad():
        grids = model._rasterize(batch, device=torch.device("cpu"), dtype=torch.float32)

    grid, cell_mask = grids["s2"]
    observed_index = model.rasterizers["s2"].channel_layout()["cell_observed"][0]

    assert torch.equal(grid[:, observed_index] > 0.5, cell_mask)


# --- the Lightning explainer ------------------------------------------------


def _shap_result(model, **config_overrides) -> ShapResult:
    settings = {"RANDOM_SEED": 42, "EXPLAIN_MAX_SAMPLES": 8, "EXPLAIN_BACKGROUND_SAMPLES": 16}
    settings.update(config_overrides)
    datamodule = SimpleNamespace(
        test_dataloader=lambda: [_batch(16, seed=1), _batch(16, seed=2)],
        setup=lambda stage: None,
    )
    results = build_shap_results(
        config=SimpleNamespace(**settings),
        backend="lightning",
        model=model,
        bundle=SimpleNamespace(datamodule=datamodule),
        target="organic_matter_pct",
    )
    assert len(results) == 1
    return results[0]


def test_lightning_shap_covers_static_temporal_and_auxiliary_in_one_feature_space() -> None:
    result = _shap_result(_model())

    for expected in STATIC_NAMES + ["texture"] + S2_BANDS + CLIM_BANDS + AUXILIARY:
        assert expected in result.feature_names

    assert result.values.shape == (result.n_samples, result.n_features)
    assert set(result.blocks) == {"static", "categorical", "s2", "clim", "auxiliary"}


def test_lightning_shap_values_are_additive() -> None:
    """sum of SHAP over features ~= f(x) - E[f(background)], the defining SHAP property."""
    model = _model()
    result = _shap_result(model)

    batches = [_batch(16, seed=1), _batch(16, seed=2)]
    with torch.no_grad():
        merged = {
            key: (
                torch.cat([batches[0][key], batches[1][key]], dim=0)
                if torch.is_tensor(batches[0][key])
                else {
                    name: torch.cat([batches[0][key][name], batches[1][key][name]], dim=0)
                    for name in batches[0][key]
                }
            )
            for key in batches[0]
        }
        parts, _groups = model.explanation_parts(merged)
        predictions = model.forward_from_parts(parts).squeeze(-1).numpy()

    background_mean = predictions[:16].mean()
    explained = predictions[16 : 16 + result.n_samples]

    attributed = result.values.sum(axis=1)
    expected = explained - background_mean

    assert np.corrcoef(attributed, expected)[0, 1] > 0.9
    assert np.abs(attributed - expected).mean() < max(0.25 * expected.std(), 1e-3)


def test_shap_values_are_in_standardized_space_not_target_units() -> None:
    """Attribution is taken on forward(), which stops before inverse_transform_targets."""
    assert _shap_result(_model()).output_space == "standardized_log1p"


def test_beeswarm_colours_are_de_standardized_into_real_units() -> None:
    result = _shap_result(_model())
    clay = result.data[:, result.feature_names.index("clay_pct")]

    # static_mean 10.0, static_scale 2.0 against standard-normal inputs.
    assert 2.0 < float(np.nanmean(clay)) < 18.0


def test_positional_rows_have_no_colour_value() -> None:
    result = _shap_result(_model())
    positional = result.data[:, result.feature_names.index("s2_month_positional")]

    assert np.isnan(positional).all()


def test_temporal_colours_use_the_bands_own_scale() -> None:
    result = _shap_result(_model())
    precip = result.data[:, result.feature_names.index("CLIM_precip")]

    # sequence_mean 50.0, sequence_scale 10.0.
    assert 10.0 < float(np.nanmean(precip)) < 90.0


def test_a_model_without_the_seam_is_refused_by_name() -> None:
    with pytest.raises(TypeError, match="attribution seam"):
        build_shap_results(
            config=SimpleNamespace(),
            backend="lightning",
            model=SimpleNamespace(training=False, eval=lambda: None),
            bundle=SimpleNamespace(datamodule=SimpleNamespace()),
            target="om",
        )


def test_unknown_backend_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown explain backend"):
        build_shap_results(config=SimpleNamespace(), backend="tensorflow")


def test_explaining_leaves_the_model_in_the_mode_it_was_found_in() -> None:
    model = _model()
    model.train()
    _shap_result(model)
    assert model.training


# --- ShapResult -------------------------------------------------------------


def _toy_result() -> ShapResult:
    return ShapResult(
        values=np.array([[1.0, -2.0, 0.5], [3.0, 1.0, -0.5]]),
        data=np.array([[10.0, 20.0, np.nan], [11.0, 21.0, np.nan]]),
        feature_names=["a", "b", "c"],
        blocks=["static", "s2", "s2"],
        target_name="om",
        output_space="original_units",
    )


def test_mean_abs_and_ranking() -> None:
    result = _toy_result()

    assert np.allclose(result.mean_abs(), [2.0, 1.5, 0.5])
    assert result.ranking() == [0, 1, 2]


def test_block_rollup_sums_within_a_block_before_taking_the_absolute_value() -> None:
    """Two features in one block that cancel on a sample contribute nothing jointly."""
    result = _toy_result()
    blocks = result.block_mean_abs()

    assert np.isclose(blocks["static"], 2.0)
    # s2 per sample: (-2.0 + 0.5) = -1.5 and (1.0 + -0.5) = 0.5 -> mean(|.|) = 1.0
    assert np.isclose(blocks["s2"], 1.0)


def test_to_frame_is_long_and_complete() -> None:
    frame = _toy_result().to_frame()

    assert len(frame) == 2 * 3
    assert set(frame.columns) == {
        "sample",
        "feature",
        "block",
        "shap_value",
        "feature_value",
        "target",
        "output_space",
    }


def test_summary_reports_provenance_and_ranking() -> None:
    summary = _toy_result().summary(top_n=2)

    assert summary["output_space"] == "original_units"
    assert [entry["feature"] for entry in summary["ranking"]] == ["a", "b"]
    assert set(summary["blocks"]) == {"static", "s2"}


def test_shape_mismatches_are_refused() -> None:
    with pytest.raises(ValueError, match="same shape"):
        ShapResult(
            values=np.zeros((2, 3)),
            data=np.zeros((2, 2)),
            feature_names=["a", "b", "c"],
            target_name="om",
            output_space="original_units",
        )

    with pytest.raises(ValueError, match="feature name"):
        ShapResult(
            values=np.zeros((2, 3)),
            data=np.zeros((2, 3)),
            feature_names=["a", "b"],
            target_name="om",
            output_space="original_units",
        )


# --- plots ------------------------------------------------------------------


def test_plots_return_figures_rather_than_drawing_and_returning_none() -> None:
    result = _shap_result(_model())

    for figure in (shap_beeswarm(result, 5), shap_bar(result, 5), shap_block_bar(result)):
        assert figure is not None
        assert hasattr(figure, "savefig")


def test_plots_return_a_message_figure_when_there_is_nothing_to_draw() -> None:
    empty = ShapResult(
        values=np.empty((0, 0)),
        data=np.empty((0, 0)),
        feature_names=[],
        target_name="om",
        output_space="original_units",
    )

    assert shap_beeswarm(empty) is not None
    assert shap_bar(empty) is not None
    assert shap_block_bar(empty) is not None
