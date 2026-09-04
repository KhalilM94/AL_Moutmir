"""SHAP for the Lightning path, by expected gradients over the model's own attribution seam.

The CNN takes a dict batch of ragged, date-stamped observations, so there is no flat feature matrix
to hand a masking explainer. What there is, is a clean cut: ``CalendarGridRasterizer`` scatters the
ragged sequences onto a ``(years x months)`` grid, and everything downstream of that grid -
convolutions, pooling, the fusion gate, the head - is differentiable. So the grids become explainer
inputs, and ``shap.GradientExplainer`` attributes straight through the CNNs to individual channels.

Cutting there rather than at the encoder outputs is what makes the beeswarm useful. Encoder outputs
would give one row per modality, so ``S2`` would appear as a single opaque bar next to sixty
individually named static features - and it would out-rank all of them for structural reasons,
because it aggregates ten bands' worth of contribution. Cutting at the grid gives every band its own
row, so static features, categorical features, temporal bands and auxiliary lab columns all sit in
one flat, comparable feature space.

Attribution is summed within each group (over embedding dimensions, over a band's value and validity
channels, and over the year and month axes). Summing is the correct reduction: SHAP values are
additive, so a group's contribution is the sum of its parts', and the additivity check in
tests/test_explain.py is what pins that.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yg_eo_soilnet.explain.result import STANDARDIZED_LOG1P, ShapResult

# Torch is imported lazily throughout: this module is imported by explain/__init__, which the logger
# reaches only when EXPLAIN_ENABLED is true, but keeping the heavy imports inside functions means an
# import of the package itself stays cheap.


def _collect_batches(datamodule, limit: int) -> list[dict]:
    """Test batches until ``limit`` samples are gathered, or the loader is exhausted."""
    loader_factory = getattr(datamodule, "test_dataloader", None)
    if loader_factory is None:
        raise TypeError("The datamodule has no test_dataloader(); nothing to explain.")

    try:
        datamodule.setup("test")
    except Exception:
        # Already set up, or this datamodule does not need a test stage. Either way the loader
        # below is the real check.
        pass

    batches: list[dict] = []
    gathered = 0
    for batch in loader_factory():
        batches.append(batch)
        gathered += int(np.asarray(batch["x_static"]).shape[0])
        if gathered >= limit:
            break
    return batches


def _concatenate_parts(per_batch: list[list], torch) -> list:
    """Stack each part across batches, requiring the trailing dimensions to agree.

    They can disagree only when ``grid_years`` was left unset, in which case the rasterizer sizes the
    grid from each batch's own longest history. Falling back to the first batch is better than
    padding: a padded grid would carry fabricated empty years that the attribution would then spread
    importance across.
    """
    if len(per_batch) == 1:
        return per_batch[0]

    reference = per_batch[0]
    for parts in per_batch[1:]:
        if any(a.shape[1:] != b.shape[1:] for a, b in zip(reference, parts)):
            return reference

    return [torch.cat([parts[index] for parts in per_batch], dim=0) for index in range(len(reference))]


def _group_values(part_values: list[np.ndarray], groups: list[dict]) -> np.ndarray:
    """Fold raw per-element SHAP values into one column per feature group.

    Each array in ``part_values`` is ``(n_samples, *part_dims)``. A group names one part and a set of
    column indices along that part's first non-sample axis; everything after it (the year and month
    axes of a grid) is summed out entirely.
    """
    columns = []
    for group in groups:
        array = part_values[group["part"]]
        selected = array[:, group["columns"], ...]
        summed = selected.reshape(selected.shape[0], -1).sum(axis=1)
        columns.append(summed)
    return np.stack(columns, axis=1) if columns else np.empty((0, 0))


def _destandardize(values: np.ndarray, mean: list[float], scale: list[float]) -> np.ndarray:
    """Undo the datamodule's train-only standardization so colours read in real units."""
    if len(mean) != values.shape[-1] or len(scale) != values.shape[-1]:
        return values
    return values * np.asarray(scale, dtype=np.float64) + np.asarray(mean, dtype=np.float64)


def _colour_values(model, parts, groups: list[dict], categorical_codes, state: dict, torch) -> np.ndarray:
    """One scalar per (sample, feature) for the beeswarm's colour axis.

    A group that spans several columns has no single value of its own, so each kind gets the scalar
    that actually means something for it: the de-standardized reading for a static feature, the
    integer code for a categorical one, the point's mean observed value for a temporal band, the raw
    lab value for an auxiliary column, and the coordinate in DEGREES for a spatial one. The month
    positional pair gets NaN, which shap renders grey - it is a coordinate, not a measurement.

    ``context`` is coloured exactly like ``static`` because that is exactly what it is: the group
    lives in the same part, standardized by the same statistics, and differs only in being named.
    """
    n_samples = int(parts[0].shape[0])
    colours = np.full((n_samples, len(groups)), np.nan, dtype=np.float64)

    static_values = _destandardize(
        parts[0].detach().cpu().numpy().astype(np.float64),
        state.get("static_mean") or [],
        state.get("static_scale") or [],
    )

    sequence_mean = state.get("sequence_mean") or {}
    sequence_scale = state.get("sequence_scale") or {}

    label_mean = np.asarray(state.get("label_mean") or [], dtype=np.float64)
    label_scale = np.asarray(state.get("label_scale") or [], dtype=np.float64)
    # For the residual base block, which is published in the target's space rather than the lab
    # standardizer's and so inverts with these instead.
    target_mean = np.asarray(state.get("target_mean") or [], dtype=np.float64)
    target_scale = np.asarray(state.get("target_scale") or [], dtype=np.float64)
    coord_min = np.asarray(state.get("coord_min") or [], dtype=np.float64)
    coord_max = np.asarray(state.get("coord_max") or [], dtype=np.float64)
    auxiliary_index = getattr(model, "auxiliary_index", None)
    auxiliary_index = None if auxiliary_index is None else auxiliary_index.detach().cpu().numpy()

    categorical_names = list(getattr(getattr(model, "static_encoder", None), "embeddings", None).feature_names) if (
        getattr(model, "has_static_features", False)
        and getattr(getattr(model, "static_encoder", None), "embeddings", None) is not None
    ) else []

    for column, group in enumerate(groups):
        kind = group["kind"]

        if kind in ("static", "context"):
            colours[:, column] = static_values[:, group["columns"][0]]

        elif kind == "spatial":
            # Back to degrees, undoing the train-bbox mapping the datamodule applied. Colouring by
            # the normalized value would work but reads as a meaningless -1..1; in degrees the
            # beeswarm's colour axis is literally "how far north", which is the thing worth seeing.
            normalized = parts[group["part"]].detach().cpu().numpy().astype(np.float64)
            position = group["columns"][0]
            value = normalized[:, position]
            if position < coord_min.size and position < coord_max.size:
                span = coord_max[position] - coord_min[position]
                value = (value + 1.0) / 2.0 * span + coord_min[position]
            colours[:, column] = value

        elif kind == "categorical" and categorical_codes is not None:
            if group["name"] in categorical_names:
                colours[:, column] = categorical_codes[:, categorical_names.index(group["name"])]

        elif kind == "temporal":
            modality = group["modality"]
            layout = model.rasterizers[modality].channel_layout()
            if group["columns"][0] not in layout["values"]:
                continue  # the month positional row: a coordinate, left as NaN
            band = layout["values"].index(group["columns"][0])
            grid = parts[group["part"]].detach().cpu().numpy().astype(np.float64)
            observed = grid[:, layout["cell_observed"][0]] > 0.5
            readings = grid[:, group["columns"][0]]
            counts = observed.reshape(observed.shape[0], -1).sum(axis=1)
            totals = (readings * observed).reshape(readings.shape[0], -1).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                means = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
            mean_values = sequence_mean.get(modality) or []
            scale_values = sequence_scale.get(modality) or []
            if band < len(mean_values) and band < len(scale_values):
                means = means * float(scale_values[band]) + float(mean_values[band])
            colours[:, column] = means

        elif kind == "auxiliary":
            selected = parts[group["part"]].detach().cpu().numpy().astype(np.float64)
            position = group["columns"][0]
            value = selected[:, position]
            if auxiliary_index is not None and position < len(auxiliary_index):
                roster_index = int(auxiliary_index[position])
                if roster_index < label_mean.size and roster_index < label_scale.size:
                    value = value * label_scale[roster_index] + label_mean[roster_index]
            colours[:, column] = value

        elif kind == "residual_base":
            # Already in the TARGET's space, not the lab standardizer's - that is the whole point of
            # the block - so it inverts with the target statistics, not label_mean/label_scale. The
            # result is the base prediction in original units, which is the number a reader
            # comparing the correction against what it corrected actually wants.
            block = parts[group["part"]].detach().cpu().numpy().astype(np.float64)
            position = group["columns"][0]
            value = block[:, position]
            if position < target_mean.size and position < target_scale.size:
                value = value * target_scale[position] + target_mean[position]
            if bool(getattr(model, "targets_are_log1p", False)):
                value = np.expm1(value / 10.0)
            colours[:, column] = value

    return colours


def lightning_shap_results(*, config, model, bundle, target: str) -> list[ShapResult]:
    """One ShapResult per model output, in output order.

    ``target`` is only a naming fallback for when the model carries no ``target_names``: a joint
    module is explained once and every one of its outputs comes back. See ``build_shap_results``.
    """
    import shap
    import torch
    from torch import nn

    if not hasattr(model, "explanation_parts") or not hasattr(model, "forward_from_parts"):
        raise TypeError(
            f"{type(model).__name__} has no attribution seam. SoilCNNLightningModule and "
            "SoilSequenceLightningModule implement explanation_parts/forward_from_parts; "
            "SoilGraphLightningModule deliberately does not."
        )

    max_samples = int(getattr(config, "EXPLAIN_MAX_SAMPLES", 500))
    background_samples = int(getattr(config, "EXPLAIN_BACKGROUND_SAMPLES", 100))

    was_training = model.training
    model.eval()
    try:
        batches = _collect_batches(bundle.datamodule, max_samples + background_samples)
        if not batches:
            return []

        per_batch_parts = []
        groups: list[dict] = []
        categorical_chunks = []
        with torch.no_grad():
            for batch in batches:
                parts, groups = model.explanation_parts(batch)
                per_batch_parts.append([part.detach() for part in parts])
                codes = batch.get("x_categorical")
                if codes is not None:
                    categorical_chunks.append(codes.detach().cpu().numpy())

        parts = _concatenate_parts(per_batch_parts, torch)
        categorical_codes = (
            np.concatenate(categorical_chunks, axis=0)[: int(parts[0].shape[0])]
            if categorical_chunks
            else None
        )

        total = int(parts[0].shape[0])
        if total < 2 or not groups:
            return []

        # Background first, explained rows after it, so the two never overlap: a point explained
        # against itself contributes a zero-length interval to the expected-gradients integral.
        split = min(max(1, background_samples), total - 1)
        background = [part[:split] for part in parts]
        explain = [part[split : split + max_samples] for part in parts]
        if int(explain[0].shape[0]) == 0:
            return []
        # The colour values describe the EXPLAINED rows, so the codes follow the same slice. Using
        # the full array here silently misaligns every categorical colour by `split` rows.
        if categorical_codes is not None:
            categorical_codes = categorical_codes[split : split + max_samples]

        class _PartsModule(nn.Module):
            """Makes forward_from_parts look like an ordinary multi-input nn.Module to shap."""

            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, *inputs):
                return self.wrapped.forward_from_parts(list(inputs))

        explainer = shap.GradientExplainer(_PartsModule(model), background)
        raw_values = explainer.shap_values(explain)
        if not isinstance(raw_values, list):
            raw_values = [raw_values]

        state = model.get_preprocessing_state() if hasattr(model, "get_preprocessing_state") else {}
        colours = _colour_values(model, explain, groups, categorical_codes, state, torch)

        feature_names = [str(group["name"]) for group in groups]
        blocks = [
            group["modality"] if group["kind"] == "temporal" else group["kind"] for group in groups
        ]

        target_names = list(getattr(model, "target_names", None) or []) or [target]
        n_outputs = _output_count(raw_values, explain)

        results = []
        for output_index in range(n_outputs):
            per_part = [
                _slice_output(array, part, output_index)
                for array, part in zip(raw_values, explain)
            ]
            values = _group_values(per_part, groups)
            results.append(
                ShapResult(
                    values=values,
                    data=colours,
                    feature_names=feature_names,
                    blocks=blocks,
                    target_name=str(
                        target_names[output_index] if output_index < len(target_names) else output_index
                    ),
                    # Attribution is taken on forward(), which is what the loss sees, so these are
                    # contributions in STANDARDIZED LOG1P space - not in the target's own units.
                    # Only predict_step inverts the transform, and it is downstream of the seam.
                    output_space=STANDARDIZED_LOG1P,
                )
            )
        return results
    finally:
        if was_training:
            model.train()


def _output_count(raw_values: list, parts: list) -> int:
    """How many model outputs shap returned values for.

    Decided by comparing against the INPUT shape rather than by counting dimensions: shap appends a
    trailing output axis only for multi-output models, and the parts themselves are a mix of 2-D
    (static, embeddings, auxiliary) and 4-D (grids), so a bare ndim test cannot tell the two apart.
    """
    reference = np.asarray(raw_values[0])
    part_shape = tuple(parts[0].shape)
    if reference.shape == part_shape:
        return 1
    if reference.ndim == len(part_shape) + 1:
        return int(reference.shape[-1])
    return 1


def _slice_output(array: Any, part, output_index: int) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.shape == tuple(part.shape):
        return values
    if values.ndim == len(tuple(part.shape)) + 1:
        return values[..., output_index]
    return values
