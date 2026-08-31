from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn

from yg_eo_soilnet.models.lightningmodules._regression_base import (
    SoilRegressionLightningBase,
    as_float_list,
    as_float_matrix,
    batch_get,
)
from yg_eo_soilnet.models.lightningmodules.mlp import build_mlp_stack
from yg_eo_soilnet.models.lightningmodules.tabular_encoders import TabularStaticEncoder
from yg_eo_soilnet.models.lightningmodules.temporal_cnn_encoders import (
    AnnualGrid2DEncoder,
    CalendarGridRasterizer,
    ConcatGatedFusion,
    DilatedTempCNNEncoder,
)


def _as_width_list(value: Any) -> list[int]:
    """`64` and `[64]` both mean one 64-wide block, so a scalar in the config still works."""
    if isinstance(value, (list, tuple)):
        return [int(width) for width in value]
    return [int(value)]


class SoilCNNLightningModule(SoilRegressionLightningBase):
    """Static covariates plus one convolutional encoder per modality over a calendar grid.

        x_static ---> static_encoder --------------------------------.
                                                                      \\
        sequences[m] + mask + time + validity                          >-- ConcatGatedFusion
                 |                                                    /            |
                 '--> CalendarGridRasterizer ---> CNN_m ---> concat --'             v
                      (ragged -> years x months)                        LayerNorm+GELU MLP head

    Convolution replaces recurrence, so every month is processed in parallel and annual seasonality
    is a property of the receptive field rather than something the network has to learn to remember.
    Two variants share the rasteriser and differ only in how they read the grid:

    * ``dilated_tempcnn`` flattens it and uses ``dilation=12`` to link month *t* to month *t-12*;
    * ``annual_grid2d`` keeps it 2D and factorises the kernel into a months pass and a years pass.

    Any number of modalities of any width is supported: everything is driven by ``modality_dims``.

    ``auxiliary_label_columns`` optionally appends MEASURED lab values to the fused vector, just
    before the head. This is an explicit opt-out of the rule that a label is never a feature, and it
    is only sound when the named values are genuinely available at inference time too - predicting
    organic matter for a sample whose texture and pH were measured, say. Naming a column that is
    also being fitted raises rather than being filtered out.
    """

    def __init__(
        self,
        static_dim: int,
        target_dim: int,
        target_names: Optional[Sequence[str]] = None,
        fitted_target_names: Optional[Sequence[str]] = None,
        categorical_cardinalities: Optional[Sequence[int]] = None,
        categorical_vocabularies: Optional[Sequence[Sequence[str]]] = None,
        categorical_feature_names: Optional[Sequence[str]] = None,
        embedding_dims: Any = None,
        embedding_dropout: float = 0.0,
        embedding_max_dim: int = 50,
        continuous_norm: str = "none",
        modality_dims: Optional[Mapping[str, int]] = None,
        temporal_enabled: bool = True,
        grid_years: Optional[int] = None,
        temporal_encoder: str = "dilated_tempcnn",
        cnn_hidden_dims: Any = (32,),
        modality_embed_dim: Any = 32,
        cnn_norm: str = "batch",
        pool: str = "masked_avg",
        month_positional: bool = True,
        use_validity_channels: bool = True,
        auxiliary_label_columns: Optional[Sequence[str]] = None,
        auxiliary_available_names: Optional[Sequence[str]] = None,
        auxiliary_validity_channels: bool = True,
        auxiliary_hidden_dims: Optional[Sequence[int]] = None,
        auxiliary_dropout: float = 0.0,
        static_hidden_dims: Sequence[int] = (64,),
        head_hidden_dims: Sequence[int] = (128, 64),
        head_norm_final: bool = False,
        dropout: float = 0.1,
        loss_name: str = "mse",
        huber_delta: float = 1.0,
        # --- structure-aware losses ---------------------------------------------------------
        # Inert unless loss_name is mahalanobis / correlation_penalty / cosine; see
        # lightningmodules/losses.py. target_covariance is injected by LightningConfigFactory
        # from the datamodule's training split, exactly as target_mean/target_scale are.
        loss_base: str = "mse",
        loss_lambda: float = 0.1,
        loss_shrinkage: float = 0.05,
        loss_min_batch: int = 16,
        cosine_space: str = "original",
        target_covariance: Optional[Any] = None,
        learning_rate: float = 1e-3,
        optimizer_name: str = "adamw",
        weight_decay: float = 1e-4,
        scheduler_type: str = "plateau",
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 5,
        scheduler_min_lr: float = 1e-6,
        scheduler_monitor: str = "val_loss",
        target_mean: Optional[Any] = None,
        target_scale: Optional[Any] = None,
        target_transform: Optional[str] = None,
        # Emit (mu, log var) instead of mu alone, and train with beta-NLL. Set by the factory from
        # uncertainty.heteroscedastic; see SoilRegressionLightningBase._beta_nll_loss.
        predict_variance: bool = False,
        beta_nll: float = 0.5,
    ):
        super().__init__()
        # Coerce BEFORE save_hyperparameters(): it captures this frame's locals, and a numpy array
        # stored in hyper_parameters makes the checkpoint unloadable under torch.load's
        # weights_only=True default (PyTorch >= 2.6).
        target_mean = as_float_list(target_mean)
        target_scale = as_float_list(target_scale)
        target_covariance = as_float_matrix(target_covariance)
        head_hidden_dims = [int(width) for width in head_hidden_dims]
        static_hidden_dims = [int(width) for width in static_hidden_dims]
        # Same reason: plain builtins only in hyper_parameters. The vocabularies live here rather
        # than in the datamodule so the checkpoint carries its own label->index mapping and can be
        # applied to a frame it has never seen.
        categorical_cardinalities = [int(value) for value in (categorical_cardinalities or [])]
        categorical_vocabularies = [
            [str(category) for category in vocabulary] for vocabulary in (categorical_vocabularies or [])
        ]
        categorical_feature_names = [str(name) for name in (categorical_feature_names or [])]
        # Same reason again, plus one of its own: the selected names and the roster they were
        # resolved against both travel in hyper_parameters, so the checkpoint records which lab
        # columns it expects instead of re-deriving positions from whatever frame it is handed.
        auxiliary_label_columns = [str(name) for name in (auxiliary_label_columns or [])]
        auxiliary_available_names = [str(name) for name in (auxiliary_available_names or [])]
        auxiliary_hidden_dims = [int(width) for width in (auxiliary_hidden_dims or [])]
        target_names = [str(name) for name in (target_names or [])]
        # Every target the RUN fits, which under per-target grouping is a superset of this model's
        # outputs. Defaults to target_names so a hand-built module keeps the old behaviour.
        fitted_target_names = [str(name) for name in (fitted_target_names or target_names)]
        self.save_hyperparameters()

        self._init_regression_targets(
            target_dim=target_dim,
            target_names=target_names,
            target_mean=target_mean,
            target_scale=target_scale,
            target_transform=target_transform,
            predict_variance=predict_variance,
            beta_nll=beta_nll,
            loss_name=loss_name,
            huber_delta=huber_delta,
            loss_base=loss_base,
            loss_lambda=loss_lambda,
            loss_shrinkage=loss_shrinkage,
            loss_min_batch=loss_min_batch,
            cosine_space=cosine_space,
            target_covariance=target_covariance,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            scheduler_type=scheduler_type,
            scheduler_factor=scheduler_factor,
            scheduler_patience=scheduler_patience,
            scheduler_min_lr=scheduler_min_lr,
            scheduler_monitor=scheduler_monitor,
        )

        # static_dim counts the CONTINUOUS covariates only; the categorical ones arrive separately as
        # indices and contribute their embedding widths instead.
        self.static_dim = int(static_dim)
        self.static_hidden_dims = list(static_hidden_dims)
        # The fused vector is sized off the static branch's OUTPUT width, which is its last block.
        self.static_hidden_dim = self.static_hidden_dims[-1]
        # A checkpoint may carry vocabularies without cardinalities; they are redundant by
        # construction (cardinality == len(vocabulary) + 1), so derive rather than demand both.
        if not categorical_cardinalities and categorical_vocabularies:
            categorical_cardinalities = [len(vocabulary) + 1 for vocabulary in categorical_vocabularies]
        self.categorical_cardinalities = list(categorical_cardinalities)
        self.categorical_vocabularies = list(categorical_vocabularies)
        self.categorical_feature_names = list(categorical_feature_names) or [
            f"categorical_{index}" for index in range(len(self.categorical_cardinalities))
        ]
        self.temporal_encoder_name = str(temporal_encoder).lower()
        if self.temporal_encoder_name not in {"dilated_tempcnn", "annual_grid2d"}:
            raise ValueError(
                f"temporal_encoder must be 'dilated_tempcnn' or 'annual_grid2d', got {temporal_encoder!r}"
            )
        # None means "infer the span from each batch". Safe because masked pooling makes an
        # embedding independent of how many empty year-rows a grid carries, but an injected
        # grid_years keeps the grid identical from batch to batch, which is one less thing to reason
        # about when comparing runs.
        self.grid_years = None if grid_years in (None, 0) else max(1, int(grid_years))

        self.modality_dims = {
            str(name).lower(): int(dim)
            for name, dim in dict(modality_dims or {}).items()
            if dim is not None and int(dim) > 0
        }
        self.temporal_enabled = bool(temporal_enabled) and bool(self.modality_dims)
        self._cnn_hidden_dims = cnn_hidden_dims
        self._modality_embed_dim = modality_embed_dim

        self.static_encoder = self._build_static_encoder(
            dropout,
            embedding_dims=embedding_dims,
            embedding_dropout=embedding_dropout,
            embedding_max_dim=embedding_max_dim,
            continuous_norm=continuous_norm,
        )

        self.rasterizers = nn.ModuleDict()
        self.temporal_encoders = nn.ModuleDict()
        if self.temporal_enabled:
            for modality_name, modality_dim in self.modality_dims.items():
                rasterizer = CalendarGridRasterizer(
                    num_channels=modality_dim,
                    grid_years=self.grid_years,
                    use_validity_channels=use_validity_channels,
                    month_positional=month_positional,
                )
                self.rasterizers[modality_name] = rasterizer
                encoder_cls = (
                    DilatedTempCNNEncoder
                    if self.temporal_encoder_name == "dilated_tempcnn"
                    else AnnualGrid2DEncoder
                )
                self.temporal_encoders[modality_name] = encoder_cls(
                    num_channels=rasterizer.output_channels,
                    output_dim=int(
                        self._per_modality_value(
                            self._modality_embed_dim, modality_name, "modality_embed_dim"
                        )
                    ),
                    hidden_dims=_as_width_list(
                        self._per_modality_value(self._cnn_hidden_dims, modality_name, "cnn_hidden_dims")
                    ),
                    dropout=dropout,
                    norm=cnn_norm,
                    pool=pool,
                )

        self.target_names = list(target_names)
        self.fitted_target_names = list(fitted_target_names)
        self.auxiliary_validity_channels = bool(auxiliary_validity_channels)
        self.auxiliary_encoder = self._build_auxiliary_encoder(
            auxiliary_label_columns,
            auxiliary_available_names,
            auxiliary_hidden_dims,
            auxiliary_dropout,
        )

        # The static branch keeps its width even when static_dim is 0, so the fused vector has a
        # fixed shape regardless of whether covariates are present.
        temporal_dim = sum(encoder.output_dim for encoder in self.temporal_encoders.values())
        self.fusion = ConcatGatedFusion(self.static_hidden_dim, temporal_dim)
        self.output_head = build_mlp_stack(
            self.fusion.output_dim + self.auxiliary_output_dim,
            head_hidden_dims,
            # head_output_dim, not target_dim: a heteroscedastic head is twice as wide
            # because it emits a log variance beside every mean.
            self.head_output_dim,
            dropout=dropout,
            activation="gelu",
            norm_final=bool(head_norm_final),
        )

    # --- construction helpers ----------------------------------------------

    def _per_modality_value(self, setting: Any, modality_name: str, label: str) -> Any:
        """One setting for every modality, or a Mapping that must name every modality.

        Returns the raw value; callers coerce. `cnn_hidden_dims` is a *list* per modality, so
        coercing to int here would be wrong for it.
        """
        if isinstance(setting, Mapping):
            value = setting.get(modality_name, setting.get(str(modality_name).lower()))
            if value is None:
                # Silently defaulting here would hand a newly added modality the old width,
                # undoing the per-branch sizing without any signal.
                raise ValueError(
                    f"Modality {modality_name!r} has no entry in {label} "
                    f"(configured: {sorted(setting)}). Add one for it, or use a single value "
                    "to apply the same setting to every modality."
                )
            return value
        return setting

    def _build_auxiliary_encoder(
        self,
        selected: list[str],
        available: list[str],
        hidden_dims: list[int],
        dropout: float,
    ) -> nn.Module:
        """Resolve the named lab columns to positions and build the branch that reads them.

        The selection is by NAME against the roster the datamodule offers, resolved once into a
        buffer. Positions cannot be configured directly: the bundle's column order follows
        LABEL_COLUMNS, so an index would silently point at a different measurement the moment that
        list is reordered.
        """
        self.auxiliary_label_columns = list(selected)
        self.auxiliary_available_names = list(available)
        self.auxiliary_output_dim = 0
        # Registered even when empty so state_dict keys do not depend on the configuration, and a
        # checkpoint trained without auxiliary columns still loads into a module that declares them.
        self.register_buffer("auxiliary_index", torch.zeros(0, dtype=torch.long), persistent=True)
        if not selected:
            return nn.Identity()

        if not self.fitted_target_names:
            # Without the target roster the leakage check below cannot run, and silently skipping
            # it is how a model ends up reading its own answer. The config factory always supplies
            # these, so this only fires on hand-built modules.
            raise ValueError(
                "auxiliary_label_columns requires target_names so a selected column can be checked "
                "against what is being fitted; pass target_names explicitly."
            )

        # Checked against every target the RUN fits, not just this model's outputs. Under
        # per-target grouping the two differ, and checking the narrower list would admit a sibling
        # target as an input - which leaks the answer just as surely, via whatever correlation the
        # two share.
        leaking = [name for name in selected if name in set(self.fitted_target_names)]
        if leaking:
            raise ValueError(
                f"auxiliary_label_columns may not name a column being fitted: {sorted(leaking)} "
                f"also appear(s) in the run's targets {sorted(self.fitted_target_names)}. The model "
                "would read a target as an input."
            )

        if not available:
            # Distinguished from the unknown-name case below because the fix is somewhere else
            # entirely: the columns may well exist and be correctly declared, and still not have
            # been carried. Reporting "unknown column" here sends the reader to audit a config that
            # is already right.
            raise ValueError(
                f"auxiliary_label_columns names {sorted(selected)} but no lab columns are being "
                "carried with the data. Set CARRY_LABEL_COLUMNS: true in data_spec.yml to make the "
                "LABEL_COLUMNS entries available as auxiliary inputs."
            )

        unknown = [name for name in selected if name not in set(available)]
        if unknown:
            raise ValueError(
                f"auxiliary_label_columns names column(s) the data does not carry: {sorted(unknown)}. "
                f"Available: {sorted(available)}. A lab column must be listed in LABEL_COLUMNS and "
                "present in the static or targets source to be selectable."
            )

        duplicates = sorted({name for name in selected if selected.count(name) > 1})
        if duplicates:
            raise ValueError(f"auxiliary_label_columns lists duplicate column(s): {duplicates}")

        self.auxiliary_index = torch.as_tensor([available.index(name) for name in selected], dtype=torch.long)
        # Validity doubles the width: one measured/filled flag per selected column, so the network
        # can discount a train-median fill instead of reading it as a measurement.
        input_dim = len(selected) * (2 if self.auxiliary_validity_channels else 1)
        # An empty hidden_dims makes this an Identity, which IS the raw-concat mode - the two
        # options are one code path, and forward() needs no branch between them.
        encoder = build_mlp_stack(
            input_dim,
            hidden_dims,
            None,
            dropout=dropout,
            activation="gelu",
            use_layer_norm=True,
            # This block feeds the head's input vector rather than a readout, so both are on - the
            # case build_mlp_stack's docstring describes as "feeding a fusion".
            norm_final=True,
            dropout_final=True,
        )
        self.auxiliary_output_dim = hidden_dims[-1] if hidden_dims else input_dim
        return encoder

    @property
    def has_auxiliary_labels(self) -> bool:
        return bool(self.auxiliary_label_columns)

    @property
    def has_static_features(self) -> bool:
        return self.static_dim > 0 or bool(self.categorical_cardinalities)

    def _build_static_encoder(
        self,
        dropout: float,
        *,
        embedding_dims: Any,
        embedding_dropout: float,
        embedding_max_dim: int,
        continuous_norm: str,
    ) -> nn.Module:
        if not self.has_static_features:
            return nn.Identity()
        # No output projection: this branch keeps its full final width, because ConcatGatedFusion
        # gates the concatenation rather than interpolating at a shared width.
        return TabularStaticEncoder(
            num_continuous=self.static_dim,
            hidden_dims=self.static_hidden_dims,
            cardinalities=self.categorical_cardinalities,
            embedding_dims=embedding_dims,
            embedding_dropout=embedding_dropout,
            embedding_max_dim=embedding_max_dim,
            feature_names=self.categorical_feature_names,
            dropout=dropout,
            activation="gelu",
            use_layer_norm=True,
            continuous_norm=continuous_norm,
        )

    # --- forward -----------------------------------------------------------

    def _encode_static(
        self, x_static: torch.Tensor, x_categorical: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if not self.has_static_features:
            return torch.zeros(
                (x_static.size(0), self.static_hidden_dim), device=x_static.device, dtype=x_static.dtype
            )
        return self.static_encoder(x_static, x_categorical)

    def _select_auxiliary(self, batch: Any, device, dtype) -> Optional[torch.Tensor]:
        """The auxiliary lab block as it enters ``auxiliary_encoder``: values, then validity flags.

        Separated from :meth:`_encode_auxiliary` so an explainer can attribute to these raw
        per-column inputs rather than to the encoder's output, which has no per-column meaning.
        """
        if not self.has_auxiliary_labels:
            return None

        values = batch_get(batch, "x_labels")
        if values is None:
            raise KeyError(
                "Batch is missing 'x_labels'; auxiliary_label_columns needs the measured lab values "
                f"({self.auxiliary_label_columns}) that the sequence datamodule collates."
            )
        values = values.to(device=device, dtype=dtype)
        if values.size(-1) != len(self.auxiliary_available_names):
            # Positions were resolved against the roster this model was BUILT with. A batch of a
            # different width means it is not that roster, and index_select would then quietly read
            # whichever measurement now sits at that position.
            raise ValueError(
                f"Batch carries {values.size(-1)} lab column(s) but this model resolved its "
                f"auxiliary columns against {len(self.auxiliary_available_names)}; the data no "
                "longer matches the checkpoint's label roster."
            )

        selected = values.index_select(-1, self.auxiliary_index)
        if self.auxiliary_validity_channels:
            validity = batch_get(batch, "x_label_validity")
            if validity is None:
                raise KeyError(
                    "Batch is missing 'x_label_validity'; set auxiliary_validity_channels=False to "
                    "run without the measured-vs-filled flags."
                )
            validity = validity.to(device=device, dtype=dtype).index_select(-1, self.auxiliary_index)
            selected = torch.cat([selected, validity], dim=-1)

        return selected

    def _encode_auxiliary(self, batch: Any, device, dtype) -> Optional[torch.Tensor]:
        selected = self._select_auxiliary(batch, device=device, dtype=dtype)
        if selected is None:
            return None
        return self.auxiliary_encoder(selected)

    def _rasterize(self, batch: Any, device, dtype) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """One ``(grid, cell_mask)`` per modality: the ragged sequences laid onto a calendar grid.

        This is the boundary between the non-differentiable part of the temporal branch and the
        differentiable one. The scatter that builds the grid cannot be attributed through, but
        everything downstream of it is convolution and pooling, so a gradient explainer takes these
        grids as its inputs and reaches every individual band.
        """
        sequences = batch_get(batch, "sequences", {}) or {}
        masks = batch_get(batch, "sequence_mask", {}) or {}
        times = batch_get(batch, "sequence_time", {}) or {}
        validities = batch_get(batch, "sequence_validity", {}) or {}

        grids: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for modality_name in self.temporal_encoders:
            values = sequences.get(modality_name)
            mask = masks.get(modality_name)
            modality_times = times.get(modality_name)
            if values is None or mask is None or modality_times is None:
                raise KeyError(
                    f"Batch is missing sequence data for modality '{modality_name}'; expected it in "
                    "'sequences', 'sequence_mask' and 'sequence_time'"
                )
            validity = validities.get(modality_name)
            grids[modality_name] = self.rasterizers[modality_name](
                values.to(device=device, dtype=dtype),
                mask.to(device=device),
                modality_times.to(device=device),
                None if validity is None else validity.to(device=device),
            )
        return grids

    def _encode_temporal_from_grids(
        self, grids: Mapping[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> Optional[torch.Tensor]:
        if not self.temporal_encoders:
            return None

        embeddings = []
        for modality_name, encoder in self.temporal_encoders.items():
            grid, cell_mask = grids[modality_name]
            embedding = encoder(grid, cell_mask)
            # A point with nothing in the window contributes nothing rather than a bias-shaped
            # artefact that the gate would then have to learn to suppress.
            embeddings.append(embedding * cell_mask.flatten(1).any(dim=1, keepdim=True).to(dtype=embedding.dtype))

        return torch.cat(embeddings, dim=-1)

    def _encode_temporal(self, batch: Any, device, dtype) -> Optional[torch.Tensor]:
        if not self.temporal_encoders:
            return None
        return self._encode_temporal_from_grids(self._rasterize(batch, device=device, dtype=dtype))

    def forward(self, batch: Any) -> torch.Tensor:
        x_static = batch_get(batch, "x_static")
        if x_static is None:
            raise KeyError("Batch is missing 'x_static'")

        reference = next(self.parameters())
        device, dtype = reference.device, reference.dtype
        x_static = x_static.to(device=device, dtype=dtype)
        x_categorical = batch_get(batch, "x_categorical")
        if x_categorical is not None:
            x_categorical = x_categorical.to(device=device)

        static_features = self._encode_static(x_static, x_categorical)
        temporal_features = self._encode_temporal(batch, device=device, dtype=dtype)
        fused = self.fusion(static_features, temporal_features)

        # Appended AFTER the gate, so a measured lab value reaches the head at full strength rather
        # than being traded off against the branches that had to infer it.
        auxiliary_features = self._encode_auxiliary(batch, device=device, dtype=dtype)
        if auxiliary_features is not None:
            fused = torch.cat([fused, auxiliary_features], dim=-1)
        return self.output_head(fused)

    # --- attribution seam ---------------------------------------------------
    # explanation_parts() splits a batch into the tensors an explainer perturbs, and
    # forward_from_parts() rebuilds the prediction from exactly those tensors. The pair must agree:
    # forward_from_parts(explanation_parts(batch)[0]) has to equal forward(batch) exactly, because
    # any drift between the two silently attributes importance to a model that is not the one being
    # scored. tests/test_explain.py pins that equality.
    #
    # Neither method is called by forward, _shared_step or predict_step. Training is bit-identical
    # whether or not anything ever explains the model.

    def explanation_parts(self, batch: Any) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
        """``(parts, groups)`` - the attribution inputs and how to fold them back into features.

        Each group describes one contiguous run of columns in one part tensor, and names the
        feature that run belongs to. Summing a group's SHAP values gives that feature's
        contribution, which is valid because SHAP values are additive.
        """
        reference = next(self.parameters())
        device, dtype = reference.device, reference.dtype

        x_static = batch_get(batch, "x_static")
        if x_static is None:
            raise KeyError("Batch is missing 'x_static'")
        x_static = x_static.to(device=device, dtype=dtype)

        parts: list[torch.Tensor] = [x_static]
        groups: list[dict[str, Any]] = [
            {
                "part": 0,
                "kind": "static",
                "name": name,
                "columns": [index],
            }
            for index, name in enumerate(self._static_feature_names(x_static.size(-1)))
        ]

        # Categorical: the embedding, not the int64 index, because an index has no gradient. One
        # group per feature, spanning that feature's embedding dimensions.
        if self.has_static_features and self.static_encoder.embeddings.num_features:
            x_categorical = batch_get(batch, "x_categorical")
            if x_categorical is None:
                raise KeyError("Batch is missing 'x_categorical'")
            embedded = self.static_encoder.embeddings(x_categorical.to(device=device)).to(dtype=dtype)
            part_index = len(parts)
            parts.append(embedded)
            cursor = 0
            for name, width in zip(
                self.static_encoder.embeddings.feature_names,
                self.static_encoder.embeddings.embedding_dims,
            ):
                groups.append(
                    {
                        "part": part_index,
                        "kind": "categorical",
                        "name": name,
                        "columns": list(range(cursor, cursor + int(width))),
                    }
                )
                cursor += int(width)

        # Temporal: one part per modality, the rasterized grid. Groups fold each band's value
        # channel together with its validity channel - they describe the same band - and keep the
        # month sin/cos pair as one row of its own.
        grids = self._rasterize(batch, device=device, dtype=dtype) if self.temporal_encoders else {}
        for modality_name in self.temporal_encoders:
            grid, _cell_mask = grids[modality_name]
            part_index = len(parts)
            parts.append(grid)
            layout = self.rasterizers[modality_name].channel_layout()
            column_names = self._modality_column_names(modality_name, len(layout["values"]))
            for band_index, band_name in enumerate(column_names):
                columns = [layout["values"][band_index]]
                if layout["validity"]:
                    columns.append(layout["validity"][band_index])
                groups.append(
                    {
                        "part": part_index,
                        "kind": "temporal",
                        "modality": modality_name,
                        "name": band_name,
                        "columns": columns,
                    }
                )
            if layout["month_positional"]:
                groups.append(
                    {
                        "part": part_index,
                        "kind": "temporal",
                        "modality": modality_name,
                        "name": f"{modality_name}_month_positional",
                        "columns": list(layout["month_positional"]),
                    }
                )

        # Auxiliary lab block: raw values, followed by validity flags when they are enabled.
        selected = self._select_auxiliary(batch, device=device, dtype=dtype)
        if selected is not None:
            part_index = len(parts)
            parts.append(selected)
            count = len(self.auxiliary_label_columns)
            for index, name in enumerate(self.auxiliary_label_columns):
                columns = [index]
                if self.auxiliary_validity_channels:
                    columns.append(count + index)
                groups.append(
                    {
                        "part": part_index,
                        "kind": "auxiliary",
                        "name": name,
                        "columns": columns,
                    }
                )

        return parts, groups

    def forward_from_parts(self, parts: Sequence[torch.Tensor]) -> torch.Tensor:
        """Rebuild a prediction from :meth:`explanation_parts` output, and nothing else.

        A pure function of ``parts`` on purpose. A gradient explainer evaluates the model on
        interpolations between a sample and random background rows, so anything the forward pass
        needs has to be derivable from the perturbed tensors themselves - it cannot be captured from
        the original batch, because the row count and the row identities both change.

        ``cell_mask`` is therefore read back out of the grid rather than passed alongside it: the
        rasterizer already writes it as the ``cell_observed`` channel, so the grid is self-contained.
        """
        cursor = 0
        x_static = parts[cursor]
        cursor += 1

        embedded = None
        if self.has_static_features and self.static_encoder.embeddings.num_features:
            embedded = parts[cursor]
            cursor += 1

        if self.has_static_features:
            static_features = self.static_encoder.forward_with_embedding(x_static, embedded)
        else:
            static_features = torch.zeros(
                (x_static.size(0), self.static_hidden_dim), device=x_static.device, dtype=x_static.dtype
            )

        temporal_features = None
        if self.temporal_encoders:
            grids = {}
            for modality_name in self.temporal_encoders:
                grid = parts[cursor]
                cursor += 1
                observed_index = self.rasterizers[modality_name].channel_layout()["cell_observed"][0]
                grids[modality_name] = (grid, grid[:, observed_index] > 0.5)
            temporal_features = self._encode_temporal_from_grids(grids)

        fused = self.fusion(static_features, temporal_features)

        if self.has_auxiliary_labels:
            fused = torch.cat([fused, self.auxiliary_encoder(parts[cursor])], dim=-1)
            cursor += 1

        # The MEAN only. On a heteroscedastic head the readout is 2*target_dim wide, and returning
        # it whole would hand the explainer a second block of outputs that are log variances - which
        # it would attribute and label as targets, producing a SHAP plot with twice the targets the
        # model has, half of them explaining a quantity nobody asked about.
        return self._split_head_output(self.output_head(fused))[0]

    def _static_feature_names(self, width: int) -> list[str]:
        """Names for the continuous static block, falling back to positions when none were stored."""
        names = list(getattr(self, "static_feature_names", None) or [])
        if len(names) == width:
            return names
        return [f"static_{index}" for index in range(width)]

    def _modality_column_names(self, modality_name: str, width: int) -> list[str]:
        """Band names for one modality, falling back to positions when none were stored."""
        stored = (getattr(self, "modality_column_names", None) or {}).get(modality_name)
        names = list(stored or [])
        if len(names) == width:
            return names
        return [f"{modality_name}_{index}" for index in range(width)]
