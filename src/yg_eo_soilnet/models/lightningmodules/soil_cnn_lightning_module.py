from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn

from yg_eo_soilnet.models.lightningmodules._regression_base import (
    SoilRegressionLightningBase,
    as_float_list,
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
    """

    def __init__(
        self,
        static_dim: int,
        target_dim: int,
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
        static_hidden_dims: Sequence[int] = (64,),
        head_hidden_dims: Sequence[int] = (128, 64),
        head_norm_final: bool = False,
        dropout: float = 0.1,
        loss_name: str = "mse",
        huber_delta: float = 1.0,
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
    ):
        super().__init__()
        # Coerce BEFORE save_hyperparameters(): it captures this frame's locals, and a numpy array
        # stored in hyper_parameters makes the checkpoint unloadable under torch.load's
        # weights_only=True default (PyTorch >= 2.6).
        target_mean = as_float_list(target_mean)
        target_scale = as_float_list(target_scale)
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
        self.save_hyperparameters()

        self._init_regression_targets(
            target_dim=target_dim,
            target_mean=target_mean,
            target_scale=target_scale,
            target_transform=target_transform,
            loss_name=loss_name,
            huber_delta=huber_delta,
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

        # The static branch keeps its width even when static_dim is 0, so the fused vector has a
        # fixed shape regardless of whether covariates are present.
        temporal_dim = sum(encoder.output_dim for encoder in self.temporal_encoders.values())
        self.fusion = ConcatGatedFusion(self.static_hidden_dim, temporal_dim)
        self.output_head = build_mlp_stack(
            self.fusion.output_dim,
            head_hidden_dims,
            self.target_dim,
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

    def _encode_temporal(self, batch: Any, device, dtype) -> Optional[torch.Tensor]:
        if not self.temporal_encoders:
            return None

        sequences = batch_get(batch, "sequences", {}) or {}
        masks = batch_get(batch, "sequence_mask", {}) or {}
        times = batch_get(batch, "sequence_time", {}) or {}
        validities = batch_get(batch, "sequence_validity", {}) or {}

        embeddings = []
        for modality_name, encoder in self.temporal_encoders.items():
            values = sequences.get(modality_name)
            mask = masks.get(modality_name)
            modality_times = times.get(modality_name)
            if values is None or mask is None or modality_times is None:
                raise KeyError(
                    f"Batch is missing sequence data for modality '{modality_name}'; expected it in "
                    "'sequences', 'sequence_mask' and 'sequence_time'"
                )
            validity = validities.get(modality_name)
            grid, cell_mask = self.rasterizers[modality_name](
                values.to(device=device, dtype=dtype),
                mask.to(device=device),
                modality_times.to(device=device),
                None if validity is None else validity.to(device=device),
            )
            embedding = encoder(grid, cell_mask)
            # A point with nothing in the window contributes nothing rather than a bias-shaped
            # artefact that the gate would then have to learn to suppress.
            embeddings.append(embedding * cell_mask.flatten(1).any(dim=1, keepdim=True).to(dtype=embedding.dtype))

        return torch.cat(embeddings, dim=-1)

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
        return self.output_head(fused)
