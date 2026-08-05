from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn

from yg_eo_soilnet.models.lightningmodules._regression_base import (
    SoilRegressionLightningBase,
    as_float_list,
    batch_get,
)
from yg_eo_soilnet.models.lightningmodules.temporal_cnn_encoders import (
    AnnualGrid2DEncoder,
    CalendarGridRasterizer,
    ConcatGatedFusion,
    DilatedTempCNNEncoder,
)


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
        modality_dims: Optional[Mapping[str, int]] = None,
        temporal_enabled: bool = True,
        grid_years: Optional[int] = None,
        temporal_encoder: str = "dilated_tempcnn",
        cnn_hidden_dim: Any = 32,
        modality_embed_dim: Any = 32,
        num_blocks: int = 1,
        cnn_norm: str = "batch",
        pool: str = "masked_avg",
        month_positional: bool = True,
        use_validity_channels: bool = True,
        static_hidden_dim: int = 64,
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

        self.static_dim = int(static_dim)
        self.static_hidden_dim = int(static_hidden_dim)
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
        self._cnn_hidden_dim = cnn_hidden_dim
        self._modality_embed_dim = modality_embed_dim

        self.static_encoder = self._build_static_encoder(dropout)

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
                    output_dim=self._per_modality_value(
                        self._modality_embed_dim, modality_name, "modality_embed_dim"
                    ),
                    hidden_dim=self._per_modality_value(self._cnn_hidden_dim, modality_name, "cnn_hidden_dim"),
                    num_blocks=num_blocks,
                    dropout=dropout,
                    norm=cnn_norm,
                    pool=pool,
                )

        # The static branch keeps its width even when static_dim is 0, so the fused vector has a
        # fixed shape regardless of whether covariates are present.
        temporal_dim = sum(encoder.output_dim for encoder in self.temporal_encoders.values())
        self.fusion = ConcatGatedFusion(self.static_hidden_dim, temporal_dim)
        self.output_head = self._build_output_head(
            self.fusion.output_dim, head_hidden_dims, bool(head_norm_final), dropout
        )

    # --- construction helpers ----------------------------------------------

    def _per_modality_value(self, setting: Any, modality_name: str, label: str) -> int:
        """A scalar applies one width everywhere; a Mapping must name every modality."""
        if isinstance(setting, Mapping):
            value = setting.get(modality_name, setting.get(str(modality_name).lower()))
            if value is None:
                # Silently defaulting here would hand a newly added modality the old width,
                # undoing the per-branch sizing without any signal.
                raise ValueError(
                    f"Modality {modality_name!r} has no width in {label} "
                    f"(configured: {sorted(setting)}). Add an entry for it, or use a scalar value "
                    "to apply one width to every modality."
                )
            return int(value)
        return int(setting)

    def _build_static_encoder(self, dropout: float) -> nn.Module:
        if self.static_dim <= 0:
            return nn.Identity()
        return nn.Sequential(
            nn.Linear(self.static_dim, self.static_hidden_dim),
            nn.LayerNorm(self.static_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _build_output_head(
        self, input_dim: int, hidden_dims: Sequence[int], norm_final: bool, dropout: float
    ) -> nn.Module:
        if not hidden_dims:
            return nn.Linear(input_dim, self.target_dim)

        layers: list[nn.Module] = []
        dim = input_dim
        for index, width in enumerate(hidden_dims):
            width = int(width)
            layers.append(nn.Linear(dim, width))
            is_last_block = index == len(hidden_dims) - 1
            # LayerNorm + GELU per block, except on the block feeding the readout unless asked for.
            # Normalising there forces the penultimate vector to unit variance, leaving the final
            # Linear only its direction - and magnitude is what a regressor needs to reach the
            # tails. Dropout there is likewise minimized under MSE by shrinking the readout toward
            # its bias, i.e. toward the target mean.
            if not is_last_block or norm_final:
                layers.append(nn.LayerNorm(width))
            layers.append(nn.GELU())
            if not is_last_block:
                layers.append(nn.Dropout(dropout))
            dim = width
        layers.append(nn.Linear(dim, self.target_dim))
        return nn.Sequential(*layers)

    # --- forward -----------------------------------------------------------

    def _encode_static(self, x_static: torch.Tensor) -> torch.Tensor:
        if self.static_dim <= 0:
            return torch.zeros(
                (x_static.size(0), self.static_hidden_dim), device=x_static.device, dtype=x_static.dtype
            )
        return self.static_encoder(x_static)

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

        static_features = self._encode_static(x_static)
        temporal_features = self._encode_temporal(batch, device=device, dtype=dtype)
        fused = self.fusion(static_features, temporal_features)
        return self.output_head(fused)
