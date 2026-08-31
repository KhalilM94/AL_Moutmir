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
from yg_eo_soilnet.models.lightningmodules.temporal_encoders import (
    GatedFusion,
    TemporalTransformerEncoder,
    TimeAwareLSTMEncoder,
)


class SoilSequenceLightningModule(SoilRegressionLightningBase):
    """Static covariates plus one sequence encoder per modality, blended by a learned gate.

        x_static ---> static_encoder ------------------------------.
                                                                    \\
        sequences[s2]   ---> encoder_s2   ---.                       >--- GatedFusion
        sequences[s1]   ---> encoder_s1   ---+                      /         |
        sequences[soil] ---> encoder_soil ---+-- concat -> project -'          v
        sequences[ag]   ---> encoder_ag   ---+                            fusion_norm
        sequences[clim] ---> encoder_clim ---'                                 |
                                                                               v
                                                                          output_head

    There is no graph here: no edges, no coordinates, no message passing. Each point is encoded from
    its own covariates and its own observation history.

    The model is length- and era-agnostic. No parameter is sized by the number of observations, and
    time enters only through relative and cyclical features, so a checkpoint trained on 2017-2025
    monthly data runs unchanged on a shorter, longer, or differently-dated series.
    """

    def __init__(
        self,
        static_dim: int,
        target_dim: int,
        target_names: Optional[Sequence[str]] = None,
        categorical_cardinalities: Optional[Sequence[int]] = None,
        categorical_vocabularies: Optional[Sequence[Sequence[str]]] = None,
        categorical_feature_names: Optional[Sequence[str]] = None,
        embedding_dims: Any = None,
        embedding_dropout: float = 0.0,
        embedding_max_dim: int = 50,
        continuous_norm: str = "none",
        modality_dims: Optional[Mapping[str, int]] = None,
        temporal_enabled: bool = True,
        temporal_encoder: str = "time_transformer",
        modality_embed_dim: Any = 32,
        fusion_dim: int = 64,
        static_hidden_dims: Sequence[int] = (64,),
        d_model: int = 48,
        nhead: int = 4,
        num_layers: int = 2,
        time2vec_dim: int = 8,
        lstm_hidden_dim: Any = 32,
        lstm_num_layers: int = 1,
        lstm_bidirectional: bool = False,
        temporal_pooling: str = "mean_max_last",
        span_cap_years: float = 8.0,
        delta_cap_months: float = 24.0,
        dropout: float = 0.1,
        head_hidden_dims: Sequence[int] = (64, 32),
        use_layer_norm: bool = True,
        fusion_norm_type: str = "batch",
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
        self.fusion_dim = int(fusion_dim)
        self.static_hidden_dims = list(static_hidden_dims)
        # A checkpoint may carry vocabularies without cardinalities; they are redundant by
        # construction (cardinality == len(vocabulary) + 1), so derive rather than demand both.
        if not categorical_cardinalities and categorical_vocabularies:
            categorical_cardinalities = [len(vocabulary) + 1 for vocabulary in categorical_vocabularies]
        self.categorical_cardinalities = list(categorical_cardinalities)
        self.categorical_vocabularies = list(categorical_vocabularies)
        self.categorical_feature_names = list(categorical_feature_names) or [
            f"categorical_{index}" for index in range(len(self.categorical_cardinalities))
        ]
        self.use_layer_norm = bool(use_layer_norm)
        self.temporal_encoder_name = str(temporal_encoder).lower()
        if self.temporal_encoder_name not in {"time_transformer", "time_lstm"}:
            raise ValueError(
                f"temporal_encoder must be 'time_transformer' or 'time_lstm', got {temporal_encoder!r}"
            )
        self.fusion_norm_type = str(fusion_norm_type).lower()
        if self.fusion_norm_type not in {"batch", "layer", "none"}:
            raise ValueError("fusion_norm_type must be 'batch', 'layer' or 'none'")

        self.modality_dims = {
            str(name).lower(): int(dim)
            for name, dim in dict(modality_dims or {}).items()
            if dim is not None and int(dim) > 0
        }
        self.temporal_enabled = bool(temporal_enabled) and bool(self.modality_dims)
        self._modality_embed_dim = modality_embed_dim
        self._lstm_hidden_dim = lstm_hidden_dim

        self.static_encoder = self._build_static_encoder(
            dropout,
            embedding_dims=embedding_dims,
            embedding_dropout=embedding_dropout,
            embedding_max_dim=embedding_max_dim,
            continuous_norm=continuous_norm,
        )

        self.temporal_encoders = nn.ModuleDict()
        if self.temporal_enabled:
            for modality_name, modality_dim in self.modality_dims.items():
                self.temporal_encoders[modality_name] = self._build_temporal_encoder(
                    modality_name,
                    modality_dim,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    time2vec_dim=time2vec_dim,
                    lstm_num_layers=lstm_num_layers,
                    lstm_bidirectional=lstm_bidirectional,
                    temporal_pooling=temporal_pooling,
                    span_cap_years=span_cap_years,
                    delta_cap_months=delta_cap_months,
                    dropout=dropout,
                )

        if self.temporal_encoders:
            concatenated_dim = sum(encoder.output_dim for encoder in self.temporal_encoders.values())
            self.temporal_projection = nn.Sequential(
                nn.Linear(concatenated_dim, self.fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.fusion = GatedFusion(self.fusion_dim)
        else:
            self.temporal_projection = None
            self.fusion = None

        # BatchNorm1d by default: it normalizes each feature across the batch and leaves per-sample
        # magnitude intact, whereas LayerNorm normalizes across features within a sample and erases
        # the overall level of the vector - two points differing by a global offset become
        # identical, which caps how far predictions can move from the target mean.
        self.fusion_norm = self._build_fusion_norm()
        self.output_head = build_mlp_stack(
            self.fusion_dim,
            head_hidden_dims,
            # head_output_dim, not target_dim: a heteroscedastic head is twice as wide
            # because it emits a log variance beside every mean.
            self.head_output_dim,
            dropout=dropout,
            use_layer_norm=self.use_layer_norm,
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

    def _build_temporal_encoder(
        self,
        modality_name: str,
        modality_dim: int,
        *,
        d_model: int,
        nhead: int,
        num_layers: int,
        time2vec_dim: int,
        lstm_num_layers: int,
        lstm_bidirectional: bool,
        temporal_pooling: str,
        span_cap_years: float,
        delta_cap_months: float,
        dropout: float,
    ) -> nn.Module:
        output_dim = self._per_modality_value(self._modality_embed_dim, modality_name, "modality_embed_dim")
        if self.temporal_encoder_name == "time_lstm":
            return TimeAwareLSTMEncoder(
                input_dim=modality_dim,
                output_dim=output_dim,
                hidden_dim=self._per_modality_value(self._lstm_hidden_dim, modality_name, "lstm_hidden_dim"),
                num_layers=lstm_num_layers,
                dropout=dropout,
                bidirectional=lstm_bidirectional,
                pooling=temporal_pooling,
                span_cap_years=span_cap_years,
                delta_cap_months=delta_cap_months,
            )
        return TemporalTransformerEncoder(
            input_dim=modality_dim,
            output_dim=output_dim,
            d_model=self._per_modality_value(d_model, modality_name, "d_model"),
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            time2vec_dim=time2vec_dim,
            span_cap_years=span_cap_years,
            delta_cap_months=delta_cap_months,
        )

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
        # Projects to fusion_dim, because the gate blends static against temporal at a shared width.
        return TabularStaticEncoder(
            num_continuous=self.static_dim,
            hidden_dims=self.static_hidden_dims,
            output_dim=self.fusion_dim,
            cardinalities=self.categorical_cardinalities,
            embedding_dims=embedding_dims,
            embedding_dropout=embedding_dropout,
            embedding_max_dim=embedding_max_dim,
            feature_names=self.categorical_feature_names,
            dropout=dropout,
            activation="relu",
            use_layer_norm=self.use_layer_norm,
            continuous_norm=continuous_norm,
        )

    def _build_fusion_norm(self) -> nn.Module:
        if self.fusion_norm_type == "none":
            return nn.Identity()
        if self.fusion_norm_type == "batch":
            return nn.BatchNorm1d(self.fusion_dim)
        return nn.LayerNorm(self.fusion_dim)

    # --- forward -----------------------------------------------------------

    def _encode_static(
        self, x_static: torch.Tensor, x_categorical: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if not self.has_static_features:
            return torch.zeros(
                (x_static.size(0), self.fusion_dim), device=x_static.device, dtype=x_static.dtype
            )
        return self.static_encoder(x_static, x_categorical)

    def _encode_temporal(self, batch: Any, device, dtype) -> Optional[torch.Tensor]:
        if not self.temporal_encoders:
            return None

        sequences = batch_get(batch, "sequences", {}) or {}
        masks = batch_get(batch, "sequence_mask", {}) or {}
        times = batch_get(batch, "sequence_time", {}) or {}

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
            embeddings.append(
                encoder(
                    values.to(device=device, dtype=dtype),
                    mask.to(device=device),
                    modality_times.to(device=device),
                )
            )

        if not embeddings:
            return None
        return self.temporal_projection(torch.cat(embeddings, dim=-1))

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

        fused = static_features if self.fusion is None else self.fusion(static_features, temporal_features)
        fused = self.fusion_norm(fused)
        return self.output_head(fused)
