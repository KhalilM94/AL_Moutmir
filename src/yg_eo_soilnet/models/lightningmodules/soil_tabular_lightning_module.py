from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from yg_eo_soilnet.models.lightningmodules._regression_base import (
    SoilRegressionLightningBase,
    as_float_list,
    batch_get,
)
from yg_eo_soilnet.models.lightningmodules.mlp import build_mlp_stack
from yg_eo_soilnet.models.lightningmodules.tabular_encoders import TabularStaticEncoder


class SoilTabularLightningModule(SoilRegressionLightningBase):
    """Static covariates only: entity embeddings + scaled numerics -> MLP -> target.

        x_static ------.
                        >-- TabularStaticEncoder --> output_head
        x_categorical -'      (embeddings + concat)

    No temporal branch at all. It reads the same sequence datamodule as the other two models and
    simply ignores every sequence key, which is the point: it answers "how much does the temporal
    branch actually buy" against an otherwise identical static pathway, loss and optimizer.

    It also exercises ``TabularStaticEncoder`` on its own, so a regression in the shared block shows
    up here without a temporal encoder in the way.
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
        static_hidden_dims: Sequence[int] = (64,),
        head_hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.1,
        activation: str = "relu",
        use_layer_norm: bool = True,
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
        if not categorical_cardinalities and categorical_vocabularies:
            categorical_cardinalities = [len(vocabulary) + 1 for vocabulary in categorical_vocabularies]
        self.categorical_cardinalities = list(categorical_cardinalities)
        self.categorical_vocabularies = list(categorical_vocabularies)
        self.categorical_feature_names = list(categorical_feature_names) or [
            f"categorical_{index}" for index in range(len(self.categorical_cardinalities))
        ]

        if self.static_dim <= 0 and not self.categorical_cardinalities:
            raise ValueError(
                "SoilTabularLightningModule has no features: static_dim is 0 and no categorical "
                "cardinalities were given. This model has no temporal branch to fall back on."
            )

        self.static_encoder = TabularStaticEncoder(
            num_continuous=self.static_dim,
            hidden_dims=static_hidden_dims,
            cardinalities=self.categorical_cardinalities,
            embedding_dims=embedding_dims,
            embedding_dropout=embedding_dropout,
            embedding_max_dim=embedding_max_dim,
            feature_names=self.categorical_feature_names,
            dropout=dropout,
            activation=activation,
            use_layer_norm=use_layer_norm,
            continuous_norm=continuous_norm,
        )
        self.output_head = build_mlp_stack(
            self.static_encoder.output_dim,
            head_hidden_dims,
            self.target_dim,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, batch: Any) -> torch.Tensor:
        x_static = batch_get(batch, "x_static")
        if x_static is None:
            raise KeyError("Batch is missing 'x_static'")

        reference = next(self.parameters())
        x_static = x_static.to(device=reference.device, dtype=reference.dtype)
        x_categorical = batch_get(batch, "x_categorical")
        if x_categorical is not None:
            x_categorical = x_categorical.to(device=reference.device)

        return self.output_head(self.static_encoder(x_static, x_categorical))
