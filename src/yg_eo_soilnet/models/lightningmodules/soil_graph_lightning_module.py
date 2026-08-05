from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from torch import nn

from lightning.pytorch import LightningModule


def _batch_get(batch: Any, key: str, default=None):
    if isinstance(batch, Mapping):
        return batch.get(key, default)
    if hasattr(batch, "get"):
        return batch.get(key, default)
    if hasattr(batch, "__getitem__"):
        try:
            return batch[key]
        except Exception:
            return default
    return getattr(batch, key, default)


def _as_float_list(value: Any) -> Optional[list[float]]:
    """Normalize array-likes to plain Python floats so hyperparameters stay pickle-safe."""
    if value is None:
        return None
    return [float(item) for item in torch.as_tensor(value, dtype=torch.float32).flatten().tolist()]


class ResidualGraphBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_attr_dim: int = 0, dropout: float = 0.0, use_layer_norm: bool = True):
        super().__init__()
        self.edge_gate = nn.Linear(edge_attr_dim, hidden_dim) if edge_attr_dim > 0 else None
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

    def forward(self, node_features, edge_index, edge_attr=None):
        if edge_index is None or edge_index.numel() == 0:
            aggregated = node_features
        else:
            source_nodes = edge_index[0].long()
            target_nodes = edge_index[1].long()
            messages = node_features[source_nodes]
            if edge_attr is not None and self.edge_gate is not None and edge_attr.numel() > 0:
                gate = torch.sigmoid(self.edge_gate(edge_attr))
                messages = messages * gate

            aggregated = torch.zeros_like(node_features)
            aggregated.index_add_(0, target_nodes, messages)
            degree = torch.zeros(node_features.size(0), device=node_features.device, dtype=node_features.dtype)
            degree.index_add_(0, target_nodes, torch.ones(target_nodes.size(0), device=node_features.device, dtype=node_features.dtype))
            aggregated = aggregated / degree.clamp_min(1.0).unsqueeze(-1)

        # Genuinely residual: the skip is what keeps gradients flowing past ~2 stacked blocks.
        update = self.node_update(torch.cat([node_features, aggregated], dim=-1))
        return self.norm(node_features + update)


class SoilGraphLightningModule(LightningModule):
    def __init__(
        self,
        static_dim: int,
        target_dim: int,
        hidden_dim: int = 64,
        temporal_hidden_dim: int = 32,
        modality_dims: Optional[Mapping[str, int]] = None,
        temporal_steps: Optional[int] = None,
        edge_attr_dim: int = 0,
        num_graph_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        temporal_enabled: bool = True,
        static_hidden_dim: Optional[int] = None,
        head_num_layers: int = 0,
        head_hidden_dim: Optional[int] = None,
        head_min_hidden_dim: int = 16,
        use_layer_norm: bool = True,
        fusion_norm_type: str = "batch",
        temporal_lstm_hidden_dim: Optional[Any] = None,
        temporal_lstm_num_layers: int = 1,
        temporal_lstm_dropout: float = 0.0,
        temporal_lstm_bidirectional: bool = False,
        temporal_pooling: str = "last",
        spatial_graph_enabled: bool = True,
        optimizer_name: str = "adamw",
        weight_decay: float = 1e-4,
        scheduler_type: str = "plateau",
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 5,
        scheduler_min_lr: float = 1e-6,
        scheduler_monitor: str = "val_loss",
        loss_name: str = "mse",
        huber_delta: float = 1.0,
        target_mean: Optional[Any] = None,
        target_scale: Optional[Any] = None,
        target_transform: Optional[str] = None,
    ):
        if torch is None or nn is None:
            raise ImportError("torch is required to instantiate SoilGraphLightningModule")

        super().__init__()
        # Coerce BEFORE save_hyperparameters(): it captures this frame's locals, and a numpy array
        # stored in hyper_parameters makes the checkpoint unloadable under torch.load's
        # weights_only=True default (PyTorch >= 2.6).
        target_mean = _as_float_list(target_mean)
        target_scale = _as_float_list(target_scale)
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters()

        self.static_dim = int(static_dim)
        self.target_dim = int(target_dim)
        self.hidden_dim = int(hidden_dim)
        self.temporal_hidden_dim = int(temporal_hidden_dim)
        self.edge_attr_dim = int(edge_attr_dim)
        self.learning_rate = float(learning_rate)
        self.temporal_enabled = bool(temporal_enabled)
        self.temporal_steps = temporal_steps
        self.static_hidden_dim = int(static_hidden_dim or hidden_dim)
        self.head_num_layers = max(0, int(head_num_layers))
        self.head_hidden_dim = int(head_hidden_dim or hidden_dim)
        self.head_min_hidden_dim = max(1, int(head_min_hidden_dim))
        self.use_layer_norm = bool(use_layer_norm)
        self.fusion_norm_type = str(fusion_norm_type).lower()
        if self.fusion_norm_type not in {"batch", "layer", "none"}:
            raise ValueError("fusion_norm_type must be 'batch', 'layer' or 'none'")
        # int -> same width everywhere (previous behaviour); Mapping -> per-modality width.
        self.temporal_lstm_hidden_dim = temporal_lstm_hidden_dim if isinstance(
            temporal_lstm_hidden_dim, Mapping
        ) else int(temporal_lstm_hidden_dim or temporal_hidden_dim)
        self.temporal_lstm_num_layers = max(1, int(temporal_lstm_num_layers))
        self.temporal_lstm_dropout = float(temporal_lstm_dropout)
        self.temporal_lstm_bidirectional = bool(temporal_lstm_bidirectional)
        self.temporal_pooling = str(temporal_pooling).lower()
        self.spatial_graph_enabled = bool(spatial_graph_enabled)
        self.optimizer_name = str(optimizer_name).lower()
        self.weight_decay = float(weight_decay)
        self.scheduler_type = str(scheduler_type).lower()
        self.scheduler_factor = float(scheduler_factor)
        self.scheduler_patience = max(0, int(scheduler_patience))
        self.scheduler_min_lr = float(scheduler_min_lr)
        self.scheduler_monitor = str(scheduler_monitor)
        if self.temporal_pooling not in {"last", "attention"}:
            raise ValueError("temporal_pooling must be 'last' or 'attention'")

        # Target standardization stats from the datamodule. The loss is computed in standardized
        # space; predict_step inverts so downstream evaluation sees original units.
        self.register_buffer(
            "target_mean",
            torch.zeros(self.target_dim) if target_mean is None else torch.as_tensor(target_mean, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "target_scale",
            torch.ones(self.target_dim) if target_scale is None else torch.as_tensor(target_scale, dtype=torch.float32),
            persistent=True,
        )
        # A buffer, not a plain attribute: it must round-trip through state_dict. If it were derived
        # from init args alone, a checkpoint restore that lost the hyperparameters would silently
        # skip the inverse transform and report predictions in standardized units.
        self.register_buffer(
            "targets_are_standardized",
            torch.tensor(target_mean is not None and target_scale is not None),
            persistent=True,
        )
        # Same reasoning as above: the datamodule may have applied log1p to the targets before
        # fitting the standardization stats, and a checkpoint restore that lost the hyperparameters
        # must not silently skip half the inverse and report predictions on the log scale.
        self.target_transform = None if target_transform is None else str(target_transform).lower()
        if self.target_transform not in {None, "none", "log1p"}:
            raise ValueError("target_transform must be None or 'log1p'")
        self.register_buffer(
            "targets_are_log1p",
            torch.tensor(self.target_transform == "log1p"),
            persistent=True,
        )

        self.static_encoder = self._build_static_encoder(self.static_dim, self.static_hidden_dim, dropout)

        self.modality_dims = {
            str(name).lower(): int(dim)
            for name, dim in dict(modality_dims or {}).items()
            if dim is not None
        }

        self.temporal_modalities = dict(self.modality_dims)
        self.temporal_encoders = nn.ModuleDict()
        self.temporal_attention = nn.ModuleDict()
        self.temporal_embedding_dims: dict[str, int] = {}
        for modality_name, modality_dim in self.temporal_modalities.items():
            if modality_dim is None or not self.temporal_enabled:
                continue
            modality_hidden = self._temporal_hidden(modality_name)
            embedding_dim = modality_hidden * (2 if self.temporal_lstm_bidirectional else 1)
            self.temporal_embedding_dims[modality_name] = embedding_dim
            self.temporal_encoders[modality_name] = nn.LSTM(
                input_size=int(modality_dim),
                hidden_size=modality_hidden,
                num_layers=self.temporal_lstm_num_layers,
                dropout=self.temporal_lstm_dropout if self.temporal_lstm_num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=self.temporal_lstm_bidirectional,
            )
            if self.temporal_pooling == "attention":
                self.temporal_attention[modality_name] = nn.Linear(embedding_dim, 1)

        fusion_input_dim = self.static_hidden_dim + sum(self.temporal_embedding_dims.values())
        self.fusion_input_dim = fusion_input_dim
        # The static branch and each pooled temporal branch arrive on unrelated scales before being
        # combined by a single linear map, so the fused vector needs normalizing. Prefer "batch":
        # BatchNorm1d normalizes each feature across the batch and leaves per-sample magnitude
        # intact, whereas LayerNorm normalizes across features within a sample and erases the
        # overall level of the feature vector - two samples differing by a global offset become
        # identical, which caps how far predictions can move from the mean.
        self.fusion_norm = self._build_fusion_norm(fusion_input_dim)
        self.graph_blocks = nn.ModuleList(
            [
                ResidualGraphBlock(
                    fusion_input_dim,
                    edge_attr_dim=self.edge_attr_dim,
                    dropout=dropout,
                    use_layer_norm=self.use_layer_norm,
                )
                for _ in range(max(1, int(num_graph_layers)))
            ]
        )
        self.output_head = self._build_output_head(
            fusion_input_dim, self.head_hidden_dim, self.target_dim, self.head_num_layers, dropout
        )
        # NOTE: val_loss is only comparable across runs that share loss_name - it is the monitor for
        # early stopping, checkpoint selection and the LR scheduler.
        self.loss_name = str(loss_name).lower()
        self.huber_delta = float(huber_delta)
        self.loss_fn = self._build_loss_fn(self.loss_name, self.huber_delta)

    def _temporal_hidden(self, modality_name: str) -> int:
        """Per-modality LSTM width. A scalar config applies the same width to every modality."""
        configured = self.temporal_lstm_hidden_dim
        if isinstance(configured, Mapping):
            value = configured.get(modality_name, configured.get(str(modality_name).lower()))
            if value is None:
                # Silently defaulting here would hand a newly added modality the old oversized
                # width, undoing the per-branch sizing without any signal.
                raise ValueError(
                    f"Modality {modality_name!r} has no width in temporal_lstm_hidden_dim "
                    f"(configured: {sorted(configured)}). Add an entry for it, or use a scalar "
                    "value to apply one width to every modality."
                )
            return int(value)
        return int(configured)

    @staticmethod
    def _build_loss_fn(loss_name: str, huber_delta: float):
        if loss_name in {"mse", "l2"}:
            return nn.MSELoss()
        if loss_name == "huber":
            return nn.HuberLoss(delta=huber_delta)
        if loss_name in {"smooth_l1", "smoothl1"}:
            return nn.SmoothL1Loss(beta=huber_delta)
        raise ValueError(f"Unknown loss_name '{loss_name}'; expected one of mse, huber, smooth_l1")

    def _build_fusion_norm(self, fusion_input_dim: int):
        if self.fusion_norm_type == "none" or fusion_input_dim <= 0:
            return nn.Identity()
        if self.fusion_norm_type == "batch":
            return nn.BatchNorm1d(fusion_input_dim)
        return nn.LayerNorm(fusion_input_dim)

    def _build_static_encoder(self, input_dim: int, hidden_dim: int, dropout: float):
        if input_dim <= 0:
            return nn.Identity()
        layers = [nn.Linear(input_dim, hidden_dim)]
        if self.use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers += [nn.ReLU(), nn.Dropout(dropout)]
        return nn.Sequential(*layers)

    def _build_output_head(self, input_dim: int, hidden_dim: int, target_dim: int, num_layers: int, dropout: float):
        """Fusion -> target. `num_layers=0` keeps the original single linear readout."""
        if num_layers <= 0:
            return nn.Linear(input_dim, target_dim)

        layers: list[nn.Module] = []
        dim = input_dim
        for index in range(num_layers):
            # Halve each layer, but never below the floor - an unbounded taper collapses a deep
            # head to a handful of dimensions and undoes the depth it is adding.
            width = max(self.head_min_hidden_dim, hidden_dim // (2 ** index))
            layers.append(nn.Linear(dim, width))
            is_last_block = index == num_layers - 1
            # No LayerNorm or Dropout on the block feeding the readout. LayerNorm there forces the
            # penultimate vector to unit variance, leaving the final Linear only its direction - and
            # magnitude is what a regressor needs to reach the tails. Dropout on the same vector is
            # minimized under MSE by shrinking the readout toward its bias, i.e. the target mean.
            if not is_last_block:
                if self.use_layer_norm:
                    layers.append(nn.LayerNorm(width))
                layers += [nn.ReLU(), nn.Dropout(dropout)]
            else:
                layers.append(nn.ReLU())
            dim = width
        layers.append(nn.Linear(dim, target_dim))
        return nn.Sequential(*layers)

    def _encode_static(self, x_static):
        if self.static_dim <= 0:
            batch_size = x_static.size(0)
            # Must match the width fusion_input_dim was computed from, not the generic hidden_dim.
            return torch.zeros((batch_size, self.static_hidden_dim), device=x_static.device, dtype=x_static.dtype)
        return self.static_encoder(x_static)

    def _encode_temporal(self, batch: Mapping[str, Any], device) -> Optional[Any]:
        if not self.temporal_enabled or not self.temporal_encoders:
            return None

        encoded_modalities = []
        temporal_features = _batch_get(batch, "temporal_features", {}) or {}
        temporal_lengths = _batch_get(batch, "temporal_lengths", {}) or {}
        temporal_masks = _batch_get(batch, "temporal_masks", {}) or {}
        for modality_name, encoder in self.temporal_encoders.items():
            modality_tensor = temporal_features.get(modality_name)
            if modality_tensor is None:
                continue
            modality_tensor = modality_tensor.to(device)
            if modality_tensor.ndim == 2:
                modality_tensor = modality_tensor.unsqueeze(1)
            if modality_tensor.ndim != 3:
                raise ValueError(
                    f"Temporal modality '{modality_name}' must have rank 2 or 3, got {modality_tensor.ndim}"
                )

            step_mask = self._resolve_step_mask(modality_name, temporal_masks, modality_tensor, device)
            lengths_tensor = self._resolve_temporal_lengths(modality_name, temporal_lengths, step_mask, device)

            if step_mask is not None:
                empty_mask = ~step_mask.any(dim=1)
            elif lengths_tensor is not None:
                empty_mask = lengths_tensor <= 0
            else:
                empty_mask = None

            outputs, hidden_state = self._run_lstm(encoder, modality_tensor, step_mask, lengths_tensor)

            if self.temporal_pooling == "attention":
                temporal_embedding = self._attention_pool(modality_name, outputs, step_mask, lengths_tensor, device)
            else:
                temporal_embedding = self._last_hidden_state(encoder, hidden_state, outputs, step_mask)

            if empty_mask is not None and bool(empty_mask.any()):
                temporal_embedding = temporal_embedding.clone()
                temporal_embedding[empty_mask] = 0.0

            encoded_modalities.append(temporal_embedding)

        if not encoded_modalities:
            return None

        return torch.cat(encoded_modalities, dim=-1)

    @staticmethod
    def _resolve_step_mask(modality_name, temporal_masks, modality_tensor, device):
        """The (batch, time) boolean observation mask, or None when the batch carries no mask."""
        mask_value = temporal_masks.get(modality_name)
        if mask_value is None:
            return None
        mask_tensor = torch.as_tensor(mask_value, device=device)
        if mask_tensor.ndim != 2 or mask_tensor.shape != modality_tensor.shape[:2]:
            return None
        return mask_tensor.to(dtype=torch.bool)

    @staticmethod
    def _resolve_temporal_lengths(modality_name, temporal_lengths, step_mask, device):
        """Observed-step counts. Only a valid sequence length when observations are a dense prefix."""
        lengths_value = temporal_lengths.get(modality_name)
        if lengths_value is not None:
            return torch.as_tensor(lengths_value, device=device, dtype=torch.long)
        if step_mask is not None:
            return step_mask.to(dtype=torch.long).sum(dim=1)
        return None

    def _run_lstm(self, encoder, modality_tensor, step_mask, lengths_tensor):
        # With a per-timestep mask the whole sequence is run: observations sit at absolute positions
        # on the shared date axis, so packing to the observed COUNT would truncate every point whose
        # readings are not a dense prefix. The datamodule has already zeroed the gaps.
        if step_mask is not None:
            outputs, (hidden_state, _) = encoder(modality_tensor)
            return outputs, hidden_state

        if lengths_tensor is not None:
            safe_lengths = lengths_tensor.clamp_min(1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                modality_tensor,
                lengths=safe_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_outputs, (hidden_state, _) = encoder(packed)
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_outputs,
                batch_first=True,
                total_length=modality_tensor.size(1),
            )
            return outputs, hidden_state

        outputs, (hidden_state, _) = encoder(modality_tensor)
        return outputs, hidden_state

    def _last_hidden_state(self, encoder, hidden_state, outputs=None, step_mask=None):
        # Masked case: the sequence ran to the end over zero-filled gaps, so the final hidden state
        # is not the last observation. Read the output at each row's last observed index instead.
        if step_mask is not None and outputs is not None:
            last_index = self._last_observed_index(step_mask)
            gather_index = last_index.view(-1, 1, 1).expand(-1, 1, outputs.size(-1))
            return outputs.gather(1, gather_index).squeeze(1)

        directions = 2 if encoder.bidirectional else 1
        hidden_layers = hidden_state.view(
            encoder.num_layers,
            directions,
            hidden_state.size(1),
            encoder.hidden_size,
        )
        last_layer_hidden = hidden_layers[-1]
        if directions == 2:
            return torch.cat([last_layer_hidden[0], last_layer_hidden[1]], dim=-1)
        return last_layer_hidden[0]

    @staticmethod
    def _last_observed_index(step_mask):
        """Index of the last True per row; 0 for all-False rows (their embedding is zeroed anyway)."""
        time_steps = step_mask.size(1)
        flipped = torch.flip(step_mask.to(dtype=torch.long), dims=[1])
        last_index = time_steps - 1 - flipped.argmax(dim=1)
        return torch.where(step_mask.any(dim=1), last_index, torch.zeros_like(last_index))

    def _attention_pool(self, modality_name, outputs, step_mask, lengths_tensor, device):
        scores = self.temporal_attention[modality_name](outputs).squeeze(-1)

        valid_mask = step_mask
        if valid_mask is None and lengths_tensor is not None:
            time_steps = outputs.size(1)
            valid_mask = torch.arange(time_steps, device=device).unsqueeze(0) < lengths_tensor.unsqueeze(1)

        if valid_mask is None:
            return torch.sum(outputs * torch.softmax(scores, dim=1).unsqueeze(-1), dim=1)

        has_observation = valid_mask.any(dim=1)
        scores = scores.masked_fill(~valid_mask, float("-inf"))
        # A row with nothing observed would softmax over all -inf and produce NaN; give it a uniform
        # distribution and zero the pooled vector afterwards.
        scores = scores.masked_fill(~has_observation.unsqueeze(1), 0.0)

        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = torch.sum(outputs * weights, dim=1)
        return pooled * has_observation.to(dtype=pooled.dtype).unsqueeze(-1)

    def _move_batch_to_device(self, batch: Any, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device=device)
        if isinstance(batch, Mapping):
            moved_batch = {}
            for key, value in batch.items():
                moved_batch[key] = self._move_batch_to_device(value, device=device)
            return moved_batch
        if isinstance(batch, list):
            return [self._move_batch_to_device(value, device=device) for value in batch]
        if hasattr(batch, "keys") and callable(getattr(batch, "keys")):
            moved_batch = {}
            for key in batch.keys():
                if hasattr(batch, "get"):
                    value = batch.get(key)
                else:
                    value = getattr(batch, key, None)
                moved_batch[key] = self._move_batch_to_device(value, device=device)
            return moved_batch
        return batch

    def forward(self, batch: Mapping[str, Any]):
        x_static = _batch_get(batch, "x_static")

        if x_static is None:
            raise KeyError("Graph batch is missing 'x_static'")

        device = next(self.parameters()).device
        batch_for_model = self._move_batch_to_device(batch, device=device)
        x_static = _batch_get(batch_for_model, "x_static")
        edge_index = _batch_get(batch_for_model, "edge_index")
        edge_attr = _batch_get(batch_for_model, "edge_attr")

        node_features = self._encode_static(x_static)
        temporal_context = self._encode_temporal(batch_for_model, device=node_features.device)
        if temporal_context is not None:
            node_features = torch.cat([node_features, temporal_context], dim=-1)
        node_features = self.fusion_norm(node_features)

        if self.spatial_graph_enabled:
            for block in self.graph_blocks:
                node_features = block(node_features, edge_index, edge_attr=edge_attr)

        return self.output_head(node_features)

    def _shared_step(self, batch: Mapping[str, Any], stage: str):
        predictions = self.forward(batch)
        targets = _batch_get(batch, "y")
        if targets is None:
            raise KeyError("Graph batch is missing 'y'")
        if not torch.isfinite(targets).all():
            raise ValueError(f"Non-finite target values encountered during {stage} step.")
        if not torch.isfinite(predictions).all():
            raise ValueError(f"Non-finite prediction values encountered during {stage} step.")
        loss = self.loss_fn(predictions, targets)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss encountered during {stage} step.")
        # The real node count, not 1: Lightning weights the epoch mean by batch_size, so a constant
        # 1 makes the trailing partial batch count as much as a full one. This metric drives early
        # stopping, checkpoint selection and the LR scheduler.
        self.log(
            f"{stage}_loss",
            loss,
            batch_size=int(targets.shape[0]) if targets.ndim else 1,
            prog_bar=stage != "train",
            on_step=False,
            on_epoch=True,
        )
        return loss

    def training_step(self, batch: Mapping[str, Any], batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int):
        return self._shared_step(batch, "val")

    def test_step(self, batch: Mapping[str, Any], batch_idx: int):
        return self._shared_step(batch, "test")

    def predict_step(self, batch: Mapping[str, Any], batch_idx: int, dataloader_idx: int = 0):
        return self.inverse_transform_targets(self.forward(batch))

    def inverse_transform_targets(self, predictions):
        """Map standardized predictions back to the target's original units.

        Un-standardize first, then undo log1p: the datamodule fits the standardization stats on
        already-transformed targets, so the two must be inverted in the opposite order.
        """
        if bool(self.targets_are_standardized):
            predictions = (
                predictions * self.target_scale.to(predictions.device)
                + self.target_mean.to(predictions.device)
            )
        if bool(self.targets_are_log1p):
            # Mirrors LogTransformer in yg_eo_soilnet.utils: forward is 10 * log1p(y).
            predictions = torch.expm1(predictions / 10.0)
        return predictions

    def configure_optimizers(self):
        if self.optimizer_name in {"adamw", "adam_w"}:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.scheduler_type in {"plateau", "reducelronplateau", "reduce_on_plateau"}:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                min_lr=self.scheduler_min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.scheduler_monitor,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return optimizer