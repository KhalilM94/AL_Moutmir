"""Reusable tabular blocks: learned entity embeddings plus a static feature encoder.

Plain ``nn.Module``s, deliberately not ``LightningModule``s, for the same reason
``temporal_encoders`` and ``temporal_cnn_encoders`` are: the sequence model, the calendar-grid CNN
and any future tabular model all need this block, and a LightningModule cannot be composed into
three different parents. Everything training-related - loss, optimizer, target inversion - stays in
``SoilRegressionLightningBase``.

The counterpart on the data side is ``datamodules.categorical``, which produces the integer codes
these embeddings look up. The contract between them is narrow on purpose: an ``(B, K)`` int64 tensor
whose column *i* holds values in ``[0, cardinalities[i])``, with 0 reserved for unknown and missing.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn

from yg_eo_soilnet.models.lightningmodules.mlp import build_mlp_stack


#: Widest embedding the heuristic will produce, regardless of cardinality.
DEFAULT_EMBEDDING_MAX_DIM = 50


def embedding_dim_for(cardinality: int, max_dim: int = DEFAULT_EMBEDDING_MAX_DIM) -> int:
    """Standard entity-embedding heuristic: ``min(max_dim, (cardinality + 1) // 2)``.

    Half the cardinality gives a small column a nearly one-hot-sized space and forces a large one to
    share structure, which is the whole point of embedding it rather than one-hot encoding it. The
    cap stops a high-cardinality column from dominating the concatenated static vector.

    ``cardinality`` is the embedding's row count - the fitted vocabulary **plus** its reserved OOV
    slot - so this matches what ``CategoricalEncoder.cardinalities`` reports.
    """
    cardinality = int(cardinality)
    if cardinality <= 0:
        raise ValueError(f"cardinality must be positive, got {cardinality}")
    return max(1, min(int(max_dim), (cardinality + 1) // 2))


def resolve_embedding_dims(
    cardinalities: Sequence[int],
    embedding_dims: Any = None,
    *,
    max_dim: int = DEFAULT_EMBEDDING_MAX_DIM,
    feature_names: Optional[Sequence[str]] = None,
) -> list[int]:
    """Per-column embedding widths from ``auto``, a scalar, a per-column sequence or a mapping.

    Mirrors the ``_per_modality_value`` contract the sequence model already uses for its temporal
    branch: a scalar applies one width everywhere, and anything per-column must cover every column
    rather than silently defaulting the ones it forgot.
    """
    cardinalities = [int(value) for value in cardinalities]
    names = list(feature_names) if feature_names is not None else [str(i) for i in range(len(cardinalities))]

    if embedding_dims is None or (isinstance(embedding_dims, str) and embedding_dims.lower() == "auto"):
        return [embedding_dim_for(cardinality, max_dim) for cardinality in cardinalities]

    if isinstance(embedding_dims, Mapping):
        resolved = []
        for name, cardinality in zip(names, cardinalities):
            value = embedding_dims.get(name, embedding_dims.get(str(name).lower()))
            if value is None:
                raise ValueError(
                    f"Categorical feature {name!r} has no width in embedding_dims "
                    f"(configured: {sorted(embedding_dims)}). Add an entry for it, or use a scalar "
                    "to apply one width to every feature."
                )
            resolved.append(int(value))
        return resolved

    if isinstance(embedding_dims, (int, float)) and not isinstance(embedding_dims, bool):
        return [int(embedding_dims)] * len(cardinalities)

    resolved = [int(value) for value in embedding_dims]
    if len(resolved) != len(cardinalities):
        raise ValueError(
            f"embedding_dims has {len(resolved)} entry/ies but there are {len(cardinalities)} "
            f"categorical feature(s) ({', '.join(names) or 'none'})"
        )
    return resolved


class EntityEmbeddingBlock(nn.Module):
    """One ``nn.Embedding`` per categorical feature, concatenated into a single vector.

    Individual tables rather than one shared one: the columns have unrelated vocabularies, and a
    shared table would force ``texture_20cm`` and ``landform_class`` to compete for the same rows.
    """

    def __init__(
        self,
        cardinalities: Sequence[int],
        embedding_dims: Any = None,
        *,
        dropout: float = 0.0,
        max_dim: int = DEFAULT_EMBEDDING_MAX_DIM,
        feature_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.cardinalities = [int(value) for value in cardinalities]
        if any(cardinality <= 0 for cardinality in self.cardinalities):
            raise ValueError(f"Every cardinality must be positive, got {self.cardinalities}")

        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else [f"categorical_{index}" for index in range(len(self.cardinalities))]
        )
        if len(self.feature_names) != len(self.cardinalities):
            raise ValueError(
                f"Got {len(self.feature_names)} feature name(s) but "
                f"{len(self.cardinalities)} cardinality/ies"
            )

        self.embedding_dims = resolve_embedding_dims(
            self.cardinalities, embedding_dims, max_dim=max_dim, feature_names=self.feature_names
        )
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, dim)
            for cardinality, dim in zip(self.cardinalities, self.embedding_dims)
        )
        # Dropout on the concatenated vector, not per table: zeroing whole coordinates of the joint
        # categorical representation is the regularizer this is meant to be, whereas dropping inside
        # a single lookup just adds noise to one feature's code.
        self.dropout = nn.Dropout(float(dropout))
        self.output_dim = int(sum(self.embedding_dims))

    @property
    def num_features(self) -> int:
        return len(self.embeddings)

    def forward(self, x_categorical: torch.Tensor) -> torch.Tensor:
        """``(B, K)`` int64 indices -> ``(B, sum(embedding_dims))``."""
        if self.num_features == 0:
            return x_categorical.new_zeros((x_categorical.size(0), 0), dtype=torch.float32)

        if x_categorical.dim() != 2:
            raise ValueError(
                f"x_categorical must be 2-D (batch, features), got {x_categorical.dim()}-D"
            )
        if x_categorical.size(1) != self.num_features:
            raise ValueError(
                f"x_categorical has {x_categorical.size(1)} column(s) but this block embeds "
                f"{self.num_features} feature(s) ({', '.join(self.feature_names)})"
            )

        # Values are guaranteed in range by CategoricalEncoder (0 is the reserved slot), so there is
        # no bounds check here: it would cost a device sync on every batch to re-verify an invariant
        # the producing side already holds.
        indices = x_categorical.long()
        embedded = [
            embedding(indices[:, column]) for column, embedding in enumerate(self.embeddings)
        ]
        return self.dropout(torch.cat(embedded, dim=-1))


def _build_continuous_norm(kind: str, num_features: int) -> nn.Module:
    key = str(kind).lower()
    if key == "none":
        return nn.Identity()
    if key == "batch":
        return nn.BatchNorm1d(num_features)
    if key == "layer":
        return nn.LayerNorm(num_features)
    raise ValueError(f"continuous_norm must be 'none', 'batch' or 'layer', got {kind!r}")


class TabularStaticEncoder(nn.Module):
    """Entity embeddings + scaled continuous features -> one static representation.

    Replaces the near-identical private ``_build_static_encoder`` in the sequence and CNN modules.
    The two differ only in activation family and in whether they project to a separate fusion width,
    both of which are parameters here, so each keeps its current output shape exactly.

    ``output_dim=None`` means "no projection": the representation is ``hidden_dims[-1]`` wide. Passing
    an int appends a final ``Linear`` to that width.

    ``hidden_dims`` is a list because this block is an MLP like any other, not a fixed projection: a
    single width is ``[64]``, and a deeper static branch is ``[128, 64]``. Every block here carries
    its norm and its dropout, including the last - unlike an output head, nothing downstream is a
    readout whose magnitude they would erase.
    """

    def __init__(
        self,
        num_continuous: int,
        hidden_dims: Sequence[int],
        *,
        output_dim: Optional[int] = None,
        cardinalities: Sequence[int] = (),
        embedding_dims: Any = None,
        embedding_dropout: float = 0.0,
        embedding_max_dim: int = DEFAULT_EMBEDDING_MAX_DIM,
        feature_names: Optional[Sequence[str]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_layer_norm: bool = True,
        continuous_norm: str = "none",
    ):
        super().__init__()
        self.num_continuous = int(num_continuous)
        if self.num_continuous < 0:
            raise ValueError(f"num_continuous must be non-negative, got {num_continuous}")

        self.hidden_dims = [int(width) for width in hidden_dims]
        if not self.hidden_dims:
            raise ValueError(
                "TabularStaticEncoder needs at least one hidden width. A zero-layer static branch "
                "would feed the raw concatenation straight into the fusion, which is a different "
                "model rather than a smaller one."
            )
        self.embeddings = EntityEmbeddingBlock(
            cardinalities,
            embedding_dims,
            dropout=embedding_dropout,
            max_dim=embedding_max_dim,
            feature_names=feature_names,
        )

        self.input_dim = self.num_continuous + self.embeddings.output_dim
        if self.input_dim <= 0:
            raise ValueError(
                "TabularStaticEncoder needs at least one feature: got no continuous columns and no "
                "categorical ones. Callers with no static features should skip this encoder."
            )

        # Applied to the continuous block alone. The datamodule already standardizes it train-only,
        # so 'none' is the default; 'batch' is the classic entity-embedding recipe and is safe here
        # because the train loader drops a trailing batch of one.
        self.continuous_norm = (
            _build_continuous_norm(continuous_norm, self.num_continuous)
            if self.num_continuous > 0
            else nn.Identity()
        )

        self.encoder = build_mlp_stack(
            self.input_dim,
            self.hidden_dims,
            output_dim,
            dropout=float(dropout),
            use_layer_norm=use_layer_norm,
            activation=activation,
            # Every block is a full block: this branch feeds a fusion, not a readout, so there is no
            # magnitude for a trailing norm or dropout to strip.
            norm_final=True,
            dropout_final=True,
        )

        self.output_dim = self.hidden_dims[-1] if output_dim is None else int(output_dim)

    @property
    def embedding_dims(self) -> list[int]:
        return list(self.embeddings.embedding_dims)

    def forward(
        self, x_static: torch.Tensor, x_categorical: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """``(B, num_continuous)`` float + ``(B, K)`` int64 -> ``(B, output_dim)``."""
        if self.embeddings.num_features and x_categorical is None:
            raise KeyError(
                f"This encoder embeds {self.embeddings.num_features} categorical feature(s) "
                f"({', '.join(self.embeddings.feature_names)}) but the batch carried no "
                "'x_categorical'"
            )

        embedded = self.embeddings(x_categorical) if self.embeddings.num_features else None
        return self.forward_with_embedding(x_static, embedded)

    def forward_with_embedding(
        self, x_static: torch.Tensor, embedded: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Same as :meth:`forward` but taking an ALREADY-EMBEDDED categorical block.

        Split out for gradient-based attribution. Embedding lookups are indexed by int64 and are
        not differentiable with respect to their input, so an explainer cannot perturb
        ``x_categorical`` directly; it perturbs the embedding vectors instead and sums the
        attribution back over each feature's slice. Passing the embedding in is what makes that
        possible without duplicating this concatenation.
        """
        parts: list[torch.Tensor] = []
        if self.num_continuous > 0:
            parts.append(self.continuous_norm(x_static))
        if embedded is not None:
            parts.append(embedded.to(dtype=x_static.dtype))

        features = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        return self.encoder(features)
