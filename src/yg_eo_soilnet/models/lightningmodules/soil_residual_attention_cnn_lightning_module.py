from __future__ import annotations

from typing import Any, Sequence

from torch import nn

from yg_eo_soilnet.models.lightningmodules.soil_residual_cnn_lightning_module import (
    SoilResidualCNNLightningModule,
)
from yg_eo_soilnet.models.lightningmodules.temporal_cnn_encoders import AttentionFusion

STATIC_TOKEN_MODES = ("summary", "per_feature")


class SoilResidualAttentionCNNLightningModule(SoilResidualCNNLightningModule):
    """SoilResidualCNN with its ConcatGatedFusion replaced by attention over branch tokens.

        x_static ---------> static token(s) ---.
        CNN_m, one per modality -> m token ----+--> AttentionFusion --> fused --.
        HarmonicPositionEncoder -> loc token --'    (cls | mean | flatten)      |
        auxiliary lab block ----------------------------------------------------+--> head --> + base
        residual base block ----------------------------------------------------'

    Only the fusion changes. The base offset, the base and auxiliary blocks appended after the
    fusion, the variance head, the coverage guard and the serving columns are all inherited, so a
    comparison against soil_residual_cnn measures the fusion and nothing else.

    ``attention_static_tokens`` picks how the static covariates become tokens:

    * ``summary`` - TabularStaticEncoder's output is ONE token: the branch soil_residual_cnn already
      has, read by attention instead of by a gate.
    * ``per_feature`` - every continuous column is its own token and every categorical embedding
      another, FT-Transformer style. The encoder's MLP is replaced by an identity, which leaves
      ``static_encoder`` returning exactly the per-column block the tokens are cut from; its
      categorical embeddings and its ``continuous_norm`` are kept, so the two modes differ in
      tokenization only. ``static_hidden_dims`` has no effect in this mode.

    Replacing the MLP rather than bypassing the encoder is what keeps the attribution seam intact
    without a line of new code: ``forward`` reaches the encoder through ``static_encoder(...)`` and
    ``forward_from_parts`` through ``forward_with_embedding``, and both now return the same block.
    """

    def __init__(
        self,
        *,
        attention_static_tokens: str = "summary",
        attention_d_model: int = 64,
        attention_nhead: int = 4,
        attention_num_layers: int = 2,
        attention_ff_multiplier: int = 2,
        attention_dropout: float = 0.1,
        attention_readout: str = "cls",
        # Declared here, as the residual parent does, so the head can be rebuilt around the new
        # fusion from the values themselves. All three are forwarded to the parent unchanged.
        head_hidden_dims: Sequence[int] = (128, 64),
        head_norm_final: bool = False,
        dropout: float = 0.1,
        **kwargs: Any,
    ):
        # Coerced BEFORE save_hyperparameters() for the reason the base class documents: plain
        # builtins only in hyper_parameters, or the checkpoint stops loading under weights_only=True.
        attention_static_tokens = str(attention_static_tokens).lower()
        if attention_static_tokens not in STATIC_TOKEN_MODES:
            raise ValueError(
                f"attention_static_tokens must be one of {list(STATIC_TOKEN_MODES)}, "
                f"got {attention_static_tokens!r}"
            )
        attention_d_model = int(attention_d_model)
        attention_nhead = int(attention_nhead)
        attention_num_layers = int(attention_num_layers)
        attention_ff_multiplier = int(attention_ff_multiplier)
        attention_dropout = float(attention_dropout)
        attention_readout = str(attention_readout).lower()
        head_hidden_dims = [int(width) for width in head_hidden_dims]

        super().__init__(
            head_hidden_dims=head_hidden_dims,
            head_norm_final=head_norm_final,
            dropout=dropout,
            **kwargs,
        )
        # A third call, merging this frame into the two the parents saved, so the attention settings
        # round-trip through the checkpoint instead of reverting to their defaults on reload.
        self.save_hyperparameters()

        self.attention_static_tokens = attention_static_tokens
        if attention_static_tokens == "per_feature" and self.has_static_features:
            # Unused in this mode, and left in place it would ship dead weights in every checkpoint
            # and inflate every parameter count. The identity makes the encoder's output the raw
            # [continuous_norm(x_static), embedded] block, so its width is its input width.
            self.static_encoder.encoder = nn.Identity()
            self.static_encoder.output_dim = self.static_encoder.input_dim

        # The parent built a ConcatGatedFusion and a head sized to it; both are replaced here, which
        # is the "rebuild after super" pattern the residual head already follows.
        self.fusion = AttentionFusion(
            self._static_token_dims(),
            # ModuleDict order is the order _encode_temporal_from_grids concatenates in.
            [encoder.output_dim for encoder in self.temporal_encoders.values()],
            self.coordinate_output_dim,
            d_model=attention_d_model,
            nhead=attention_nhead,
            num_layers=attention_num_layers,
            ff_multiplier=attention_ff_multiplier,
            dropout=attention_dropout,
            readout=attention_readout,
        )
        self.output_head = self._build_output_head(head_hidden_dims, head_norm_final, dropout)

    def _static_token_dims(self) -> list[int]:
        """How the static vector reaching the fusion is cut into tokens.

        ``summary`` keeps the parent's behaviour exactly, including a zero-filled token when there
        are no static features at all. ``per_feature`` with no static features has nothing to cut,
        and the fusion then reads no static tokens.
        """
        if self.attention_static_tokens == "summary":
            return [self.static_hidden_dim]
        if not self.has_static_features:
            return []
        return [1] * self.static_dim + list(self.static_encoder.embedding_dims)
