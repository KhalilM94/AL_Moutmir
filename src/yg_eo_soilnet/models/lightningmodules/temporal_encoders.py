"""Encoders for irregular, date-stamped observation sequences.

Every module here consumes ``(values, mask, times)`` where ``times`` are decimal years attached to
real observations, and returns one fixed-size embedding per point. Two invariants hold throughout,
and both are what let a trained checkpoint run on a time window it never saw:

* **No parameter is sized by sequence length.** Nothing indexes a per-step table, so the same
  instance runs on a 12-token batch and a 90-token batch.
* **No feature encodes an absolute epoch.** Time enters only as position-within-the-year, as an
  offset from each point's own latest observation, and as the gap since its previous observation.
  Shifting every date by ten years leaves the computed features unchanged.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

# How many time-feature channels `sequence_time_features` appends per token.
TIME_FEATURE_DIM = 4


def sequence_time_features(
    times: torch.Tensor,
    mask: torch.Tensor,
    *,
    span_cap_years: float = 8.0,
    delta_cap_months: float = 24.0,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Per-token time features from decimal years: ``(B, L) -> (B, L, 4)``.

    Columns are ``[sin(year fraction), cos(year fraction), relative age, gap since previous]``:

    * the sine/cosine pair carries seasonality and repeats every calendar year, so February 2018 and
      February 2033 land on the same point of the circle;
    * ``relative age`` is the offset from *this point's own latest observation*, scaled by
      ``span_cap_years`` into ``[-1, 0]`` - a position within the point's own history rather than a
      position on any calendar;
    * ``gap`` is the months since the previous observation, log-compressed into ``[0, 1]``, which is
      what tells the network that a reading follows a five-month hole.

    Padding is zeroed, and a point with no observations yields all zeros.

    Computed in float64 regardless of the incoming dtype: subtracting ``floor(times)`` off a value
    near 2020 cancels four significant digits, and in float32 what survives is coarser than the
    monthly spacing this is meant to resolve. ``dtype`` selects the returned precision.
    """
    if times.ndim != 2 or mask.shape != times.shape:
        raise ValueError(
            f"times and mask must both be (batch, length) and agree; got {tuple(times.shape)} and {tuple(mask.shape)}"
        )

    output_dtype = dtype if dtype is not None else times.dtype
    times = times.to(dtype=torch.float64)
    mask = mask.to(dtype=torch.bool)
    float_mask = mask.to(dtype=times.dtype)
    span_cap_years = max(float(span_cap_years), 1e-6)
    delta_cap_months = max(float(delta_cap_months), 1e-6)

    # Position within the calendar year. Computed on the raw decimal year, whose integer part is the
    # calendar year and whose fractional part is exactly the position within it.
    year_fraction = times - torch.floor(times)
    sin_year = torch.sin(2.0 * math.pi * year_fraction)
    cos_year = torch.cos(2.0 * math.pi * year_fraction)

    # Each point's own latest observation is the origin. Using a fixed epoch here instead is what
    # would weld the model to one date range.
    neg_inf = torch.finfo(times.dtype).min
    latest = times.masked_fill(~mask, neg_inf).max(dim=1, keepdim=True).values
    has_observation = mask.any(dim=1, keepdim=True)
    latest = torch.where(has_observation, latest, torch.zeros_like(latest))
    relative_age = ((times - latest) / span_cap_years).clamp(-1.0, 0.0)

    # Tokens are consecutive observations, so the gap is a plain first difference. The first token
    # has no predecessor and takes the cap, the honest encoding of "unknown, and long".
    gap_years = torch.empty_like(times)
    gap_years[:, :1] = delta_cap_months / 12.0
    if times.size(1) > 1:
        gap_years[:, 1:] = times[:, 1:] - times[:, :-1]
    gap_months = (gap_years * 12.0).clamp(0.0, delta_cap_months)
    gap_normalized = torch.log1p(gap_months) / math.log1p(delta_cap_months)

    features = torch.stack([sin_year, cos_year, relative_age, gap_normalized], dim=-1)
    return (features * float_mask.unsqueeze(-1)).to(dtype=output_dtype)


def compact_observations(
    values: torch.Tensor, mask: torch.Tensor, times: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move real observations to the front of each row, preserving their order.

    The datamodule already right-pads, so this is usually a no-op reordering. It exists because two
    things downstream would be silently wrong on a scattered mask, and "silently" is the problem:

    * ``pack_padded_sequence`` takes the *first* ``mask.sum(1)`` tokens, so any observation sitting
      past that many slots would be dropped without a word - the same count-as-length mistake that
      previously turned this project's temporal branch into noise;
    * the gap feature is a first difference between adjacent slots, which is only the gap between
      consecutive observations once the observations are adjacent.

    Compacting makes the mask a dense prefix by construction, so neither can happen regardless of
    how a caller assembled the batch.
    """
    order = torch.argsort((~mask).to(torch.uint8), dim=1, stable=True)
    gather_values = order.unsqueeze(-1).expand(-1, -1, values.size(-1))
    return values.gather(1, gather_values), mask.gather(1, order), times.gather(1, order)


def masked_pool(outputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Concatenate masked mean, masked max and the last real token: ``(B, L, H) -> (B, 3H)``."""
    float_mask = mask.to(dtype=outputs.dtype).unsqueeze(-1)
    counts = float_mask.sum(dim=1).clamp_min(1.0)
    mean_pooled = (outputs * float_mask).sum(dim=1) / counts

    max_pooled = outputs.masked_fill(~mask.unsqueeze(-1), torch.finfo(outputs.dtype).min).max(dim=1).values
    has_observation = mask.any(dim=1, keepdim=True)
    max_pooled = torch.where(has_observation, max_pooled, torch.zeros_like(max_pooled))

    last_index = last_observed_index(mask)
    gather_index = last_index.view(-1, 1, 1).expand(-1, 1, outputs.size(-1))
    last_pooled = outputs.gather(1, gather_index).squeeze(1)

    return torch.cat([mean_pooled, max_pooled, last_pooled], dim=-1)


def last_observed_index(mask: torch.Tensor) -> torch.Tensor:
    """Index of the last True per row; 0 for all-False rows, whose embedding is zeroed anyway."""
    length = mask.size(1)
    flipped = torch.flip(mask.to(dtype=torch.long), dims=[1])
    index = length - 1 - flipped.argmax(dim=1)
    return torch.where(mask.any(dim=1), index, torch.zeros_like(index))


class Time2Vec(nn.Module):
    """Continuous positional encoding: one linear term plus ``k-1`` learned sinusoids.

    Applied to the *relative* age, never to a raw decimal year, so the learned frequencies describe
    spacing rather than a calendar era. The parameter count depends only on ``embed_dim``, which is
    what keeps the encoder length-agnostic.
    """

    def __init__(self, embed_dim: int = 8):
        super().__init__()
        if embed_dim < 1:
            raise ValueError(f"Time2Vec embed_dim must be >= 1, got {embed_dim}")
        self.embed_dim = int(embed_dim)
        self.linear_weight = nn.Parameter(torch.randn(1) * 0.1)
        self.linear_bias = nn.Parameter(torch.zeros(1))
        self.periodic_weight = nn.Parameter(torch.randn(self.embed_dim - 1) * 2.0 * math.pi)
        self.periodic_bias = nn.Parameter(torch.zeros(self.embed_dim - 1))

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        times = times.unsqueeze(-1)
        linear = times * self.linear_weight + self.linear_bias
        if self.embed_dim == 1:
            return linear
        periodic = torch.sin(times * self.periodic_weight + self.periodic_bias)
        return torch.cat([linear, periodic], dim=-1)


class TimeAwareLSTMEncoder(nn.Module):
    """Option 1: an LSTM over observation tokens carrying explicit calendar and gap channels.

    Packing by ``mask.sum(1)`` is correct *here* and only here: in this representation every
    unmasked token is a real observation, so the count is a genuine length. The dense-axis graph
    encoder must never do this - there the count is the number of readings scattered across a fixed
    axis, and packing to it would silently discard everything past that many steps.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        pooling: str = "mean_max_last",
        span_cap_years: float = 8.0,
        delta_cap_months: float = 24.0,
    ):
        super().__init__()
        self.pooling = str(pooling).lower()
        if self.pooling not in {"mean_max_last", "last", "attention"}:
            raise ValueError(
                f"pooling must be 'mean_max_last', 'last' or 'attention', got {pooling!r}"
            )
        self.span_cap_years = float(span_cap_years)
        self.delta_cap_months = float(delta_cap_months)

        self.lstm = nn.LSTM(
            input_size=int(input_dim) + TIME_FEATURE_DIM,
            hidden_size=int(hidden_dim),
            num_layers=max(1, int(num_layers)),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=True,
            bidirectional=bool(bidirectional),
        )

        state_dim = int(hidden_dim) * (2 if bidirectional else 1)
        self.attention = nn.Linear(state_dim, 1) if self.pooling == "attention" else None
        pooled_dim = state_dim * 3 if self.pooling == "mean_max_last" else state_dim
        # Project so the branch width is set by output_dim rather than by the pooling choice.
        self.projection = nn.Linear(pooled_dim, int(output_dim))
        self.output_dim = int(output_dim)

    def forward(self, values: torch.Tensor, mask: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=torch.bool)
        values, mask, times = compact_observations(values, mask, times)
        time_features = sequence_time_features(
            times,
            mask,
            span_cap_years=self.span_cap_years,
            delta_cap_months=self.delta_cap_months,
            dtype=values.dtype,
        )
        tokens = torch.cat([values, time_features], dim=-1)

        lengths = mask.sum(dim=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            tokens,
            lengths=lengths.clamp_min(1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=tokens.size(1)
        )

        if self.pooling == "mean_max_last":
            pooled = masked_pool(outputs, mask)
        elif self.pooling == "last":
            index = last_observed_index(mask)
            gather_index = index.view(-1, 1, 1).expand(-1, 1, outputs.size(-1))
            pooled = outputs.gather(1, gather_index).squeeze(1)
        else:
            pooled = self._attention_pool(outputs, mask)

        embedding = self.projection(pooled)
        # A point with nothing observed contributes nothing rather than a bias-shaped artefact.
        return embedding * mask.any(dim=1, keepdim=True).to(dtype=embedding.dtype)

    def _attention_pool(self, outputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attention(outputs).squeeze(-1)
        has_observation = mask.any(dim=1)
        scores = scores.masked_fill(~mask, float("-inf"))
        # An empty row would softmax over all -inf and produce NaN; give it a uniform distribution
        # and zero the result afterwards.
        scores = scores.masked_fill(~has_observation.unsqueeze(1), 0.0)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (outputs * weights).sum(dim=1)
        return pooled * has_observation.to(dtype=pooled.dtype).unsqueeze(-1)


class TemporalTransformerEncoder(nn.Module):
    """Option 2: a Transformer encoder treating each observation as a token.

    ``src_key_padding_mask`` means attention never attends to padding, and Time2Vec on the relative
    age means the spacing between tokens is read from the timestamps rather than assumed uniform.
    A learned ``[CLS]`` token pools the variable-length sequence into one fixed-size embedding, and
    is never masked - so even a point with zero observations has one valid key and cannot produce a
    NaN softmax.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 48,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        time2vec_dim: int = 8,
        span_cap_years: float = 8.0,
        delta_cap_months: float = 24.0,
    ):
        super().__init__()
        d_model = int(d_model)
        nhead = int(nhead)
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model must be divisible by nhead for multi-head attention; got d_model={d_model} "
                f"and nhead={nhead}"
            )
        self.span_cap_years = float(span_cap_years)
        self.delta_cap_months = float(delta_cap_months)

        # The value projection sees the seasonal pair and the gap; absolute-ish position enters
        # separately through Time2Vec, additively, as a positional encoding should.
        self.value_projection = nn.Linear(int(input_dim) + 3, d_model)
        self.time2vec = Time2Vec(int(time2vec_dim))
        self.time_projection = nn.Linear(int(time2vec_dim), d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is incompatible with norm_first and would only warn; say so explicitly.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=max(1, int(num_layers)), enable_nested_tensor=False
        )
        self.projection = nn.Linear(d_model, int(output_dim))
        self.output_dim = int(output_dim)

    def forward(self, values: torch.Tensor, mask: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=torch.bool)
        # Attention is order-invariant given the positional encoding, but the gap feature is a first
        # difference between adjacent slots, so observations must be adjacent for it to mean anything.
        values, mask, times = compact_observations(values, mask, times)
        time_features = sequence_time_features(
            times,
            mask,
            span_cap_years=self.span_cap_years,
            delta_cap_months=self.delta_cap_months,
            dtype=values.dtype,
        )
        sin_year, cos_year, relative_age, gap = time_features.unbind(dim=-1)

        tokens = self.value_projection(
            torch.cat([values, sin_year.unsqueeze(-1), cos_year.unsqueeze(-1), gap.unsqueeze(-1)], dim=-1)
        )
        tokens = tokens + self.time_projection(self.time2vec(relative_age))

        batch_size = tokens.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls, tokens], dim=1)
        # False = attend. The CLS column is always attendable, which is what keeps an
        # observation-free row from softmaxing over an entirely masked set of keys.
        padding_mask = torch.cat(
            [torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device), ~mask], dim=1
        )

        encoded = self.encoder(sequence, src_key_padding_mask=padding_mask)
        embedding = self.projection(encoded[:, 0])
        return embedding * mask.any(dim=1, keepdim=True).to(dtype=embedding.dtype)


class GatedFusion(nn.Module):
    """Blend the static and dynamic branches with a learned per-feature gate.

    The gate lets the network decide, per feature and per point, how much to trust the sequence
    against the site covariates - which matters when observation counts vary by an order of
    magnitude across points.
    """

    def __init__(self, fusion_dim: int):
        super().__init__()
        self.gate = nn.Linear(2 * int(fusion_dim), int(fusion_dim))

    def forward(self, static_features: torch.Tensor, temporal_features: Optional[torch.Tensor]) -> torch.Tensor:
        if temporal_features is None:
            return static_features
        gate = torch.sigmoid(self.gate(torch.cat([static_features, temporal_features], dim=-1)))
        return gate * temporal_features + (1.0 - gate) * static_features
