"""Convolutional encoders over a rasterised calendar grid.

Convolution needs a regular lattice, but the observations arrive ragged and date-stamped. The
rasteriser here is what bridges the two: it scatters each reading into the (year, month) cell its
date names, leaving unfilled cells empty and flagged. Everything downstream then convolves over a
proper grid without any of it having been faked into existence.

Two properties are preserved from the sequence path and are load-bearing:

* **No parameter is sized by the grid.** Pooling is global, so the same weights run on a three-year
  grid and a twelve-year one.
* **No feature encodes an absolute epoch.** Rows are counted back from each point's own latest
  observation and columns are calendar months, so shifting every date by a decade changes nothing.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import nn

MONTHS_PER_YEAR = 12

# Day-of-year on which each month starts in a non-leap year, 1-indexed.
_MONTH_START_DAY = (1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)
_DAYS_PER_YEAR = 365.25


def decimal_year_to_month_index(times: torch.Tensor) -> torch.Tensor:
    """Decimal years -> calendar month index in [0, 11].

    Deliberately not ``floor(frac * 12)``, which is wrong wherever month lengths drift from
    365.25/12: 1 November is decimal .8323, and ``floor(.8323 * 12) == 9`` puts it in October. This
    converts back to a day-of-year and looks the month up.

    The table is leap-adjusted per year. Without that, every month from March on starts a day late in
    a leap year, so the last day of each of those months lands on the next month's tabulated
    boundary - 10 misplaced days per leap year. Harmless for the first-of-month stamps this dataset
    uses, wrong for anything sampled daily.
    """
    times = times.to(dtype=torch.float64)
    year = torch.floor(times)
    day_of_year = (times - year) * _DAYS_PER_YEAR + 1.0

    starts = torch.tensor(_MONTH_START_DAY, dtype=torch.float64, device=times.device)
    is_leap = (((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)).to(dtype=torch.float64)
    leap_shift = torch.zeros(MONTHS_PER_YEAR, dtype=torch.float64, device=times.device)
    leap_shift[2:] = 1.0  # March onward starts a day later once February has 29 days
    starts = starts + is_leap.unsqueeze(-1) * leap_shift

    # The tolerance is not cosmetic. to_decimal_year divides the day-of-year by 365.25 and this
    # multiplies it back, so 1 February round-trips to 31.999999... and would test as falling before
    # its own month's 32-day boundary - putting every first-of-month reading in the previous month.
    # Real day-of-year values are integers, so a 1e-6 slack cannot reach the next boundary.
    month = ((day_of_year.unsqueeze(-1) + 1e-6) >= starts).sum(dim=-1) - 1
    return month.clamp(0, MONTHS_PER_YEAR - 1).to(dtype=torch.long)


class CalendarGridRasterizer(nn.Module):
    """Ragged date-stamped observations -> a dense (years x months) calendar grid.

    ``grid_years`` rows are counted back from each point's **own** latest observation, so the grid
    describes a point's recent history rather than a fixed slice of the calendar. Columns are
    calendar months, which is what makes a convolution across the month axis mean "phenology"
    rather than "whatever twelve steps happened to follow each other".

    Output channels per modality, in order:
    ``C`` standardized readings, ``C`` per-channel validity, 1 cell-observed flag, and optionally
    2 month sin/cos. The positional pair earns its place: convolution is translation-equivariant
    along the month axis and global pooling discards position, so without it the network can learn
    the *shape* of a seasonal transition but never which column is January.
    """

    def __init__(
        self,
        num_channels: int,
        grid_years: Optional[int] = None,
        *,
        use_validity_channels: bool = True,
        month_positional: bool = True,
    ):
        super().__init__()
        self.num_channels = int(num_channels)
        self.grid_years = None if grid_years in (None, 0) else max(1, int(grid_years))
        self.use_validity_channels = bool(use_validity_channels)
        self.month_positional = bool(month_positional)

        if self.month_positional:
            months = torch.arange(MONTHS_PER_YEAR, dtype=torch.float32)
            angle = 2.0 * math.pi * months / MONTHS_PER_YEAR
            self.register_buffer("month_sin", torch.sin(angle), persistent=False)
            self.register_buffer("month_cos", torch.cos(angle), persistent=False)

    @property
    def output_channels(self) -> int:
        channels = self.num_channels + 1
        if self.use_validity_channels:
            channels += self.num_channels
        if self.month_positional:
            channels += 2
        return channels

    def channel_layout(self) -> dict[str, list[int]]:
        """Which output channel index carries what, in the order ``forward`` writes them.

        Defined here rather than reconstructed by callers because this class is the only thing that
        decides the order, and an explainer that guessed it wrong would attribute a band's
        importance to a different band - a mistake that produces a plausible-looking plot instead of
        an error. The keys mirror the class docstring: ``values`` then ``validity`` then the
        cell-observed flag then the optional month sin/cos pair.
        """
        layout: dict[str, list[int]] = {"values": list(range(self.num_channels))}
        cursor = self.num_channels

        if self.use_validity_channels:
            layout["validity"] = list(range(cursor, cursor + self.num_channels))
            cursor += self.num_channels
        else:
            layout["validity"] = []

        layout["cell_observed"] = [cursor]
        cursor += 1

        layout["month_positional"] = list(range(cursor, cursor + 2)) if self.month_positional else []
        return layout

    def resolve_grid_years(self, times: torch.Tensor, mask: torch.Tensor) -> int:
        """The configured span, or the batch's own longest history when none was configured."""
        if self.grid_years is not None:
            return self.grid_years
        if not bool(mask.any()):
            return 1
        years = torch.floor(times.to(dtype=torch.float64))
        largest = years.masked_fill(~mask, float("-inf")).max(dim=1).values
        smallest = years.masked_fill(~mask, float("inf")).min(dim=1).values
        spans = (largest - smallest + 1.0)[mask.any(dim=1)]
        return max(1, int(spans.max().item()))

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        times: torch.Tensor,
        validity: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B,L,C) -> grid (B, output_channels, Y, 12)`` plus ``cell_mask (B, Y, 12)``."""
        batch_size, length, channels = values.shape
        if channels != self.num_channels:
            raise ValueError(
                f"CalendarGridRasterizer was built for {self.num_channels} channel(s) but received {channels}"
            )
        mask = mask.to(dtype=torch.bool)
        device = values.device
        grid_years = self.resolve_grid_years(times, mask)
        cells = grid_years * MONTHS_PER_YEAR

        if validity is None:
            validity = torch.ones_like(values, dtype=torch.bool)
        validity = validity.to(dtype=torch.bool) & mask.unsqueeze(-1)

        years = torch.floor(times.to(dtype=torch.float64))
        latest = years.masked_fill(~mask, float("-inf")).max(dim=1, keepdim=True).values
        latest = torch.where(mask.any(dim=1, keepdim=True), latest, torch.zeros_like(latest))

        row = (years - latest).to(dtype=torch.long) + (grid_years - 1)
        column = decimal_year_to_month_index(times)
        flat = row * MONTHS_PER_YEAR + column

        # Observations older than the window, and padding, are routed to a trailing bin that is
        # sliced off below. Clamping them into a real cell instead would fabricate readings.
        keep = mask & (row >= 0) & (row < grid_years)
        flat = torch.where(keep, flat, torch.full_like(flat, cells))

        index = flat.unsqueeze(1).expand(-1, channels, -1)
        source = (values * keep.unsqueeze(-1)).transpose(1, 2)

        totals = torch.zeros(batch_size, channels, cells + 1, device=device, dtype=values.dtype)
        totals.scatter_add_(2, index, source)

        counts = torch.zeros(batch_size, 1, cells + 1, device=device, dtype=values.dtype)
        counts.scatter_add_(2, flat.unsqueeze(1), keep.unsqueeze(1).to(dtype=values.dtype))

        valid_totals = torch.zeros(batch_size, channels, cells + 1, device=device, dtype=values.dtype)
        valid_totals.scatter_add_(2, index, (validity & keep.unsqueeze(-1)).to(dtype=values.dtype).transpose(1, 2))

        totals, counts, valid_totals = totals[..., :cells], counts[..., :cells], valid_totals[..., :cells]

        # Repeated readings in one cell average. This never fires on the current dataset - the
        # (point, year, month) scatter is injective there - but the encoder must not depend on that.
        occupancy = counts.clamp_min(1.0)
        readings = totals / occupancy
        cell_mask = counts.squeeze(1) > 0

        layers = [readings]
        if self.use_validity_channels:
            layers.append(valid_totals / occupancy)
        layers.append(cell_mask.unsqueeze(1).to(dtype=values.dtype))

        grid = torch.cat(layers, dim=1).view(batch_size, -1, grid_years, MONTHS_PER_YEAR)

        if self.month_positional:
            positional = torch.stack([self.month_sin, self.month_cos]).to(dtype=grid.dtype, device=device)
            positional = positional.view(1, 2, 1, MONTHS_PER_YEAR).expand(batch_size, -1, grid_years, -1)
            grid = torch.cat([grid, positional], dim=1)

        return grid, cell_mask.view(batch_size, grid_years, MONTHS_PER_YEAR)


def masked_global_pool(features: torch.Tensor, cell_mask: torch.Tensor, *, masked: bool = True) -> torch.Tensor:
    """Global average over the trailing spatial dims: ``(B, C, *grid) -> (B, C)``.

    With ``masked`` the average runs over occupied cells only. Averaging over the whole grid instead
    scales a point's embedding by its coverage, so a point with 55 of 108 cells filled arrives at
    roughly half the magnitude of a full one - a systematic pull toward zero, and therefore toward
    the target mean, for exactly the sparse points a regressor already struggles with.
    """
    spatial_dims = tuple(range(2, features.ndim))
    if not masked:
        return features.mean(dim=spatial_dims)

    weights = cell_mask.to(dtype=features.dtype).unsqueeze(1)
    occupied = weights.sum(dim=spatial_dims)
    pooled = (features * weights).sum(dim=spatial_dims) / occupied.clamp_min(1.0)
    return pooled * (occupied > 0).to(dtype=features.dtype)


def _build_norm(kind: str, num_features: int, dims: int) -> nn.Module:
    kind = str(kind).lower()
    if kind == "none":
        return nn.Identity()
    if kind == "group":
        # Batch statistics are polluted by the empty cells that padding leaves behind; GroupNorm is
        # computed per sample and sidesteps that entirely.
        return nn.GroupNorm(num_groups=math.gcd(8, num_features) or 1, num_channels=num_features)
    if kind == "batch":
        return nn.BatchNorm1d(num_features) if dims == 1 else nn.BatchNorm2d(num_features)
    raise ValueError(f"norm must be 'batch', 'group' or 'none', got {kind!r}")


class _MaskedConvStack(nn.Module):
    """Convolution stages with the empty cells re-zeroed after each one.

    Without this an empty cell is not neutral: a convolution has a bias, so an unoccupied cell emits
    ``activation(bias)`` and its occupied neighbours read that as signal on the next layer. Two
    consequences, both bad. A gap in the record starts contributing a learned constant, which is the
    "missing looks like data" failure this whole representation exists to avoid. And an occupied cell
    at the edge of the data sees ``activation(bias)`` from an interior empty neighbour but a true zero
    from outside the tensor, so simply widening the grid with empty years changes the answer.

    Re-zeroing makes an interior empty cell behave exactly like the padding beyond the tensor edge,
    which is what lets the grid span be inferred rather than frozen.
    """

    def __init__(self, stages: list[nn.Module], mask_between: bool = True):
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.mask_between = bool(mask_between)

    def forward(self, features: torch.Tensor, cell_mask: torch.Tensor) -> torch.Tensor:
        weights = cell_mask.unsqueeze(1).to(dtype=features.dtype)
        if self.mask_between:
            features = features * weights
        for stage in self.stages:
            features = stage(features)
            if self.mask_between:
                features = features * weights
        return features


def _validate_hidden_dims(hidden_dims: Sequence[int], owner: str) -> list[int]:
    """One width per block, walked in order. The list IS the stack - there is no separate count.

    A conv stack conventionally widens as it pools, which a single shared width could not express;
    `[64, 128]` can. An empty list is refused rather than treated as zero blocks: that would pool the
    raw rasterized channels straight into the projection, which is a different model, not a smaller
    one.
    """
    widths = [int(width) for width in hidden_dims]
    if not widths:
        raise ValueError(f"{owner} needs at least one hidden width; got an empty hidden_dims.")
    if any(width <= 0 for width in widths):
        raise ValueError(f"{owner} hidden_dims must all be positive, got {widths}.")
    return widths


def _conv_stages(conv_factory, hidden_dim: int, norm: str, dropout: float, dims: int) -> list[nn.Module]:
    first, second = conv_factory()
    return [
        nn.Sequential(first, _build_norm(norm, hidden_dim, dims=dims), nn.GELU(), nn.Dropout(dropout)),
        nn.Sequential(second, _build_norm(norm, hidden_dim, dims=dims), nn.GELU()),
    ]


class DilatedTempCNNEncoder(nn.Module):
    """Architecture 1: dilated 1D convolutions over the flattened calendar grid.

    The second layer is the point of the design. With ``kernel_size=3`` and ``dilation=12`` its taps
    sit on months ``t-12``, ``t`` and ``t+12``, so a single weight compares a month against the same
    month in the neighbouring years. ``padding=12`` makes that exactly length-preserving.
    """

    def __init__(
        self,
        num_channels: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (32,),
        dropout: float = 0.1,
        norm: str = "batch",
        pool: str = "masked_avg",
        mask_between_blocks: bool = True,
    ):
        super().__init__()
        self.pool = str(pool).lower()
        if self.pool not in {"masked_avg", "avg"}:
            raise ValueError(f"pool must be 'masked_avg' or 'avg', got {pool!r}")

        hidden_dims = _validate_hidden_dims(hidden_dims, type(self).__name__)
        stages: list[nn.Module] = []
        in_channels = int(num_channels)
        for width in hidden_dims:
            stages += _conv_stages(
                lambda c=in_channels, w=width: (
                    # Short-term structure: adjacent months, i.e. within-quarter trend.
                    nn.Conv1d(c, w, kernel_size=3, padding=1, dilation=1),
                    # Year-over-year: month t wired directly to month t-12.
                    nn.Conv1d(
                        w,
                        w,
                        kernel_size=3,
                        padding=MONTHS_PER_YEAR,
                        dilation=MONTHS_PER_YEAR,
                    ),
                ),
                width,
                norm,
                dropout,
                dims=1,
            )
            in_channels = width
        self.hidden_dims = hidden_dims
        self.blocks = _MaskedConvStack(stages, mask_between=mask_between_blocks)
        self.projection = nn.Linear(hidden_dims[-1], int(output_dim))
        self.output_dim = int(output_dim)

    def forward(self, grid: torch.Tensor, cell_mask: torch.Tensor) -> torch.Tensor:
        batch_size = grid.size(0)
        sequence = grid.reshape(batch_size, grid.size(1), -1)
        flat_mask = cell_mask.reshape(batch_size, -1)
        features = self.blocks(sequence, flat_mask)
        pooled = masked_global_pool(features, flat_mask, masked=self.pool == "masked_avg")
        return self.projection(pooled)


class AnnualGrid2DEncoder(nn.Module):
    """Architecture 2: separable 2D convolutions over the (years x months) grid.

    Factorising the kernel is what makes the two axes interpretable. ``(1, 3)`` moves along the month
    axis within a single year and never mixes years, so it can only learn phenology; ``(3, 1)`` moves
    along the year axis at a fixed calendar month, so it can only learn how a given month drifts
    across years.
    """

    def __init__(
        self,
        num_channels: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (32,),
        dropout: float = 0.1,
        norm: str = "batch",
        pool: str = "masked_avg",
        mask_between_blocks: bool = True,
    ):
        super().__init__()
        self.pool = str(pool).lower()
        if self.pool not in {"masked_avg", "avg"}:
            raise ValueError(f"pool must be 'masked_avg' or 'avg', got {pool!r}")

        hidden_dims = _validate_hidden_dims(hidden_dims, type(self).__name__)
        stages: list[nn.Module] = []
        in_channels = int(num_channels)
        for width in hidden_dims:
            stages += _conv_stages(
                lambda c=in_channels, w=width: (
                    # Intra-annual: slides across the 12 calendar months, one year at a time.
                    nn.Conv2d(c, w, kernel_size=(1, 3), padding=(0, 1)),
                    # Inter-annual: slides across years, one calendar month at a time.
                    nn.Conv2d(w, w, kernel_size=(3, 1), padding=(1, 0)),
                ),
                width,
                norm,
                dropout,
                dims=2,
            )
            in_channels = width
        self.hidden_dims = hidden_dims
        self.blocks = _MaskedConvStack(stages, mask_between=mask_between_blocks)
        self.projection = nn.Linear(hidden_dims[-1], int(output_dim))
        self.output_dim = int(output_dim)

    def forward(self, grid: torch.Tensor, cell_mask: torch.Tensor) -> torch.Tensor:
        features = self.blocks(grid, cell_mask)
        pooled = masked_global_pool(features, cell_mask, masked=self.pool == "masked_avg")
        return self.projection(pooled)


class ConcatGatedFusion(nn.Module):
    """``Z = concat(branches) * sigmoid(Linear(concat(branches)))``.

    A self-gate over the concatenation, so the output keeps every branch at full width and the gate
    decides per feature how much of each survives. Note this is *not* the interpolating gate the
    sequence model uses, which trades one branch off against the other at a shared width.

    ``coordinate_dim`` adds a third branch. Putting position INSIDE the gate rather than appending
    it afterwards is the point of doing it here: the gate reads the concatenation, so it can damp or
    admit a static covariate or a temporal embedding *conditioned on where the point is* - a
    reflectance band that means one thing on an irrigated plain and another on a limestone slope.
    Appending after the gate would let position reach the head unmediated but would leave it unable
    to modulate anything.

    A third named parameter rather than ``*part_dims``: callers construct this by keyword, and a
    varargs signature would break them. ``coordinate_dim=0`` builds exactly the two-branch module
    this class has always been - same ``output_dim``, same ``gate`` shape, same state_dict keys.
    """

    def __init__(self, static_dim: int, temporal_dim: int, coordinate_dim: int = 0):
        super().__init__()
        self.output_dim = int(static_dim) + int(temporal_dim) + int(coordinate_dim)
        self.gate = nn.Linear(self.output_dim, self.output_dim)

    def forward(
        self,
        static_features: torch.Tensor,
        temporal_features: Optional[torch.Tensor],
        coordinate_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = [
            part
            for part in (static_features, temporal_features, coordinate_features)
            if part is not None
        ]
        joined = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        return joined * torch.sigmoid(self.gate(joined))


class AttentionFusion(nn.Module):
    """Transformer self-attention over one token per branch chunk - a drop-in for ConcatGatedFusion.

    The gate can only rescale a branch's features, reading the others to decide by how much. Here
    every chunk is projected to a ``d_model``-wide token by its own linear layer and the tokens attend
    to one another, so a branch's representation is REBUILT from the others before the head sees it:
    the S2 token can be re-read in light of the climate token, the static token or the position.

    ``static_dims`` says how the static vector is cut. ``[64]`` makes it one token - the summary a
    TabularStaticEncoder produced. ``[1, 1, ..., 4]`` makes every continuous column its own token and
    each categorical embedding another, which is FT-Transformer's tokenizer: a width-1 chunk through a
    Linear is exactly its ``x * w + b``. An empty ``static_dims`` reads no static tokens at all, so a
    caller with no static features may hand over any placeholder.

    The call signature is ConcatGatedFusion's on purpose: SoilCNNLightningModule's ``_fuse`` and
    ``_fuse_from_parts`` drive either one without knowing which they hold.

    There is no attention mask. A modality with no observations arrives as an exactly-zero embedding
    (``_encode_temporal_from_grids`` zeroes it) and becomes a learned constant token - the same thing
    the gate sees. A mask derived from "is this chunk all zeros" would also be discontinuous under the
    interpolations a gradient explainer runs, which is a second reason not to build one.
    """

    READOUTS = ("cls", "mean", "flatten")

    def __init__(
        self,
        static_dims: Sequence[int],
        temporal_dims: Sequence[int],
        coordinate_dim: int = 0,
        *,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        ff_multiplier: int = 2,
        dropout: float = 0.1,
        readout: str = "cls",
    ):
        super().__init__()
        d_model, nhead = int(d_model), int(nhead)
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model must be divisible by nhead for multi-head attention; got d_model={d_model} "
                f"and nhead={nhead}"
            )
        if int(ff_multiplier) < 1:
            raise ValueError(f"ff_multiplier must be at least 1, got {ff_multiplier}")
        self.readout = str(readout).lower()
        if self.readout not in self.READOUTS:
            raise ValueError(f"readout must be one of {list(self.READOUTS)}, got {readout!r}")

        self.static_dims = [int(width) for width in static_dims]
        self.temporal_dims = [int(width) for width in temporal_dims]
        self.coordinate_dim = int(coordinate_dim or 0)
        widths = self.static_dims + self.temporal_dims + ([self.coordinate_dim] if self.coordinate_dim > 0 else [])
        if not widths:
            raise ValueError(
                "AttentionFusion needs at least one input token; got no static, temporal or coordinate chunks."
            )
        if any(width <= 0 for width in widths):
            raise ValueError(f"AttentionFusion chunk widths must all be positive, got {widths}")

        self.d_model = d_model
        self.num_tokens = len(widths)
        self.tokenizers = nn.ModuleList(nn.Linear(width, d_model) for width in widths)
        # Every slot has its own projection already, but a learned per-slot vector keeps two chunks
        # that happen to project alike distinguishable to attention, which is otherwise order-blind.
        self.token_type = nn.Parameter(torch.zeros(self.num_tokens, d_model))
        nn.init.normal_(self.token_type, std=0.02)
        if self.readout == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        # Same layer as TemporalTransformerEncoder, plus a closing LayerNorm: a pre-norm stack leaves
        # its last residual stream unnormalized, and this output feeds a head rather than another block.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(ff_multiplier) * d_model,
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is incompatible with norm_first and would only warn; say so explicitly.
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=max(1, int(num_layers)),
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.output_dim = d_model * self.num_tokens if self.readout == "flatten" else d_model

    @staticmethod
    def _split(features: Optional[torch.Tensor], dims: list[int], label: str) -> list[torch.Tensor]:
        if not dims:
            return []
        if features is None:
            raise ValueError(f"AttentionFusion was built with {label} chunks {dims} but received no {label} features")
        if features.size(-1) != sum(dims):
            # The alternative is a silent re-slicing: the wrong width would still split, with every
            # chunk after the first reading a neighbour's columns.
            raise ValueError(
                f"AttentionFusion expected {sum(dims)} {label} feature(s) (chunks {dims}), got {features.size(-1)}"
            )
        return list(torch.split(features, dims, dim=-1))

    def forward(
        self,
        static_features: torch.Tensor,
        temporal_features: Optional[torch.Tensor],
        coordinate_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        chunks = self._split(static_features, self.static_dims, "static")
        chunks += self._split(temporal_features, self.temporal_dims, "temporal")
        if self.coordinate_dim > 0:
            chunks += self._split(coordinate_features, [self.coordinate_dim], "coordinate")

        tokens = torch.stack(
            [tokenizer(chunk) for tokenizer, chunk in zip(self.tokenizers, chunks)], dim=1
        ) + self.token_type
        if self.readout == "cls":
            tokens = torch.cat([self.cls_token.expand(tokens.size(0), -1, -1), tokens], dim=1)

        encoded = self.encoder(tokens)
        if self.readout == "cls":
            return encoded[:, 0]
        if self.readout == "mean":
            return encoded.mean(dim=1)
        return encoded.flatten(1)
