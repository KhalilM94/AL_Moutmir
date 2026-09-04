"""Positional encoding for point coordinates.

A plain ``nn.Module``, deliberately not a ``LightningModule``, for the reason
``tabular_encoders`` states about its own blocks: a LightningModule cannot be composed into several
parents, and everything training-related - loss, optimizer, target inversion - stays in
``SoilRegressionLightningBase``. This is a branch, not a model.

The counterpart on the data side is ``SoilSequenceDataModule._normalize_coords``, which maps
lat/lon onto the ``[-1, 1]`` interval this module's frequencies are defined on, using the bounding
box of the TRAINING split. The contract between them is narrow on purpose: a ``(B, 2)`` float
tensor whose columns are ``(lat, lon)`` normalized against that box. Nothing here fits anything, so
a checkpoint of this module is fully described by its hyperparameters.

Why coordinates are treated differently from dates. The temporal branch is built so that *nothing
encodes an absolute epoch* - rows are counted back from each point's own latest observation, so
shifting every date by a decade changes no input. Position is the opposite case: absolute location
is precisely the signal, and two points at the same relative offset from their neighbours are not
interchangeable. So this branch does encode absolute position, and the train bounding box that
makes it absolute travels in the checkpoint.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import nn

from yg_eo_soilnet.models.lightningmodules.mlp import build_mlp_stack


class HarmonicPositionEncoder(nn.Module):
    """Normalized coordinates -> a multi-frequency sine/cosine embedding.

    Frequency ``k`` is ``2**k * pi``, so on the ``[-1, 1]`` interval it completes ``2**k`` cycles
    and therefore resolves roughly ``1/2**k`` of the study area. That is what makes
    ``num_frequencies`` a *resolution* knob rather than a dataset-specific constant: the same
    setting means "down to a 64th of the area" on a province and on a continent.

    Two low frequencies alone would let the network express a smooth regional trend and nothing
    else; the high ones give it the capacity to separate nearby points. Both are wanted, which is
    why the bank is geometric rather than a single scale.

    ``include_input`` keeps the normalized coordinates themselves beside the harmonics. They are the
    zero-frequency term in all but name - a monotone north-south or east-west gradient, which the
    sine bank alone can only approximate - and they cost two channels.

    Output channels, in the order :meth:`forward` writes them:

    * ``2`` normalized inputs, when ``include_input``
    * then, per frequency ``k`` in ascending order, ``4`` channels:
      ``sin(f_k * lat), cos(f_k * lat), sin(f_k * lon), cos(f_k * lon)``

    :meth:`channel_layout` reports that mapping. It is defined here rather than reconstructed by
    callers for the reason ``CalendarGridRasterizer.channel_layout`` gives: this class is the only
    thing that decides the order, and an explainer that guessed it wrong would attribute one
    coordinate's importance to the other and produce a plausible-looking plot instead of an error.
    """

    def __init__(
        self,
        num_coordinates: int = 2,
        num_frequencies: int = 6,
        include_input: bool = True,
        hidden_dims: Optional[Sequence[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_coordinates = int(num_coordinates)
        if self.num_coordinates <= 0:
            raise ValueError(
                f"num_coordinates must be positive, got {num_coordinates}. Callers with no "
                "coordinates should skip this encoder entirely rather than build a zero-wide one."
            )
        self.num_frequencies = int(num_frequencies)
        if self.num_frequencies <= 0:
            raise ValueError(
                f"num_frequencies must be positive, got {num_frequencies}. A bank with no "
                "frequencies encodes nothing; switch the branch off with USE_HARMONIC_COORDS "
                "instead, which also stops the coordinates being carried at all."
            )
        self.include_input = bool(include_input)

        # Non-persistent: fully determined by num_frequencies, which travels in hyper_parameters.
        # Keeping it out of the state_dict means a checkpoint stays loadable when the schedule's
        # derivation changes, and keeps the tensor out of the weights_only=True round trip.
        exponents = torch.arange(self.num_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", (2.0**exponents) * math.pi, persistent=False)

        self.embedding_dim = self.num_coordinates * (
            2 * self.num_frequencies + (1 if self.include_input else 0)
        )
        hidden_dims = [int(width) for width in (hidden_dims or [])]
        self.hidden_dims = hidden_dims
        # An empty hidden_dims makes this an Identity, which IS the raw-concat mode - the two
        # options are one code path and forward() needs no branch between them. The same shape the
        # auxiliary label block uses, and for the same reason.
        self.projection = build_mlp_stack(
            self.embedding_dim,
            hidden_dims,
            None,
            dropout=float(dropout),
            activation="gelu",
            use_layer_norm=True,
            # This block feeds a fusion rather than a readout, so both are on - the case
            # build_mlp_stack's docstring describes as "feeding a fusion".
            norm_final=True,
            dropout_final=True,
        )
        self.output_dim = hidden_dims[-1] if hidden_dims else self.embedding_dim

    def channel_layout(self) -> dict[str, list[int]]:
        """Which channel of the RAW embedding carries what, before any projection.

        Reported against ``embedding_dim``, not ``output_dim``: once ``hidden_dims`` mixes the
        channels there is no per-channel meaning left to report, which is exactly why an explainer
        attributes to the encoder's *input* coordinates rather than to these.
        """
        layout: dict[str, list[int]] = {"input": [], "sin": [], "cos": []}
        cursor = 0
        if self.include_input:
            layout["input"] = list(range(cursor, cursor + self.num_coordinates))
            cursor += self.num_coordinates
        for _ in range(self.num_frequencies):
            for _ in range(self.num_coordinates):
                layout["sin"].append(cursor)
                layout["cos"].append(cursor + 1)
                cursor += 2
        return layout

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """``(B, num_coordinates)`` normalized coordinates -> ``(B, output_dim)``."""
        if coords.dim() != 2:
            raise ValueError(f"coords must be 2-D (batch, coordinates), got {coords.dim()}-D")
        if coords.size(-1) != self.num_coordinates:
            raise ValueError(
                f"coords has {coords.size(-1)} column(s) but this encoder was built for "
                f"{self.num_coordinates}"
            )

        parts: list[torch.Tensor] = [coords] if self.include_input else []
        # (B, C, K): every coordinate against every frequency, then interleaved sin/cos per
        # frequency so the channel order matches channel_layout above.
        angles = coords.unsqueeze(-1) * self.frequencies.to(dtype=coords.dtype)
        for index in range(self.num_frequencies):
            angle = angles[..., index]
            parts.append(torch.stack([angle.sin(), angle.cos()], dim=-1).flatten(1))
        return self.projection(torch.cat(parts, dim=-1))
