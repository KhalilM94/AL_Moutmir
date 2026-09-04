from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn

from yg_eo_soilnet.models.lightningmodules._regression_base import as_float_list, batch_get
from yg_eo_soilnet.models.lightningmodules.mlp import build_mlp_stack
from yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module import SoilCNNLightningModule

logger = logging.getLogger(__name__)


class SoilResidualCNNLightningModule(SoilCNNLightningModule):
    """SoilCNN, but the head learns a CORRECTION to an existing prediction rather than the value.

    One column of the lab roster is nominated as the *base* for each fitted target - in practice a
    ``<target>__<model>`` column written by ``predictions_export`` and merged back into the static or
    targets source. The base is mapped into the target's own space and added to the readout::

        b_raw = x_labels[:, i] * label_scale[i] + label_mean[i]      # undo the LAB standardizer
        b_std = (10*log1p(b_raw) - target_mean) / target_scale       # into the TARGET's space
        pred  = b_std + output_head(fused)

    The offset lives entirely inside ``forward``, which is the point: ``y`` is already in that
    space, so the loss, the epoch metrics, ``predict_step``'s inversion, the uncertainty ensemble
    and the serving path all work unchanged and the head simply has an easier job.

    The base ALSO enters the network as an input, through a block of its own concatenated after the
    fusion gate beside the auxiliary block. Without it the head would be correcting blind - it could
    not tell whether it was adjusting a large base or a small one, and the right correction is
    rarely the same for both.

    ``val_loss`` here is a residual-space loss. It is not comparable to a non-residual entry's, for
    the same reason the base class warns that it is only comparable across runs sharing a
    ``loss_name``. Compare ``r2_test`` / ``rmse_test`` instead, which are in original units.

    LEAKAGE: a base column clears the inherited auxiliary check only because
    ``c_e_c_meq_100g__soil_cnn`` is a different string from ``c_e_c_meq_100g``. That is right in
    principle - a prediction exists at inference time and the measurement does not - but it is only
    SOUND if those predictions were produced out-of-fold with respect to the split this run uses,
    which nothing here can verify. A warning naming each base column is emitted at construction.
    """

    def __init__(
        self,
        *,
        residual_base_columns: Optional[Mapping[str, str]] = None,
        residual_base_hidden_dims: Optional[Sequence[int]] = None,
        residual_base_dropout: float = 0.0,
        residual_base_validity_channels: bool = True,
        residual_base_max_missing: float = 0.05,
        # Full-roster lab standardization stats, injected by LightningConfigFactory exactly as
        # target_mean/target_scale are. They are what maps the base out of the lab standardizer;
        # without them the offset would be a z-score.
        auxiliary_label_mean: Optional[Any] = None,
        auxiliary_label_scale: Optional[Any] = None,
        # Declared here rather than read back out of hparams so the head can be rebuilt from the
        # values themselves. All three are forwarded to the parent unchanged.
        head_hidden_dims: Sequence[int] = (128, 64),
        head_norm_final: bool = False,
        dropout: float = 0.1,
        **kwargs: Any,
    ):
        # Coerced BEFORE super().__init__ for the reason the parent documents: save_hyperparameters
        # stores these verbatim, and a numpy array in hyper_parameters makes the checkpoint
        # unloadable under torch.load's weights_only=True default.
        residual_base_columns = {
            str(target): str(column) for target, column in dict(residual_base_columns or {}).items()
        }
        residual_base_hidden_dims = [int(width) for width in (residual_base_hidden_dims or [])]
        # `or []` because as_float_list passes None straight through, and the width check below -
        # the one thing standing between a missing statistic and an offset expressed as a z-score -
        # needs a length rather than a TypeError.
        auxiliary_label_mean = as_float_list(auxiliary_label_mean) or []
        auxiliary_label_scale = as_float_list(auxiliary_label_scale) or []
        head_hidden_dims = [int(width) for width in head_hidden_dims]

        # Keyword-only throughout, so nothing can arrive positionally: Lightning's
        # save_hyperparameters drops *args, and a positional static_dim would then be missing from
        # the checkpoint's hyper_parameters and absent when load_from_checkpoint re-calls __init__.
        super().__init__(
            head_hidden_dims=head_hidden_dims,
            head_norm_final=head_norm_final,
            dropout=dropout,
            **kwargs,
        )
        # The parent's save_hyperparameters() captured its own frame, which does not include these
        # arguments. Calling it again merges this frame in (Lightning updates rather than replaces),
        # so residual_base_columns round-trips through the checkpoint instead of being rebuilt as an
        # empty mapping on reload.
        self.save_hyperparameters()

        self.residual_base_columns = residual_base_columns
        self.residual_base_validity_channels = bool(residual_base_validity_channels)
        self.residual_base_max_missing = float(residual_base_max_missing)
        self.base_encoder = self._build_residual_base_encoder(
            auxiliary_label_mean,
            auxiliary_label_scale,
            residual_base_hidden_dims,
            float(residual_base_dropout),
        )

        # Rebuilt rather than sized up front: the parent's __init__ has already built a head for its
        # own fused width, and the base block widens it. head_output_dim, not target_dim - a
        # heteroscedastic head is twice as wide because it emits a log variance beside every mean.
        self.output_head = build_mlp_stack(
            self.fusion.output_dim + self.auxiliary_output_dim + self.residual_base_output_dim,
            head_hidden_dims,
            self.head_output_dim,
            dropout=dropout,
            activation="gelu",
            norm_final=bool(head_norm_final),
        )

    # --- construction -------------------------------------------------------

    def _build_residual_base_encoder(
        self,
        label_mean: list[float],
        label_scale: list[float],
        hidden_dims: list[int],
        dropout: float,
    ) -> nn.Module:
        """Resolve one base column per fitted target and build the block that reads them.

        Resolution is by name against the same roster the auxiliary branch uses, and for the same
        reason: the roster's order follows LABEL_COLUMNS, so a configured index would point at a
        different measurement the moment that list is reordered.
        """
        self.residual_base_output_dim = 0
        # Registered unconditionally so state_dict keys never depend on the configuration.
        self.register_buffer("residual_base_index", torch.zeros(0, dtype=torch.long), persistent=True)
        self.register_buffer("residual_base_label_mean", torch.zeros(0), persistent=True)
        self.register_buffer("residual_base_label_scale", torch.ones(0), persistent=True)

        selected = self.residual_base_columns
        if not selected:
            # Unlike the auxiliary branch, whose empty case is the ordinary one, an empty mapping
            # here means the architecture has nothing to be a residual OF. Silently degrading to
            # plain SoilCNN would make two registry entries that train identically under different
            # names, with only the metrics to say which one actually ran.
            raise ValueError(
                "residual_base_columns is required: SoilResidualCNNLightningModule anchors its head "
                "on a base prediction per target. Use SoilCNNLightningModule for a model with no "
                "base."
            )

        missing = [name for name in self.target_names if name not in selected]
        if missing:
            raise ValueError(
                f"residual_base_columns has no entry for target(s) {sorted(missing)}; it must name a "
                f"base column for every target this model fits ({sorted(self.target_names)}). A "
                "partially anchored head would learn residuals for some outputs and absolute values "
                "for others, against a single loss."
            )

        extra = [name for name in selected if name not in set(self.target_names)]
        if extra:
            raise ValueError(
                f"residual_base_columns names target(s) this model does not fit: {sorted(extra)}. "
                f"This model's targets are {sorted(self.target_names)}."
            )

        # The measured target itself, rather than a prediction of it. The inherited auxiliary check
        # never sees this mapping, so the same rail has to be laid here.
        leaking = [column for column in selected.values() if column in set(self.fitted_target_names)]
        if leaking:
            raise ValueError(
                f"residual_base_columns may not name a column being fitted: {sorted(leaking)} "
                f"also appear(s) in the run's targets {sorted(self.fitted_target_names)}. The base "
                "must be a PREDICTION of the target, not the target."
            )

        shared = sorted(set(selected.values()) & set(self.auxiliary_label_columns))
        if shared:
            raise ValueError(
                f"Column(s) {shared} appear in both residual_base_columns and "
                "auxiliary_label_columns. A base column already reaches the head through its own "
                "block; listing it twice feeds it twice and lets the two roles be configured apart."
            )

        available = list(self.auxiliary_available_names)
        if not available:
            raise ValueError(
                f"residual_base_columns names {sorted(set(selected.values()))} but no lab columns "
                "are being carried with the data. Set CARRY_LABEL_COLUMNS: true in data_spec.yml, "
                "and list the base column(s) in LABEL_COLUMNS."
            )

        unknown = [column for column in selected.values() if column not in set(available)]
        if unknown:
            raise ValueError(
                f"residual_base_columns names column(s) the data does not carry: {sorted(unknown)}. "
                f"Available: {sorted(available)}. A base column must be listed in LABEL_COLUMNS and "
                "present in the static or targets source."
            )

        if len(label_mean) != len(available) or len(label_scale) != len(available):
            # Without these the base cannot leave the lab standardizer's space, and adding it as-is
            # would offset the prediction by a z-score. The factory supplies them from the training
            # split; this only fires on a hand-built module.
            raise ValueError(
                "residual_base_columns requires auxiliary_label_mean and auxiliary_label_scale at "
                f"the roster's width ({len(available)}); got {len(label_mean)} and "
                f"{len(label_scale)}. They are what maps the base out of the lab standardizer."
            )

        # Ordered by target_names, so position j of the offset lines up with output column j.
        indices = [available.index(selected[name]) for name in self.target_names]
        self.residual_base_index = torch.as_tensor(indices, dtype=torch.long)
        self.residual_base_label_mean = torch.as_tensor(
            [label_mean[index] for index in indices], dtype=torch.float32
        )
        self.residual_base_label_scale = torch.as_tensor(
            [label_scale[index] for index in indices], dtype=torch.float32
        )

        logger.warning(
            "SoilResidualCNNLightningModule anchors on %s. These must be OUT-OF-FOLD predictions "
            "for the split this run uses: nothing in the pipeline can verify that, and in-fold "
            "predictions will inflate every reported metric.",
            {name: selected[name] for name in self.target_names},
        )

        input_dim = self.target_dim * (2 if self.residual_base_validity_channels else 1)
        # An empty hidden_dims makes this an Identity, which IS the raw-concat mode - the same shape
        # the auxiliary and harmonic blocks use, and for the same reason.
        encoder = build_mlp_stack(
            input_dim,
            hidden_dims,
            None,
            dropout=dropout,
            activation="gelu",
            use_layer_norm=True,
            # This block feeds the head's input vector rather than a readout, so both are on.
            norm_final=True,
            dropout_final=True,
        )
        self.residual_base_output_dim = hidden_dims[-1] if hidden_dims else input_dim
        return encoder

    # --- serving ------------------------------------------------------------

    @property
    def serving_label_columns(self) -> list[str]:
        """The auxiliary columns plus the base columns - everything a request has to supply.

        The base is the one column a served request must never omit: an absent one arrives NaN, is
        median-filled from the training split, and the head then corrects a base that is the same
        constant for every point.
        """
        auxiliary = list(self.auxiliary_label_columns)
        base = [self.residual_base_columns[name] for name in self.target_names]
        return auxiliary + [column for column in base if column not in set(auxiliary)]

    # --- coverage guard -----------------------------------------------------

    def on_fit_start(self) -> None:
        """Refuse to train on a base column the data mostly does not carry.

        Nothing upstream checks this: ``assert_columns_are_dense_enough`` inspects only the
        continuous covariates, and lab columns are removed from that block by ``filter_schema``. A
        sparse base is median-filled, so the failure is silent - the head learns to correct a
        constant and the run merely looks mediocre.
        """
        super().on_fit_start()
        datamodule = getattr(self.trainer, "datamodule", None) if self.trainer is not None else None
        bundle = getattr(datamodule, "sequence_bundle", None)
        if bundle is None or not hasattr(bundle, "label_missing_fraction"):
            return

        offenders = {}
        for name in self.target_names:
            column = self.residual_base_columns[name]
            fraction = float(bundle.label_missing_fraction(column))
            if fraction > self.residual_base_max_missing:
                offenders[column] = round(fraction, 4)
        if offenders:
            raise ValueError(
                f"Base column(s) {offenders} exceed residual_base_max_missing="
                f"{self.residual_base_max_missing}. A missing base is median-filled, so the head "
                "would be correcting the training median rather than a prediction for that point. "
                "Regenerate the predictions over the full population, or raise the threshold "
                "deliberately."
            )

    # --- forward ------------------------------------------------------------

    def _residual_base(self, batch: Any, *, device, dtype) -> torch.Tensor:
        """``[base_in_target_space, validity]`` - the offset and the block, in one tensor.

        One tensor rather than two so :meth:`explanation_parts` can publish a single part from which
        BOTH the encoder input and the additive offset are derived. Splitting them across two parts,
        or reading the offset off the batch, would break the equality
        ``forward_from_parts(explanation_parts(batch)[0]) == forward(batch)`` that
        tests/test_explain.py pins.
        """
        values = batch_get(batch, "x_labels")
        if values is None:
            raise KeyError(
                "Batch is missing 'x_labels'; residual_base_columns needs the base prediction "
                f"column(s) {sorted(set(self.residual_base_columns.values()))} that the sequence "
                "datamodule collates."
            )
        values = values.to(device=device, dtype=dtype)
        if values.size(-1) != len(self.auxiliary_available_names):
            raise ValueError(
                f"Batch carries {values.size(-1)} lab column(s) but this model resolved its base "
                f"columns against {len(self.auxiliary_available_names)}; the data no longer matches "
                "the checkpoint's label roster."
            )

        base = values.index_select(-1, self.residual_base_index)
        base = base * self.residual_base_label_scale + self.residual_base_label_mean
        if bool(self.targets_are_log1p):
            # Mirrors SoilSequenceDataModule._apply_target_transform, clipping included: log1p is
            # undefined below -1 and these targets are non-negative, so a base that came back
            # slightly negative is clipped exactly as a measured value would be.
            base = 10.0 * torch.log1p(base.clamp_min(0.0))
        if bool(self.targets_are_standardized):
            base = (base - self.target_mean) / self.target_scale

        if not self.residual_base_validity_channels:
            return base

        validity = batch_get(batch, "x_label_validity")
        if validity is None:
            raise KeyError(
                "Batch is missing 'x_label_validity'; set residual_base_validity_channels=False to "
                "run without the measured-vs-filled flags."
            )
        validity = validity.to(device=device, dtype=dtype).index_select(-1, self.residual_base_index)
        return torch.cat([base, validity], dim=-1)

    def _head_from_base(self, fused: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
        """Widen the fused vector with the base block, run the head, and add the offset back.

        Returns the readout in its raw shape, so ``_split_head_output`` still applies.
        """
        fused = torch.cat([fused, self.base_encoder(block)], dim=-1)
        mean, log_variance = self._split_head_output(self.output_head(fused))
        mean = mean + block[..., : self.target_dim]
        if log_variance is None:
            return mean
        # The offset belongs to the MEAN half only. On a heteroscedastic head the readout is
        # 2*target_dim wide, and adding it to the whole tensor would shift the log variances too -
        # turning a base of 40 into a predicted variance of exp(40).
        return torch.cat([mean, log_variance], dim=-1)

    def forward(self, batch: Any) -> torch.Tensor:
        reference = next(self.parameters())
        device, dtype = reference.device, reference.dtype
        fused = self._fuse(batch, device=device, dtype=dtype)
        block = self._residual_base(batch, device=device, dtype=dtype)
        return self._head_from_base(fused, block)

    # --- attribution seam ---------------------------------------------------

    def explanation_parts(self, batch: Any) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
        parts, groups = super().explanation_parts(batch)
        reference = next(self.parameters())
        block = self._residual_base(batch, device=reference.device, dtype=reference.dtype)
        part_index = len(parts)
        parts.append(block)
        for index, name in enumerate(self.target_names):
            columns = [index]
            if self.residual_base_validity_channels:
                columns.append(self.target_dim + index)
            groups.append(
                {
                    "part": part_index,
                    "kind": "residual_base",
                    "name": self.residual_base_columns[name],
                    "target": name,
                    "columns": columns,
                }
            )
        return parts, groups

    def forward_from_parts(self, parts: Sequence[torch.Tensor]) -> torch.Tensor:
        fused, cursor = self._fuse_from_parts(parts)
        # The MEAN only, for the reason the parent gives: a heteroscedastic head's log variances
        # would otherwise be attributed and labelled as targets the model does not have.
        return self._split_head_output(self._head_from_base(fused, parts[cursor]))[0]
