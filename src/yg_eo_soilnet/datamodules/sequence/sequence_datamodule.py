from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from lightning.pytorch import LightningDataModule

from yg_eo_soilnet.datamodules.categorical import CategoricalEncoder
from yg_eo_soilnet.datamodules.loaders import build_loader
from yg_eo_soilnet.datamodules.sequence.sequence_bundle import SoilSequenceBundle
from yg_eo_soilnet.datamodules.splitting import SplitPlan
from yg_eo_soilnet.targets import select_target_columns


class _PointDataset(Dataset):
    """Yields point indices; the DataLoader batches them and collate assembles the padded batch."""

    def __init__(self, point_indices: np.ndarray):
        self.point_indices = np.asarray(point_indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.point_indices.size)

    def __getitem__(self, index: int) -> int:
        return int(self.point_indices[index])


class SoilSequenceDataModule(LightningDataModule):
    """Serves static covariates plus ragged, date-stamped observation sequences.

    Two properties distinguish this from the graph datamodule, and both are load-bearing:

    * **Padding is a batch-local artefact.** Sequences are stored ragged and padded only to the
      current batch's longest series, so ``L`` differs from batch to batch. Nothing downstream may
      assume a fixed number of steps.
    * **``sequence_mask.sum(1)`` is a genuine length.** Every unmasked token is a real observation,
      so the count-versus-length confusion that dogged the dense-axis representation cannot arise
      here, and packing an RNN by that count is correct.

    Batches are plain dicts of tensors, so Lightning moves them to the accelerator by itself; the
    model needs no device-transfer override.
    """

    def __init__(
        self,
        sequence_bundle: "SoilSequenceBundle | Mapping[str, Any]",
        batch_size: int = 32,
        val_size: float = 0.2,
        test_size: float = 0.2,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        seed: int = 42,
        shuffle: bool = False,
        target_transform: Optional[str] = None,
        max_sequence_length: Optional[int] = None,
        split_plan: Optional["SplitPlan"] = None,
        active_targets: Optional[list[str]] = None,
    ):
        super().__init__()
        self.sequence_bundle = deepcopy(SoilSequenceBundle.from_mapping(sequence_bundle))
        # The subset of the bundle's targets this run fits. None keeps all of them, which is the
        # joint head. Narrowed HERE, immediately after the copy, so everything below - target_dim,
        # the per-column scaler, the y frames, the batch `y`, the serving state - follows from the
        # two fields it rewrites and needs no per-target branch of its own.
        self.active_targets = list(active_targets) if active_targets else None
        narrowed, target_names, _ = select_target_columns(
            self.sequence_bundle.targets, self.sequence_bundle.target_names, self.active_targets
        )
        self.sequence_bundle.targets = narrowed
        self.sequence_bundle.target_names = target_names
        self.target_transform = None if target_transform is None else str(target_transform).lower()
        if self.target_transform not in {None, "none", "log1p"}:
            raise ValueError("target_transform must be None or 'log1p'")
        if self.target_transform == "none":
            self.target_transform = None

        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        # The run's shared split. When present it decides train/val/test and `val_size`/`test_size`
        # are ignored - the point being that this datamodule and the sklearn family hold out the
        # same points. None keeps the legacy ratio carve, which standalone and test use rely on.
        self.split_plan = split_plan
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed
        self.shuffle = shuffle
        self.max_sequence_length = None if max_sequence_length is None else max(1, int(max_sequence_length))

        self._is_setup = False
        self.train_idx_: Optional[np.ndarray] = None
        self.val_idx_: Optional[np.ndarray] = None
        self.test_idx_: Optional[np.ndarray] = None
        self.X_train_frame_ = None
        self.y_train_frame_ = None
        self.X_val_frame_ = None
        self.y_val_frame_ = None
        self.X_test_frame_ = None
        self.y_test_frame_ = None

        # Standardization statistics, fitted on the train split only in setup().
        self.static_mean_: Optional[np.ndarray] = None
        self.static_scale_: Optional[np.ndarray] = None
        # The FILL value for a missing covariate, kept separately from the mean for the same reason
        # label_median_ below is: a skewed column's mean sits somewhere no sample actually is.
        self.static_median_: Optional[np.ndarray] = None
        # Categorical vocabulary, fitted on the train split only in setup() for the same reason the
        # scaler is: a category seen only in validation or test must reach the model as "unknown",
        # exactly as an unseen category would at inference time.
        self.categorical_encoder_: Optional[CategoricalEncoder] = None
        self.categorical_codes_: Optional[np.ndarray] = None
        self.sequence_mean_: dict[str, np.ndarray] = {}
        self.sequence_scale_: dict[str, np.ndarray] = {}
        self.target_mean_: Optional[np.ndarray] = None
        self.target_scale_: Optional[np.ndarray] = None
        # Covariance of the STANDARDIZED training targets, i.e. their correlation matrix. Fitted
        # here rather than derived by the model because this is the only place holding the whole
        # training split at once; the config factory offers it to whichever loss wants it.
        self.target_covariance_: Optional[np.ndarray] = None
        # Lab-value statistics, train-only for the same reason as everything above. The median is
        # kept separately from the mean because it is the FILL value, not a centring constant: a
        # skewed column's mean sits somewhere no sample actually is.
        self.label_mean_: Optional[np.ndarray] = None
        self.label_scale_: Optional[np.ndarray] = None
        self.label_median_: Optional[np.ndarray] = None
        # The TRAIN-split bounding box, which is what maps lat/lon onto the [-1, 1] the harmonic
        # encoder's frequencies are defined on. Train-only for the same reason every statistic here
        # is, and stored in preprocessing_state for a reason specific to this one: a serving request
        # can be a SINGLE point, whose own bounding box is degenerate, so a re-fit would normalize
        # every served point to the centre of itself.
        self.coord_min_: Optional[np.ndarray] = None
        self.coord_max_: Optional[np.ndarray] = None

        # The shape contract the Lightning config factory reads off the datamodule. Note the
        # deliberate absence of `temporal_steps` and `edge_attr_dim`: the model is length-agnostic
        # and graph-free, and the factory only injects attributes that actually exist here.
        # Validity channels widen the static block, so static_dim counts them: the model's first
        # Linear must match what _collate_points actually hands it. Only covariates that genuinely
        # have gaps carry a flag, so a complete dataset leaves this exactly as it was.
        self.static_validity_names = list(self.sequence_bundle.static_validity_names)
        self.static_dim = int(
            np.asarray(self.sequence_bundle.static_features).shape[1] + len(self.static_validity_names)
        )
        self.target_dim = int(np.asarray(self.sequence_bundle.targets).shape[1])
        self.static_feature_names = list(self.sequence_bundle.static_feature_names)
        # The subset of the above declared as spatial context. Purely descriptive: these columns are
        # already inside static_features and are standardized with the rest, so this changes no
        # shape. It travels so the explainer can roll them up as a block instead of scattering them
        # among the ordinary covariates.
        self.context_feature_names = list(self.sequence_bundle.context_feature_names)
        # 2 when USE_HARMONIC_COORDS put coordinates on the bundle, 0 otherwise. The config factory
        # reads it as coord_dim, and 0 is what keeps the model's coordinate branch an nn.Identity
        # with no parameters and no state_dict keys.
        self.coord_dim = int(self.sequence_bundle.coord_dim)
        self.coord_names = list(self.sequence_bundle.coord_names)
        # Categorical shape contract. The names are known now, but the cardinalities are not: they
        # depend on the vocabulary, which depends on the train split, which setup() decides. The
        # config factory reads these AFTER calling setup(), so by then they are filled in.
        self.categorical_feature_names = list(self.sequence_bundle.categorical_feature_names)
        self.categorical_cardinalities: list[int] = []
        self.categorical_vocabularies: list[list[str]] = []
        self.target_names = list(self.sequence_bundle.target_names)
        # Every lab column the bundle carries, offered to the model so it can resolve the subset
        # named in auxiliary_label_columns. Nothing is selected here: the choice belongs to the
        # architecture, and this datamodule serves several.
        self.label_feature_names = list(self.sequence_bundle.label_feature_names)
        self.label_dim = int(self.sequence_bundle.label_dim)
        self.modality_dims = dict(self.sequence_bundle.modality_dims)
        self.temporal_enabled = bool(self.sequence_bundle.temporal_enabled and self.modality_dims)
        self.grid_years = self._infer_grid_years()

    def preprocessing_state(self) -> dict:
        """Everything fitted in :meth:`setup` that a saved model needs in order to serve raw data.

        The scalers here are fitted on the TRAIN SPLIT ONLY, and until this method existed they
        lived nowhere but on this object. A checkpoint therefore restored the weights and the target
        inverse-transform (both are buffers on the module) but not the input standardization, so a
        reloaded model could only ever be fed data that some datamodule had already scaled - which
        is to say, it could not be deployed. Attaching this to the module closes that gap.

        Everything is returned as plain builtins. That is not cosmetic: numpy arrays in a Lightning
        module's ``hyper_parameters`` make the checkpoint unloadable under ``torch.load``'s
        ``weights_only=True`` default from PyTorch 2.6 on, which is the same constraint that put
        ``as_float_list`` in the model constructors.
        """

        def as_list(values) -> list[float]:
            if values is None:
                return []
            return [float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)]

        return {
            "static_mean": as_list(self.static_mean_),
            "static_scale": as_list(self.static_scale_),
            "static_median": as_list(self.static_median_),
            "static_feature_names": list(self.static_feature_names),
            # Which covariates carry a measured-vs-filled flag, and therefore how wide x_static is.
            # Without this a served model would rebuild the flags from whatever the request happens
            # to be missing and hand the network a different width than it was trained on.
            "static_validity_names": list(self.static_validity_names),
            # Which of those covariates are the declared spatial-context group. Carried so a
            # restored model can label its attributions the way the training run did.
            "context_feature_names": list(self.context_feature_names),
            # The train bounding box, without which a served point cannot be placed on the same
            # [-1, 1] interval the model was trained on. Empty when coordinates are switched off.
            "coord_min": as_list(self.coord_min_),
            "coord_max": as_list(self.coord_max_),
            "coord_names": list(self.coord_names),
            "sequence_mean": {name: as_list(values) for name, values in self.sequence_mean_.items()},
            "sequence_scale": {name: as_list(values) for name, values in self.sequence_scale_.items()},
            "modality_column_names": {
                name: list(columns) for name, columns in self.sequence_bundle.modality_columns.items()
            },
            "label_mean": as_list(self.label_mean_),
            "label_scale": as_list(self.label_scale_),
            "label_median": as_list(self.label_median_),
            "label_feature_names": list(self.label_feature_names),
            "categorical_feature_names": list(self.categorical_feature_names),
            "categorical_vocabularies": [
                [str(category) for category in vocabulary] for vocabulary in self.categorical_vocabularies
            ],
            "target_mean": as_list(self.target_mean_),
            "target_scale": as_list(self.target_scale_),
            "target_names": list(self.target_names),
        }

    def apply_preprocessing_state(self, state: Mapping[str, Any]) -> None:
        """Install statistics fitted ELSEWHERE, instead of fitting them from this data.

        The inference counterpart of :meth:`setup`. At serving time the incoming points are not a
        training split - they may be a single point - so re-fitting a scaler on them would
        standardize each request against itself and produce predictions that drift with batch
        composition. This installs the statistics the model was trained with, which is the only
        correct choice, and is why :meth:`preprocessing_state` puts them in the checkpoint.

        Categorical codes are re-derived through ``CategoricalEncoder.from_vocabularies``, so a
        category this data has but training did not lands on the reserved out-of-vocabulary index
        rather than shifting every other code.
        """
        if not state:
            raise ValueError("apply_preprocessing_state needs the state produced by preprocessing_state()")

        def as_array(values, dtype=np.float32):
            array = np.asarray(list(values or []), dtype=dtype)
            return None if array.size == 0 else array

        self.static_mean_ = as_array(state.get("static_mean"))
        self.static_scale_ = as_array(state.get("static_scale"))
        self.static_median_ = as_array(state.get("static_median"))
        # Restored rather than re-derived, so x_static keeps the width the model was trained with
        # even when the incoming request happens to be missing a different set of covariates.
        if "static_validity_names" in state:
            self.static_validity_names = [str(name) for name in (state.get("static_validity_names") or [])]
        if "context_feature_names" in state:
            self.context_feature_names = [str(name) for name in (state.get("context_feature_names") or [])]
        # Installed rather than re-fitted, which is the whole point of this method: a serving batch
        # can be one point, and a bounding box fitted on one point is a single location that
        # normalizes to the centre of itself. Every request must be placed on the interval the model
        # was trained on, so a point outside the training extent lands outside [-1, 1] - correct, and
        # not clipped: the encoder is periodic and handles it.
        self.coord_min_ = as_array(state.get("coord_min"), dtype=np.float64)
        self.coord_max_ = as_array(state.get("coord_max"), dtype=np.float64)
        if "coord_names" in state:
            self.coord_names = [str(name) for name in (state.get("coord_names") or [])]
        self.target_mean_ = as_array(state.get("target_mean"))
        self.target_scale_ = as_array(state.get("target_scale"))
        self.label_mean_ = as_array(state.get("label_mean"))
        self.label_scale_ = as_array(state.get("label_scale"))
        self.label_median_ = as_array(state.get("label_median"))
        self.sequence_mean_ = {
            name: np.asarray(values, dtype=np.float32)
            for name, values in (state.get("sequence_mean") or {}).items()
        }
        self.sequence_scale_ = {
            name: np.asarray(values, dtype=np.float32)
            for name, values in (state.get("sequence_scale") or {}).items()
        }

        vocabularies = state.get("categorical_vocabularies") or []
        names = state.get("categorical_feature_names") or []
        raw_categoricals = np.asarray(self.sequence_bundle.static_categoricals, dtype=object)
        if vocabularies and names and raw_categoricals.size:
            self.categorical_encoder_ = CategoricalEncoder.from_vocabularies(names, vocabularies)
            self.categorical_codes_ = self.categorical_encoder_.transform(raw_categoricals)
            self.categorical_vocabularies = [list(vocabulary) for vocabulary in vocabularies]
            self.categorical_cardinalities = [len(vocabulary) + 1 for vocabulary in vocabularies]
        else:
            self.categorical_encoder_ = None
            self.categorical_codes_ = np.zeros((raw_categoricals.shape[0], 0), dtype=np.int64)

        self._is_setup = True

    def collate(self, point_indices) -> dict[str, Any]:
        """Public entry to the batch builder, so serving code need not reach for a private name."""
        return self._collate_points(point_indices)

    def _infer_grid_years(self) -> int:
        """How many calendar years a per-point grid must span to hold every observation.

        Derived from the data rather than configured, so a one-year dataset produces a one-row grid
        and a decade produces ten. Consumers that rasterise onto a calendar grid read this; the
        sequence encoders ignore it entirely.

        The span is measured per point and maximised, not measured across the dataset: a grid is
        anchored on each point's own latest observation, so what matters is the longest individual
        history, not the calendar range the dataset as a whole happens to cover.
        """
        longest = 0
        has_observations = False
        for per_point_times in (self.sequence_bundle.sequence_times or {}).values():
            for times in per_point_times:
                times = np.asarray(times)
                if not times.size:
                    continue
                has_observations = True
                # Counted in CALENDAR years, not decimal ones. A point spanning 2017.9 to 2018.1 is
                # 0.2 decimal years but occupies two rows, so measuring the decimal span would
                # under-allocate the grid and silently drop the earlier reading.
                longest = max(longest, int(np.floor(times[-1])) - int(np.floor(times[0])) + 1)
        if not has_observations:
            return 0
        return max(1, longest)

    # --- lifecycle ---------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        if self._is_setup:
            return

        train_idx, val_idx, test_idx = self._split_indices(self.sequence_bundle.num_points)
        self._fit_normalization(train_idx)
        self._fit_categoricals(train_idx)
        self.train_idx_, self.val_idx_, self.test_idx_ = train_idx, val_idx, test_idx
        self._is_setup = True

        self.X_train_frame_, self.y_train_frame_ = self._build_split_frames(train_idx)
        self.X_val_frame_, self.y_val_frame_ = self._build_split_frames(val_idx)
        self.X_test_frame_, self.y_test_frame_ = self._build_split_frames(test_idx)

    def train_dataloader(self):
        if not self._is_setup:
            self.setup("fit")
        # drop_last on TRAIN only: a trailing batch of one point makes BatchNorm1d raise, and a
        # handful of points is a noisy gradient regardless. Val/test/predict must keep every sample.
        drop_last = self.train_idx_ is not None and self.train_idx_.size > max(1, int(self.batch_size))
        return self._make_loader(self.train_idx_, shuffle=self.shuffle, drop_last=drop_last)

    def val_dataloader(self):
        if not self._is_setup:
            self.setup("fit")
        return self._make_loader(self.val_idx_)

    def test_dataloader(self):
        if not self._is_setup:
            self.setup("test")
        return self._make_loader(self.test_idx_)

    def predict_dataloader(self):
        if not self._is_setup:
            self.setup("predict")
        # Never shuffle: the trainer aligns predictions positionally with y_test_frame_.
        return self._make_loader(self.test_idx_)

    def _make_loader(self, point_indices, *, shuffle: bool = False, drop_last: bool = False):
        point_indices = np.asarray(point_indices, dtype=np.int64)
        batch_size = max(1, int(self.batch_size)) if point_indices.size else 1
        # build_loader, not a bare DataLoader: it keeps num_workers and persistent_workers from
        # shifting the global RNG stream, so they cannot change what a run trains to.
        return build_loader(
            _PointDataset(point_indices),
            batch_size=batch_size,
            drop_last=bool(drop_last),
            shuffle=bool(shuffle) and point_indices.size > 1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._collate_points,
        )

    # --- standardization (fitted on the train split only) ------------------

    @staticmethod
    def _finite(array: np.ndarray) -> np.ndarray:
        """Replace non-finite values with 0 so a single inf cannot poison a column statistic."""
        return np.nan_to_num(np.asarray(array, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _safe_scale(scale: np.ndarray) -> np.ndarray:
        scale = np.asarray(scale, dtype=np.float32)
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        return scale

    def _fit_normalization(self, train_idx: np.ndarray) -> None:
        indices = np.asarray(train_idx, dtype=np.int64)
        if indices.size == 0:
            return

        coords = np.asarray(self.sequence_bundle.coords)
        if coords.size and coords.shape[1]:
            train_coords = np.asarray(coords[indices], dtype=np.float64)
            self.coord_min_ = train_coords.min(axis=0)
            self.coord_max_ = train_coords.max(axis=0)

        static_features = np.asarray(self.sequence_bundle.static_features)
        if static_features.size:
            # Median-filled from the TRAIN rows only, exactly as the lab values below are: a
            # covariate gap costs one imputed value rather than the whole soil sample, and the fill
            # value must not be able to see validation or test. Statistics are computed AFTER the
            # fill so the model's inputs and the standardizer agree about where a filled cell lands.
            train_static = np.asarray(static_features[indices], dtype=np.float64)
            measured = np.isfinite(train_static)
            counts = measured.sum(axis=0)
            median = np.zeros(train_static.shape[1], dtype=np.float64)
            for column in range(train_static.shape[1]):
                if counts[column]:
                    median[column] = np.median(train_static[measured[:, column], column])
            filled = np.where(measured, train_static, median)
            self.static_median_ = median.astype(np.float32)
            self.static_mean_ = filled.mean(axis=0).astype(np.float32)
            self.static_scale_ = self._safe_scale(filled.std(axis=0))

        targets = np.asarray(self.sequence_bundle.targets)
        if targets.size:
            # Fit on ALREADY-transformed targets so the two stages compose; the model inverts them
            # in the opposite order.
            train_targets = self._apply_target_transform(self._finite(targets[indices]))
            self.target_mean_ = train_targets.mean(axis=0).astype(np.float32)
            self.target_scale_ = self._safe_scale(train_targets.std(axis=0))
            # In the SAME space the loss runs in, which is what makes this usable as-is: the loss
            # never sees raw units, so a covariance fitted on raw units would describe a different
            # geometry than the one the errors live in. Standardizing first also means the diagonal
            # comes out at 1 and this IS the correlation matrix.
            #
            # ddof=0 to match np.std above; ddof=1 would put n/(n-1) on the diagonal instead of 1.
            if indices.size > 1 and train_targets.shape[1] > 1:
                standardized = (train_targets - self.target_mean_) / self.target_scale_
                self.target_covariance_ = np.atleast_2d(
                    np.cov(standardized, rowvar=False, ddof=0)
                ).astype(np.float64)

        label_features = np.asarray(self.sequence_bundle.label_features)
        if label_features.size:
            # Measured cells only, exactly as for the sequence channels below: a column that is 37%
            # absent would otherwise have its own fill value counted into the mean it was derived
            # from, shrinking the spread and inflating every real reading on standardization.
            train_labels = np.asarray(label_features[indices], dtype=np.float64)
            measured = np.isfinite(train_labels)
            counts = measured.sum(axis=0)
            # A column with nothing measured in the train split cannot be filled from the data;
            # 0.0 with unit scale makes it a constant the model can only ignore, which is the
            # honest degenerate answer rather than a fabricated centre.
            median = np.zeros(train_labels.shape[1], dtype=np.float64)
            for column in range(train_labels.shape[1]):
                if counts[column]:
                    median[column] = np.median(train_labels[measured[:, column], column])
            # Statistics are computed AFTER the fill, so the model's inputs and the standardizer
            # agree about what a filled cell looks like: it lands wherever the train median lands,
            # not at an arbitrary offset from a mean fitted on a different population.
            filled = np.where(measured, train_labels, median)
            self.label_median_ = median.astype(np.float32)
            self.label_mean_ = filled.mean(axis=0).astype(np.float32)
            self.label_scale_ = self._safe_scale(filled.std(axis=0))

        # Per-modality, per-channel statistics over the train points' real observations. With no
        # padding at rest there is nothing to mask out - every row here is a genuine reading.
        for modality_name, per_point_values in (self.sequence_bundle.sequences or {}).items():
            channels = len(self.sequence_bundle.modality_columns.get(modality_name, []))
            selected = [
                index for index in indices if index < len(per_point_values) and len(per_point_values[index])
            ]
            if not selected:
                self.sequence_mean_[modality_name] = np.zeros(channels, dtype=np.float32)
                self.sequence_scale_[modality_name] = np.ones(channels, dtype=np.float32)
                continue

            stacked = self._finite(np.concatenate([per_point_values[index] for index in selected], axis=0))
            valid = np.concatenate(
                [self.sequence_bundle.validity_for(modality_name, index) for index in selected], axis=0
            )
            # Statistics over measured cells only. Median-filled cells are all equal to one value by
            # construction, so counting them pulls the mean toward that median and shrinks the
            # standard deviation - which would then inflate every real reading on standardization.
            counts = valid.sum(axis=0)
            mean = np.where(counts > 0, (stacked * valid).sum(axis=0) / np.maximum(counts, 1), 0.0)
            variance = np.where(
                counts > 1,
                (((stacked - mean) ** 2) * valid).sum(axis=0) / np.maximum(counts - 1, 1),
                0.0,
            )
            self.sequence_mean_[modality_name] = mean.astype(np.float32)
            self.sequence_scale_[modality_name] = self._safe_scale(np.sqrt(variance))

    def _fit_categoricals(self, train_idx: np.ndarray) -> None:
        """Fit the vocabulary on the train split, then encode every point against it.

        Encoding the full array against a train-only vocabulary is the point: a category that occurs
        only in validation or test is not in the vocabulary, so it lands on the reserved index and
        the model meets it exactly as it will meet a genuinely new category at inference.
        """
        raw = np.asarray(self.sequence_bundle.static_categoricals, dtype=object)
        names = self.categorical_feature_names
        if raw.ndim != 2 or raw.shape[1] == 0 or not names:
            self.categorical_encoder_ = None
            self.categorical_codes_ = np.zeros((raw.shape[0] if raw.ndim == 2 else 0, 0), dtype=np.int64)
            self.categorical_cardinalities = []
            self.categorical_vocabularies = []
            return

        indices = np.asarray(train_idx, dtype=np.int64)
        # With no train split there is nothing to fit a vocabulary from; an empty one sends every
        # category to the reserved index, which is the correct degenerate behaviour rather than an
        # excuse to fall back on the full frame.
        fit_rows = raw[indices] if indices.size else raw[:0]

        encoder = CategoricalEncoder().fit(fit_rows, names)
        self.categorical_encoder_ = encoder
        self.categorical_codes_ = encoder.transform(raw)
        self.categorical_cardinalities = encoder.cardinalities
        self.categorical_vocabularies = encoder.vocabularies

    def _standardize_static(self, values: np.ndarray, validity: Optional[np.ndarray] = None) -> np.ndarray:
        """Covariates -> standardized, missing cells filled from the train median.

        `_finite` is not the fill: it maps NaN to 0, which after centring is a fabricated value
        sitting wherever 0 happens to fall. The train median is where the column actually is. The
        flags for the gappy columns are appended here so the model can tell the two apart, and are
        computed from the RAW values before the fill, since afterwards the information is gone.
        """
        if values.size == 0:
            return np.asarray(values, dtype=np.float32)

        raw = np.asarray(values, dtype=np.float64)
        measured = np.isfinite(raw)
        filled = raw if self.static_median_ is None else np.where(measured, raw, self.static_median_)

        if self.static_mean_ is None:
            standardized = self._finite(filled).astype(np.float32)
        else:
            standardized = ((self._finite(filled) - self.static_mean_) / self.static_scale_).astype(np.float32)

        flags = self._static_validity_channels(measured, validity)
        if flags.size:
            standardized = np.concatenate([standardized, flags], axis=1)
        return standardized

    def _static_validity_channels(
        self, measured: np.ndarray, validity: Optional[np.ndarray]
    ) -> np.ndarray:
        """The measured-vs-filled flags, restricted to the covariates that actually have gaps."""
        if not self.static_validity_names:
            return np.empty((measured.shape[0], 0), dtype=np.float32)
        if validity is not None and np.asarray(validity).size:
            return np.asarray(validity, dtype=np.float32).reshape(measured.shape[0], -1)
        # No precomputed mask (a caller standardizing raw rows, e.g. at serving time): derive the
        # flags from the values themselves, taking the same columns in the same order.
        positions = [self.static_feature_names.index(name) for name in self.static_validity_names]
        return measured[:, positions].astype(np.float32)

    def _normalize_coords(self, values: np.ndarray) -> np.ndarray:
        """lat/lon -> the [-1, 1] interval the harmonic frequencies are defined on.

        A min-max onto the TRAIN bounding box, not a z-score. The distinction matters: the encoder's
        frequency k is ``2**k * pi``, so it resolves about 1/2**k of the interval, and that is a
        statement about the study area only while the study area IS the interval. Standardizing
        instead would make the same k mean a different distance on every dataset.

        A point outside the training extent lands outside [-1, 1] and is NOT clipped. Clipping would
        collapse every point beyond the edge onto the boundary, making a distant location
        indistinguishable from one just outside; sine and cosine are periodic and handle the
        overflow without any special case.
        """
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0 or self.coord_min_ is None or self.coord_max_ is None:
            return values.astype(np.float32)

        span = np.asarray(self.coord_max_, dtype=np.float64) - np.asarray(self.coord_min_, dtype=np.float64)
        # A degenerate axis - every training point on one meridian, or a single training point -
        # carries no positional information at all. Mapping it to a constant 0 is the honest answer;
        # dividing by it would produce inf and poison every downstream channel.
        safe_span = np.where(np.isfinite(span) & (np.abs(span) > 1e-12), span, 1.0)
        normalized = 2.0 * (values - self.coord_min_) / safe_span - 1.0
        normalized = np.where(np.abs(span) > 1e-12, normalized, 0.0)
        return normalized.astype(np.float32)

    def _standardize_labels(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lab values -> ``(standardized, validity)``, filling what is missing from the train median.

        Validity is captured BEFORE the fill - afterwards the information is gone for good, and a
        filled cell would be indistinguishable from a measured one. That is the same failure the
        time-series validity channels exist to prevent.
        """
        values = np.asarray(values, dtype=np.float64)
        validity = np.isfinite(values)
        if values.size == 0 or self.label_median_ is None:
            return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), validity

        filled = np.where(validity, values, self.label_median_)
        return ((filled - self.label_mean_) / self.label_scale_).astype(np.float32), validity

    def _apply_target_transform(self, values: np.ndarray) -> np.ndarray:
        """Forward target transform. Mirrors LogTransformer in yg_eo_soilnet.utils (10 * log1p)."""
        if self.target_transform != "log1p" or values.size == 0:
            return values

        if np.any(values < 0):
            import warnings

            warnings.warn(
                f"target_transform='log1p' clipped {int((values < 0).sum())} negative target value(s) "
                "to 0; log1p is undefined below -1 and these targets are expected to be non-negative.",
                RuntimeWarning,
                stacklevel=2,
            )
            values = np.maximum(values, 0.0)
        return 10.0 * np.log1p(values)

    def _standardize_targets(self, values: np.ndarray) -> np.ndarray:
        transformed = self._apply_target_transform(self._finite(values))
        if self.target_mean_ is None or values.size == 0:
            return transformed.astype(np.float32)
        return ((transformed - self.target_mean_) / self.target_scale_).astype(np.float32)

    def _standardize_sequence(self, modality_name: str, values: np.ndarray) -> np.ndarray:
        mean = self.sequence_mean_.get(modality_name)
        if mean is None or values.size == 0:
            return self._finite(values).astype(np.float32)
        return ((self._finite(values) - mean) / self.sequence_scale_[modality_name]).astype(np.float32)

    # --- batch assembly ----------------------------------------------------

    def _collate_points(self, point_indices) -> dict[str, Any]:
        indices = np.asarray(point_indices, dtype=np.int64)
        bundle = self.sequence_bundle

        static_features = np.asarray(bundle.static_features)
        static_validity = np.asarray(bundle.static_validity)
        targets = np.asarray(bundle.targets)
        x_static = (
            self._standardize_static(
                static_features[indices],
                static_validity[indices] if static_validity.size else None,
            )
            if static_features.size
            else np.empty((indices.size, 0), dtype=np.float32)
        )
        y = self._standardize_targets(targets[indices]) if targets.size else np.empty(
            (indices.size, 0), dtype=np.float32
        )

        # Indices, never scaled: they address an embedding table rather than measuring anything.
        # Always well-shaped, so a dataset with no categoricals needs no None branch downstream.
        if self.categorical_codes_ is not None and self.categorical_codes_.shape[1]:
            x_categorical = self.categorical_codes_[indices]
        else:
            x_categorical = np.zeros((indices.size, 0), dtype=np.int64)

        # ALL lab columns, always, in bundle order. The model index_selects the ones it was built
        # for; sending only a selected subset would make the batch depend on which architecture is
        # training, and this datamodule is shared and cached across several.
        label_features = np.asarray(bundle.label_features)
        if label_features.size:
            x_labels, x_label_validity = self._standardize_labels(label_features[indices])
        else:
            x_labels = np.zeros((indices.size, 0), dtype=np.float32)
            x_label_validity = np.zeros((indices.size, 0), dtype=bool)

        # Normalized against the TRAIN bounding box, never against this batch. Always well-shaped
        # and zero-width when coordinates are switched off, exactly as x_categorical is, so nothing
        # downstream needs a None branch.
        coords = np.asarray(bundle.coords)
        if coords.size and coords.shape[1]:
            x_coords = self._normalize_coords(coords[indices])
        else:
            x_coords = np.zeros((indices.size, 0), dtype=np.float32)

        batch: dict[str, Any] = {
            "x_static": torch.as_tensor(x_static, dtype=torch.float32),
            "x_categorical": torch.as_tensor(x_categorical, dtype=torch.long),
            "x_coords": torch.as_tensor(x_coords, dtype=torch.float32),
            "x_labels": torch.as_tensor(x_labels, dtype=torch.float32),
            "x_label_validity": torch.as_tensor(x_label_validity, dtype=torch.bool),
            "y": torch.as_tensor(y, dtype=torch.float32),
            "point_ids": [bundle.point_ids[index] for index in indices.tolist()],
            "target_names": list(bundle.target_names),
            "label_feature_names": list(bundle.label_feature_names),
            "sequences": {},
            "sequence_mask": {},
            "sequence_time": {},
            "sequence_validity": {},
        }

        for modality_name, per_point_values in (bundle.sequences or {}).items():
            per_point_times = bundle.sequence_times.get(modality_name, [])
            channels = len(bundle.modality_columns.get(modality_name, []))

            selected_values = []
            selected_times = []
            selected_validity = []
            for index in indices.tolist():
                values = np.asarray(per_point_values[index], dtype=np.float32)
                # float64 throughout: see to_decimal_year on why float32 blurs the seasonal signal.
                times = np.asarray(per_point_times[index], dtype=np.float64)
                validity = bundle.validity_for(modality_name, index)
                if self.max_sequence_length is not None and len(times) > self.max_sequence_length:
                    # Keep the most recent readings: they sit closest to the sampling date the
                    # target was measured at.
                    values = values[-self.max_sequence_length :]
                    times = times[-self.max_sequence_length :]
                    validity = validity[-self.max_sequence_length :]
                selected_values.append(values)
                selected_times.append(times)
                selected_validity.append(validity)

            # Pad to this batch's longest series, never to a global constant. At least one column
            # so a batch where nothing was observed still produces well-shaped tensors.
            max_length = max((len(times) for times in selected_times), default=0)
            max_length = max(1, max_length)

            padded = np.zeros((indices.size, max_length, channels), dtype=np.float32)
            mask = np.zeros((indices.size, max_length), dtype=bool)
            times_padded = np.zeros((indices.size, max_length), dtype=np.float64)
            validity_padded = np.zeros((indices.size, max_length, channels), dtype=bool)

            for row, (values, times, validity) in enumerate(
                zip(selected_values, selected_times, selected_validity)
            ):
                length = len(times)
                if length == 0:
                    continue
                padded[row, :length] = self._standardize_sequence(modality_name, values)
                mask[row, :length] = True
                times_padded[row, :length] = times
                validity_padded[row, :length] = validity

            batch["sequences"][modality_name] = torch.as_tensor(padded, dtype=torch.float32)
            batch["sequence_mask"][modality_name] = torch.as_tensor(mask, dtype=torch.bool)
            # float64, matching the array above. Downcasting here would undo the precision the
            # decimal-year representation exists to preserve.
            batch["sequence_time"][modality_name] = torch.as_tensor(times_padded, dtype=torch.float64)
            batch["sequence_validity"][modality_name] = torch.as_tensor(validity_padded, dtype=torch.bool)

        return batch

    # --- splits and evaluation frames --------------------------------------

    def _build_split_frames(self, point_indices):
        indices = np.asarray(point_indices, dtype=np.int64)
        static_features = np.asarray(self.sequence_bundle.static_features)
        targets = np.asarray(self.sequence_bundle.targets)

        if self.static_feature_names and len(self.static_feature_names) == static_features.shape[1]:
            feature_columns = list(self.static_feature_names)
        else:
            feature_columns = [f"feature_{index}" for index in range(static_features.shape[1])]

        if self.target_names and len(self.target_names) == targets.shape[1]:
            target_columns = list(self.target_names)
        else:
            target_columns = [f"target_{index}" for index in range(targets.shape[1])]

        if indices.size == 0:
            return pd.DataFrame(columns=feature_columns), pd.DataFrame(columns=target_columns)

        # Raw, untransformed values: the trainer compares predictions against these in original units.
        x_frame = pd.DataFrame(static_features[indices], columns=feature_columns)
        y_frame = pd.DataFrame(targets[indices], columns=target_columns)
        return x_frame, y_frame

    def _split_indices(self, num_rows: int):
        """Train/val/test point indices.

        Resolved from the run's shared :class:`SplitPlan` when one was supplied, so this family and
        the sklearn family hold out the same points. Otherwise it falls back to the local ratio
        carve, in which `test_size` and `val_size` apply sequentially.
        """
        empty = np.array([], dtype=np.int64)
        indices = np.arange(num_rows, dtype=np.int64)
        if num_rows <= 1:
            return indices, empty, empty

        if self.split_plan is not None:
            return self._planned_split_indices(num_rows)

        test_idx, train_val_idx = self._carve_out(indices, self.test_size)
        if train_val_idx.size <= 1:
            return train_val_idx, empty, test_idx

        val_idx, train_idx = self._carve_out(train_val_idx, self.val_size)
        return train_idx, val_idx, test_idx

    def _planned_split_indices(self, num_rows: int):
        """Resolve the shared plan against this bundle's own point ordering."""
        point_ids = list(self.sequence_bundle.point_ids)
        if len(point_ids) != num_rows:
            raise ValueError(
                f"The bundle carries {len(point_ids)} point id(s) for {num_rows} row(s); the shared "
                f"split cannot be resolved."
            )
        train_idx, val_idx, test_idx = self.split_plan.split_indices(point_ids)
        assigned = train_idx.size + val_idx.size + test_idx.size
        if assigned < num_rows:
            # Expected under population_policy=intersect: these are points the other families
            # could not use, so the shared plan deliberately gives them no split.
            self._log(
                f"{num_rows - assigned} of {num_rows} point(s) are outside the shared split "
                f"population and are used by no split "
                f"(population_policy={self.split_plan.population_policy})."
            )
        if train_idx.size == 0:
            raise ValueError(
                "The shared split plan left this datamodule with no training points. Check "
                "split.population_policy and the eligibility of this family's rows."
            )
        return train_idx, val_idx, test_idx

    def _log(self, message: str) -> None:
        """Best-effort info logging; a LightningDataModule has no logger of its own."""
        import logging

        logging.getLogger(__name__).info(message)

    def _carve_out(self, indices: np.ndarray, fraction: float):
        """Split off `fraction` of `indices`, returning (held_out, remainder).

        A fraction of 0 means "no holdout", which train_test_split rejects outright rather than
        treating as empty - so handle it here and keep val_size=0 or test_size=0 usable.
        """
        fraction = min(max(float(fraction), 0.0), 0.9)
        if fraction <= 0.0:
            return np.array([], dtype=np.int64), np.asarray(indices, dtype=np.int64)

        from sklearn.model_selection import train_test_split

        remainder, held_out = train_test_split(
            indices, test_size=fraction, random_state=self.seed, shuffle=True
        )
        return np.asarray(held_out, dtype=np.int64), np.asarray(remainder, dtype=np.int64)
