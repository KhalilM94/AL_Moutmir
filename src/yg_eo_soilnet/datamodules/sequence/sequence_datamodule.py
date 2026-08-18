from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import LightningDataModule

from yg_eo_soilnet.datamodules.categorical import CategoricalEncoder
from yg_eo_soilnet.datamodules.sequence.sequence_bundle import SoilSequenceBundle


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
    ):
        super().__init__()
        self.sequence_bundle = deepcopy(SoilSequenceBundle.from_mapping(sequence_bundle))
        self.target_transform = None if target_transform is None else str(target_transform).lower()
        if self.target_transform not in {None, "none", "log1p"}:
            raise ValueError("target_transform must be None or 'log1p'")
        if self.target_transform == "none":
            self.target_transform = None

        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
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
        # Categorical vocabulary, fitted on the train split only in setup() for the same reason the
        # scaler is: a category seen only in validation or test must reach the model as "unknown",
        # exactly as an unseen category would at inference time.
        self.categorical_encoder_: Optional[CategoricalEncoder] = None
        self.categorical_codes_: Optional[np.ndarray] = None
        self.sequence_mean_: dict[str, np.ndarray] = {}
        self.sequence_scale_: dict[str, np.ndarray] = {}
        self.target_mean_: Optional[np.ndarray] = None
        self.target_scale_: Optional[np.ndarray] = None
        # Lab-value statistics, train-only for the same reason as everything above. The median is
        # kept separately from the mean because it is the FILL value, not a centring constant: a
        # skewed column's mean sits somewhere no sample actually is.
        self.label_mean_: Optional[np.ndarray] = None
        self.label_scale_: Optional[np.ndarray] = None
        self.label_median_: Optional[np.ndarray] = None

        # The shape contract the Lightning config factory reads off the datamodule. Note the
        # deliberate absence of `temporal_steps` and `edge_attr_dim`: the model is length-agnostic
        # and graph-free, and the factory only injects attributes that actually exist here.
        self.static_dim = int(np.asarray(self.sequence_bundle.static_features).shape[1])
        self.target_dim = int(np.asarray(self.sequence_bundle.targets).shape[1])
        self.static_feature_names = list(self.sequence_bundle.static_feature_names)
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
            "static_feature_names": list(self.static_feature_names),
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
        return DataLoader(
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

        static_features = np.asarray(self.sequence_bundle.static_features)
        if static_features.size:
            train_static = self._finite(static_features[indices])
            self.static_mean_ = train_static.mean(axis=0).astype(np.float32)
            self.static_scale_ = self._safe_scale(train_static.std(axis=0))

        targets = np.asarray(self.sequence_bundle.targets)
        if targets.size:
            # Fit on ALREADY-transformed targets so the two stages compose; the model inverts them
            # in the opposite order.
            train_targets = self._apply_target_transform(self._finite(targets[indices]))
            self.target_mean_ = train_targets.mean(axis=0).astype(np.float32)
            self.target_scale_ = self._safe_scale(train_targets.std(axis=0))

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

    def _standardize_static(self, values: np.ndarray) -> np.ndarray:
        if self.static_mean_ is None or values.size == 0:
            return self._finite(values).astype(np.float32)
        return ((self._finite(values) - self.static_mean_) / self.static_scale_).astype(np.float32)

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
        targets = np.asarray(bundle.targets)
        x_static = self._standardize_static(static_features[indices]) if static_features.size else np.empty(
            (indices.size, 0), dtype=np.float32
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

        batch: dict[str, Any] = {
            "x_static": torch.as_tensor(x_static, dtype=torch.float32),
            "x_categorical": torch.as_tensor(x_categorical, dtype=torch.long),
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
        """Split points into train/val/test. `test_size` and `val_size` apply sequentially."""
        empty = np.array([], dtype=np.int64)
        indices = np.arange(num_rows, dtype=np.int64)
        if num_rows <= 1:
            return indices, empty, empty

        test_idx, train_val_idx = self._carve_out(indices, self.test_size)
        if train_val_idx.size <= 1:
            return train_val_idx, empty, test_idx

        val_idx, train_idx = self._carve_out(train_val_idx, self.val_size)
        return train_idx, val_idx, test_idx

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
