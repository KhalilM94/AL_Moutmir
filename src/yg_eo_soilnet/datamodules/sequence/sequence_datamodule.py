from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import LightningDataModule

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
        self.sequence_mean_: dict[str, np.ndarray] = {}
        self.sequence_scale_: dict[str, np.ndarray] = {}
        self.target_mean_: Optional[np.ndarray] = None
        self.target_scale_: Optional[np.ndarray] = None

        # The shape contract the Lightning config factory reads off the datamodule. Note the
        # deliberate absence of `temporal_steps` and `edge_attr_dim`: the model is length-agnostic
        # and graph-free, and the factory only injects attributes that actually exist here.
        self.static_dim = int(np.asarray(self.sequence_bundle.static_features).shape[1])
        self.target_dim = int(np.asarray(self.sequence_bundle.targets).shape[1])
        self.static_feature_names = list(self.sequence_bundle.static_feature_names)
        self.target_names = list(self.sequence_bundle.target_names)
        self.modality_dims = dict(self.sequence_bundle.modality_dims)
        self.temporal_enabled = bool(self.sequence_bundle.temporal_enabled and self.modality_dims)
        self.grid_years = self._infer_grid_years()

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

    def _standardize_static(self, values: np.ndarray) -> np.ndarray:
        if self.static_mean_ is None or values.size == 0:
            return self._finite(values).astype(np.float32)
        return ((self._finite(values) - self.static_mean_) / self.static_scale_).astype(np.float32)

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

        batch: dict[str, Any] = {
            "x_static": torch.as_tensor(x_static, dtype=torch.float32),
            "y": torch.as_tensor(y, dtype=torch.float32),
            "point_ids": [bundle.point_ids[index] for index in indices.tolist()],
            "target_names": list(bundle.target_names),
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
