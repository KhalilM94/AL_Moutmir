from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import LightningDataModule

from yg_eo_soilnet.datamodules.lightning.spatiotemporal_graph import SpatiotemporalGraph
from yg_eo_soilnet.datamodules.splitting import SplitPlan
from yg_eo_soilnet.targets import select_target_columns


def _as_tensor(value, dtype=None):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if dtype is not None else value
    return torch.as_tensor(value, dtype=dtype)

@dataclass
class GraphSample:
    x_static: Any
    y: Any
    edge_index: Any
    edge_attr: Any
    coords: Any
    point_ids: list[Any]
    temporal_features: dict[str, Any]
    temporal_lengths: dict[str, Any]
    temporal_masks: dict[str, Any]
    node_indices: Any
    target_names: list[str]
    temporal_enabled: bool

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def keys(self):
        return (
            "x_static",
            "y",
            "edge_index",
            "edge_attr",
            "coords",
            "point_ids",
            "temporal_features",
            "temporal_lengths",
            "temporal_masks",
            "node_indices",
            "target_names",
            "temporal_enabled",
        )


class _NodeIndexDataset(Dataset):
    """Yields node indices; the DataLoader batches them and collate builds the subgraph sample.

    This is what turns one giant full-batch step per epoch into ceil(n_nodes / batch_size) steps.
    """

    def __init__(self, node_indices: np.ndarray):
        self.node_indices = np.asarray(node_indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.node_indices.size)

    def __getitem__(self, index: int) -> int:
        return int(self.node_indices[index])


class SingleNodeGraphDataModule(LightningDataModule):
    def __init__(
        self,
        spatiotemporal_graph: "SpatiotemporalGraph | Mapping[str, Any]",
        batch_size: int = 1,
        val_size: float = 0.2,
        test_size: float = 0.2,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        seed: int = 42,
        shuffle: bool = False,
        target_transform: Optional[str] = None,
        split_plan: Optional["SplitPlan"] = None,
        active_targets: Optional[list[str]] = None,
    ):
        super().__init__()
        self.spatiotemporal_graph = deepcopy(SpatiotemporalGraph.from_mapping(spatiotemporal_graph))
        # See the sequence datamodule for the contract: None keeps every target (the joint head),
        # a list narrows to what this run fits, and everything downstream follows from the fields
        # rewritten here. Residuals are narrowed alongside because they are validated against
        # target_names and would otherwise disagree with it.
        self.active_targets = list(active_targets) if active_targets else None
        all_target_names = list(self.spatiotemporal_graph.target_names)
        narrowed, target_names, indices = select_target_columns(
            self.spatiotemporal_graph.targets, all_target_names, self.active_targets
        )
        self.spatiotemporal_graph.targets = narrowed
        self.spatiotemporal_graph.target_names = target_names
        if indices is not None:
            residuals = np.asarray(self.spatiotemporal_graph.residuals)
            if residuals.ndim == 2 and residuals.shape[1] == len(all_target_names):
                self.spatiotemporal_graph.residuals = residuals[:, indices]
        self.target_transform = None if target_transform is None else str(target_transform).lower()
        if self.target_transform not in {None, "none", "log1p"}:
            raise ValueError("target_transform must be None or 'log1p'")
        if self.target_transform == "none":
            self.target_transform = None
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        # The run's shared split; when present it decides train/val/test and the ratios above are
        # ignored. See the sequence datamodule for the same contract.
        self.split_plan = split_plan
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed
        self.shuffle = shuffle

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
        self.temporal_mean_: dict[str, np.ndarray] = {}
        self.temporal_scale_: dict[str, np.ndarray] = {}
        self.target_mean_: Optional[np.ndarray] = None
        self.target_scale_: Optional[np.ndarray] = None

        self.static_dim = int(np.asarray(self.spatiotemporal_graph.get("static_features", np.empty((0, 0)))).shape[1])
        self.target_dim = int(np.asarray(self.spatiotemporal_graph.get("targets", np.empty((0, 0)))).shape[1])
        self.static_feature_names = list(self.spatiotemporal_graph.get("static_feature_names", []))
        self.target_names = list(self.spatiotemporal_graph.get("target_names", []))
        temporal_bundle = dict(self.spatiotemporal_graph.get("temporal_features", {}) or {})
        self.temporal_lengths = dict(self.spatiotemporal_graph.get("temporal_lengths", {}) or {})
        self.temporal_masks = dict(self.spatiotemporal_graph.get("temporal_masks", {}) or {})
        self.temporal_enabled = bool(self.spatiotemporal_graph.get("temporal_enabled", False) and temporal_bundle)
        self.temporal_steps = None
        self.modality_dims: dict[str, int] = {}
        if temporal_bundle:
            for modality_name, array in temporal_bundle.items():
                modality_array = np.asarray(array)
                if modality_array.ndim == 3:
                    if self.temporal_steps is None:
                        self.temporal_steps = int(modality_array.shape[1])
                    modality_dim = int(modality_array.shape[2])
                elif modality_array.ndim == 2:
                    if self.temporal_steps is None:
                        self.temporal_steps = 1
                    modality_dim = int(modality_array.shape[1])
                else:
                    modality_dim = 0

                normalized_name = str(modality_name).lower()
                self.modality_dims[normalized_name] = modality_dim

        edge_attr = np.asarray(self.spatiotemporal_graph.get("edge_attr", np.empty((0, self.target_dim or 1))))
        self.edge_attr_dim = int(edge_attr.shape[1]) if edge_attr.ndim == 2 and edge_attr.size else 0

    def setup(self, stage: Optional[str] = None) -> None:
        if self._is_setup:
            return

        train_idx, val_idx, test_idx = self._split_indices(len(self.spatiotemporal_graph.static_features))
        self._fit_normalization(train_idx)

        self.train_idx_, self.val_idx_, self.test_idx_ = train_idx, val_idx, test_idx
        # Samples are built per batch by _collate_nodes; building full-split ones here was
        # leftover from the full-batch design and their contents were never read.
        self._warn_if_batching_breaks_the_graph(train_idx)
        self._is_setup = True
        self.X_train_frame_, self.y_train_frame_ = self._build_split_frames(train_idx)
        self.X_val_frame_, self.y_val_frame_ = self._build_split_frames(val_idx)
        self.X_test_frame_, self.y_test_frame_ = self._build_split_frames(test_idx)

    def train_dataloader(self):
        if not self._is_setup:
            self.setup("fit")
        # drop_last on TRAIN only: a trailing batch of one node makes BatchNorm1d raise, and a
        # handful of nodes is a noisy gradient regardless. Val/test/predict must keep every sample -
        # _build_evaluation_frame aligns predictions against y_test_frame_ row by row.
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
        # Never shuffle: _build_evaluation_frame aligns predictions positionally with y_test_frame_.
        return self._make_loader(self.test_idx_)

    def _warn_if_batching_breaks_the_graph(self, train_idx: np.ndarray) -> None:
        """Node mini-batching keeps only edges whose BOTH endpoints land in the same batch.

        For a radius graph over thousands of nodes that is close to zero, so message passing goes
        silently inert and the "graph" model is really an MLP. Only meaningful when the spatial
        graph is on, so the check is skipped otherwise.
        """
        if not bool(self.spatiotemporal_graph.get("spatial_graph_enabled", False)):
            return

        edge_index = np.asarray(self.spatiotemporal_graph.get("edge_index", np.zeros((2, 0))))
        if edge_index.size == 0 or train_idx.size == 0:
            return

        batch_size = max(1, int(self.batch_size))
        if batch_size >= train_idx.size:
            return  # full-batch: every edge survives

        # Expected retention for a random partition: both endpoints in the same batch.
        retained = (batch_size - 1) / max(1, train_idx.size - 1)
        import warnings

        warnings.warn(
            f"spatial_graph_enabled=True but batch_size={batch_size} over {train_idx.size} train "
            f"nodes retains only ~{retained:.1%} of edges per batch; message passing is effectively "
            "inert. Use full-batch training or neighbour sampling. NOTE: edge_attr also carries "
            "target residuals computed before the split (known leakage) - graph-enabled metrics "
            "are not trustworthy until that is fixed.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _collate_nodes(self, node_indices) -> GraphSample:
        return self._build_graph_sample(np.asarray(node_indices, dtype=np.int64))

    def _make_loader(self, node_indices, *, shuffle: bool = False, drop_last: bool = False):
        node_indices = np.asarray(node_indices, dtype=np.int64)
        # A split can be empty (e.g. val_size=0); fall back to one empty batch.
        batch_size = max(1, int(self.batch_size)) if node_indices.size else 1
        return DataLoader(
            _NodeIndexDataset(node_indices),
            batch_size=batch_size,
            drop_last=bool(drop_last),
            shuffle=bool(shuffle) and node_indices.size > 1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._collate_nodes,
        )

    # --- standardization (fitted on the train split only) ---------------------

    @staticmethod
    def _finite(array: np.ndarray) -> np.ndarray:
        """Replace non-finite values with 0 so a single inf cannot poison a column statistic."""
        return np.nan_to_num(np.asarray(array, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _safe_scale(scale: np.ndarray) -> np.ndarray:
        scale = np.asarray(scale, dtype=np.float32)
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        return scale

    def _observed_mask(self, modality_name: str, indices: np.ndarray, expected_shape) -> Optional[np.ndarray]:
        """The (n_selected, T) boolean mask for `indices`, or None when no usable mask exists."""
        mask = self.temporal_masks.get(modality_name)
        if mask is None:
            return None
        mask_array = np.asarray(mask)
        if mask_array.ndim != 2 or mask_array.shape[0] <= int(indices.max()):
            return None
        selected = mask_array[indices].astype(bool)
        return selected if selected.shape == tuple(expected_shape) else None

    def _fit_normalization(self, train_idx) -> None:
        """Fit feature and target statistics on training nodes only, to avoid leakage."""
        indices = np.asarray(train_idx, dtype=np.int64)
        if indices.size == 0:
            return

        static_features = np.asarray(self.spatiotemporal_graph.static_features)
        if static_features.size:
            train_static = self._finite(static_features[indices])
            self.static_mean_ = train_static.mean(axis=0).astype(np.float32)
            self.static_scale_ = self._safe_scale(train_static.std(axis=0))

        targets = np.asarray(self.spatiotemporal_graph.targets)
        if targets.size:
            # Fit the standardization on ALREADY-transformed targets so the two stages compose;
            # the module inverts them in the opposite order.
            train_targets = self._apply_target_transform(self._finite(targets[indices]))
            self.target_mean_ = train_targets.mean(axis=0).astype(np.float32)
            self.target_scale_ = self._safe_scale(train_targets.std(axis=0))

        # Per-modality, per-channel statistics over observed timesteps only, so padding does not
        # drag the mean toward zero.
        for modality_name, array in (self.spatiotemporal_graph.temporal_features or {}).items():
            modality_array = np.asarray(array)
            if modality_array.ndim != 3 or modality_array.shape[0] <= int(indices.max()):
                continue
            values = self._finite(modality_array[indices])
            # Slice the mask to the train split BEFORE comparing shapes: the mask spans every node
            # while `values` is already subset, so comparing the two raw shapes is always False and
            # silently falls through to counting the padding.
            observed = self._observed_mask(modality_name, indices, values.shape[:2])
            if observed is None:
                observed = np.ones(values.shape[0] * values.shape[1], dtype=bool)
            else:
                observed = observed.reshape(-1)
            flat = values.reshape(-1, values.shape[2])[observed]
            if flat.size == 0:
                flat = values.reshape(-1, values.shape[2])
            self.temporal_mean_[modality_name] = flat.mean(axis=0).astype(np.float32)
            self.temporal_scale_[modality_name] = self._safe_scale(flat.std(axis=0))

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

    def _standardize_modality(
        self,
        modality_name: str,
        values: np.ndarray,
        observed: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        mean = self.temporal_mean_.get(modality_name)
        if mean is None or values.size == 0:
            standardized = self._finite(values).astype(np.float32)
        else:
            standardized = (
                (self._finite(values) - mean) / self.temporal_scale_[modality_name]
            ).astype(np.float32)
        # An unobserved step is a raw 0, which standardizes to -mean/scale - a multi-sigma spike the
        # encoder cannot tell apart from a real reading. Re-zero it so "missing" stays neutral.
        if observed is not None and standardized.ndim == 3:
            standardized = standardized * observed[..., None].astype(np.float32)
        return standardized

    def _build_graph_sample(self, node_indices) -> GraphSample:
        indices = np.asarray(node_indices, dtype=np.int64)
        static_features = np.asarray(self.spatiotemporal_graph.get("static_features", np.empty((0, 0))))
        targets = np.asarray(self.spatiotemporal_graph.get("targets", np.empty((0, 0))))
        coords = np.asarray(self.spatiotemporal_graph.get("coords", np.empty((0, 0))))
        point_ids = list(self.spatiotemporal_graph.get("point_ids", []))
        temporal_bundle = dict(self.spatiotemporal_graph.get("temporal_features", {}) or {})
        temporal_lengths = dict(self.spatiotemporal_graph.get("temporal_lengths", {}) or {})
        temporal_masks = dict(self.spatiotemporal_graph.get("temporal_masks", {}) or {})

        edge_index = np.asarray(self.spatiotemporal_graph.get("edge_index", np.zeros((2, 0), dtype=np.int64)))
        edge_attr = np.asarray(self.spatiotemporal_graph.get("edge_attr", np.empty((0, self.target_dim or 1))))
        if edge_index.size:
            source_nodes, target_nodes = edge_index
            keep_mask = np.isin(source_nodes, indices) & np.isin(target_nodes, indices)
            edge_index = edge_index[:, keep_mask]
            edge_attr = edge_attr[keep_mask]
            reindex = {int(old_index): new_index for new_index, old_index in enumerate(indices.tolist())}
            edge_index = np.asarray(
                [[reindex[int(source)] for source in edge_index[0]], [reindex[int(target)] for target in edge_index[1]]],
                dtype=np.int64,
            )

        # SpatiotemporalGraph guarantees temporal arrays are in static point order, so the same
        # node indices used for static_features apply directly here.
        subset_temporal: dict[str, Any] = {}
        subset_temporal_lengths: dict[str, Any] = {}
        subset_temporal_masks: dict[str, Any] = {}
        max_index = int(indices.max()) if indices.size else -1
        for modality_name, modality_array in temporal_bundle.items():
            modality_array = np.asarray(modality_array)
            if modality_array.size == 0 or modality_array.ndim == 0:
                continue
            if modality_array.shape[0] <= max_index:
                continue
            selected = modality_array[indices]
            observed = self._observed_mask(modality_name, indices, selected.shape[:2])
            subset_temporal[modality_name] = self._standardize_modality(modality_name, selected, observed)
            if modality_name in temporal_lengths:
                subset_temporal_lengths[modality_name] = np.asarray(temporal_lengths[modality_name])[indices]
            if modality_name in temporal_masks:
                subset_temporal_masks[modality_name] = np.asarray(temporal_masks[modality_name])[indices]

        # Standardized for the network; _build_split_frames keeps the raw arrays for evaluation.
        x_static = self._standardize_static(static_features[indices])
        y = self._standardize_targets(targets[indices])
        coords = coords[indices] if coords.size else coords
        selected_point_ids = [point_ids[index] for index in indices.tolist()] if point_ids else indices.tolist()

        tensor_dtype = getattr(torch, "float32", None)
        long_dtype = getattr(torch, "long", None)

        return GraphSample(
            x_static=_as_tensor(x_static, dtype=tensor_dtype),
            y=_as_tensor(y, dtype=tensor_dtype),
            edge_index=_as_tensor(edge_index, dtype=long_dtype),
            edge_attr=_as_tensor(edge_attr, dtype=tensor_dtype),
            coords=_as_tensor(coords, dtype=tensor_dtype),
            point_ids=selected_point_ids,
            temporal_features={key: _as_tensor(value, dtype=tensor_dtype) for key, value in subset_temporal.items()},
            temporal_lengths={key: _as_tensor(value, dtype=long_dtype) for key, value in subset_temporal_lengths.items()},
            temporal_masks={key: _as_tensor(value, dtype=long_dtype) for key, value in subset_temporal_masks.items()},
            node_indices=_as_tensor(indices, dtype=long_dtype),
            target_names=list(self.spatiotemporal_graph.get("target_names", [])),
            temporal_enabled=self.temporal_enabled,
        )

    def _build_split_frames(self, node_indices):
        indices = np.asarray(node_indices, dtype=np.int64)
        static_features = np.asarray(self.spatiotemporal_graph.get("static_features", np.empty((0, 0))))
        targets = np.asarray(self.spatiotemporal_graph.get("targets", np.empty((0, 0))))

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

        x_frame = pd.DataFrame(static_features[indices], columns=feature_columns)
        y_frame = pd.DataFrame(targets[indices], columns=target_columns)
        return x_frame, y_frame

    def _split_indices(self, num_rows: int):
        """Train/val/test node indices.

        Resolved from the run's shared :class:`SplitPlan` when one was supplied, so the graph family
        holds out the same points as sklearn and the sequence family. Otherwise it falls back to the
        local ratio carve, in which `test_size` and `val_size` apply sequentially.
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

    def _carve_out(self, indices: np.ndarray, fraction: float):
        """Split off `fraction` of `indices`, returning ``(held_out, remainder)``.

        A fraction of 0 means "no holdout", which ``train_test_split`` rejects outright rather than
        treating as empty. Handling it here is what keeps ``val_size=0``/``test_size=0`` usable -
        this module used to call ``train_test_split`` directly and raised on both.
        """
        fraction = min(max(float(fraction), 0.0), 0.9)
        if fraction <= 0.0:
            return np.array([], dtype=np.int64), np.asarray(indices, dtype=np.int64)

        from sklearn.model_selection import train_test_split

        remainder, held_out = train_test_split(
            indices, test_size=fraction, random_state=self.seed, shuffle=True
        )
        return np.asarray(held_out, dtype=np.int64), np.asarray(remainder, dtype=np.int64)

    def _planned_split_indices(self, num_rows: int):
        """Resolve the shared plan against this graph's own node ordering."""
        point_ids = list(self.spatiotemporal_graph.point_ids)
        if len(point_ids) != num_rows:
            raise ValueError(
                f"The graph carries {len(point_ids)} point id(s) for {num_rows} node(s); the shared "
                f"split cannot be resolved."
            )
        train_idx, val_idx, test_idx = self.split_plan.split_indices(point_ids)
        if train_idx.size == 0:
            raise ValueError(
                "The shared split plan left this datamodule with no training nodes. Check "
                "split.population_policy and the eligibility of this family's rows."
            )
        return train_idx, val_idx, test_idx