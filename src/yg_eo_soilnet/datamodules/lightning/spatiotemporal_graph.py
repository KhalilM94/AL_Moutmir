from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np


@dataclass
class SpatiotemporalGraph:
    """The graph bundle exchanged between the builder and the Lightning graph datamodule.

    Construction establishes one invariant: every temporal array is indexed by **static point
    order**, so consumers can index modalities with the same node indices they use for
    ``static_features``. Alignment happens here and nowhere else - neither the builder nor the
    datamodule re-checks it.

    Node splits deliberately live outside this type: splitting is a dataloading concern owned by
    the datamodule.
    """

    point_ids: list[Any] = field(default_factory=list)
    coords: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    coords_crs: str = "EPSG:4326"
    static_features: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    static_feature_names: list[str] = field(default_factory=list)
    targets: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    target_names: list[str] = field(default_factory=list)
    residuals: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    edge_index: np.ndarray = field(default_factory=lambda: np.zeros((2, 0), dtype=np.int64))
    edge_attr: np.ndarray = field(default_factory=lambda: np.empty((0, 1), dtype=np.float32))
    temporal_features: dict[str, np.ndarray] = field(default_factory=dict)
    temporal_lengths: dict[str, np.ndarray] = field(default_factory=dict)
    temporal_masks: dict[str, np.ndarray] = field(default_factory=dict)
    temporal_metadata: dict[str, Any] = field(default_factory=dict)
    temporal_enabled: bool = False
    spatial_graph_enabled: bool = True

    def __post_init__(self) -> None:
        temporal_point_ids = self._temporal_point_ids()
        if not temporal_point_ids or not self.point_ids:
            return
        if list(temporal_point_ids) == list(self.point_ids):
            return

        for attribute in ("temporal_features", "temporal_lengths", "temporal_masks"):
            arrays = getattr(self, attribute) or {}
            setattr(
                self,
                attribute,
                {
                    name: self._align_to_static_points(array, temporal_point_ids)
                    for name, array in arrays.items()
                },
            )

        self.temporal_metadata = {**(self.temporal_metadata or {}), "point_ids": list(self.point_ids)}

    def _temporal_point_ids(self) -> Optional[list[Any]]:
        metadata = self.temporal_metadata or {}
        if isinstance(metadata, Mapping):
            point_ids = metadata.get("point_ids")
            if point_ids is not None:
                return list(point_ids)
        return None

    def _align_to_static_points(self, values: Any, temporal_point_ids: Sequence[Any]) -> np.ndarray:
        array = np.asarray(values)
        if array.size == 0 or array.ndim == 0:
            return array

        target_shape = (len(self.point_ids), *array.shape[1:]) if array.ndim > 1 else (len(self.point_ids),)
        aligned = np.zeros(target_shape, dtype=array.dtype)
        lookup = {point_id: index for index, point_id in enumerate(temporal_point_ids) if point_id is not None}
        for static_index, point_id in enumerate(self.point_ids):
            temporal_index = lookup.get(point_id)
            if temporal_index is None or temporal_index >= array.shape[0]:
                continue
            aligned[static_index] = array[temporal_index]
        return aligned

    # --- Mapping-style access, so dict-based callers and tests keep working ---

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def keys(self):
        return tuple(self.__dataclass_fields__)

    @classmethod
    def from_mapping(cls, value: "SpatiotemporalGraph | Mapping[str, Any]") -> "SpatiotemporalGraph":
        if isinstance(value, cls):
            return value
        known = set(cls.__dataclass_fields__)
        return cls(**{key: item for key, item in dict(value).items() if key in known})

    # --- validation -------------------------------------------------------

    def validate(self) -> None:
        """Raise if any numeric field carries non-finite values, naming the offending cell."""
        self._validate_numeric_array("coords", self.coords, column_labels=["lat", "lon"])
        self._validate_numeric_array("static_features", self.static_features, column_labels=self.static_feature_names)
        self._validate_numeric_array("targets", self.targets, column_labels=self.target_names)
        self._validate_numeric_array("residuals", self.residuals, column_labels=self.target_names)
        self._validate_numeric_array("edge_index", self.edge_index, row_labels=None)
        self._validate_numeric_array("edge_attr", self.edge_attr, row_labels=None)

        for name, array in (self.temporal_features or {}).items():
            self._validate_numeric_array(f"temporal_features.{name}", array, row_labels=None)
        for name, array in (self.temporal_lengths or {}).items():
            self._validate_numeric_array(f"temporal_lengths.{name}", array, row_labels=None)
        for name, array in (self.temporal_masks or {}).items():
            self._validate_numeric_array(f"temporal_masks.{name}", array, row_labels=None)

    def _validate_numeric_array(
        self,
        name: str,
        array: Any,
        *,
        row_labels: Optional[list[Any]] = "point_ids",  # type: ignore[assignment]
        column_labels: Optional[list[str]] = None,
    ) -> None:
        labels = list(self.point_ids) if row_labels == "point_ids" else row_labels
        first_bad_index = self._first_non_finite_index(array)
        if first_bad_index is None:
            return

        if len(first_bad_index) == 2 and column_labels is not None:
            row_index, column_index = first_bad_index
            row_label = labels[row_index] if labels is not None and row_index < len(labels) else row_index
            column_label = column_labels[column_index] if column_index < len(column_labels) else column_index
            raise ValueError(
                f"Non-finite values found in graph bundle field '{name}' at row {row_label}, column '{column_label}'"
            )

        if labels is not None and len(first_bad_index) >= 1:
            row_index = first_bad_index[0]
            row_label = labels[row_index] if row_index < len(labels) else row_index
            raise ValueError(f"Non-finite values found in graph bundle field '{name}' at row {row_label}")

        raise ValueError(f"Non-finite values found in graph bundle field '{name}' at index {first_bad_index}")

    @staticmethod
    def _first_non_finite_index(array: Any) -> Optional[tuple[int, ...]]:
        values = np.asarray(array)
        if values.size == 0 or not np.issubdtype(values.dtype, np.number):
            return None
        finite_mask = np.isfinite(values)
        if bool(finite_mask.all()):
            return None
        first_bad = np.argwhere(~finite_mask)[0]
        return tuple(int(index) for index in first_bad.tolist())
