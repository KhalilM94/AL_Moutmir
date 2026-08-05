from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass
class SoilSequenceBundle:
    """The bundle exchanged between :class:`SoilSequenceBuilder` and the sequence datamodule.

    Every point carries only its **actual observations**, each paired with its own date. There is no
    shared time axis, no fixed number of steps and no zero-filled gap: ``sequences[modality][i]`` has
    one row per reading that point genuinely has for that modality, and
    ``sequence_times[modality][i]`` holds the matching decimal years.

    That is what makes a model built on this bundle transferable. Nothing here encodes *which* years
    the data came from, so a 2030-2035 series is the same kind of object as a 2017-2025 one, and
    points with 40 readings sit beside points with 90 without any padding at rest.

    Deliberately absent, because there is no graph: coords, edges, edge attributes and residuals.
    """

    point_ids: list[Any] = field(default_factory=list)
    static_features: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    static_feature_names: list[str] = field(default_factory=list)
    targets: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    target_names: list[str] = field(default_factory=list)
    # modality -> one (n_i, C_m) array per point, in static point order
    sequences: dict[str, list[np.ndarray]] = field(default_factory=dict)
    # modality -> one (n_i,) array of decimal years per point, in static point order
    sequence_times: dict[str, list[np.ndarray]] = field(default_factory=dict)
    # modality -> one (n_i, C_m) bool array per point: True where the cell was measured rather than
    # median-filled. Empty dict means "no imputation was tracked", which reads as all-True.
    sequence_validity: dict[str, list[np.ndarray]] = field(default_factory=dict)
    modality_columns: dict[str, list[str]] = field(default_factory=dict)
    temporal_enabled: bool = False

    # --- Mapping-style access, so dict-based callers and tests keep working ---

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def keys(self):
        return tuple(self.__dataclass_fields__)

    @classmethod
    def from_mapping(cls, value: "SoilSequenceBundle | Mapping[str, Any]") -> "SoilSequenceBundle":
        if isinstance(value, cls):
            return value
        known = set(cls.__dataclass_fields__)
        return cls(**{key: item for key, item in dict(value).items() if key in known})

    @property
    def num_points(self) -> int:
        return len(self.point_ids)

    @property
    def modality_dims(self) -> dict[str, int]:
        """Channel count per modality, taken from the column lists rather than any point's data.

        Reading it off an array would break on a bundle whose first point has no observations.
        """
        return {name: len(columns) for name, columns in self.modality_columns.items()}

    def observation_counts(self, modality: str) -> np.ndarray:
        return np.asarray([len(values) for values in self.sequences.get(modality, [])], dtype=np.int64)

    def validity_for(self, modality: str, index: int) -> np.ndarray:
        """Per-channel validity for one point, defaulting to all-True when none was tracked."""
        per_point = (self.sequence_validity or {}).get(modality)
        if per_point is not None and index < len(per_point):
            return np.asarray(per_point[index], dtype=bool)
        return np.ones(np.asarray(self.sequences[modality][index]).shape, dtype=bool)

    def imputed_fraction(self, modality: str) -> float:
        """Share of this modality's cells that were median-filled rather than measured."""
        per_point = (self.sequence_validity or {}).get(modality)
        if not per_point:
            return 0.0
        total = sum(array.size for array in per_point)
        if total == 0:
            return 0.0
        return float(sum(int((~np.asarray(array, dtype=bool)).sum()) for array in per_point) / total)

    # --- validation -------------------------------------------------------

    def validate(self) -> None:
        """Raise if the bundle is internally inconsistent or carries non-finite values.

        Every message names the offending point id (and column where known), because a bare
        "non-finite value" on a 5,700-point bundle is not actionable.
        """
        self._validate_numeric_array("static_features", self.static_features, self.static_feature_names)
        self._validate_numeric_array("targets", self.targets, self.target_names)

        num_points = len(self.point_ids)
        if self.static_features.size and self.static_features.shape[0] != num_points:
            raise ValueError(
                f"static_features has {self.static_features.shape[0]} row(s) but there are {num_points} point(s)"
            )
        if self.targets.size and self.targets.shape[0] != num_points:
            raise ValueError(f"targets has {self.targets.shape[0]} row(s) but there are {num_points} point(s)")

        for modality, per_point_values in (self.sequences or {}).items():
            per_point_times = (self.sequence_times or {}).get(modality)
            if per_point_times is None:
                raise ValueError(f"Modality '{modality}' has sequences but no sequence_times")
            if len(per_point_values) != num_points or len(per_point_times) != num_points:
                raise ValueError(
                    f"Modality '{modality}' covers {len(per_point_values)} point(s) and "
                    f"{len(per_point_times)} time array(s) but there are {num_points} point(s)"
                )

            per_point_validity = (self.sequence_validity or {}).get(modality)
            if per_point_validity is not None and len(per_point_validity) != num_points:
                raise ValueError(
                    f"Modality '{modality}' has {len(per_point_validity)} validity array(s) "
                    f"but there are {num_points} point(s)"
                )

            expected_channels = len(self.modality_columns.get(modality, []))
            for index, (values, times) in enumerate(zip(per_point_values, per_point_times)):
                point_label = self.point_ids[index] if index < len(self.point_ids) else index
                self._validate_point_sequence(modality, point_label, values, times, expected_channels)
                if per_point_validity is not None:
                    validity = np.asarray(per_point_validity[index])
                    if validity.shape != np.asarray(values).shape:
                        raise ValueError(
                            f"sequence_validity['{modality}'] at point {point_label} has shape "
                            f"{validity.shape}, expected {np.asarray(values).shape} to match the readings"
                        )

    def _validate_point_sequence(
        self,
        modality: str,
        point_label: Any,
        values: np.ndarray,
        times: np.ndarray,
        expected_channels: int,
    ) -> None:
        values = np.asarray(values)
        times = np.asarray(times)

        if values.ndim != 2:
            raise ValueError(
                f"sequences['{modality}'] at point {point_label} must be 2-D (observations, channels), "
                f"got {values.ndim}-D"
            )
        if len(times) != values.shape[0]:
            raise ValueError(
                f"Modality '{modality}' at point {point_label} has {values.shape[0]} observation(s) "
                f"but {len(times)} timestamp(s)"
            )
        if expected_channels and values.shape[1] != expected_channels:
            raise ValueError(
                f"Modality '{modality}' at point {point_label} has {values.shape[1]} channel(s), "
                f"expected {expected_channels}"
            )
        if values.size and not np.isfinite(values).all():
            row, column = (int(index) for index in np.argwhere(~np.isfinite(values))[0])
            columns = self.modality_columns.get(modality, [])
            column_label = columns[column] if column < len(columns) else column
            raise ValueError(
                f"Non-finite value in sequences['{modality}'] at point {point_label}, "
                f"observation {row}, column '{column_label}'"
            )
        if times.size:
            if not np.isfinite(times).all():
                raise ValueError(f"Non-finite timestamp in sequence_times['{modality}'] at point {point_label}")
            # Strictly ascending, because the encoders read the gap between consecutive tokens as a
            # first difference. An unsorted or duplicated timestamp would silently produce a
            # negative or zero gap and corrupt the time features.
            if times.size > 1 and not bool(np.all(np.diff(times) > 0)):
                raise ValueError(
                    f"sequence_times['{modality}'] at point {point_label} is not strictly ascending; "
                    "observations must be sorted by date with no duplicate timestamps"
                )

    def _validate_numeric_array(self, name: str, array: Any, column_labels: list[str]) -> None:
        values = np.asarray(array)
        if values.size == 0 or not np.issubdtype(values.dtype, np.number):
            return
        finite_mask = np.isfinite(values)
        if bool(finite_mask.all()):
            return

        first_bad = np.argwhere(~finite_mask)[0]
        row_index = int(first_bad[0])
        point_label = self.point_ids[row_index] if row_index < len(self.point_ids) else row_index
        if len(first_bad) == 2:
            column_index = int(first_bad[1])
            column_label = column_labels[column_index] if column_index < len(column_labels) else column_index
            raise ValueError(f"Non-finite value in '{name}' at point {point_label}, column '{column_label}'")
        raise ValueError(f"Non-finite value in '{name}' at point {point_label}")
