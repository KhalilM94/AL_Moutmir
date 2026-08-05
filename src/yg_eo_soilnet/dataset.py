from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd


@dataclass
class SoilDataset:
    """A loaded dataset, normalized so no caller has to know how it was stored on disk.

    Static features and targets are already joined into ``tabular``, whether they arrived as one
    joint file or as separate files/folders. Time-series always arrives separately and is loaded
    on first access to ``timeseries`` - a sklearn-only run never touches it.
    """

    tabular: pd.DataFrame
    point_id_column: str
    lat_column: str
    lon_column: str
    target_columns: list[str]
    temporal_enabled: bool
    _load_timeseries: Callable[[], Optional[pd.DataFrame]]

    _timeseries: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _timeseries_loaded: bool = field(default=False, init=False, repr=False)

    @property
    def timeseries(self) -> Optional[pd.DataFrame]:
        """The time-series frame, loaded on first access and memoized after."""
        if not self._timeseries_loaded:
            self._timeseries = self._load_timeseries()
            self._timeseries_loaded = True
        return self._timeseries

    @property
    def has_timeseries(self) -> bool:
        """Whether temporal data is configured - answers without triggering the load."""
        return self.temporal_enabled

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        loaded = "loaded" if self._timeseries_loaded else "not loaded"
        return (
            f"SoilDataset(tabular={self.tabular.shape}, targets={self.target_columns}, "
            f"temporal_enabled={self.temporal_enabled}, timeseries={loaded})"
        )
