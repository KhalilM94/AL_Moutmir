from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from yg_eo_soilnet.dataset import SoilDataset


class DataManager:
    """Loads raw datasets and filters non-feature columns out of them.

    Each source - static, targets, time-series - is one configured path that may be a file or a
    folder of files, optionally listed in a manifest. ``load_dataset`` resolves all of them into a
    single normalized :class:`SoilDataset`.

    Everything downstream of that is framework-specific and lives elsewhere:
    ``datamodules.scikit`` owns preprocessing/splitting, ``datamodules.lightning`` owns graph
    construction and dataloaders. This class deliberately knows nothing about either.
    """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self._cache: dict[str, pd.DataFrame] = {}

    # --- schema filtering -------------------------------------------------

    def metadata_columns(self, target_columns: Optional[list[str]] = None, *, include_targets: bool = True) -> set[str]:
        """Columns that identify or measure a sample rather than describe it."""
        columns = {
            getattr(self.config, "POINT_ID_COLUMN", "point_id"),
            getattr(self.config, "LAT_COLUMN", "lat"),
            getattr(self.config, "LON_COLUMN", "lon"),
            "geometry",
        }
        if include_targets:
            active = target_columns if target_columns is not None else list(getattr(self.config, "TARGET_COLUMNS", []))
            # LABEL_COLUMNS is added regardless of which targets are being fitted: a measured label
            # is never a feature. Without it, commenting a target out of TARGET_COLUMNS silently
            # promotes it to a predictor of the remaining targets - which a joint static+targets
            # file makes live immediately.
            columns.update(active)
            columns.update(getattr(self.config, "LABEL_COLUMNS", []) or [])
        return columns

    def excluded_columns(self, columns: Iterable[str]) -> set[str]:
        """Configured non-feature columns: eliminated, excluded categorical and ignored bands."""
        return (
            set(getattr(self.config, "ELIMINATED_FEATURES", []) or [])
            | set(getattr(self.config, "EXCLUDE_CATEGORICAL", []) or [])
            | self.hyperspectral_drop_columns(columns)
        )

    def filter_schema(self, frame: pd.DataFrame, target_columns: Optional[list[str]] = None) -> pd.DataFrame:
        """Drop metadata and configured non-feature columns, leaving a feature-only frame."""
        drop_columns = self.metadata_columns(target_columns) | self.excluded_columns(frame.columns)
        columns_to_drop = sorted(column for column in frame.columns if column in drop_columns)
        if not columns_to_drop:
            return frame.copy()
        return frame.drop(columns=columns_to_drop, errors="ignore")

    def hyperspectral_drop_columns(self, columns: Iterable[str]) -> set[str]:
        drop_columns: set[str] = set()
        existing_hs = self.normalize_mapping(getattr(self.config, "EXISTING_HS_FEATURES", {}))
        if existing_hs.get("enabled", False) and existing_hs.get("ignore", False):
            band_names = existing_hs.get("band_names", [])
            if isinstance(band_names, (list, tuple, set)):
                drop_columns.update({str(name) for name in band_names if name})

            prefix = existing_hs.get("prefix", "")
            if prefix:
                drop_columns.update({str(column) for column in columns if str(column).startswith(str(prefix))})

        legacy_ignore_bands = getattr(self.config, "IGNORE_BANDS", [])
        if isinstance(legacy_ignore_bands, (list, tuple, set)):
            drop_columns.update({str(column) for column in columns if column in legacy_ignore_bands})

        return drop_columns

    def temporal_config(self) -> dict[str, Any]:
        return self.normalize_mapping(getattr(self.config, "TEMPORAL_FEATURES", {}) or {})

    @staticmethod
    def normalize_mapping(value: Any) -> dict:
        """Coerce a config value that may be a dict or a JSON string into a dict."""
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except Exception:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    # --- path resolution and file reading ---------------------------------

    @staticmethod
    def _is_tabular_file(file_name: str) -> bool:
        lower_name = file_name.lower()
        return lower_name.endswith((".csv", ".parquet", ".pq"))

    def _resolve_data_path(self, path_value: Optional[str]) -> Optional[str]:
        if not path_value:
            return None
        if os.path.isabs(path_value):
            return path_value
        return os.path.join(self.config.DATA_FOLDER, path_value)

    def _read_tabular_file(self, file_path: str, missing_label: str = "Tabular file") -> pd.DataFrame:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"{missing_label} not found: {file_path}")
        lower_path = file_path.lower()
        if lower_path.endswith((".parquet", ".pq")):
            return pd.read_parquet(file_path)
        if lower_path.endswith(".csv"):
            return pd.read_csv(file_path)
        raise ValueError(f"Unsupported tabular file format: {file_path}")

    def _load_data_index_manifest(self) -> Optional[dict[str, Any]]:
        manifest_path = getattr(self.config, "DATA_INDEX_MANIFEST_PATH", None)
        if not manifest_path:
            return None
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Data index manifest not found: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        if not isinstance(manifest, dict):
            raise ValueError(f"Data index manifest must contain a JSON object: {manifest_path}")
        return manifest

    def _discover_tabular_files_one_level(self, folder_path: str) -> list[str]:
        discovered: list[str] = []
        if not os.path.isdir(folder_path):
            return discovered

        with os.scandir(folder_path) as entries:
            for entry in entries:
                if entry.is_file() and self._is_tabular_file(entry.name):
                    discovered.append(entry.path)
                elif entry.is_dir():
                    with os.scandir(entry.path) as nested_entries:
                        for nested_entry in nested_entries:
                            if nested_entry.is_file() and self._is_tabular_file(nested_entry.name):
                                discovered.append(nested_entry.path)

        return sorted(discovered)

    def _normalize_manifest_paths(self, paths: Any) -> list[str]:
        if paths is None:
            return []
        if isinstance(paths, str):
            return [paths]
        if isinstance(paths, (list, tuple)):
            return [str(path) for path in paths if path]
        return []

    def _load_tabular_paths(self, paths: list[str]) -> pd.DataFrame:
        if not paths:
            raise FileNotFoundError("No tabular files were discovered")
        frames = [self._read_tabular_file(path) for path in paths]
        return pd.concat(frames, ignore_index=True, sort=False)

    def _load_folder_or_file(self, source_path: Optional[str], manifest_paths: Optional[list[str]] = None) -> pd.DataFrame:
        if manifest_paths:
            absolute_paths = [self._resolve_data_path(path) for path in manifest_paths]
            return self._load_tabular_paths([path for path in absolute_paths if path])

        resolved_path = self._resolve_data_path(source_path)
        if not resolved_path:
            raise FileNotFoundError("Data source path is not configured")

        if os.path.isfile(resolved_path):
            return self._read_tabular_file(resolved_path)

        if not os.path.isdir(resolved_path):
            raise FileNotFoundError(f"Data source folder not found: {resolved_path}")

        discovered_paths = self._discover_tabular_files_one_level(resolved_path)
        return self._load_tabular_paths(discovered_paths)

    def _load_manifest_source_paths(self, manifest: dict[str, Any], manifest_key: str) -> list[str]:
        if not manifest:
            return []
        for key in (manifest_key, manifest_key.lower(), manifest_key.upper()):
            if key in manifest:
                return self._normalize_manifest_paths(manifest[key])
        return []

    def _load_source(self, source_path: Optional[str], manifest_key: str, *, label: str) -> pd.DataFrame:
        """Resolve one configured source: manifest entry, folder scan, or single file.

        The manifest is consulted first so it works for file-based configs too, not only when the
        configured path happens to be a directory.
        """
        cache_key = f"{manifest_key}:{source_path}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        manifest = self._load_data_index_manifest()
        manifest_paths = self._load_manifest_source_paths(manifest or {}, manifest_key)
        if not manifest_paths and not source_path:
            raise FileNotFoundError(f"{label} source is not configured")

        self.logger.info(f"Loading {label} from {'manifest' if manifest_paths else source_path}")
        frame = self._load_folder_or_file(source_path, manifest_paths=manifest_paths)
        self._cache[cache_key] = frame
        return frame

    # --- loaders ----------------------------------------------------------

    def load_dataset(self) -> SoilDataset:
        """Load the dataset, whatever shape it arrives in.

        Static and targets may be one joint file or two separate files/folders; time-series always
        arrives separately and is loaded lazily. Every framework consumes the returned bundle.
        """
        return SoilDataset(
            tabular=self.load_tabular_data(),
            point_id_column=getattr(self.config, "POINT_ID_COLUMN", "point_id"),
            lat_column=getattr(self.config, "LAT_COLUMN", "lat"),
            lon_column=getattr(self.config, "LON_COLUMN", "lon"),
            target_columns=list(getattr(self.config, "TARGET_COLUMNS", [])),
            temporal_enabled=self._temporal_enabled(),
            _load_timeseries=self.load_timeseries_data,
        )

    def reload(self) -> None:
        """Drop cached frames so the next load re-reads from disk."""
        self._cache.clear()

    def _static_source(self) -> Optional[str]:
        return (
            getattr(self.config, "STATIC_SOURCE", None)
            or getattr(self.config, "STATIC_FEATURES_FOLDER", None)
            or getattr(self.config, "STATIC_CSV_PATH", None)
            or self._resolve_data_path(getattr(self.config, "DATA_FILE", "data.csv"))
        )

    def _targets_source(self) -> Optional[str]:
        # Falling back to the static path is safe: load_tabular_data only reaches here when the
        # static frame lacks the targets, i.e. when the two are genuinely separate.
        return (
            getattr(self.config, "TARGETS_SOURCE", None)
            or getattr(self.config, "TARGETS_FOLDER", None)
            or getattr(self.config, "TARGETS_CSV_PATH", None)
            or self._resolve_data_path(
                getattr(self.config, "TARGETS_FILE", getattr(self.config, "DATA_FILE", "data.csv"))
            )
        )

    def _timeseries_source(self) -> Optional[str]:
        temporal_config = self.temporal_config()
        return (
            temporal_config.get("timeseries_folder")
            or temporal_config.get("timeseries_file")
            or temporal_config.get("timeseries_csv_path")
            or getattr(self.config, "TIMESERIES_SOURCE", None)
            or getattr(self.config, "TIMESERIES_FOLDER", None)
            or getattr(self.config, "TIMESERIES_CSV_PATH", None)
        )

    def _temporal_enabled(self) -> bool:
        temporal_config = self.temporal_config()
        enabled = bool(
            temporal_config.get("enabled", getattr(self.config, "TEMPORAL_FEATURES_ENABLED", False))
        )
        return enabled and bool(self._timeseries_source())

    def _load_static(self) -> pd.DataFrame:
        """Static features as configured - targets may or may not be present in this frame."""
        return self._load_source(self._static_source(), "static_features", label="static data")

    def _load_targets(self) -> pd.DataFrame:
        return self._load_source(self._targets_source(), "targets", label="targets")

    def load_tabular_data(self) -> pd.DataFrame:
        """Static features with targets joined in, whether they arrive together or separately."""
        static_df = self._load_static()
        target_columns = list(getattr(self.config, "TARGET_COLUMNS", []))
        if not target_columns:
            raise ValueError("TARGET_COLUMNS must be configured for tabular preprocessing")

        if all(column in static_df.columns for column in target_columns):
            return static_df

        targets_df = self._load_targets()
        point_col = getattr(self.config, "POINT_ID_COLUMN", "point_id")
        if point_col not in static_df.columns:
            raise KeyError(f"Point id column '{point_col}' not found in static features file")
        if point_col not in targets_df.columns:
            raise KeyError(f"Point id column '{point_col}' not found in targets file")

        missing_targets = [column for column in target_columns if column not in targets_df.columns]
        if missing_targets:
            raise KeyError(f"Target column(s) missing from targets file: {', '.join(missing_targets)}")

        # Coordinates are per-sample metadata both frameworks need, but a covariates file often
        # carries only the point id. Carry them across when only the targets file has them.
        coordinate_columns = [
            column
            for column in (getattr(self.config, "LAT_COLUMN", "lat"), getattr(self.config, "LON_COLUMN", "lon"))
            if column not in static_df.columns and column in targets_df.columns
        ]
        if coordinate_columns:
            self.logger.info(f"Joining coordinate column(s) {coordinate_columns} from the targets source")

        # Measured lab values beyond the ones being fitted, carried only when CARRY_LABEL_COLUMNS
        # asks for them. Without this the split-file layout silently offers a model just the active
        # targets while the joint-file layout offers every label, so the same LABEL_COLUMNS
        # declaration means different things depending on how the data happens to be filed.
        #
        # Joining them does NOT make them predictors: metadata_columns adds every LABEL_COLUMNS
        # entry to the drop set regardless of which targets are active, and all three paths select
        # their features through filter_schema. Only a model that names one in
        # auxiliary_label_columns ever reads it.
        label_columns = []
        if getattr(self.config, "CARRY_LABEL_COLUMNS", False):
            label_columns = [
                column
                for column in (getattr(self.config, "LABEL_COLUMNS", []) or [])
                if column in targets_df.columns and column not in static_df.columns
            ]
            if label_columns:
                self.logger.info(
                    f"Joining {len(label_columns)} measured label column(s) from the targets source "
                    "as auxiliary inputs; they remain excluded from the feature set"
                )

        # dict.fromkeys, not a bare list: a column that is both a target and a label - which every
        # active target is - would otherwise be selected twice, and targets_df[join_columns] would
        # return a frame with a duplicated column that fails on merge for no legible reason.
        join_columns = list(dict.fromkeys([point_col, *target_columns, *label_columns, *coordinate_columns]))
        dedup_targets = targets_df[join_columns].drop_duplicates(subset=[point_col])
        return static_df.merge(dedup_targets, on=point_col, how="inner")

    def load_timeseries_data(self) -> Optional[pd.DataFrame]:
        if not self._temporal_enabled():
            return None

        source = self._timeseries_source()
        resolved_path = self._resolve_data_path(source)
        if not resolved_path:
            return None
        if os.path.isdir(resolved_path):
            return self._load_timeseries_folder(resolved_path)

        cache_key = f"timeseries:{resolved_path}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        self.logger.info(f"Loading time-series from file: {resolved_path}")
        frame = self._read_tabular_file(resolved_path, missing_label="Time-series CSV file")
        self._cache[cache_key] = frame
        return frame

    def _load_timeseries_folder(self, timeseries_root: str) -> pd.DataFrame:
        resolved_root = self._resolve_data_path(timeseries_root)
        if not resolved_root or not os.path.isdir(resolved_root):
            raise FileNotFoundError(f"Time-series folder not found: {resolved_root or timeseries_root}")

        cache_key = f"timeseries:{resolved_root}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        manifest = self._load_data_index_manifest()
        manifest_timeseries = (manifest or {}).get("timeseries", {}) if manifest else {}

        modality_frames: list[pd.DataFrame] = []
        if isinstance(manifest_timeseries, dict) and manifest_timeseries:
            for modality_name, modality_paths in manifest_timeseries.items():
                resolved_paths = [self._resolve_data_path(path) for path in self._normalize_manifest_paths(modality_paths)]
                frame_paths = [path for path in resolved_paths if path]
                if not frame_paths:
                    continue
                modality_frames.append(self._load_tabular_paths(frame_paths))
        else:
            with os.scandir(resolved_root) as modality_entries:
                for modality_entry in sorted(modality_entries, key=lambda entry: entry.name):
                    if modality_entry.is_dir():
                        shard_paths = self._discover_tabular_files_one_level(modality_entry.path)
                        if shard_paths:
                            modality_frames.append(self._load_tabular_paths(shard_paths))
                    elif modality_entry.is_file() and self._is_tabular_file(modality_entry.name):
                        modality_frames.append(self._read_tabular_file(modality_entry.path))

        if not modality_frames:
            raise FileNotFoundError(f"No time-series files were discovered under {resolved_root}")

        combined_frame = modality_frames[0]
        point_col = getattr(self.config, "POINT_ID_COLUMN", "point_id")
        time_col = self.temporal_config().get("time_column", getattr(self.config, "TIME_COLUMN", "date"))
        for frame in modality_frames[1:]:
            combined_frame = combined_frame.merge(frame, on=[point_col, time_col], how="outer")
        self._cache[cache_key] = combined_frame
        return combined_frame
