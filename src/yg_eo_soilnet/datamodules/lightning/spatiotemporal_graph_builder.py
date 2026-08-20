from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from yg_eo_soilnet.datamodules.frame_cleaning import (
    assert_columns_are_dense_enough,
    build_finite_row_mask,
    drop_non_finite_rows,
    encode_categorical_features,
    sanitize_numeric_columns,
)
from yg_eo_soilnet.datamodules.lightning.spatiotemporal_graph import SpatiotemporalGraph

try:  # pragma: no cover - optional dependency shim
    from pyproj import CRS, Transformer
except ImportError:  # pragma: no cover
    CRS = None  # type: ignore[assignment]
    Transformer = None  # type: ignore[assignment]


class SpatiotemporalGraphBuilder:
    """Builds the :class:`SpatiotemporalGraph` consumed by the Lightning graph datamodule.

    Depends on the DataManager only for raw loading; every graph-specific transform lives here.
    Deliberately torch-free, so graph construction runs and tests without a torch install.
    Node splitting is not done here - that is the datamodule's concern.
    """

    def __init__(self, config, logger, data_manager):
        self.config = config
        self.logger = logger
        self.data_manager = data_manager

    def clean_static_frame(self, static_df, dataset):
        """Schema-filter, ordinal-encode categoricals, gate sparse covariates and drop bad rows.

        Split out of :meth:`build` so :meth:`usable_point_ids` applies the *same* rule rather than a
        second copy of it. Note this family is stricter than the sequence one: it also requires
        finite coordinates, because a node with no position cannot be placed in the graph.

        It is also the one family that still DELETES a row for a missing covariate rather than
        filling and flagging it. The sparsity gate below caps how bad that can get - a column past
        the threshold stops the run everywhere - and the per-family population guard in
        SplitPlanProvider makes any remaining shrinkage loud rather than silent. Converting this
        path properly means threading validity through the graph bundle and its baseline residuals;
        `soil_graph` is enabled:false and superseded by `soil_sequence`, so that is deliberately not
        done here. Enable the graph family and this asymmetry becomes real - check the population
        the guard reports before trusting a comparison against the other two.
        """
        point_col = dataset.point_id_column
        lat_col = dataset.lat_column
        lon_col = dataset.lon_column

        target_columns = list(dataset.target_columns)
        missing_targets = [column for column in target_columns if column not in static_df.columns]
        if missing_targets:
            raise KeyError(f"Missing target columns in static CSV: {', '.join(missing_targets)}")

        # Use the one schema filter DataManager owns, so the graph and sklearn paths train on the
        # same predictors. Re-implementing it here silently skipped EXCLUDE_CATEGORICAL and the
        # hyperspectral drops.
        feature_frame = self.data_manager.filter_schema(static_df, target_columns)
        static_df, feature_columns = self._encode_categorical_features(static_df, feature_frame.columns)

        assert_columns_are_dense_enough(
            static_df,
            feature_columns,
            max_missing_ratio=float(getattr(self.config, "MAX_MISSING_COLUMN_RATIO", 0.2)),
            label="the static graph source",
            logger=self.logger,
            allow=getattr(self.config, "ALLOW_SPARSE_COLUMNS", ()) or (),
            fail=bool(getattr(self.config, "FAIL_ON_SPARSE_COLUMNS", True)),
        )

        static_df = self._drop_non_finite_rows(
            static_df,
            label="static graph source",
            required_columns=[point_col, lat_col, lon_col, *target_columns],
            numeric_columns=[lat_col, lon_col, *feature_columns, *target_columns],
        )
        return static_df, feature_columns

    def usable_point_ids(self, static_df=None):
        """Point ids that survive this family's cleaning. Consumed by the unified splitter."""
        import pandas as pd

        dataset = self.data_manager.load_dataset()
        frame = dataset.tabular if static_df is None else static_df
        cleaned, _ = self.clean_static_frame(frame, dataset)
        point_col = dataset.point_id_column
        if point_col not in cleaned.columns:
            return pd.Index(range(len(cleaned)))
        return pd.Index(cleaned[point_col].to_numpy())

    def build(self, graph_data_args: Optional[Mapping[str, Any]] = None) -> SpatiotemporalGraph:
        graph_data_args = dict(graph_data_args or {})
        dataset = self.data_manager.load_dataset()
        point_col = dataset.point_id_column
        lat_col = dataset.lat_column
        lon_col = dataset.lon_column
        target_columns = list(dataset.target_columns)

        static_df, feature_columns = self.clean_static_frame(dataset.tabular, dataset)

        convert_coordinates_to_utm = bool(graph_data_args.get("convert_coordinates_to_utm", False))
        coordinate_crs = graph_data_args.get("coordinate_crs", "EPSG:4326")
        if convert_coordinates_to_utm:
            coords, graph_coordinate_crs = self._convert_coordinate_columns_to_utm(
                static_df,
                lat_col=lat_col,
                lon_col=lon_col,
                source_crs=coordinate_crs,
            )
            self.logger.info(
                f"Converted coordinate columns '{lat_col}'/'{lon_col}' from {coordinate_crs} to {graph_coordinate_crs} before graph building"
            )
            self.logger.info(
                "UTM coordinate bounds for graph building: "
                f"xmin={float(np.min(coords[:, 0])):.2f}, ymin={float(np.min(coords[:, 1])):.2f}, "
                f"xmax={float(np.max(coords[:, 0])):.2f}, ymax={float(np.max(coords[:, 1])):.2f}"
            )
        else:
            coords = static_df[[lat_col, lon_col]].to_numpy(dtype=np.float32)
            graph_coordinate_crs = coordinate_crs
            self.logger.info(
                f"Using coordinate columns '{lat_col}'/'{lon_col}' without CRS conversion ({graph_coordinate_crs}) before graph building"
            )
        static_features = static_df[feature_columns].to_numpy(dtype=np.float32) if feature_columns else np.empty((len(static_df), 0), dtype=np.float32)
        targets = static_df[target_columns].to_numpy(dtype=np.float32)
        point_ids = static_df[point_col].tolist() if point_col in static_df.columns else list(range(len(static_df)))

        temporal_bundle: dict[str, Any] = {}
        temporal_metadata: dict[str, Any] = {}
        timeseries_df = dataset.timeseries
        if timeseries_df is not None and not timeseries_df.empty:
            temporal_config = self.data_manager.temporal_config()
            prefix_map = self.data_manager.normalize_mapping(
                temporal_config.get("modality_prefix_map", getattr(self.config, "MODALITY_PREFIX_MAP", {}))
            )
            explicit_columns = self.data_manager.normalize_mapping(temporal_config.get("modality_columns", {}))
            if not explicit_columns:
                explicit_columns = {
                    "s1": list(getattr(self.config, "S1_COLUMNS", []) or []),
                    "s2": list(getattr(self.config, "S2_COLUMNS", []) or []),
                    "modis": list(getattr(self.config, "MODIS_COLUMNS", []) or []),
                }
            temporal_numeric_columns: list[str] = []
            if prefix_map:
                for prefix in prefix_map.values():
                    temporal_numeric_columns.extend([column for column in timeseries_df.columns if column.startswith(str(prefix))])
            else:
                for columns in explicit_columns.values():
                    temporal_numeric_columns.extend([column for column in columns if column in timeseries_df.columns])

            # Repair the modality columns rather than dropping their rows: a handful of sparse
            # climate/soil columns would otherwise amputate ~11% of every point's series.
            timeseries_df = self._sanitize_temporal_columns(timeseries_df, temporal_numeric_columns)
            timeseries_df = self._drop_non_finite_rows(
                timeseries_df,
                label="time-series graph source",
                required_columns=[point_col, temporal_config.get("time_column", getattr(self.config, "TIME_COLUMN", "date"))],
            )
            if point_col in timeseries_df.columns and point_col in static_df.columns:
                static_point_ids = set(static_df[point_col].dropna().tolist())
                aligned_timeseries_df = timeseries_df[timeseries_df[point_col].isin(static_point_ids)].copy()
                dropped_time_rows = len(timeseries_df) - len(aligned_timeseries_df)
                if dropped_time_rows > 0:
                    self.logger.warning(
                        f"Dropped {dropped_time_rows} row(s) from time-series graph source that did not match surviving static point IDs"
                    )
                timeseries_df = aligned_timeseries_df
            temporal_split = self.split_modalities(timeseries_df)
            # Arrays stay in temporal point order here; SpatiotemporalGraph aligns them to the
            # static point order at construction, so alignment has exactly one implementation.
            temporal_metadata = {
                "point_ids": list(temporal_split.get("point_ids", []) or []),
                "time_values": temporal_split.get("time_values", []),
                "modality_columns": temporal_split.get("modality_columns", {}),
                "temporal_lengths": dict(temporal_split.get("temporal_lengths", {}) or {}),
                "temporal_masks": dict(temporal_split.get("temporal_masks", {}) or {}),
            }
            temporal_bundle = dict(temporal_split.get("modalities", {}) or {})

            covered_point_ids = set(temporal_split.get("point_ids", []) or [])
            missing_temporal = [pid for pid in point_ids if pid not in covered_point_ids]
            if missing_temporal:
                self.logger.warning(
                    f"{len(missing_temporal)} of {len(point_ids)} node(s) have no time-series rows and will be "
                    f"zero-filled with an all-false temporal mask (e.g. {missing_temporal[:3]})"
                )

        spatial_graph_enabled = bool(
            self._resolve_graph_setting(graph_data_args, "SPATIAL_GRAPH_ENABLED", "spatial_graph_enabled", True)
        )

        if spatial_graph_enabled:
            spatial_radius = self._resolve_graph_setting(graph_data_args, "SPATIAL_RADIUS", "spatial_radius", 50000)
            baseline_method = self._resolve_graph_setting(graph_data_args, "BASELINE_METHOD", "baseline_method", "knn")
            baseline_k_neighbors = self._resolve_graph_setting(
                graph_data_args, "BASELINE_K_NEIGHBORS", "baseline_k_neighbors", 5
            )
            residuals = self.compute_baseline_residuals(
                targets=targets,
                coords=coords,
                method=str(baseline_method),
                k_neighbors=int(baseline_k_neighbors),
            )
            edge_index, edge_attr = self.build_edge_index(
                coords=coords,
                radius=float(spatial_radius),
                residuals=residuals,
            )
            self.logger.info(
                f"Built spatial graph with radius={float(spatial_radius):.2f} over {len(coords)} nodes and {edge_index.shape[1]} edges"
            )
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            residual_dim = targets.shape[1] if targets.ndim > 1 else 1
            residuals = np.zeros((0, residual_dim), dtype=np.float32)
            edge_attr = np.zeros((0, residual_dim), dtype=np.float32)
            self.logger.info("Spatial graph disabled; skipping edge construction")

        graph = SpatiotemporalGraph(
            point_ids=point_ids,
            coords=coords,
            coords_crs=graph_coordinate_crs,
            static_features=static_features,
            static_feature_names=feature_columns,
            targets=targets,
            target_names=target_columns,
            temporal_features=temporal_bundle,
            temporal_lengths=temporal_metadata.get("temporal_lengths", {}) if temporal_metadata else {},
            temporal_masks=temporal_metadata.get("temporal_masks", {}) if temporal_metadata else {},
            temporal_metadata=temporal_metadata,
            residuals=residuals,
            edge_index=edge_index,
            edge_attr=edge_attr,
            temporal_enabled=bool(getattr(self.config, "TEMPORAL_FEATURES_ENABLED", False) and temporal_bundle),
            spatial_graph_enabled=spatial_graph_enabled,
        )
        graph.validate()
        return graph

    def build_edge_index(self, coords: np.ndarray, radius: float, residuals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        coords_array = np.asarray(coords, dtype=np.float32)
        residuals_array = np.asarray(residuals, dtype=np.float32)
        if residuals_array.ndim == 1:
            residuals_array = residuals_array.reshape(-1, 1)

        if len(coords_array) <= 1:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, residuals_array.shape[1]), dtype=np.float32)

        neighbor_search = NearestNeighbors(radius=float(radius), algorithm="ball_tree", metric="euclidean")
        neighbor_search.fit(coords_array)
        neighbor_distances, neighbor_indices = neighbor_search.radius_neighbors(coords_array, return_distance=True, sort_results=True)

        edge_sources: list[int] = []
        edge_targets: list[int] = []
        edge_attributes: list[np.ndarray] = []

        for target_index, source_indices in enumerate(neighbor_indices):
            for source_index in source_indices:
                if int(source_index) == target_index:
                    continue
                edge_sources.append(int(source_index))
                edge_targets.append(target_index)
                edge_attributes.append(residuals_array[int(source_index)])

        if not edge_sources:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, residuals_array.shape[1]), dtype=np.float32)

        edge_index = np.asarray([edge_sources, edge_targets], dtype=np.int64)
        edge_attr = np.asarray(edge_attributes, dtype=np.float32)
        return edge_index, edge_attr

    def _resolve_graph_setting(self, graph_data_args: Mapping[str, Any], upper_key: str, lower_key: str, default: Any) -> Any:
        if upper_key in graph_data_args:
            return graph_data_args[upper_key]
        if lower_key in graph_data_args:
            return graph_data_args[lower_key]
        return getattr(self.config, upper_key, default)

    # --- temporal assembly ------------------------------------------------
    def split_modalities(self, ts_df: pd.DataFrame) -> Dict[str, Any]:
        temporal_config = self.data_manager.temporal_config()
        point_col = getattr(self.config, "POINT_ID_COLUMN", "point_id")
        time_col = temporal_config.get("time_column", getattr(self.config, "TIME_COLUMN", "date"))
        prefix_map = self.data_manager.normalize_mapping(
            temporal_config.get("modality_prefix_map", getattr(self.config, "MODALITY_PREFIX_MAP", {}))
        )
        explicit_columns = self.data_manager.normalize_mapping(temporal_config.get("modality_columns", {}))
        if not explicit_columns:
            explicit_columns = {
                "s1": list(getattr(self.config, "S1_COLUMNS", []) or []),
                "s2": list(getattr(self.config, "S2_COLUMNS", []) or []),
                "modis": list(getattr(self.config, "MODIS_COLUMNS", []) or []),
            }

        if point_col not in ts_df.columns or time_col not in ts_df.columns:
            raise KeyError(f"Time-series CSV must contain '{point_col}' and '{time_col}' columns")

        point_ids = list(pd.Index(ts_df[point_col].dropna().unique()).sort_values())
        time_values = list(pd.Index(ts_df[time_col].dropna().unique()).sort_values())
        point_lookup = {point_id: index for index, point_id in enumerate(point_ids)}
        time_lookup = {time_value: index for index, time_value in enumerate(time_values)}

        modality_arrays: dict[str, np.ndarray] = {}
        modality_columns: dict[str, list[str]] = {}
        temporal_masks: dict[str, np.ndarray] = {}
        temporal_lengths: dict[str, np.ndarray] = {}
        if prefix_map:
            modality_entries = [
                (str(modality_name).lower(), [column for column in ts_df.columns if column.startswith(str(prefix))])
                for modality_name, prefix in prefix_map.items()
            ]
        else:
            modality_entries = []
            for modality_name, columns in explicit_columns.items():
                matched_columns = [column for column in columns if column in ts_df.columns]
                if matched_columns:
                    modality_entries.append((str(modality_name).lower(), matched_columns))

        for modality_name, columns in modality_entries:
            if not columns:
                continue

            ordered = ts_df[[point_col, time_col, *columns]].copy().sort_values([point_col, time_col])
            modality_array = np.zeros((len(point_ids), len(time_values), len(columns)), dtype=np.float32)
            observed_mask = np.zeros((len(point_ids), len(time_values)), dtype=bool)
            for record in ordered.to_dict(orient="records"):
                point_id = record[point_col]
                time_value = record[time_col]
                if point_id not in point_lookup or time_value not in time_lookup:
                    continue
                point_index = point_lookup[point_id]
                time_index = time_lookup[time_value]
                modality_array[point_index, time_index, :] = np.asarray([record.get(column, 0.0) for column in columns], dtype=np.float32)
                observed_mask[point_index, time_index] = True

            modality_arrays[modality_name] = modality_array
            modality_columns[modality_name] = columns
            modality_mask = observed_mask
            temporal_masks[modality_name] = modality_mask
            # A COUNT of observed steps, not the index of the last one. Observations are written at
            # their absolute position on the global date axis, so this equals "last index + 1" only
            # when a point's dates form a dense prefix. Consumers must mask per timestep instead of
            # treating this as a sequence length.
            temporal_lengths[modality_name] = modality_mask.sum(axis=1).astype(np.int64)
            self._warn_if_not_prefix_observed(modality_name, modality_mask)

        return {
            "point_ids": point_ids,
            "time_values": time_values,
            "modalities": modality_arrays,
            "modality_columns": modality_columns,
            "temporal_masks": temporal_masks,
            "temporal_lengths": temporal_lengths,
        }

    @staticmethod
    def _warn_if_not_prefix_observed(modality_name: str, observed_mask: np.ndarray) -> None:
        """Flag points whose observations are not a dense prefix of the global date axis.

        Any consumer that treats the observed COUNT as a sequence length (pack_padded_sequence,
        arange(T) < length) silently discards every reading past that count for these points.
        """
        if observed_mask.size == 0 or not observed_mask.any():
            return

        counts = observed_mask.sum(axis=1)
        last_index = observed_mask.shape[1] - 1 - np.argmax(observed_mask[:, ::-1], axis=1)
        offending = (counts > 0) & (counts != last_index + 1)
        if not offending.any():
            return

        import warnings

        dropped = (last_index + 1 - counts)[offending]
        warnings.warn(
            f"Modality '{modality_name}': {int(offending.sum())}/{observed_mask.shape[0]} points have "
            f"observations that are not a dense prefix of the {observed_mask.shape[1]}-step date axis "
            f"(up to {int(dropped.max())} steps sit beyond the observed count). temporal_lengths is a "
            "count, not a length - encode with the per-timestep mask, never as a padded prefix.",
            RuntimeWarning,
            stacklevel=3,
        )

    # --- spatial baseline and splitting -----------------------------------

    def compute_baseline_residuals(
        self,
        targets: np.ndarray,
        coords: np.ndarray,
        method: str = "knn",
        k_neighbors: Optional[int] = None,
    ) -> np.ndarray:
        targets_array = np.asarray(targets, dtype=np.float32)
        if targets_array.ndim == 1:
            targets_array = targets_array.reshape(-1, 1)

        if len(coords) <= 1:
            return np.zeros_like(targets_array)

        if method != "knn":
            raise NotImplementedError(f"Unsupported baseline method: {method}")

        k_neighbors = int(k_neighbors if k_neighbors is not None else getattr(self.config, "BASELINE_K_NEIGHBORS", 5))
        k_neighbors = max(1, min(k_neighbors, len(coords) - 1))
        neighbors = NearestNeighbors(n_neighbors=k_neighbors + 1)
        neighbors.fit(coords)
        _, indices = neighbors.kneighbors(coords)

        baseline_predictions = np.zeros_like(targets_array)
        for row_index, row_neighbors in enumerate(indices):
            leave_one_out_neighbors = [neighbor_index for neighbor_index in row_neighbors if neighbor_index != row_index]
            if not leave_one_out_neighbors:
                baseline_predictions[row_index] = targets_array[row_index]
                continue
            baseline_predictions[row_index] = targets_array[leave_one_out_neighbors].mean(axis=0)

        return targets_array - baseline_predictions

    # --- frame cleaning ---------------------------------------------------
    # Thin delegates to yg_eo_soilnet.datamodules.frame_cleaning, which the sequence builder shares.
    # Kept as methods so existing callers and tests keep their entry points.

    def _encode_categorical_features(self, frame: pd.DataFrame, columns: Iterable[str]) -> Tuple[pd.DataFrame, list[str]]:
        return encode_categorical_features(frame, columns, logger=self.logger)

    def _sanitize_temporal_columns(self, frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        return sanitize_numeric_columns(frame, columns, logger=self.logger)

    @staticmethod
    def _build_finite_row_mask(
        frame: pd.DataFrame,
        *,
        required_columns: Iterable[str] = (),
        numeric_columns: Iterable[str] = (),
    ) -> pd.Series:
        return build_finite_row_mask(
            frame,
            required_columns=required_columns,
            numeric_columns=numeric_columns,
        )

    def _drop_non_finite_rows(
        self,
        frame: pd.DataFrame,
        *,
        label: str,
        required_columns: Iterable[str] = (),
        numeric_columns: Iterable[str] = (),
    ) -> pd.DataFrame:
        return drop_non_finite_rows(
            frame,
            logger=self.logger,
            label=label,
            required_columns=required_columns,
            numeric_columns=numeric_columns,
        )

    # --- coordinate handling ----------------------------------------------

    def _convert_coordinate_columns_to_utm(
        self,
        frame: pd.DataFrame,
        *,
        lat_col: str,
        lon_col: str,
        source_crs: Any,
    ) -> tuple[np.ndarray, str]:
        if CRS is None or Transformer is None:  # pragma: no cover - exercised only when pyproj is missing
            raise ImportError("pyproj is required to convert coordinates to UTM")

        if lat_col not in frame.columns or lon_col not in frame.columns:
            raise KeyError(f"Coordinate columns '{lat_col}' and '{lon_col}' are required for UTM conversion")

        source_crs_value = source_crs or "EPSG:4326"
        source_crs_obj = CRS.from_user_input(source_crs_value)

        x_values = pd.to_numeric(frame[lon_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        y_values = pd.to_numeric(frame[lat_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

        wgs84_crs = CRS.from_epsg(4326)
        if source_crs_obj.to_epsg() != 4326:
            to_wgs84 = Transformer.from_crs(source_crs_obj, wgs84_crs, always_xy=True)
            lon_values, lat_values = to_wgs84.transform(x_values, y_values)
        else:
            lon_values, lat_values = x_values, y_values

        utm_zone = int((float(np.nanmean(lon_values)) + 180.0) // 6.0) + 1
        utm_epsg = 32600 + utm_zone if float(np.nanmean(lat_values)) >= 0 else 32700 + utm_zone
        utm_crs = CRS.from_epsg(utm_epsg)

        to_utm = Transformer.from_crs(wgs84_crs, utm_crs, always_xy=True)
        utm_x, utm_y = to_utm.transform(lon_values, lat_values)
        coords = np.column_stack([utm_x, utm_y]).astype(np.float32, copy=False)
        return coords, utm_crs.to_string()
