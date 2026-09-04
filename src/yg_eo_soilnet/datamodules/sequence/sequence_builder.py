from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from yg_eo_soilnet.datamodules.categorical import (
    resolve_categorical_columns,
    split_feature_blocks,
)
from yg_eo_soilnet.datamodules.frame_cleaning import (
    assert_columns_are_dense_enough,
    build_finite_row_mask,
    drop_non_finite_rows,
    sanitize_numeric_columns,
)
from yg_eo_soilnet.datamodules.sequence.sequence_bundle import SoilSequenceBundle

# Days per average Gregorian year. Only ever used to place an observation *within* its year, so the
# small drift against a real calendar is irrelevant; what matters is that the mapping is monotonic
# and identical for every year.
DAYS_PER_YEAR = 365.25

# Column-name prefix used to carry per-cell validity through the same pandas operations as the data.
_VALIDITY_PREFIX = "__valid__"


def to_decimal_year(dates: pd.Series) -> np.ndarray:
    """Timestamps -> decimal years, e.g. 2019-07-02 -> 2019.5.

    Cadence-agnostic by construction: daily, monthly and irregular sampling all map onto the same
    continuous axis, so nothing downstream has to know how often the sensor reports.

    Returned as float64 and kept that way end to end. float32 resolves only about a day near year
    2020, and the year-fraction subtraction downstream would spend most of that, blurring the
    seasonal signal these features exist to carry.
    """
    dates = pd.to_datetime(dates, errors="coerce")
    year = dates.dt.year.to_numpy(dtype=np.float64)
    day_of_year = dates.dt.dayofyear.to_numpy(dtype=np.float64)
    return year + (day_of_year - 1.0) / DAYS_PER_YEAR


class SoilSequenceBuilder:
    """Builds the :class:`SoilSequenceBundle` consumed by the sequence datamodule.

    Depends on the DataManager only for raw loading and schema filtering, so the sequence path and
    the sklearn path train on the same predictors. Deliberately torch-free, so bundle construction
    runs and tests without a torch install.

    Contains no graph concept whatsoever: no coordinates, no edges, no spatial radius, no baseline
    residuals. Splitting is not done here - that is the datamodule's concern.
    """

    def __init__(self, config, logger, data_manager):
        self.config = config
        self.logger = logger
        self.data_manager = data_manager

    def clean_static_frame(self, static_df: pd.DataFrame, dataset) -> tuple[pd.DataFrame, Any]:
        """Schema-filter, resolve the categorical blocks, gate sparse covariates and drop bad rows.

        Split out of :meth:`build` so :meth:`usable_point_ids` answers "which points can this family
        actually use?" with the *same* rule the builder applies, rather than a second copy of it
        that can drift. The unified splitter needs that answer before any bundle exists.

        A row now dies only for a missing TARGET or point id. It used to die for a missing
        *covariate* too, which meant three columns that were 99.7% blank could - and on one real
        dataset did - reduce 5761 points to 17. Covariate gaps are median-filled and flagged by the
        datamodule instead; columns too empty for that to be honest are rejected outright by
        :func:`assert_columns_are_dense_enough` above, so nothing reaching the fill is fabricated
        wholesale.
        """
        point_col = dataset.point_id_column
        target_columns = list(dataset.target_columns)
        missing_targets = [column for column in target_columns if column not in static_df.columns]
        if missing_targets:
            raise KeyError(f"Missing target columns in static CSV: {', '.join(missing_targets)}")

        feature_frame = self.data_manager.filter_schema(static_df, target_columns)
        blocks = resolve_categorical_columns(
            self.config, static_df, feature_frame.columns, logger=self.logger
        )
        self._assert_context_features_present(static_df, blocks.continuous_columns)
        feature_columns = list(blocks.continuous_columns)
        assert_columns_are_dense_enough(
            static_df,
            feature_columns,
            max_missing_ratio=float(getattr(self.config, "MAX_MISSING_COLUMN_RATIO", 0.2)),
            label="the static sequence source",
            logger=self.logger,
            allow=getattr(self.config, "ALLOW_SPARSE_COLUMNS", ()) or (),
            fail=bool(getattr(self.config, "FAIL_ON_SPARSE_COLUMNS", True)),
        )
        # Targets, and coordinates when the harmonic branch is on. A missing category becomes the
        # reserved embedding index and a missing continuous covariate becomes a train-median fill
        # plus a validity flag; neither is a reason to delete the soil sample. A missing LABEL is,
        # because there is nothing to learn from it - and so is a missing COORDINATE, because the
        # fill that rescues a covariate has no honest analogue here: a median lat/lon is a location
        # in the middle of the study area that the sample does not occupy, and the encoder would
        # read it as a confident position rather than as an absence.
        #
        # Done HERE rather than in build() on purpose. usable_point_ids() calls this method to
        # answer "which points can this family use?" for the unified splitter; deciding the coord
        # rule in build() instead would let the splitter assign points that build() then drops, and
        # under population_policy=intersect that shrinks the whole run's population silently.
        coordinate_columns = [
            column for column in self._coordinate_columns() if column in static_df.columns
        ]
        if coordinate_columns:
            # Attributed to the coordinates specifically, rather than reported as the drop that
            # drop_non_finite_rows performs below: that one also removes rows with a missing target,
            # which would have gone whatever this flag said, and blaming those on the coordinates
            # would send someone to audit their lat/lon over a lab problem.
            missing_coords = int(
                (~build_finite_row_mask(static_df, numeric_columns=coordinate_columns)).sum()
            )
            if missing_coords:
                self.logger.info(
                    f"USE_HARMONIC_COORDS is on: {missing_coords} of {len(static_df)} point(s) have "
                    f"a missing or non-finite {' / '.join(coordinate_columns)} and are dropped. "
                    "Turning the flag off restores them."
                )

        cleaned = drop_non_finite_rows(
            static_df,
            logger=self.logger,
            label="static sequence source",
            required_columns=[point_col, *target_columns, *coordinate_columns],
            numeric_columns=[*target_columns, *coordinate_columns],
        )
        return cleaned, blocks

    def usable_point_ids(self, static_df: Optional[pd.DataFrame] = None) -> pd.Index:
        """Point ids that survive this family's cleaning. Consumed by the unified splitter."""
        dataset = self.data_manager.load_dataset()
        frame = dataset.tabular if static_df is None else static_df
        cleaned, _ = self.clean_static_frame(frame, dataset)
        point_col = dataset.point_id_column
        if point_col not in cleaned.columns:
            return pd.Index(range(len(cleaned)))
        return pd.Index(cleaned[point_col].to_numpy())

    def build(self, sequence_data_args: Optional[Mapping[str, Any]] = None) -> SoilSequenceBundle:
        sequence_data_args = dict(sequence_data_args or {})
        dataset = self.data_manager.load_dataset()
        point_col = dataset.point_id_column
        target_columns = list(dataset.target_columns)

        static_df, blocks = self.clean_static_frame(dataset.tabular, dataset)

        static_features, static_categoricals = split_feature_blocks(static_df, blocks)
        static_validity, static_validity_names = self._static_validity(
            static_df, list(blocks.continuous_columns)
        )
        targets = static_df[target_columns].to_numpy(dtype=np.float32)
        label_features, label_feature_names = self._extract_label_features(static_df)
        coords, coord_names = self._extract_coordinates(static_df)
        # Order taken from continuous_columns, not from the config: these names index positions in
        # static_features, and a list in declaration order would mislabel them the moment the config
        # and the frame disagree about ordering.
        context_columns = set(self.data_manager.context_feature_columns())
        context_feature_names = [
            column for column in blocks.continuous_columns if column in context_columns
        ]
        point_ids = (
            static_df[point_col].tolist() if point_col in static_df.columns else list(range(len(static_df)))
        )

        sequences: dict[str, list[np.ndarray]] = {}
        sequence_times: dict[str, list[np.ndarray]] = {}
        sequence_validity: dict[str, list[np.ndarray]] = {}
        modality_columns: dict[str, list[str]] = {}

        timeseries_df = dataset.timeseries
        if timeseries_df is not None and not timeseries_df.empty:
            sequences, sequence_times, sequence_validity, modality_columns = self._build_sequences(
                timeseries_df, point_ids=point_ids, point_col=point_col
            )

        bundle = SoilSequenceBundle(
            point_ids=point_ids,
            static_features=static_features,
            static_feature_names=list(blocks.continuous_columns),
            static_validity=static_validity,
            static_validity_names=static_validity_names,
            static_categoricals=static_categoricals,
            categorical_feature_names=list(blocks.categorical_columns),
            context_feature_names=context_feature_names,
            coords=coords,
            coord_names=coord_names,
            targets=targets,
            target_names=target_columns,
            label_features=label_features,
            label_feature_names=label_feature_names,
            sequences=sequences,
            sequence_times=sequence_times,
            sequence_validity=sequence_validity,
            modality_columns=modality_columns,
            temporal_enabled=bool(getattr(self.config, "TEMPORAL_FEATURES_ENABLED", False) and sequences),
        )
        bundle.validate()
        self._log_summary(bundle)
        return bundle

    # --- spatial context group --------------------------------------------

    def _assert_context_features_present(
        self, static_df: pd.DataFrame, continuous_columns: list[str]
    ) -> None:
        """Refuse a CONTEXT_FEATURES entry the data does not carry, or that something else dropped.

        Mirrors ``resolve_categorical_columns``' treatment of CATEGORICAL_FEATURES, and for the same
        reason: a declared column absent from the frame is nearly always a config left over from a
        retired dataset, and skipping it quietly means the run trains without the context features
        while the config says it has them.

        Only checked when the group is ON. With it off the columns are *supposed* to be gone - that
        is the ablation - so demanding their presence would make the switch unusable.
        """
        selected = self.data_manager.context_feature_columns()
        if not selected:
            return

        absent = [column for column in selected if column not in static_df.columns]
        if absent:
            raise KeyError(
                f"CONTEXT_FEATURES names column(s) the static source does not carry: "
                f"{', '.join(sorted(absent))}. Add them to the static CSV, remove them from the "
                "list, or set USE_CONTEXT_FEATURES: false to run without the group."
            )

        # Present in the frame but filtered out before reaching the features. Distinguished from the
        # case above because the fix is elsewhere entirely: the column exists and is spelled right,
        # and something else - IGNORED_COLUMNS, LABEL_COLUMNS, CATEGORICAL_FEATURES - is removing it.
        withheld = [
            column
            for column in selected
            if column in static_df.columns and column not in set(continuous_columns)
        ]
        if withheld:
            raise ValueError(
                f"CONTEXT_FEATURES names column(s) that are present but not continuous features: "
                f"{', '.join(sorted(withheld))}. Something else is removing them - check "
                "IGNORED_COLUMNS/ELIMINATED_FEATURES, LABEL_COLUMNS and CATEGORICAL_FEATURES."
            )

    # --- covariate gaps ---------------------------------------------------

    def _static_validity(
        self, static_df: pd.DataFrame, feature_columns: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Measured-vs-missing flags, for the continuous covariates that actually have gaps.

        Only gappy columns get a flag. A fully populated column would contribute a constant-True
        channel: it doubles the static width, carries no information, and costs the model parameters
        to ignore it. This mirrors what sklearn's ``SimpleImputer(add_indicator=True)`` does on the
        other family with ``features='missing-only'``, so both sides emit the same set.
        """
        rows = len(static_df)
        gappy = [
            column
            for column in feature_columns
            if column in static_df.columns
            and not bool(build_finite_row_mask(static_df, numeric_columns=[column]).all())
        ]
        if not gappy:
            return np.empty((rows, 0), dtype=bool), []

        validity = np.column_stack(
            [build_finite_row_mask(static_df, numeric_columns=[column]).to_numpy(dtype=bool) for column in gappy]
        )
        self.logger.info(
            f"Carrying measured-vs-filled flags for {len(gappy)} continuous covariate(s) with gaps: "
            + ", ".join(gappy)
        )
        return validity, gappy

    # --- measured lab values ----------------------------------------------

    def _extract_label_features(self, static_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Every LABEL_COLUMNS value the static frame carries, kept OUT of the feature blocks.

        Deliberately bypasses ``filter_schema``, whose whole job is to remove these. What authorises
        that is ``CARRY_LABEL_COLUMNS``: with the flag off this returns nothing, so the bundle and
        every batch collated from it are identical to a build that had never heard of lab values.
        Carrying them does not make them predictors either - nothing reads them unless a model names
        them in ``auxiliary_label_columns``, which is an explicit, per-column opt-out of that rail
        for the case where the value is genuinely measured at inference time too.

        Reading the flag here rather than trusting the frame is what keeps the joint-file and
        split-file layouts in agreement. DataManager carries the labels across the targets join
        under the same flag, so "what may a model select?" is answered by the flag plus
        LABEL_COLUMNS, never by which files the data happens to be split into.

        Rows are NOT dropped for a missing value. Lab coverage varies from complete to ~37% absent
        across these columns, so requiring finiteness here would let the choice of an auxiliary
        column silently delete a third of the dataset. The datamodule median-fills from the train
        split and passes a validity flag instead.
        """
        if not getattr(self.config, "CARRY_LABEL_COLUMNS", False):
            return np.empty((len(static_df), 0), dtype=np.float32), []

        label_columns = [
            str(column)
            for column in (getattr(self.config, "LABEL_COLUMNS", []) or [])
            if str(column) in static_df.columns
        ]
        label_columns = list(dict.fromkeys(label_columns))
        if not label_columns:
            return np.empty((len(static_df), 0), dtype=np.float32), []

        values = np.column_stack(
            [pd.to_numeric(static_df[column], errors="coerce").to_numpy(dtype=np.float64) for column in label_columns]
        )
        # Infinities join the missing: an inf reaching the standardizer would poison the column
        # statistic, and there is no meaningful lab value it could represent.
        values[~np.isfinite(values)] = np.nan
        return values.astype(np.float32), label_columns

    # --- coordinates ------------------------------------------------------

    def _coordinate_columns(self) -> list[str]:
        """The lat/lon column names when the harmonic branch wants them, else nothing.

        The single place the flag is read. Both :meth:`clean_static_frame` - which must drop rows
        with no coordinate - and :meth:`_extract_coordinates` - which reads the values - go through
        it, so the row rule and the value rule can never disagree about whether coordinates are in
        play. That matters more than it looks: ``usable_point_ids`` calls the former and feeds the
        unified splitter, so a disagreement would have the splitter assign points this builder then
        deletes.
        """
        if not getattr(self.config, "USE_HARMONIC_COORDS", False):
            return []
        return list(self.data_manager.coordinate_columns())

    def _extract_coordinates(self, static_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """lat/lon as a dedicated block, kept OUT of the feature frame.

        Deliberately bypasses ``filter_schema``, whose whole job is to remove these. What authorises
        that is ``USE_HARMONIC_COORDS`` - the same shape of opt-in ``CARRY_LABEL_COLUMNS`` gives the
        lab values, and with the flag off this returns nothing, so the bundle and every batch
        collated from it are identical to a build that had never heard of coordinates.

        Carrying them does not make them predictors. They never enter ``blocks.continuous_columns``,
        are never standardized alongside the covariates, and are read by exactly one consumer: the
        CNN's harmonic branch, which normalizes them against the train bounding box. Passing them
        through ``filter_schema`` instead would have handed raw degrees to every sklearn model as an
        ordinary column, and would have z-scored them - which is the wrong transform for a
        positional encoding, whose frequencies are defined on a known interval.

        float64 throughout: see ``to_decimal_year`` for the same argument. float32 resolves about a
        metre at this latitude and the normalization downstream subtracts two nearby numbers.
        """
        coordinate_columns = self._coordinate_columns()
        if not coordinate_columns:
            return np.empty((len(static_df), 0), dtype=np.float64), []

        missing = [column for column in coordinate_columns if column not in static_df.columns]
        if missing:
            raise KeyError(
                f"USE_HARMONIC_COORDS is on but the static frame has no {', '.join(missing)} "
                f"column(s). Coordinates often live in the TARGETS source rather than the covariates "
                f"one; DataManager joins them across automatically, so check that "
                f"LAT_COLUMN/LON_COLUMN name columns that exist in one of the two. "
                f"Available: {', '.join(map(str, static_df.columns))}"
            )

        values = np.column_stack(
            [
                pd.to_numeric(static_df[column], errors="coerce").to_numpy(dtype=np.float64)
                for column in coordinate_columns
            ]
        )
        return values, coordinate_columns

    # --- temporal assembly ------------------------------------------------

    def _modality_entries(self, timeseries_df: pd.DataFrame) -> list[tuple[str, list[str]]]:
        """Modality name -> its columns, resolved from the prefix map or an explicit column list."""
        temporal_config = self.data_manager.temporal_config()
        prefix_map = self.data_manager.normalize_mapping(
            temporal_config.get("modality_prefix_map", getattr(self.config, "MODALITY_PREFIX_MAP", {}))
        )
        if prefix_map:
            return [
                (
                    str(name).lower(),
                    [column for column in timeseries_df.columns if column.startswith(str(prefix))],
                )
                for name, prefix in prefix_map.items()
            ]

        explicit_columns = self.data_manager.normalize_mapping(temporal_config.get("modality_columns", {}))
        if not explicit_columns:
            explicit_columns = {
                "s1": list(getattr(self.config, "S1_COLUMNS", []) or []),
                "s2": list(getattr(self.config, "S2_COLUMNS", []) or []),
                "modis": list(getattr(self.config, "MODIS_COLUMNS", []) or []),
            }

        entries: list[tuple[str, list[str]]] = []
        for name, columns in explicit_columns.items():
            matched = [column for column in columns if column in timeseries_df.columns]
            if matched:
                entries.append((str(name).lower(), matched))
        return entries

    def _build_sequences(
        self,
        timeseries_df: pd.DataFrame,
        *,
        point_ids: list[Any],
        point_col: str,
    ) -> tuple[
        dict[str, list[np.ndarray]],
        dict[str, list[np.ndarray]],
        dict[str, list[np.ndarray]],
        dict[str, list[str]],
    ]:
        temporal_config = self.data_manager.temporal_config()
        time_col = temporal_config.get("time_column", getattr(self.config, "TIME_COLUMN", "date"))

        if point_col not in timeseries_df.columns or time_col not in timeseries_df.columns:
            raise KeyError(f"Time-series source must contain '{point_col}' and '{time_col}' columns")

        modality_entries = [(name, columns) for name, columns in self._modality_entries(timeseries_df) if columns]
        if not modality_entries:
            self.logger.warning("No temporal modality columns matched the time-series source")
            return {}, {}, {}, {}

        all_modality_columns = [column for _, columns in modality_entries for column in columns]
        timeseries_df, validity_df = sanitize_numeric_columns(
            timeseries_df, all_modality_columns, logger=self.logger, return_validity=True
        )

        unique_columns = list(dict.fromkeys(all_modality_columns))
        working = timeseries_df[[point_col, time_col, *unique_columns]].copy()
        # Carry validity as ordinary columns so every filter, sort and groupby below applies to it
        # identically - keeping it in a side array would silently desynchronise on the first reorder.
        for column in unique_columns:
            working[_VALIDITY_PREFIX + column] = validity_df[column].to_numpy(dtype=np.float32)
        working = working[working[point_col].notna()]

        # Parse dates before anything else: a row whose date cannot be read has no place on a
        # continuous time axis, and silently keeping it would corrupt the gap features.
        decimal_year = to_decimal_year(working[time_col])
        unparsed = int(np.isnan(decimal_year).sum())
        if unparsed:
            self.logger.warning(
                f"Dropped {unparsed} time-series row(s) whose '{time_col}' could not be parsed as a date"
            )
        working = working.loc[np.isfinite(decimal_year)].copy()
        working["__decimal_year__"] = decimal_year[np.isfinite(decimal_year)]

        # Keep only points that survived static cleaning, then collapse duplicate readings. A
        # duplicate timestamp would make the gap between consecutive tokens zero, which the bundle
        # rejects; averaging is the least surprising repair.
        known_points = set(point_ids)
        before = len(working)
        working = working[working[point_col].isin(known_points)]
        if len(working) < before:
            self.logger.warning(
                f"Dropped {before - len(working)} time-series row(s) that did not match a surviving point id"
            )

        grouped_keys = [point_col, "__decimal_year__"]
        duplicates = int(working.duplicated(subset=grouped_keys).sum())
        if duplicates:
            self.logger.warning(
                f"Averaged {duplicates} duplicate (point, date) time-series row(s) so timestamps stay strictly ascending"
            )
            working = working.groupby(grouped_keys, as_index=False, sort=True).mean(numeric_only=True)

        working = working.sort_values(grouped_keys, kind="stable")

        sequences: dict[str, list[np.ndarray]] = {}
        sequence_times: dict[str, list[np.ndarray]] = {}
        sequence_validity: dict[str, list[np.ndarray]] = {}
        modality_columns: dict[str, list[str]] = {}

        # Row positions per point, computed once and reused by every modality. `working` is already
        # sorted by (point, date), so each group's positions are in ascending time order and the
        # per-point slices below need no further sorting.
        row_groups: dict[Any, np.ndarray] = {
            point_id: np.asarray(rows, dtype=np.int64)
            for point_id, rows in working.groupby(point_col, sort=False).indices.items()
        }
        times_all = working["__decimal_year__"].to_numpy(dtype=np.float64)

        for modality_name, columns in modality_entries:
            values_all = working[columns].to_numpy(dtype=np.float32)
            # A duplicate-date groupby averages the validity flags too; require every contributing
            # row to have been measured before calling the averaged cell measured.
            validity_all = working[[_VALIDITY_PREFIX + column for column in columns]].to_numpy() >= 1.0

            per_point_values: list[np.ndarray] = []
            per_point_times: list[np.ndarray] = []
            per_point_validity: list[np.ndarray] = []
            for point_id in point_ids:
                rows = row_groups.get(point_id)
                if rows is None or rows.size == 0:
                    per_point_values.append(np.empty((0, len(columns)), dtype=np.float32))
                    per_point_times.append(np.empty((0,), dtype=np.float64))
                    per_point_validity.append(np.empty((0, len(columns)), dtype=bool))
                    continue
                per_point_values.append(values_all[rows])
                per_point_times.append(times_all[rows])
                per_point_validity.append(validity_all[rows])

            sequences[modality_name] = per_point_values
            sequence_times[modality_name] = per_point_times
            sequence_validity[modality_name] = per_point_validity
            modality_columns[modality_name] = list(columns)

        return sequences, sequence_times, sequence_validity, modality_columns

    def _log_summary(self, bundle: SoilSequenceBundle) -> None:
        self.logger.info(
            f"Built sequence bundle over {bundle.num_points} point(s) with "
            f"{bundle.static_features.shape[1] if bundle.static_features.size else 0} static feature(s)"
        )
        if bundle.label_dim:
            # Reported with the missing share because that is what decides whether a column is
            # usable as an auxiliary input: a 37%-absent column is mostly train-median by the time
            # the model sees it, which is a different feature from the one its name suggests.
            sparse = sorted(
                (
                    (name, bundle.label_missing_fraction(name))
                    for name in bundle.label_feature_names
                    if bundle.label_missing_fraction(name) > 0
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            self.logger.info(
                f"  {bundle.label_dim} measured lab column(s) available as auxiliary inputs"
                + (
                    f"; most incomplete: {', '.join(f'{name} ({share:.1%} missing)' for name, share in sparse[:3])}"
                    if sparse
                    else "; all complete"
                )
            )
        for modality_name in sorted(bundle.sequences):
            counts = bundle.observation_counts(modality_name)
            if counts.size == 0:
                continue
            empty = int((counts == 0).sum())
            imputed = bundle.imputed_fraction(modality_name)
            self.logger.info(
                f"  modality '{modality_name}': {len(bundle.modality_columns[modality_name])} channel(s), "
                f"observations per point min={int(counts.min())} median={int(np.median(counts))} "
                f"max={int(counts.max())}"
                + (f", {empty} point(s) with no observations" if empty else "")
                # Surfaced per modality because a median-filled cell is a repair, not a measurement,
                # and only the reader can judge whether that share is acceptable.
                + (f", {imputed:.1%} of cells median-filled" if imputed > 0 else "")
            )
