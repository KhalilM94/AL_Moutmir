"""One train/val/test split, decided once, consumed by every training family.

Before this module existed each family split for itself: :class:`SklearnDataSplitter` carved 70/30
off the tabular frame, while the Lightning datamodules carved 64/16/20 off a bundle built
independently from the same CSVs. The two holdouts overlapped, so a Lightning *test* point was very
likely an sklearn *training* point - and ``metrics.py`` publishes ``rmse_test`` under one name for
both families, which put those two numbers on one leaderboard axis.

The split here is decided over ``POINT_ID_COLUMN`` rather than over row positions, which is what
makes it shareable: the families disagree about which rows are usable (the sequence builder drops
non-finite rows the tabular preprocessor keeps), so positional indices into one family's arrays mean
nothing to the other. A point id means the same thing everywhere.

Two conventions differ deliberately from the code this replaces:

* **``test_size`` and ``val_size`` are fractions of the whole population**, not of the remainder.
  The old sequential carve is why ``lightning_registry.yml`` carried the comment "without this,
  train was 0.7*0.7=0.49"; asking for 0.2 and getting 0.16 is a footgun, so it is gone.
* **The split is a plan, not four dataframes.** It is persisted as one
  ``point_id -> split`` table, so a finished run can be re-keyed to its source data - the previous
  ``data_splits/*.parquet`` artifacts were written with ``index=False`` and with ``uuid`` already
  stripped by ``filter_schema``, so they carried no join key at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

TRAIN = "train"
VAL = "val"
TEST = "test"
SPLIT_NAMES = (TRAIN, VAL, TEST)

RANDOM = "random"
SPATIAL_GROUP = "spatial_group"
STRATEGIES = (RANDOM, SPATIAL_GROUP)

INTERSECT = "intersect"
ASSIGN_ALL = "assign_all"
POPULATION_POLICIES = (INTERSECT, ASSIGN_ALL)


@dataclass(frozen=True)
class SplitPlan:
    """Which split every point belongs to, plus the provenance needed to reproduce it.

    ``assignments`` is indexed by point id. A point absent from it belongs to no split and is
    therefore invisible to every family - that is how ``population_policy: intersect`` excludes the
    points only one family can use.
    """

    assignments: pd.Series
    strategy: str
    test_size: float
    val_size: float
    seed: int
    population_policy: str
    eligibility: Mapping[str, frozenset] = field(default_factory=dict)
    clusters: Optional[pd.Series] = None

    def point_ids_for(self, split: str) -> pd.Index:
        """Every point id assigned to `split`."""
        _validate_split_name(split)
        return self.assignments.index[self.assignments.to_numpy() == split]

    def labels_for(self, point_ids: Sequence) -> pd.Series:
        """The split label of each id in `point_ids`, in that order. Unassigned ids give NaN."""
        return self.assignments.reindex(pd.Index(_as_index(point_ids)))

    def indices_for(self, point_ids: Sequence, split: str) -> np.ndarray:
        """Positional indices into `point_ids` whose assignment is `split`.

        `point_ids` is the consuming family's own ordering - a bundle's ``point_ids`` list or a
        frame's id column - so each family resolves the shared plan against its own array layout.
        """
        _validate_split_name(split)
        labels = self.labels_for(point_ids).to_numpy()
        return np.flatnonzero(labels == split).astype(np.int64)

    def split_indices(self, point_ids: Sequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(train_idx, val_idx, test_idx)`` for `point_ids`, resolved in one pass."""
        labels = self.labels_for(point_ids).to_numpy()
        return tuple(np.flatnonzero(labels == name).astype(np.int64) for name in SPLIT_NAMES)

    def counts(self) -> dict[str, int]:
        return {name: int((self.assignments.to_numpy() == name).sum()) for name in SPLIT_NAMES}

    def to_frame(self) -> pd.DataFrame:
        """The persisted form: one row per point, with a join key and the eligibility flags."""
        frame = pd.DataFrame(
            {
                "point_id": self.assignments.index.to_numpy(),
                "split": self.assignments.to_numpy(),
            }
        )
        if self.clusters is not None:
            frame["cluster"] = self.clusters.reindex(self.assignments.index).to_numpy()
        for family, ids in sorted(self.eligibility.items()):
            frame[f"eligible_{family}"] = frame["point_id"].isin(ids).to_numpy()
        return frame

    def describe(self) -> dict[str, Any]:
        """Flat, log-friendly provenance. Every value is an MLflow-loggable scalar."""
        counts = self.counts()
        total = sum(counts.values()) or 1
        described: dict[str, Any] = {
            "split_strategy": self.strategy,
            "split_test_size": self.test_size,
            "split_val_size": self.val_size,
            "split_seed": self.seed,
            "split_population_policy": self.population_policy,
            "split_n_total": sum(counts.values()),
        }
        for name in SPLIT_NAMES:
            described[f"split_n_{name}"] = counts[name]
            described[f"split_fraction_{name}"] = round(counts[name] / total, 6)
        for family, ids in sorted(self.eligibility.items()):
            described[f"split_n_eligible_{family}"] = len(ids)
            described[f"split_n_excluded_{family}"] = len(
                set(self.assignments.index) - set(ids)
            )
        return described

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, **provenance: Any) -> "SplitPlan":
        """Rebuild a plan from :meth:`to_frame` output, for ``split.plan_path``."""
        missing = {"point_id", "split"} - set(frame.columns)
        if missing:
            raise KeyError(f"A split plan frame needs columns {sorted(missing)}; got {list(frame.columns)}")
        assignments = pd.Series(
            frame["split"].to_numpy(), index=pd.Index(frame["point_id"].to_numpy(), name="point_id")
        )
        unknown = sorted(set(assignments.unique()) - set(SPLIT_NAMES))
        if unknown:
            raise ValueError(f"Split plan carries unknown split name(s) {unknown}; expected {list(SPLIT_NAMES)}")
        clusters = None
        if "cluster" in frame.columns:
            clusters = pd.Series(frame["cluster"].to_numpy(), index=assignments.index)
        eligibility = {
            column[len("eligible_") :]: frozenset(
                frame.loc[frame[column].astype(bool), "point_id"].to_numpy()
            )
            for column in frame.columns
            if column.startswith("eligible_")
        }
        provenance.setdefault("strategy", "loaded")
        provenance.setdefault("test_size", float("nan"))
        provenance.setdefault("val_size", float("nan"))
        provenance.setdefault("seed", -1)
        provenance.setdefault("population_policy", "loaded")
        return cls(
            assignments=assignments,
            eligibility=eligibility,
            clusters=clusters,
            **provenance,
        )


class UnifiedSplitter:
    """Turns a population of point ids into a :class:`SplitPlan`.

    Knows nothing about either training family - it is handed ids and coordinates and returns
    assignments, which is what lets both families consume the same object.
    """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.strategy = str(getattr(config, "SPLIT_HOLDOUT_STRATEGY", RANDOM)).lower()
        if self.strategy not in STRATEGIES:
            raise ValueError(
                f"split.strategy must be one of {list(STRATEGIES)}; got {self.strategy!r}"
            )
        self.test_size = _validated_fraction(getattr(config, "SPLIT_TEST_SIZE", 0.2), "split.test_size")
        self.val_size = _validated_fraction(getattr(config, "SPLIT_VAL_SIZE", 0.16), "split.val_size")
        if self.test_size + self.val_size >= 1.0:
            raise ValueError(
                f"split.test_size + split.val_size must leave a training set; got "
                f"{self.test_size} + {self.val_size} = {self.test_size + self.val_size}"
            )
        self.seed = int(getattr(config, "SPLIT_SEED", getattr(config, "RANDOM_SEED", 42)))
        self.population_policy = str(
            getattr(config, "SPLIT_POPULATION_POLICY", INTERSECT)
        ).lower()
        if self.population_policy not in POPULATION_POLICIES:
            raise ValueError(
                f"split.population_policy must be one of {list(POPULATION_POLICIES)}; "
                f"got {self.population_policy!r}"
            )
        self.cluster_strategy_ = None

    def build_plan(
        self,
        point_ids: Sequence,
        *,
        coordinates: Optional[pd.DataFrame] = None,
        eligibility: Optional[Mapping[str, Iterable]] = None,
    ) -> SplitPlan:
        """Assign every id in `point_ids` to train, val or test.

        `coordinates` is a frame indexed by point id carrying the lat/lon columns named in the
        config. It is required for ``spatial_group`` and ignored otherwise.
        """
        ids = _as_index(point_ids)
        duplicated = ids[ids.duplicated()].unique()
        if len(duplicated):
            raise ValueError(
                f"{len(duplicated)} duplicate point id(s) in the split population, e.g. "
                f"{list(duplicated[:5])}. A split is keyed on point id, so ids must be unique."
            )
        if len(ids) == 0:
            raise ValueError("Cannot build a split plan over an empty population.")

        eligibility = {
            family: frozenset(pd.Index(values)) for family, values in (eligibility or {}).items()
        }

        if self.strategy == SPATIAL_GROUP:
            assignments, clusters = self._spatial_group_assignments(ids, coordinates)
        else:
            assignments, clusters = self._random_assignments(ids), None

        plan = SplitPlan(
            assignments=assignments,
            strategy=self.strategy,
            test_size=self.test_size,
            val_size=self.val_size,
            seed=self.seed,
            population_policy=self.population_policy,
            eligibility=eligibility,
            clusters=clusters,
        )
        counts = plan.counts()
        self.logger.info(
            f"Split plan ({self.strategy}, seed={self.seed}, policy={self.population_policy}): "
            f"train={counts[TRAIN]} | val={counts[VAL]} | test={counts[TEST]}"
        )
        return plan

    # --- strategies -----------------------------------------------------------------

    def _random_assignments(self, ids: pd.Index) -> pd.Series:
        test_ids, remainder = _carve(np.asarray(ids), self.test_size, self.seed)
        val_ids, train_ids = _carve(remainder, _remainder_fraction(self.val_size, self.test_size), self.seed)
        return _assignments_from_ids(ids, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)

    def _spatial_group_assignments(
        self, ids: pd.Index, coordinates: Optional[pd.DataFrame]
    ) -> tuple[pd.Series, pd.Series]:
        """Cluster spatially, then hold out whole clusters so no cluster straddles two splits."""
        from sklearn.model_selection import GroupShuffleSplit

        if coordinates is None:
            raise ValueError("split.strategy 'spatial_group' needs coordinates; none were provided.")

        lat_col = getattr(self.config, "LAT_COLUMN", "lat")
        lon_col = getattr(self.config, "LON_COLUMN", "lon")
        frame = self._clustering_frame(ids, coordinates, lat_col=lat_col, lon_col=lon_col)

        strategy = self._load_cluster_strategy()
        clustered = strategy.cluster(frame)
        cluster_series = pd.Series(
            clustered["cluster"].to_numpy(), index=pd.Index(clustered["point_id"].to_numpy(), name="point_id")
        )
        self.logger.info(f"Cluster value counts:\n{cluster_series.value_counts().sort_index().to_string()}")

        # A point the clusterer could not place (missing coordinates) has no spatial block, so it
        # cannot be held out honestly. Putting it in train never inflates a test score; dropping it
        # would silently shrink the dataset.
        unclustered = ids.difference(cluster_series.index)
        if len(unclustered):
            self.logger.warning(
                f"{len(unclustered)} point(s) could not be spatially clustered (missing coordinates) "
                f"and are assigned to the train split rather than held out."
            )

        clustered_ids = pd.Index(cluster_series.index)
        groups = cluster_series.to_numpy()
        placeholder = np.zeros((len(clustered_ids), 1))

        train_val_pos, test_pos = next(
            GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.seed).split(
                placeholder, groups=groups
            )
        )
        test_ids = clustered_ids[test_pos]

        val_fraction = _remainder_fraction(self.val_size, self.test_size)
        remaining_groups = len(np.unique(groups[train_val_pos]))
        if val_fraction <= 0.0 or len(train_val_pos) <= 1 or remaining_groups < 2:
            # A grouped carve cannot split a single remaining cluster without emptying train, and
            # splitting *within* a cluster would defeat the point of blocking on it. Skip the val
            # holdout rather than raise: a run with few clusters is unusual, not invalid.
            if remaining_groups < 2 and val_fraction > 0.0:
                self.logger.warning(
                    f"Only {remaining_groups} cluster(s) remain after the test holdout, so no "
                    f"grouped validation split is possible; val is empty and train keeps them all. "
                    f"Raise the cluster count or lower split.test_size to get a validation set."
                )
            val_ids = pd.Index([], dtype=clustered_ids.dtype)
            train_ids = clustered_ids[train_val_pos]
        else:
            inner_train_pos, inner_val_pos = next(
                GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=self.seed).split(
                    placeholder[train_val_pos], groups=groups[train_val_pos]
                )
            )
            val_ids = clustered_ids[train_val_pos[inner_val_pos]]
            train_ids = clustered_ids[train_val_pos[inner_train_pos]]

        train_ids = train_ids.append(unclustered)
        self._plot_split(frame, clustered, clustered_ids, test_ids)
        assignments = _assignments_from_ids(ids, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)
        return assignments, cluster_series

    def _clustering_frame(
        self, ids: pd.Index, coordinates: pd.DataFrame, *, lat_col: str, lon_col: str
    ) -> pd.DataFrame:
        missing = [column for column in (lat_col, lon_col) if column not in coordinates.columns]
        if missing:
            raise KeyError(
                f"split.strategy 'spatial_group' needs coordinate column(s) {missing} on the "
                f"population frame; got {list(coordinates.columns)}"
            )
        aligned = coordinates.reindex(ids)
        # A RangeIndex so the strategies' positional plotting helpers line up, plus explicit
        # `lat`/`lon` aliases because the shipped strategies default to those names.
        frame = pd.DataFrame(
            {
                "point_id": ids.to_numpy(),
                lat_col: aligned[lat_col].to_numpy(),
                lon_col: aligned[lon_col].to_numpy(),
            }
        )
        frame["lat"] = frame[lat_col].to_numpy()
        frame["lon"] = frame[lon_col].to_numpy()
        return frame

    def _load_cluster_strategy(self):
        from yg_eo_soilnet.clustering_utils import BaseSpatialClusterStrategy
        from yg_eo_soilnet.models import ModelConfigFactory  # local: avoids a circular import

        spec = dict(getattr(self.config, "SPLIT_GROUP_STRATEGY", {}) or {})
        if not spec.get("class_path"):
            raise ValueError(
                "split.strategy is 'spatial_group' but split.group.class_path is not set; "
                "name a BaseSpatialClusterStrategy, e.g. "
                "yg_eo_soilnet.clustering_utils.KMeansClusterStrategy"
            )
        spec.setdefault("enabled", True)
        self.logger.info(f"Clustering the split population with {spec['class_path'].rsplit('.', 1)[-1]}...")
        strategy = ModelConfigFactory(spec, self.seed).load_splitter_from_config()
        if not isinstance(strategy, BaseSpatialClusterStrategy):
            # Falling through used to produce empty train/test frames and four empty parquet
            # artifacts, surfacing much later as KeyError('groups_train').
            raise TypeError(
                "split.group did not resolve to a BaseSpatialClusterStrategy (got "
                f"{type(strategy).__name__}). Check class_path and 'enabled' in {spec!r}."
            )
        self.cluster_strategy_ = strategy
        return strategy

    def _plot_split(self, frame, clustered, clustered_ids, test_ids) -> None:
        """Best-effort split map. A plotting or MLflow failure must not lose the split."""
        if self.cluster_strategy_ is None:
            return
        try:
            position = pd.Series(np.arange(len(clustered_ids)), index=clustered_ids)
            test_pos = position.reindex(pd.Index(test_ids)).dropna().to_numpy(dtype=int)
            train_pos = np.setdiff1d(np.arange(len(clustered_ids)), test_pos)
            self.cluster_strategy_.plot_train_test(
                clustered.reset_index(drop=True),
                train_pos,
                test_pos,
                title="Spatial Group Train/Test Split",
                filename="grid_split.png",
            )
        except Exception as error:  # noqa: BLE001 - a picture is never worth failing a run over
            self.logger.warning(f"Could not plot the spatial split map: {error}")


# --- helpers ------------------------------------------------------------------------


def _validate_split_name(split: str) -> None:
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {split!r}; expected one of {list(SPLIT_NAMES)}")


def _validated_fraction(value: Any, label: str) -> float:
    fraction = float(value)
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"{label} must be in [0, 1); got {fraction}")
    return fraction


def _as_index(point_ids: Sequence) -> pd.Index:
    if isinstance(point_ids, pd.Index):
        return point_ids
    if isinstance(point_ids, pd.Series):
        return pd.Index(point_ids.to_numpy())
    return pd.Index(np.asarray(point_ids))


def _remainder_fraction(val_size: float, test_size: float) -> float:
    """`val_size` is a fraction of the WHOLE population; convert it to one of what test left over."""
    remaining = 1.0 - test_size
    if remaining <= 0.0:
        return 0.0
    return min(val_size / remaining, 0.99)


def _carve(values: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split off `fraction` of `values`, returning ``(held_out, remainder)``.

    A fraction of 0 means "no holdout", which ``train_test_split`` rejects outright rather than
    treating as empty - so it is handled here, keeping ``val_size=0`` and ``test_size=0`` usable.
    """
    values = np.asarray(values)
    if fraction <= 0.0 or values.size <= 1:
        return values[:0], values
    from sklearn.model_selection import train_test_split

    remainder, held_out = train_test_split(
        values, test_size=fraction, random_state=seed, shuffle=True
    )
    return np.asarray(held_out), np.asarray(remainder)


def _assignments_from_ids(ids: pd.Index, *, train_ids, val_ids, test_ids) -> pd.Series:
    assignments = pd.Series(TRAIN, index=pd.Index(ids, name="point_id"), dtype=object)
    assignments.loc[pd.Index(train_ids)] = TRAIN
    assignments.loc[pd.Index(val_ids)] = VAL
    assignments.loc[pd.Index(test_ids)] = TEST
    return assignments
