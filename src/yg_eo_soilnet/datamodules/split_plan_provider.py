"""Builds the one :class:`SplitPlan` a run uses, before any training family touches the data.

:mod:`yg_eo_soilnet.datamodules.splitting` is deliberately family-neutral - it turns point ids into
assignments and knows nothing about sklearn or Lightning. This module is where that neutrality is
paid for: it asks each family which points it can actually use, reconciles the answers under the
configured ``population_policy``, and hands back a single plan.

The reconciliation matters because the families genuinely disagree. The tabular preprocessor keeps
every row; the sequence builder drops rows with non-finite covariates; the graph builder drops those
*and* rows with no coordinates. Splitting each family's own population separately is exactly the bug
this replaces.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import pandas as pd

from yg_eo_soilnet.datamodules.splitting import (
    ASSIGN_ALL,
    INTERSECT,
    SplitPlan,
    UnifiedSplitter,
)

SKLEARN = "sklearn"
SEQUENCE = "sequence"
GRAPH = "graph"
KNOWN_FAMILIES = (SKLEARN, SEQUENCE, GRAPH)


class SplitPlanProvider:
    """Memoized source of truth for the run's split.

    `families` names the training families that will actually run, so a sklearn-only run never pays
    to build the graph eligibility (which ordinal-encodes a copy of the frame). Pass ``None`` to
    infer it from the enabled registry entries.
    """

    def __init__(self, config, logger, data_manager, families: Optional[Iterable[str]] = None):
        self.config = config
        self.logger = logger
        self.data_manager = data_manager
        self.families = tuple(families) if families is not None else self._infer_families(config)
        unknown = [family for family in self.families if family not in KNOWN_FAMILIES]
        if unknown:
            raise ValueError(f"Unknown training family/families {unknown}; expected {list(KNOWN_FAMILIES)}")
        self._plan: Optional[SplitPlan] = None

    def plan(self) -> SplitPlan:
        """The run's split. Built once; every later call returns the same object."""
        if self._plan is None:
            self._plan = self._build()
        return self._plan

    # --- internals ------------------------------------------------------------------

    def _build(self) -> SplitPlan:
        dataset = self.data_manager.load_dataset()
        tabular = dataset.tabular
        point_col = dataset.point_id_column
        if point_col in tabular.columns:
            all_ids = pd.Index(tabular[point_col].to_numpy(), name="point_id")
        else:
            self.logger.warning(
                f"No point id column {point_col!r} in the source frame; falling back to row position "
                f"as the split key. Every family derives it from the same frame in the same order, "
                f"so they still agree, but the plan cannot be re-keyed to a different export."
            )
            all_ids = pd.Index(range(len(tabular)), name="point_id")

        eligibility = self._eligibility(tabular)
        population = self._population(all_ids, eligibility)

        splitter = UnifiedSplitter(self.config, self.logger)
        self._log_population(all_ids, population, eligibility, splitter.population_policy)

        plan_path = getattr(self.config, "SPLIT_PLAN_PATH", None)
        if plan_path:
            return self._load_plan(plan_path, population, eligibility, splitter)

        coordinates = self._coordinates(tabular, all_ids)
        return splitter.build_plan(population, coordinates=coordinates, eligibility=eligibility)

    def _eligibility(self, tabular: pd.DataFrame) -> dict[str, frozenset]:
        builders = {
            SKLEARN: self._sklearn_usable,
            SEQUENCE: self._sequence_usable,
            GRAPH: self._graph_usable,
        }
        eligibility: dict[str, frozenset] = {}
        for family in self.families:
            eligibility[family] = frozenset(builders[family](tabular))
        return eligibility

    def _sklearn_usable(self, tabular: pd.DataFrame) -> pd.Index:
        from yg_eo_soilnet.datamodules.scikit.tabular_preprocessor import TabularPreprocessor

        return TabularPreprocessor(self.config, self.logger, self.data_manager).usable_point_ids(tabular)

    def _sequence_usable(self, tabular: pd.DataFrame) -> pd.Index:
        from yg_eo_soilnet.datamodules.sequence.sequence_builder import SoilSequenceBuilder

        return SoilSequenceBuilder(self.config, self.logger, self.data_manager).usable_point_ids(tabular)

    def _graph_usable(self, tabular: pd.DataFrame) -> pd.Index:
        from yg_eo_soilnet.datamodules.lightning.spatiotemporal_graph_builder import (
            SpatiotemporalGraphBuilder,
        )

        return SpatiotemporalGraphBuilder(self.config, self.logger, self.data_manager).usable_point_ids(
            tabular
        )

    def _population(self, all_ids: pd.Index, eligibility: Mapping[str, frozenset]) -> pd.Index:
        policy = str(getattr(self.config, "SPLIT_POPULATION_POLICY", INTERSECT)).lower()
        if policy not in (INTERSECT, ASSIGN_ALL):
            raise ValueError(
                f"split.population_policy must be one of {[INTERSECT, ASSIGN_ALL]}; got {policy!r}"
            )

        if policy == ASSIGN_ALL or not eligibility:
            # Every point gets a label; each family later selects the subset it holds. Test sets
            # then share membership but are not identical row for row.
            population = all_ids
        else:
            common: Optional[set] = None
            for ids in eligibility.values():
                common = set(ids) if common is None else (common & set(ids))
            # Preserve the source frame's order so the plan is stable under a re-run.
            population = all_ids[all_ids.isin(common or set())]

        self._guard_against_collapse(all_ids, population, eligibility, policy)
        return population

    def _guard_against_collapse(
        self,
        all_ids: pd.Index,
        population: pd.Index,
        eligibility: Mapping[str, frozenset],
        policy: str,
    ) -> None:
        """Refuse to train on a population most of the dataset has fallen out of.

        Two independent ways that happens, and both used to be silent:

        * **A family's own cleaning is too strict.** Checked per family, on every policy. This is the
          one that matters on a single-family run and under ``assign_all`` - there is no
          intersection to notice it, so the family simply trains on what is left and reports metrics
          as if nothing happened. On one real dataset that was 17 points out of 5761.
        * **Intersecting handed everyone the narrowest family's population.** Only possible under
          ``intersect``, and reported separately so the message says which of the two happened.
        """
        minimum_ratio = float(getattr(self.config, "SPLIT_MIN_POPULATION_RATIO", 0.5))
        if minimum_ratio <= 0.0 or not len(all_ids):
            return

        total = len(all_ids)
        starved = {
            family: ids
            for family, ids in eligibility.items()
            if len(ids) / total < minimum_ratio
        }
        if starved:
            listed = "\n".join(
                f"  {family}: {len(ids)} of {total} usable ({len(ids) / total:.1%})"
                for family, ids in sorted(starved.items(), key=lambda item: len(item[1]))
            )
            raise ValueError(
                f"{len(starved)} training family/families can use less than "
                f"split.min_population_ratio={minimum_ratio:.0%} of the dataset after their own "
                f"cleaning:\n{listed}\n"
                f"This is a data problem, not a split problem, and it applies whichever "
                f"population_policy is set. The builders log which columns cost the most rows (look "
                f"for 'Worst columns by rows lost'); dropping a near-empty covariate via "
                f"IGNORED_COLUMNS/ELIMINATED_FEATURES in data_spec.yml usually restores it. Lower "
                f"split.min_population_ratio to proceed anyway."
            )

        ratio = len(population) / total
        if ratio >= minimum_ratio:
            return

        breakdown = ", ".join(
            f"{family}={len(ids)}"
            for family, ids in sorted(eligibility.items(), key=lambda item: len(item[1]))
        )
        raise ValueError(
            f"split.population_policy={policy!r} left {len(population)} of {total} point(s) "
            f"({ratio:.1%}), below split.min_population_ratio={minimum_ratio}. Every family is "
            f"individually fine ({breakdown}), so it is the OVERLAP between them that is small - "
            f"they disagree about which rows are usable rather than any one of them being starved.\n"
            f"Set split.population_policy='assign_all' to let each family use what it has (test "
            f"sets then share membership without being identical), or lower "
            f"split.min_population_ratio."
        )

    def _coordinates(self, tabular: pd.DataFrame, all_ids: pd.Index) -> Optional[pd.DataFrame]:
        lat_col = getattr(self.config, "LAT_COLUMN", "lat")
        lon_col = getattr(self.config, "LON_COLUMN", "lon")
        if lat_col not in tabular.columns or lon_col not in tabular.columns:
            return None
        return pd.DataFrame(
            {
                lat_col: tabular[lat_col].to_numpy(),
                lon_col: tabular[lon_col].to_numpy(),
            },
            index=all_ids,
        )

    def _load_plan(
        self,
        plan_path: str,
        population: pd.Index,
        eligibility: Mapping[str, frozenset],
        splitter: UnifiedSplitter,
    ) -> SplitPlan:
        """Reuse a frozen assignment, so a re-run keeps yesterday's test set."""
        frame = pd.read_parquet(plan_path)
        plan = SplitPlan.from_frame(
            frame,
            strategy=splitter.strategy,
            test_size=splitter.test_size,
            val_size=splitter.val_size,
            seed=splitter.seed,
            population_policy=splitter.population_policy,
        )
        plan = SplitPlan(
            assignments=plan.assignments,
            strategy=plan.strategy,
            test_size=plan.test_size,
            val_size=plan.val_size,
            seed=plan.seed,
            population_policy=plan.population_policy,
            eligibility=dict(eligibility),
            clusters=plan.clusters,
        )
        unseen = population.difference(plan.assignments.index)
        if len(unseen):
            raise ValueError(
                f"split.plan_path {plan_path!r} does not cover {len(unseen)} point(s) in the current "
                f"population, e.g. {list(unseen[:5])}. Rebuild the plan or restore the matching data."
            )
        counts = plan.counts()
        self.logger.info(
            f"Loaded a frozen split plan from {plan_path}: "
            f"train={counts['train']} | val={counts['val']} | test={counts['test']}"
        )
        return plan

    def _log_population(
        self,
        all_ids: pd.Index,
        population: pd.Index,
        eligibility: Mapping[str, frozenset],
        policy: str,
    ) -> None:
        """Always report what each family loses. The cost of unification must be visible."""
        self.logger.info(
            f"Split population ({policy}): {len(population)} of {len(all_ids)} point(s) from the "
            f"source frame."
        )
        for family in sorted(eligibility):
            usable = eligibility[family]
            dropped_by_family = len(all_ids) - len(usable)
            lost_to_policy = len([pid for pid in usable if pid not in set(population)])
            self.logger.info(
                f"  {family}: {len(usable)} usable ({dropped_by_family} dropped by its own cleaning); "
                f"{lost_to_policy} further excluded by population_policy={policy}"
            )

    @staticmethod
    def _infer_families(config) -> tuple[str, ...]:
        """The families with at least one enabled registry entry."""
        families: list[str] = []
        sklearn_registry = getattr(config, "MODEL_REGISTRY", {}) or {}
        if any(_enabled(entry) for entry in sklearn_registry.values()):
            families.append(SKLEARN)

        lightning_registry = getattr(config, "LIGHTNING_MODEL_REGISTRY", {}) or {}
        for name, entry in lightning_registry.items():
            if name == "defaults" or not _enabled(entry):
                continue
            kind = str((entry or {}).get("input_kind", SEQUENCE)).lower()
            if kind in (SEQUENCE, GRAPH) and kind not in families:
                families.append(kind)

        # A run with nothing enabled still needs a population to split; sklearn's rule keeps
        # every row, so it is the neutral choice.
        return tuple(families) or (SKLEARN,)


def _enabled(entry: Any) -> bool:
    return bool(isinstance(entry, Mapping) and entry.get("enabled", False))
