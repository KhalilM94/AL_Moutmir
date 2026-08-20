"""The train/val/test split is decided once and shared by every training family.

Before this existed, sklearn carved 70/30 off the tabular frame while the Lightning datamodules
carved 64/16/20 off a bundle rebuilt from the same CSVs. The holdouts overlapped, so a Lightning
*test* point was usually an sklearn *training* point - and both families publish the result as
`rmse_test` on one leaderboard axis. `test_the_two_families_hold_out_the_same_points` is the test
that would have caught that.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yg_eo_soilnet.data_manager import DataManager
from yg_eo_soilnet.datamodules.scikit.scikit_datamodule import ScikitDataModule
from yg_eo_soilnet.datamodules.split_plan_provider import SplitPlanProvider
from yg_eo_soilnet.datamodules.splitting import SplitPlan, UnifiedSplitter

SPLITTER_MODULE = "yg_eo_soilnet.datamodules.scikit.sklearn_data_splitter"


def _ids(n: int) -> pd.Index:
    return pd.Index([f"p{i:04d}" for i in range(n)])


def _coordinates(ids: pd.Index, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"lat": rng.uniform(30.0, 36.0, len(ids)), "lon": rng.uniform(-9.0, -2.0, len(ids))},
        index=ids,
    )


@pytest.fixture
def split_config(toy_config):
    toy_config.SPLIT_HOLDOUT_STRATEGY = "random"
    toy_config.SPLIT_TEST_SIZE = 0.2
    toy_config.SPLIT_VAL_SIZE = 0.16
    toy_config.SPLIT_SEED = 42
    toy_config.SPLIT_POPULATION_POLICY = "intersect"
    return toy_config


# --- ratios and determinism ---------------------------------------------------------------


def test_sizes_are_fractions_of_the_whole_population_not_of_the_remainder(split_config, logger):
    """Asking for 0.2 must give 20%.

    The old code applied test_size then val_size sequentially, which is why the registry carried
    the note "without this, train was 0.7*0.7=0.49". Requesting a number and silently getting a
    smaller one is the footgun this convention removes.
    """
    plan = UnifiedSplitter(split_config, logger).build_plan(_ids(1000))

    counts = plan.counts()
    assert counts == {"train": 640, "val": 160, "test": 200}


def test_the_same_seed_and_population_give_the_same_assignment(split_config, logger):
    first = UnifiedSplitter(split_config, logger).build_plan(_ids(500))
    second = UnifiedSplitter(split_config, logger).build_plan(_ids(500))

    assert first.assignments.equals(second.assignments)


def test_a_different_seed_moves_the_test_set(split_config, logger):
    first = UnifiedSplitter(split_config, logger).build_plan(_ids(500))
    split_config.SPLIT_SEED = 7
    second = UnifiedSplitter(split_config, logger).build_plan(_ids(500))

    assert set(first.point_ids_for("test")) != set(second.point_ids_for("test"))


def test_the_three_splits_partition_the_population_without_overlap(split_config, logger):
    plan = UnifiedSplitter(split_config, logger).build_plan(_ids(300))

    train, val, test = (set(plan.point_ids_for(name)) for name in ("train", "val", "test"))
    assert train | val | test == set(_ids(300))
    assert not (train & val) and not (train & test) and not (val & test)


def test_a_zero_val_size_is_allowed_rather_than_raising(split_config, logger):
    split_config.SPLIT_VAL_SIZE = 0.0

    plan = UnifiedSplitter(split_config, logger).build_plan(_ids(100))

    assert plan.counts()["val"] == 0
    assert plan.counts()["train"] == 80


def test_sizes_that_leave_no_training_set_are_refused(split_config, logger):
    split_config.SPLIT_TEST_SIZE = 0.6
    split_config.SPLIT_VAL_SIZE = 0.5

    with pytest.raises(ValueError, match="must leave a training set"):
        UnifiedSplitter(split_config, logger)


def test_duplicate_point_ids_are_refused(split_config, logger):
    with pytest.raises(ValueError, match="duplicate point id"):
        UnifiedSplitter(split_config, logger).build_plan(pd.Index(["a", "b", "a"]))


def test_an_unknown_strategy_is_refused(split_config, logger):
    split_config.SPLIT_HOLDOUT_STRATEGY = "kfold"

    with pytest.raises(ValueError, match="split.strategy must be one of"):
        UnifiedSplitter(split_config, logger)


# --- resolving the plan against a family's own ordering -----------------------------------


def test_indices_resolve_against_each_family_own_row_order(split_config, logger):
    """Point id, not row position, is what makes the plan shareable.

    The families hold different subsets in different orders, so a positional index into one
    family's arrays means nothing to the other.
    """
    plan = UnifiedSplitter(split_config, logger).build_plan(_ids(200))

    shuffled_subset = list(_ids(200)[::3][::-1])
    train_idx, val_idx, test_idx = plan.split_indices(shuffled_subset)

    resolved_test = {shuffled_subset[i] for i in test_idx}
    assert resolved_test == set(plan.point_ids_for("test")) & set(shuffled_subset)
    assert train_idx.size + val_idx.size + test_idx.size == len(shuffled_subset)


def test_points_outside_the_population_land_in_no_split(split_config, logger):
    """That is population_policy=intersect doing its job, not a bug."""
    plan = UnifiedSplitter(split_config, logger).build_plan(_ids(100))

    with_strangers = list(_ids(100)) + ["unknown-1", "unknown-2"]
    train_idx, val_idx, test_idx = plan.split_indices(with_strangers)

    assert train_idx.size + val_idx.size + test_idx.size == 100


# --- the spatial-group strategy -----------------------------------------------------------


def test_spatial_group_never_lets_a_cluster_straddle_two_splits(split_config, logger, monkeypatch):
    """Splitting inside a cluster would defeat the point of blocking on it."""
    monkeypatch.setattr("mlflow.log_artifact", lambda *args, **kwargs: None)
    split_config.SPLIT_HOLDOUT_STRATEGY = "spatial_group"
    split_config.SPLIT_GROUP_STRATEGY = {
        "enabled": True,
        "class_path": "yg_eo_soilnet.clustering_utils.KMeansClusterStrategy",
        "params": {"n_clusters": 20},
    }
    ids = _ids(1000)

    plan = UnifiedSplitter(split_config, logger).build_plan(ids, coordinates=_coordinates(ids))

    frame = plan.to_frame()
    assert frame.groupby("cluster")["split"].nunique().max() == 1
    assert plan.counts()["test"] > 0


def test_spatial_group_without_coordinates_is_refused(split_config, logger):
    split_config.SPLIT_HOLDOUT_STRATEGY = "spatial_group"
    split_config.SPLIT_GROUP_STRATEGY = {"class_path": "yg_eo_soilnet.clustering_utils.KMeansClusterStrategy"}

    with pytest.raises(ValueError, match="needs coordinates"):
        UnifiedSplitter(split_config, logger).build_plan(_ids(50))


# --- persistence --------------------------------------------------------------------------


def test_the_plan_round_trips_through_its_frame(split_config, logger):
    """The artifact has to be re-keyable: the old data_splits parquet carried no join key at all."""
    plan = UnifiedSplitter(split_config, logger).build_plan(
        _ids(120), eligibility={"sklearn": _ids(120), "sequence": _ids(120)[:100]}
    )

    restored = SplitPlan.from_frame(plan.to_frame())

    assert restored.assignments.equals(plan.assignments)
    assert restored.eligibility["sequence"] == frozenset(_ids(120)[:100])


def test_describe_reports_the_counts_and_the_per_family_exclusions(split_config, logger):
    plan = UnifiedSplitter(split_config, logger).build_plan(
        _ids(100), eligibility={"sklearn": _ids(100), "sequence": _ids(100)[:90]}
    )

    described = plan.describe()

    assert described["split_n_total"] == 100
    assert described["split_n_test"] == 20
    assert described["split_n_excluded_sequence"] == 10
    assert described["split_n_excluded_sklearn"] == 0


# --- population policy, over a real frame the families disagree about ----------------------


def _write_dataset(tmp_path: Path, n: int = 200, unusable: int = 20) -> Path:
    """A frame whose last `unusable` rows have no measured TARGET.

    The sequence builder drops those rows - there is nothing to learn from a point with no label -
    while the tabular preprocessor keeps them and filters per target later. That is the disagreement
    population_policy exists to reconcile, and since covariate gaps became fill-and-flag it is the
    only one left.
    """
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "point_id": [f"p{i:04d}" for i in range(n)],
            "lat": rng.uniform(30.0, 36.0, n),
            "lon": rng.uniform(-9.0, -2.0, n),
            "feature": rng.normal(size=n),
            "target_a": rng.normal(size=n),
            "target_b": rng.normal(size=n),
        }
    )
    frame.loc[frame.index[-unusable:], "target_a"] = np.nan
    path = tmp_path / "static.csv"
    frame.to_csv(path, index=False)
    return path


def _configure(toy_config, tmp_path: Path, policy: str, **dataset_kwargs):
    _write_dataset(tmp_path, **dataset_kwargs)
    toy_config.DATA_FOLDER = str(tmp_path)
    toy_config.DATA_FILE = "static.csv"
    toy_config.STATIC_FEATURES_FILE = "static.csv"
    toy_config.TARGETS_FILE = "static.csv"
    toy_config.CATEGORICAL_FEATURES = []
    toy_config.SPLIT_HOLDOUT_STRATEGY = "random"
    toy_config.SPLIT_TEST_SIZE = 0.2
    toy_config.SPLIT_VAL_SIZE = 0.16
    toy_config.SPLIT_SEED = 42
    toy_config.SPLIT_POPULATION_POLICY = policy
    return toy_config


def test_intersect_splits_only_the_points_every_family_can_use(toy_config, logger, tmp_path):
    config = _configure(toy_config, tmp_path, "intersect")
    provider = SplitPlanProvider(config, logger, DataManager(config, logger), families=("sklearn", "sequence"))

    plan = provider.plan()

    # 200 rows, 20 of which the sequence family drops.
    assert len(plan.assignments) == 180
    assert plan.describe()["split_n_excluded_sequence"] == 0


def test_assign_all_labels_every_point_including_the_ones_only_sklearn_can_use(
    toy_config, logger, tmp_path
):
    config = _configure(toy_config, tmp_path, "assign_all")
    provider = SplitPlanProvider(config, logger, DataManager(config, logger), families=("sklearn", "sequence"))

    plan = provider.plan()

    assert len(plan.assignments) == 200
    # The delta is still recorded under either policy - the cost of unification stays visible.
    assert plan.describe()["split_n_excluded_sequence"] == 20


# --- the cross-family invariant -----------------------------------------------------------


def _sequence_test_ids(config, logger, plan):
    """The point ids the sequence datamodule would actually hold out."""
    from yg_eo_soilnet.datamodules.sequence.sequence_builder import SoilSequenceBuilder

    bundle = SoilSequenceBuilder(config, logger, DataManager(config, logger)).build()
    _, _, test_idx = plan.split_indices(list(bundle.point_ids))
    return {bundle.point_ids[i] for i in test_idx}


def _sklearn_test_ids(config, logger, plan, monkeypatch):
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: Path(path).write_text("f"))
    monkeypatch.setattr(f"{SPLITTER_MODULE}.mlflow.log_artifacts", lambda *args, **kwargs: None)

    module = ScikitDataModule(config, logger, DataManager(config, logger))
    processed = module.preprocess(module.load_frame())
    split = module.split(processed, plan)
    point_ids = processed["point_ids"].to_numpy()
    return set(point_ids[split["X_test"].index])


def test_the_two_families_hold_out_the_same_points(toy_config, logger, tmp_path, monkeypatch):
    """The regression test for the original bug.

    Under `intersect` the test sets must be identical row for row, so `rmse_test` from XGBoost and
    `rmse_test` from soil_cnn are measured on the same data and belong on the same axis.
    """
    config = _configure(toy_config, tmp_path, "intersect")
    plan = SplitPlanProvider(
        config, logger, DataManager(config, logger), families=("sklearn", "sequence")
    ).plan()

    sklearn_test = _sklearn_test_ids(config, logger, plan, monkeypatch)
    sequence_test = _sequence_test_ids(config, logger, plan)

    assert sklearn_test == sequence_test
    assert len(sklearn_test) == plan.counts()["test"]


def test_under_assign_all_the_sklearn_test_set_is_a_superset(toy_config, logger, tmp_path, monkeypatch):
    """The documented trade-off: shared membership, but not the identical rows."""
    config = _configure(toy_config, tmp_path, "assign_all")
    plan = SplitPlanProvider(
        config, logger, DataManager(config, logger), families=("sklearn", "sequence")
    ).plan()

    sklearn_test = _sklearn_test_ids(config, logger, plan, monkeypatch)
    sequence_test = _sequence_test_ids(config, logger, plan)

    assert sequence_test < sklearn_test


def test_the_sklearn_fit_pool_never_touches_the_shared_test_set(toy_config, logger, tmp_path, monkeypatch):
    """X_train is train ∪ val, and neither may leak a test point into GridSearchCV."""
    config = _configure(toy_config, tmp_path, "intersect")
    plan = SplitPlanProvider(
        config, logger, DataManager(config, logger), families=("sklearn", "sequence")
    ).plan()

    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: Path(path).write_text("f"))
    monkeypatch.setattr(f"{SPLITTER_MODULE}.mlflow.log_artifacts", lambda *args, **kwargs: None)
    module = ScikitDataModule(config, logger, DataManager(config, logger))
    processed = module.preprocess(module.load_frame())
    split = module.split(processed, plan)

    point_ids = processed["point_ids"].to_numpy()
    fit_pool = set(point_ids[split["X_train"].index])
    assert not fit_pool & set(plan.point_ids_for("test"))
    assert fit_pool == set(plan.point_ids_for("train")) | set(plan.point_ids_for("val"))


# --- the collapse guard -------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["intersect", "assign_all"])
def test_a_starved_family_stops_the_run_under_every_policy(toy_config, logger, tmp_path, policy):
    """A family that cannot use the data must not quietly train on the scraps.

    On the real dataset three near-empty covariates reduced the sequence family to 17 of 5761
    points and nothing said so. The check is per family and policy-independent precisely because
    `assign_all` and a single-family run have no intersection to notice it - those were the two
    blind spots that let the original case through.
    """
    config = _configure(toy_config, tmp_path, policy, n=200, unusable=180)
    provider = SplitPlanProvider(config, logger, DataManager(config, logger), families=("sklearn", "sequence"))

    with pytest.raises(ValueError, match="can use less than split.min_population_ratio"):
        provider.plan()


def test_a_starved_family_stops_a_single_family_run(toy_config, logger, tmp_path):
    """No intersection exists here at all, which is exactly why the old guard missed this."""
    config = _configure(toy_config, tmp_path, "intersect", n=200, unusable=180)
    provider = SplitPlanProvider(config, logger, DataManager(config, logger), families=("sequence",))

    with pytest.raises(ValueError, match="sequence: 20 of 200 usable"):
        provider.plan()


def test_a_small_overlap_between_healthy_families_names_the_overlap_not_a_family(toy_config, logger):
    """The other half of the guard: everyone is individually fine, they just disagree.

    `assign_all` is the documented way past this one - unlike a starved family, which no policy
    excuses.
    """
    provider = SplitPlanProvider.__new__(SplitPlanProvider)
    provider.config = toy_config
    provider.logger = logger
    all_ids = _ids(100)
    eligibility = {
        "sequence": frozenset(all_ids[:60]),
        "graph": frozenset(all_ids[40:]),
    }

    with pytest.raises(ValueError, match="it is the OVERLAP between them that is small"):
        provider._guard_against_collapse(all_ids, all_ids[40:60], eligibility, "intersect")


def test_the_guard_can_be_disabled(toy_config, logger, tmp_path):
    config = _configure(toy_config, tmp_path, "intersect", n=200, unusable=180)
    config.SPLIT_MIN_POPULATION_RATIO = 0.0

    plan = SplitPlanProvider(
        config, logger, DataManager(config, logger), families=("sklearn", "sequence")
    ).plan()

    assert len(plan.assignments) == 20


def test_a_mass_drop_names_the_columns_that_caused_it(logger, caplog):
    """"the bundle has 17 points" is not actionable; "these columns are 99% empty" is."""
    from yg_eo_soilnet.datamodules.frame_cleaning import drop_non_finite_rows

    frame = pd.DataFrame(
        {
            "keep": np.arange(100.0),
            "mostly_empty": [1.0] + [np.nan] * 99,
            "fine": np.arange(100.0),
        }
    )

    with caplog.at_level("WARNING"):
        drop_non_finite_rows(
            frame, logger=logger, label="test source", numeric_columns=["keep", "mostly_empty", "fine"]
        )

    assert "Worst columns by rows lost" in caplog.text
    assert "mostly_empty (99)" in caplog.text
