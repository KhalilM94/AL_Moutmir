"""One missingness rule, applied identically by every training family.

A blank covariate used to mean two different things depending on who read it. sklearn median-filled
it and handed the result to the model as if measured; the Lightning builders deleted every affected
row. On one real dataset three columns that were 99.7% blank cost the sequence family 5744 of 5761
points, and nothing said so.

The rule now is one decision per column:

    missing ratio >  max_missing_column_ratio  ->  the run STOPS and names the column
    missing ratio <= max_missing_column_ratio  ->  median-filled and flagged, row kept
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yg_eo_soilnet.data_manager import DataManager
from yg_eo_soilnet.datamodules.frame_cleaning import (
    SparseColumnError,
    assert_columns_are_dense_enough,
    column_missing_ratios,
)
from yg_eo_soilnet.datamodules.lightning.spatiotemporal_graph_builder import SpatiotemporalGraphBuilder
from yg_eo_soilnet.datamodules.scikit.tabular_preprocessor import TabularPreprocessor
from yg_eo_soilnet.datamodules.sequence.sequence_builder import SoilSequenceBuilder


# --- the gate itself ----------------------------------------------------------------------


def _frame(n=100, missing=0):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "point_id": [f"p{i:03d}" for i in range(n)],
            "lat": rng.uniform(30.0, 36.0, n),
            "lon": rng.uniform(-9.0, -2.0, n),
            "dense": rng.normal(size=n),
            "gappy": rng.normal(size=n),
            "target_a": rng.normal(size=n),
        }
    )
    if missing:
        frame.loc[frame.index[:missing], "gappy"] = np.nan
    return frame


def test_missing_ratios_use_the_same_finiteness_rule_as_the_cleaner(logger):
    frame = _frame(n=100, missing=30)
    frame.loc[frame.index[50], "dense"] = np.inf

    ratios = column_missing_ratios(frame, ["dense", "gappy"])

    # Infinity counts as missing, exactly as build_finite_row_mask treats it.
    assert ratios == pytest.approx({"dense": 0.01, "gappy": 0.30})


def test_a_column_over_the_threshold_stops_the_run_and_names_it(logger):
    frame = _frame(n=100, missing=99)

    with pytest.raises(SparseColumnError) as excinfo:
        assert_columns_are_dense_enough(
            frame, ["dense", "gappy"], max_missing_ratio=0.2, label="a source", logger=logger
        )

    message = str(excinfo.value)
    assert "gappy" in message and "99.0%" in message
    assert "dense" not in message
    assert [name for name, _, _ in excinfo.value.offenders] == ["gappy"]


def test_a_column_under_the_threshold_passes(logger):
    frame = _frame(n=100, missing=10)

    assert (
        assert_columns_are_dense_enough(
            frame, ["dense", "gappy"], max_missing_ratio=0.2, label="a source", logger=logger
        )
        == []
    )


def test_an_allowlisted_column_passes_however_sparse(logger):
    frame = _frame(n=100, missing=99)

    assert (
        assert_columns_are_dense_enough(
            frame,
            ["gappy"],
            max_missing_ratio=0.2,
            label="a source",
            logger=logger,
            allow=["gappy"],
        )
        == []
    )


def test_fail_false_warns_and_reports_instead_of_raising(logger, caplog):
    frame = _frame(n=100, missing=99)

    with caplog.at_level("WARNING"):
        offenders = assert_columns_are_dense_enough(
            frame, ["gappy"], max_missing_ratio=0.2, label="a source", logger=logger, fail=False
        )

    assert [name for name, _, _ in offenders] == ["gappy"]
    assert "the run continues" in caplog.text


# --- the same rule on all three entry points ----------------------------------------------


def _configure(toy_config, tmp_path: Path, frame: pd.DataFrame):
    frame.to_csv(tmp_path / "static.csv", index=False)
    toy_config.DATA_FOLDER = str(tmp_path)
    toy_config.DATA_FILE = "static.csv"
    toy_config.STATIC_FEATURES_FILE = "static.csv"
    toy_config.TARGETS_FILE = "static.csv"
    toy_config.TARGET_COLUMNS = ["target_a"]
    toy_config.CATEGORICAL_FEATURES = []
    toy_config.COLUMNS_TO_TRANSFORM = []
    toy_config.MAX_MISSING_COLUMN_RATIO = 0.2
    return toy_config


def _run_family(family: str, config, logger, frame: pd.DataFrame):
    """Drive one family's cleaning entry point over `frame`."""
    manager = DataManager(config, logger)
    if family == "sklearn":
        return TabularPreprocessor(config, logger, manager).preprocess_data(manager.load_dataset().tabular)
    if family == "sequence":
        return SoilSequenceBuilder(config, logger, manager).usable_point_ids()
    return SpatiotemporalGraphBuilder(config, logger, manager).usable_point_ids()


FAMILIES = ["sklearn", "sequence", "graph"]


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_stops_on_the_same_sparse_column(family, toy_config, logger, tmp_path):
    """The test that pins "unified for all families".

    Before this, the same frame produced three different outcomes: sklearn imputed and trained,
    the sequence builder deleted 99% of the rows, and the graph builder did too. One frame, one
    verdict.
    """
    config = _configure(toy_config, tmp_path, _frame(n=100, missing=99))

    with pytest.raises(SparseColumnError, match="gappy"):
        _run_family(family, config, logger, _frame(n=100, missing=99))


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_accepts_a_column_under_the_threshold(family, toy_config, logger, tmp_path):
    config = _configure(toy_config, tmp_path, _frame(n=100, missing=10))

    result = _run_family(family, config, logger, _frame(n=100, missing=10))

    assert len(result["X"] if family == "sklearn" else result) > 0


@pytest.mark.parametrize("family", FAMILIES)
def test_the_allowlist_works_on_every_family(family, toy_config, logger, tmp_path):
    config = _configure(toy_config, tmp_path, _frame(n=100, missing=99))
    config.ALLOW_SPARSE_COLUMNS = ["gappy"]

    result = _run_family(family, config, logger, _frame(n=100, missing=99))

    assert len(result["X"] if family == "sklearn" else result) > 0


# --- symmetry: a tolerable gap costs a value, not a row -----------------------------------


def test_sklearn_and_the_sequence_family_keep_the_same_rows(toy_config, logger, tmp_path):
    """A covariate gap must not cost the soil sample on either side.

    This is the asymmetry that made the two populations disagree in the first place: sklearn kept
    the row and the sequence builder deleted it.
    """
    config = _configure(toy_config, tmp_path, _frame(n=100, missing=10))
    manager = DataManager(config, logger)

    sklearn_ids = set(
        TabularPreprocessor(config, logger, manager).usable_point_ids(manager.load_dataset().tabular)
    )
    sequence_ids = set(SoilSequenceBuilder(config, logger, manager).usable_point_ids())

    assert sklearn_ids == sequence_ids
    assert len(sequence_ids) == 100


def test_the_sequence_bundle_flags_only_the_covariates_that_have_gaps(toy_config, logger, tmp_path):
    """A fully populated column contributes nothing, so it gets no channel.

    Matches SimpleImputer(add_indicator=True) on the sklearn side, whose default
    features='missing-only' emits an indicator for exactly the same set.
    """
    config = _configure(toy_config, tmp_path, _frame(n=100, missing=10))
    bundle = SoilSequenceBuilder(config, logger, DataManager(config, logger)).build()

    assert bundle.static_validity_names == ["gappy"]
    assert bundle.static_validity.shape == (100, 1)
    # False exactly on the rows the source left blank.
    assert bundle.static_validity[:, 0].sum() == 90
    assert not bundle.static_validity[:10, 0].any()


def test_a_complete_dataset_carries_no_validity_channels(toy_config, logger, tmp_path):
    """The shape-neutrality guarantee: nothing missing, nothing added."""
    config = _configure(toy_config, tmp_path, _frame(n=100, missing=0))
    bundle = SoilSequenceBuilder(config, logger, DataManager(config, logger)).build()

    assert bundle.static_validity_names == []
    assert bundle.static_validity.shape[1] == 0


def test_sklearn_emits_an_indicator_for_the_gappy_column_only(toy_config, logger, tmp_path):
    """The sklearn half of the same rule, via SimpleImputer's own missing-only indicator."""
    from sklearn.linear_model import Ridge

    from yg_eo_soilnet.datamodules.scikit.scikit_trainer_utils import PipelineBuilder

    frame = _frame(n=100, missing=10)
    features = frame[["dense", "gappy"]]

    pipeline = PipelineBuilder().build(Ridge(), categorical_cols=[], numeric_cols=["dense", "gappy"])
    transformed = pipeline.named_steps["preprocessor"].fit_transform(features)

    # two covariates + one indicator for `gappy`; `dense` is complete and contributes none.
    assert transformed.shape[1] == 3
