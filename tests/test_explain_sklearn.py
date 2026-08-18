"""SHAP over the sklearn path.

Two things decide whether the plot means anything, and both are easy to get wrong:

* feature names must come from ``preprocessor.get_feature_names_out()``, because encoding happens
  INSIDE the pipeline and the estimator never sees the raw column names;
* a log-transformed target puts a ``TransformedTargetRegressor`` in the way, and the estimator that
  can actually be explained predicts ``10 * log1p(y)`` - so the values are contributions in that
  space, not in the target's units.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from yg_eo_soilnet.datamodules.scikit.scikit_trainer_utils import PipelineBuilder
from yg_eo_soilnet.explain import build_shap_results


@pytest.fixture
def frame() -> tuple[pd.DataFrame, pd.Series]:
    generator = np.random.default_rng(0)
    features = pd.DataFrame(
        {
            "clay_pct": generator.normal(20, 5, 60),
            "ph": generator.normal(7, 1, 60),
            "texture": generator.choice(["sandy", "loam", "clay"], 60),
        }
    )
    # ph twice the weight of clay, so the ranking is predictable.
    target = pd.Series(
        features["clay_pct"] * 0.1 + features["ph"] * 0.5 + generator.normal(0, 0.05, 60),
        name="organic_matter_pct",
    )
    return features, target


@pytest.fixture
def config() -> SimpleNamespace:
    return SimpleNamespace(RANDOM_SEED=42, EXPLAIN_MAX_SAMPLES=30, EXPLAIN_BACKGROUND_SAMPLES=20)


def _explain(config, model, features, target, *, is_log_target: bool = False):
    pipeline = PipelineBuilder().build(
        model, is_log_target, categorical_cols=["texture"], numeric_cols=["clay_pct", "ph"]
    )
    pipeline.fit(features, target)
    results = build_shap_results(
        config=config,
        backend="sklearn",
        fitted_estimator=pipeline,
        X_train=features,
        X_test=features,
        target="organic_matter_pct",
    )
    assert len(results) == 1
    return results[0]


def test_tree_model_is_explained_with_post_transform_feature_names(config, frame) -> None:
    features, target = frame
    result = _explain(config, RandomForestRegressor(n_estimators=5, random_state=0), features, target)

    # PipelineBuilder gives tree models an OrdinalEncoder, so texture stays a single column.
    assert result.feature_names == ["num__clay_pct", "num__ph", "cat__texture"]
    assert result.values.shape == (30, 3)


def test_linear_model_gets_one_row_per_one_hot_level(config, frame) -> None:
    """Non-tree models are one-hot encoded, so the explained space is wider than the raw frame."""
    features, target = frame
    result = _explain(config, Ridge(), features, target)

    assert result.feature_names[:2] == ["num__clay_pct", "num__ph"]
    assert sum(name.startswith("cat__texture_") for name in result.feature_names) == 3


def test_the_ranking_recovers_the_generating_weights(config, frame) -> None:
    features, target = frame
    result = _explain(config, RandomForestRegressor(n_estimators=20, random_state=0), features, target)

    ranked = [result.feature_names[index] for index in result.ranking()]
    assert ranked[0] == "num__ph"  # generated with twice clay's weight
    assert ranked[1] == "num__clay_pct"


def test_a_log_transformed_target_is_reported_in_its_own_space(config, frame) -> None:
    features, target = frame
    plain = _explain(config, RandomForestRegressor(n_estimators=5, random_state=0), features, target)
    logged = _explain(
        config,
        RandomForestRegressor(n_estimators=5, random_state=0),
        features,
        target,
        is_log_target=True,
    )

    assert plain.output_space == "original_units"
    # Contributions to 10 * log1p(y), NOT to y. Silently mixing the two across models would make
    # the bar heights incomparable.
    assert logged.output_space == "log1p_x10"


def test_max_samples_caps_the_explained_rows(frame) -> None:
    features, target = frame
    result = _explain(
        SimpleNamespace(RANDOM_SEED=42, EXPLAIN_MAX_SAMPLES=7, EXPLAIN_BACKGROUND_SAMPLES=10),
        RandomForestRegressor(n_estimators=5, random_state=0),
        features,
        target,
    )

    assert result.n_samples == 7


def test_the_full_value_table_is_never_capped_by_the_display_limit(config, frame) -> None:
    features, target = frame
    result = _explain(config, RandomForestRegressor(n_estimators=5, random_state=0), features, target)

    assert len(result.to_frame()) == result.n_samples * result.n_features


def test_something_that_is_not_a_pipeline_is_refused_by_name(config) -> None:
    with pytest.raises(TypeError, match="fitted Pipeline"):
        build_shap_results(
            config=config,
            backend="sklearn",
            fitted_estimator=Ridge(),
            X_train=pd.DataFrame({"a": [1.0]}),
            X_test=pd.DataFrame({"a": [1.0]}),
            target="om",
        )


# --- block grouping ---------------------------------------------------------


def test_blocks_split_continuous_from_categorical_for_a_tree_model(config, frame) -> None:
    """Previously every sklearn feature was labelled "features", so shap_block_bar was one bar."""
    features, target = frame
    result = _explain(config, RandomForestRegressor(n_estimators=5, random_state=0), features, target)

    blocks = dict(zip(result.feature_names, result.blocks))

    assert blocks["num__clay_pct"] == "continuous"
    assert blocks["num__ph"] == "continuous"
    # A tree model ordinal-encodes texture, so it is a single categorical column.
    assert blocks["cat__texture"] == "categorical"
    assert set(result.blocks) == {"continuous", "categorical"}


def test_every_one_hot_column_lands_in_the_categorical_block(config, frame) -> None:
    """A linear model one-hot encodes texture into several columns; jointly they are one block.

    This is what makes the block view worth having: per-feature bars are not comparable between a
    tree model (one ordinal column) and a linear one (three one-hot columns), but the block totals
    are.
    """
    features, target = frame
    result = _explain(config, Ridge(), features, target)

    blocks = dict(zip(result.feature_names, result.blocks))
    one_hot = [name for name in result.feature_names if name.startswith("cat__texture_")]

    assert len(one_hot) == 3
    assert all(blocks[name] == "categorical" for name in one_hot)


def test_the_block_rollup_has_one_entry_per_block(config, frame) -> None:
    features, target = frame
    result = _explain(config, RandomForestRegressor(n_estimators=5, random_state=0), features, target)

    rollup = result.block_mean_abs()

    assert set(rollup) == {"continuous", "categorical"}
    assert all(value >= 0 for value in rollup.values())


def test_unprefixed_feature_names_fall_back_rather_than_raising() -> None:
    """A pipeline with no preprocessor yields raw column names with no transformer prefix."""
    from yg_eo_soilnet.explain.sklearn_explainer import _blocks_from_feature_names

    assert _blocks_from_feature_names(["clay_pct", "ph"]) == ["features", "features"]
    assert _blocks_from_feature_names(["num__a", "weird__b"]) == ["continuous", "features"]
