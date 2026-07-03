import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from yg_eo_soilnet.trainer_utils import CVSplitter, PipelineBuilder, TargetNanFilter


def test_cv_splitter_kfold_and_groupkfold() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3, 4]})
    groups = pd.Series([1, 1, 2, 2])

    kfold_splits = CVSplitter(cv_strategy="kfold", n_splits=2, random_state=42).create_splits(frame)
    group_splits = CVSplitter(cv_strategy="groupkfold", n_splits=2).create_splits(frame, groups=groups)

    assert len(kfold_splits) == 2
    assert len(group_splits) == 2


def test_cv_splitter_requires_groups_for_groupkfold() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3, 4]})

    with pytest.raises(ValueError, match="Groups must be provided"):
        CVSplitter(cv_strategy="groupkfold", n_splits=2).create_splits(frame)


def test_cv_splitter_rejects_unknown_strategy() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3, 4]})

    with pytest.raises(ValueError, match="Unsupported CV strategy"):
        CVSplitter(cv_strategy="unknown").create_splits(frame)


def test_target_nan_filter_drops_missing_rows() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
    target = pd.Series([1.0, None, 3.0], name="target")
    groups = pd.Series(["a", "b", "c"])

    X_clean, y_clean, groups_clean = TargetNanFilter().transform(frame, target, groups)

    assert list(X_clean.index) == [0, 2]
    assert list(y_clean.index) == [0, 2]
    assert list(groups_clean.index) == [0, 2]


def test_pipeline_builder_uses_expected_encoder_and_log_wrapper() -> None:
    builder = PipelineBuilder()
    tree_pipeline = builder.build(
        RandomForestRegressor(n_estimators=5, random_state=42),
        categorical_cols=["cat"],
        numeric_cols=["num"],
    )
    linear_pipeline = builder.build(
        LinearRegression(),
        is_log_target=True,
        categorical_cols=["cat"],
        numeric_cols=["num"],
    )

    tree_cat_encoder = tree_pipeline.named_steps["preprocessor"].transformers[1][1].named_steps["encoder"]
    linear_cat_encoder = linear_pipeline.named_steps["preprocessor"].transformers[1][1].named_steps["encoder"]

    assert tree_cat_encoder.__class__.__name__ == "OrdinalEncoder"
    assert linear_cat_encoder.__class__.__name__ == "OneHotEncoder"
    assert linear_pipeline.named_steps["model"].__class__.__name__ == "TransformedTargetRegressor"