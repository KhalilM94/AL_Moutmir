"""TabICL is wired in through the sklearn registry with no factory or trainer special-casing.

These tests read the shipped registry rather than a fixture so the entry and its expectations
cannot drift apart. Everything here is offline: TabICLRegressor's constructor does not touch the
network, the checkpoint is only fetched on the first fit(), which is why the one test that really
fits is opt-in.

    TABICL_INTEGRATION=1 pixi run -e dev pytest tests/test_tabicl_registry.py

If the download dies with "Network error: Request middleware error", the HuggingFace xet CDN is
blocked; prefix HF_HUB_DISABLE_XET=1 to fall back to the plain HTTP transfer.
"""

import os

import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

from sklearn.preprocessing import RobustScaler

from yg_eo_soilnet.datamodules.scikit.scikit_trainer_utils import PipelineBuilder
from yg_eo_soilnet.models import ModelConfigFactory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "configs" / "sklearn" / "model_registry.yml"

RUN_INTEGRATION = bool(os.environ.get("TABICL_INTEGRATION"))
requires_checkpoint = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="downloads a checkpoint from the HuggingFace hub; set TABICL_INTEGRATION=1 to run",
)


def _tabicl_spec() -> dict:
    return yaml.safe_load(REGISTRY_PATH.read_text())["TabICL"]


def test_tabicl_entry_shape() -> None:
    spec = _tabicl_spec()

    assert spec["modeltype"] == "ml"
    assert spec["import_path"] == "tabicl.TabICLRegressor"
    # A full checkpoint per worker: -1 would load one for every fold and grid point at once.
    assert spec["search_n_jobs"] == 1
    # TabICL is designed to need no tuning and every grid point is a full inference pass.
    assert spec["params"] == {}


def test_factory_builds_tabicl_from_shipped_registry() -> None:
    spec = {**_tabicl_spec(), "enabled": True}

    # A seed that is not TabICL's own default (42), so inheritance is distinguishable from it.
    configs = ModelConfigFactory(registry={"TabICL": spec}).build_model_configs(
        num_features=10, default_seed=7
    )

    model = configs["TabICL"]["model"]
    assert type(model).__name__ == "TabICLRegressor"
    assert model.get_params()["n_estimators"] == spec["init_args"]["n_estimators"]
    # The entry does not pin a seed, so it takes the run's.
    assert "random_state" not in spec["init_args"]
    assert model.get_params()["random_state"] == 7
    assert configs["TabICL"]["search_n_jobs"] == 1
    assert configs["TabICL"]["params"] == {}


def test_tabicl_uses_the_non_tree_pipeline_branch() -> None:
    """TabICL is a neural model, so it must keep the RobustScaler the tree branch drops.

    _is_tree_based_model matches on substrings of the class and module name; nothing in
    'tabiclregressor' or 'tabicl._sklearn.regressor' hits a tree marker today, and this pins that.
    """
    from tabicl import TabICLRegressor

    builder = PipelineBuilder()
    model = TabICLRegressor()

    assert builder._is_tree_based_model(model) is False

    pipeline = builder.build(model, numeric_cols=["a", "b"], categorical_cols=[])
    transformers = pipeline.named_steps["preprocessor"].transformers
    numeric_branch = next(branch for name, branch, _ in transformers if name == "num")

    assert any(isinstance(step, RobustScaler) for _, step in numeric_branch.steps)


@requires_checkpoint
def test_tabicl_fits_and_predicts_through_the_pipeline() -> None:
    from sklearn.datasets import make_regression
    from tabicl import TabICLRegressor

    X, y = make_regression(n_samples=120, n_features=8, n_informative=5, noise=0.5, random_state=42)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y = pd.Series(y)

    pipeline = PipelineBuilder().build(
        TabICLRegressor(n_estimators=2, device="cpu", random_state=42),
        numeric_cols=list(X.columns),
        categorical_cols=[],
    )
    predictions = pipeline.fit(X, y).predict(X)

    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()
    # A collapsed constant prediction would still pass the shape check above.
    assert np.std(predictions) > 0
