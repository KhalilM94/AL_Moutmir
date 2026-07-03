from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import yg_eo_soilnet.trainers.sklearn_trainer as trainer_module
from yg_eo_soilnet.trainers import ModelTrainer


class FakePipeline:
    def __init__(self) -> None:
        self.params = {}
        self.fitted = False

    def set_params(self, **kwargs):
        self.params.update(kwargs)
        return self

    def get_params(self, deep: bool = True):
        return {"params": self.params.copy()}

    def fit(self, X, y):
        self.fitted = True
        self.fit_shape = (X.shape, len(y))
        return self

    def predict(self, X):
        return pd.Series([1.0] * len(X), index=X.index)


class FakeGridSearchCV:
    instances = []

    def __init__(self, estimator, param_grid, cv, refit, scoring, n_jobs, return_train_score, verbose):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.refit = refit
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.return_train_score = return_train_score
        self.verbose = verbose
        self.best_index_ = 0
        self.best_params_ = {key: (value[0] if isinstance(value, list) else value) for key, value in param_grid.items()}
        self.cv_results_ = {
            "params": [self.best_params_],
            "mean_train_score": [0.1],
            "mean_test_score": [0.2],
        }
        self.fitted = False
        FakeGridSearchCV.instances.append(self)

    def fit(self, X, y):
        self.fitted = True
        self.fit_shape = (X.shape, len(y))
        return self


class FakeCVSplitter:
    def __init__(self, cv_strategy, n_splits, random_state):
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.random_state = random_state

    def create_splits(self, X, y=None, groups=None):
        return [(list(range(len(X) - 1)), [len(X) - 1])]


class FakeChildLogger:
    def __init__(self):
        self.log_child_run = MagicMock()


def test_validate_model_pipelines_requires_model_and_params() -> None:
    trainer = ModelTrainer(config=SimpleNamespace(), logger=MagicMock())

    with pytest.raises(ValueError, match="must have 'model' and 'params'"):
        trainer._validate_model_pipelines({"bad": {"model": object()}})


def test_should_skip_target_handles_empty_targets() -> None:
    trainer = ModelTrainer(config=SimpleNamespace(), logger=MagicMock())

    assert trainer._should_skip_target(pd.Series(dtype=float), pd.Series([1.0]), "target") is True
    assert trainer._should_skip_target(pd.Series([1.0]), pd.Series(dtype=float), "target") is True


def test_train_orchestrates_ml_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeGridSearchCV.instances.clear()

    config = SimpleNamespace(
        CATEGORICAL_FEATURES=["cat"],
        CLUSTERING_STRATEGY={"enabled": False, "params": {}},
    )
    trainer = ModelTrainer(
        config=config,
        columns_to_transform=["target_a"],
        enable_clustering=False,
        split_strategy="kfold",
        seed=42,
        n_splits=2,
        logger=MagicMock(),
    )

    class LinearRegressionModel:
        pass

    LinearRegressionModel.__module__ = "sklearn.linear_model"
    model = LinearRegressionModel()

    fake_pipeline = FakePipeline()
    fake_child_logger = FakeChildLogger()

    monkeypatch.setattr(trainer_module, "CVSplitter", FakeCVSplitter)
    monkeypatch.setattr(trainer_module, "GridSearchCV", FakeGridSearchCV)
    monkeypatch.setattr(trainer_module, "clone", lambda obj: obj)
    monkeypatch.setattr(trainer_module.PipelineBuilder, "build", lambda self, *args, **kwargs: fake_pipeline)
    monkeypatch.setattr(trainer_module, "ChildRunLogger", lambda: fake_child_logger)

    data = {
        "X_train": pd.DataFrame({"lat": [0, 1, 2], "lon": [10, 11, 12], "cat": ["a", "b", "a"], "num": [1, 2, 3]}),
        "X_test": pd.DataFrame({"lat": [3], "lon": [13], "cat": ["b"], "num": [4]}),
        "y_train": pd.DataFrame({"target_a": [1.0, 2.0, 3.0]}),
        "y_test": pd.DataFrame({"target_a": [4.0]}),
    }
    model_pipelines = {
        "linear": {
            "model": model,
            "params": {"model__alpha": [0.1, 0.2]},
            "modeltype": "ml",
        }
    }

    trainer.train(target="target_a", data=data, model_pipelines=model_pipelines)

    assert FakeGridSearchCV.instances[0].param_grid == {"model__regressor__alpha": [0.1, 0.2]}
    assert fake_pipeline.fitted is True
    assert fake_child_logger.log_child_run.called


def test_train_skips_when_target_has_no_valid_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(CATEGORICAL_FEATURES=[], CLUSTERING_STRATEGY={"enabled": False, "params": {}})
    trainer = ModelTrainer(config=config, logger=MagicMock())

    monkeypatch.setattr(trainer_module, "ChildRunLogger", lambda: FakeChildLogger())
    monkeypatch.setattr(trainer_module, "CVSplitter", FakeCVSplitter)
    monkeypatch.setattr(trainer_module, "GridSearchCV", FakeGridSearchCV)
    monkeypatch.setattr(trainer_module, "clone", lambda obj: obj)
    monkeypatch.setattr(trainer_module.PipelineBuilder, "build", lambda self, *args, **kwargs: FakePipeline())

    data = {
        "X_train": pd.DataFrame({"lat": [0], "lon": [1], "num": [1]}),
        "X_test": pd.DataFrame({"lat": [1], "lon": [2], "num": [2]}),
        "y_train": pd.DataFrame({"target_a": [None]}),
        "y_test": pd.DataFrame({"target_a": [None]}),
    }

    trainer.train(
        target="target_a",
        data=data,
        model_pipelines={"linear": {"model": object(), "params": {}, "modeltype": "ml"}},
    )

    assert trainer.logger.warning.called