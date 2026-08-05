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
    instances = []

    def __init__(self, cv_strategy, n_splits, random_state):
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.random_state = random_state
        FakeCVSplitter.instances.append(self)

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


def test_train_warns_when_valid_rows_are_too_few(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(
        CATEGORICAL_FEATURES=[],
        CLUSTERING_STRATEGY={"enabled": False, "params": {}},
        MIN_FEATURE_COUNT=5,
    )
    trainer = ModelTrainer(config=config, logger=MagicMock())

    monkeypatch.setattr(trainer_module, "ChildRunLogger", lambda: FakeChildLogger())
    monkeypatch.setattr(trainer_module, "CVSplitter", FakeCVSplitter)
    monkeypatch.setattr(trainer_module, "GridSearchCV", FakeGridSearchCV)
    monkeypatch.setattr(trainer_module, "clone", lambda obj: obj)
    monkeypatch.setattr(trainer_module.PipelineBuilder, "build", lambda self, *args, **kwargs: FakePipeline())

    data = {
        "X_train": pd.DataFrame({"lat": [0, 1, 2], "lon": [1, 2, 3], "num": [1, 2, 3]}),
        "X_test": pd.DataFrame({"lat": [3], "lon": [4], "num": [4]}),
        "y_train": pd.DataFrame({"target_a": [1.0, 2.0, 3.0]}),
        "y_test": pd.DataFrame({"target_a": [4.0]}),
    }

    trainer.train(
        target="target_a",
        data=data,
        model_pipelines={"linear": {"model": object(), "params": {}, "modeltype": "ml"}},
    )

    warning_messages = [call.args[0] for call in trainer.logger.warning.call_args_list]
    assert any("only 3 valid training rows" in message for message in warning_messages)


def test_train_uses_per_model_random_seed_for_cv(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeGridSearchCV.instances.clear()
    FakeCVSplitter.instances.clear()

    config = SimpleNamespace(CATEGORICAL_FEATURES=[], CLUSTERING_STRATEGY={"enabled": False, "params": {}})
    trainer = ModelTrainer(
        config=config,
        columns_to_transform=[],
        enable_clustering=False,
        split_strategy="kfold",
        seed=42,
        n_splits=2,
        logger=MagicMock(),
    )

    monkeypatch.setattr(trainer_module, "ChildRunLogger", lambda: FakeChildLogger())
    monkeypatch.setattr(trainer_module, "CVSplitter", FakeCVSplitter)
    monkeypatch.setattr(trainer_module, "GridSearchCV", FakeGridSearchCV)
    monkeypatch.setattr(trainer_module, "clone", lambda obj: obj)
    monkeypatch.setattr(trainer_module.PipelineBuilder, "build", lambda self, *args, **kwargs: FakePipeline())

    class LinearRegressionModel:
        pass

    LinearRegressionModel.__module__ = "sklearn.linear_model"

    data = {
        "X_train": pd.DataFrame({"lat": [0, 1, 2], "lon": [10, 11, 12], "num": [1, 2, 3]}),
        "X_test": pd.DataFrame({"lat": [3], "lon": [13], "num": [4]}),
        "y_train": pd.DataFrame({"target_a": [1.0, 2.0, 3.0]}),
        "y_test": pd.DataFrame({"target_a": [4.0]}),
    }

    trainer.train(
        target="target_a",
        data=data,
        model_pipelines={
            "linear": {
                "model": LinearRegressionModel(),
                "params": {"model__alpha": [0.1]},
                "modeltype": "ml",
                "random_seed": 99,
            }
        },
    )

    assert FakeCVSplitter.instances
    assert FakeCVSplitter.instances[0].random_state == 99


def test_model_trainer_falls_back_to_sklearn_file_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeTrainingLogger:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs

        def get_logger(self):
            return MagicMock()

    monkeypatch.setattr(trainer_module, "TrainingLogger", FakeTrainingLogger)

    config = SimpleNamespace(SKLEARN_FILE_LOGGING_ENABLED=False)
    ModelTrainer(config=config, logger=None)

    assert captured["kwargs"]["enable_file_logging"] is False

# --- failure policy and dtype handling --------------------------------------


def _minimal_trainer(**config_overrides) -> ModelTrainer:
    defaults = dict(
        MIN_FEATURE_COUNT=1, CATEGORICAL_FEATURES=[], EXCLUDE_CATEGORICAL=[],
        FAIL_ON_MODEL_ERROR=False, FAIL_IF_ALL_MODELS_FAIL_FOR_TARGET=True,
        TREE_CATEGORICAL_ENCODING="ordinal", TREE_ONEHOT_MAX_CATEGORIES=None,
    )
    defaults.update(config_overrides)
    return ModelTrainer(
        config=SimpleNamespace(**defaults), columns_to_transform=[], enable_clustering=False,
        split_strategy="kfold", seed=42, logger=MagicMock(),
    )


def _int_data():
    return {
        "X_train": pd.DataFrame({"i": pd.Series([1, 2, 3, 4], dtype="int64"), "f": [1.0, 2.0, 3.0, 4.0]}),
        "y_train": pd.DataFrame({"t": [1.0, 2.0, 3.0, 4.0]}),
        "X_test": pd.DataFrame({"i": pd.Series([5, 6], dtype="int64"), "f": [5.0, 6.0]}),
        "y_test": pd.DataFrame({"t": [5.0, 6.0]}),
    }


def test_x_test_integer_columns_are_cast_to_float(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cast keyed off X_train's already-converted dtypes, so X_test stayed int64."""
    seen = []
    original = trainer_module.TargetNanFilter

    class _Capture(original):
        def transform(self, X, y=None, groups=None):
            seen.append(dict(X.dtypes))
            return original.transform(self, X, y, groups)

    monkeypatch.setattr(trainer_module, "TargetNanFilter", _Capture)
    trainer = _minimal_trainer(FAIL_IF_ALL_MODELS_FAIL_FOR_TARGET=False)
    trainer.train(target="t", data=_int_data(), model_pipelines={})

    assert len(seen) == 2, "expected X_train and X_test to both reach the filter"
    x_train_dtypes, x_test_dtypes = seen
    assert str(x_train_dtypes["i"]) == "float64"
    assert str(x_test_dtypes["i"]) == "float64"   # int64 before the fix


def test_all_models_failing_raises_when_configured() -> None:
    """A target where every model errored used to exit 0 with an empty MLflow run."""
    trainer = _minimal_trainer(FAIL_IF_ALL_MODELS_FAIL_FOR_TARGET=True)
    pipelines = {"boom": {"model": object(), "params": {}, "modeltype": "ml"}}

    with pytest.raises(RuntimeError, match="All 1 model\\(s\\) failed to train for target 't'"):
        trainer.train(target="t", data=_int_data(), model_pipelines=pipelines)


def test_all_models_failing_is_tolerated_when_flag_is_off() -> None:
    trainer = _minimal_trainer(FAIL_IF_ALL_MODELS_FAIL_FOR_TARGET=False)
    pipelines = {"boom": {"model": object(), "params": {}, "modeltype": "ml"}}

    trainer.train(target="t", data=_int_data(), model_pipelines=pipelines)   # must not raise


def test_fail_on_model_error_reraises_immediately() -> None:
    trainer = _minimal_trainer(FAIL_ON_MODEL_ERROR=True)
    pipelines = {"boom": {"model": object(), "params": {}, "modeltype": "ml"}}

    with pytest.raises(Exception):
        trainer.train(target="t", data=_int_data(), model_pipelines=pipelines)
