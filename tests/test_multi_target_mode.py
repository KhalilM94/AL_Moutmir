"""MULTI_TARGET_MODE: how several targets become models, in both families.

Before this switch existed the two families sat at opposite hard-coded extremes. Lightning always
fitted one model with a target_dim-wide head - the sequence and graph builders read the whole of
TARGET_COLUMNS and nothing ever narrowed `y` - while sklearn always fitted one model per target.
Neither could be told to do the other thing.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from yg_eo_soilnet.datamodules.scikit.scikit_trainer_utils import TargetNanFilter
from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger
from yg_eo_soilnet.datamodules.sequence.sequence_bundle import SoilSequenceBundle
from yg_eo_soilnet.datamodules.sequence.sequence_datamodule import SoilSequenceDataModule
from yg_eo_soilnet.metrics import regression_metrics
from yg_eo_soilnet.targets import (
    join_target_names,
    resolve_target_groups,
    select_target_columns,
    split_target_names,
)


def _config(**overrides):
    base = {"TARGET_COLUMNS": ["target_a", "target_b"], "MULTI_TARGET_MODE": "joint"}
    base.update(overrides)
    return SimpleNamespace(**base)


# --- resolving the groups --------------------------------------------------


def test_joint_is_one_group_and_per_target_is_one_group_each() -> None:
    assert resolve_target_groups(_config()) == [["target_a", "target_b"]]
    assert resolve_target_groups(_config(MULTI_TARGET_MODE="per_target")) == [["target_a"], ["target_b"]]


def test_a_single_target_is_one_group_whatever_the_mode() -> None:
    for mode in ("joint", "per_target"):
        config = _config(TARGET_COLUMNS=["only"], MULTI_TARGET_MODE=mode)
        assert resolve_target_groups(config) == [["only"]]


def test_an_entry_overrides_the_global_mode() -> None:
    assert resolve_target_groups(_config(), {"multi_target": "per_target"}) == [["target_a"], ["target_b"]]
    assert resolve_target_groups(
        _config(MULTI_TARGET_MODE="per_target"), {"multi_target": "joint"}
    ) == [["target_a", "target_b"]]


def test_native_declares_capability_and_leaves_the_mode_alone() -> None:
    """`native` answers "can this estimator fit a 2-D y", not "how should targets be grouped"."""
    spec = {"multi_target": "native"}
    assert resolve_target_groups(_config(), spec) == [["target_a", "target_b"]]
    assert resolve_target_groups(_config(MULTI_TARGET_MODE="per_target"), spec) == [
        ["target_a"],
        ["target_b"],
    ]


def test_an_estimator_that_cannot_take_a_2d_y_falls_back_instead_of_failing() -> None:
    logger = SimpleNamespace(warning=lambda message: warnings.append(message))
    warnings: list[str] = []

    groups = resolve_target_groups(
        _config(), {}, require_joint_support=True, logger=logger, entry_name="GradientBoosting"
    )

    assert groups == [["target_a"], ["target_b"]]
    assert len(warnings) == 1 and "GradientBoosting" in warnings[0]


def test_an_unknown_mode_is_refused_rather_than_silently_treated_as_joint() -> None:
    with pytest.raises(ValueError, match="MULTI_TARGET_MODE"):
        resolve_target_groups(_config(MULTI_TARGET_MODE="both"))
    with pytest.raises(ValueError, match="multi_target"):
        resolve_target_groups(_config(), {"multi_target": "sometimes"})


# --- the run label ---------------------------------------------------------


def test_the_group_label_round_trips() -> None:
    assert join_target_names(["a", "b"]) == "a__b"
    assert split_target_names("a__b") == ["a", "b"]
    # A single target is its own label, so single-target runs keep the names they always had.
    assert join_target_names(["a"]) == "a"
    assert split_target_names("a") == ["a"]


def test_a_target_name_containing_the_separator_is_refused() -> None:
    """It would split into pieces naming no column, silently, at read time."""
    with pytest.raises(ValueError, match="__"):
        join_target_names(["clay__pct", "ph_water"])
    # Harmless on its own, where nothing is joined.
    assert join_target_names(["clay__pct"]) == "clay__pct"


# --- narrowing the datamodule ----------------------------------------------


def _bundle() -> SoilSequenceBundle:
    return SoilSequenceBundle(
        point_ids=list(range(8)),
        static_features=np.arange(16, dtype=np.float32).reshape(8, 2),
        static_feature_names=["f1", "f2"],
        targets=np.arange(16, dtype=np.float32).reshape(8, 2),
        target_names=["target_a", "target_b"],
    )


def test_joint_keeps_every_target_and_per_target_narrows_to_one() -> None:
    joint = SoilSequenceDataModule(_bundle(), batch_size=4, val_size=0.25, test_size=0.25, seed=0)
    joint.setup("fit")
    assert joint.target_dim == 2
    assert joint.target_names == ["target_a", "target_b"]
    assert joint._collate_points(np.arange(4))["y"].shape == (4, 2)

    narrowed = SoilSequenceDataModule(
        _bundle(), batch_size=4, val_size=0.25, test_size=0.25, seed=0, active_targets=["target_b"]
    )
    narrowed.setup("fit")
    assert narrowed.target_dim == 1
    assert narrowed.target_names == ["target_b"]
    assert narrowed._collate_points(np.arange(4))["y"].shape == (4, 1)


def test_narrowing_takes_the_named_column_not_the_first() -> None:
    narrowed = SoilSequenceDataModule(
        _bundle(), batch_size=8, val_size=0.0, test_size=0.0, seed=0, active_targets=["target_b"]
    )
    narrowed.setup("fit")

    # Column 1 of the bundle is the odd numbers; column 0 is the even ones.
    raw = np.asarray(narrowed.sequence_bundle.targets).reshape(-1)
    assert (raw % 2 == 1).all()
    # The scaler is fitted per column, so narrowing must move it too.
    assert narrowed.target_mean_.shape == (1,)


def test_narrowing_to_a_column_the_bundle_does_not_carry_is_refused() -> None:
    with pytest.raises(ValueError, match="does not carry"):
        SoilSequenceDataModule(_bundle(), active_targets=["no_such_target"])


def test_select_target_columns_reports_when_it_narrowed_nothing() -> None:
    targets = np.arange(6, dtype=np.float32).reshape(3, 2)
    _values, names, indices = select_target_columns(targets, ["a", "b"], ["a", "b"])
    assert names == ["a", "b"] and indices is None


# --- the leakage guard -----------------------------------------------------


def test_a_sibling_target_may_not_become_an_auxiliary_input() -> None:
    """The trap per-target mode opens.

    `target_names` narrows to this model's outputs, so checking auxiliary columns against it would
    admit the OTHER configured target as an input - which leaks the answer via whatever correlation
    the two share, just less obviously than reading the target itself.
    """
    from yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module import SoilCNNLightningModule

    with pytest.raises(ValueError, match="may not name a column being fitted"):
        SoilCNNLightningModule(
            static_dim=2,
            target_dim=1,
            target_names=["target_a"],
            fitted_target_names=["target_a", "target_b"],
            auxiliary_label_columns=["target_b"],
            auxiliary_available_names=["target_b", "some_other_lab_value"],
            temporal_enabled=False,
        )


def test_a_lab_value_that_is_not_being_fitted_is_still_allowed() -> None:
    from yg_eo_soilnet.models.lightningmodules.soil_cnn_lightning_module import SoilCNNLightningModule

    model = SoilCNNLightningModule(
        static_dim=2,
        target_dim=1,
        target_names=["target_a"],
        fitted_target_names=["target_a", "target_b"],
        auxiliary_label_columns=["some_other_lab_value"],
        auxiliary_available_names=["target_b", "some_other_lab_value"],
        temporal_enabled=False,
    )
    assert model.fitted_target_names == ["target_a", "target_b"]


# --- sklearn's 2-D y -------------------------------------------------------


def test_the_nan_filter_keeps_only_rows_where_every_fitted_target_was_measured() -> None:
    """One design matrix is shared by the whole group, so a row is usable or it is not.

    `pd.notna` on a DataFrame is a 2-D boolean frame, which is not a row selector - this used to
    work only because y was always a Series by the time it arrived.
    """
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
    y = pd.DataFrame({"target_a": [1.0, np.nan, 3.0, 4.0], "target_b": [1.0, 2.0, np.nan, 4.0]})

    X_clean, y_clean, _groups = TargetNanFilter().transform(X, y)

    assert X_clean["feature"].tolist() == [1.0, 4.0]
    assert y_clean.index.tolist() == [0, 3]


def test_the_nan_filter_is_unchanged_for_a_single_target() -> None:
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    y = pd.Series([1.0, np.nan, 3.0], name="target_a")

    X_clean, y_clean, _groups = TargetNanFilter().transform(X, y)

    assert X_clean["feature"].tolist() == [1.0, 3.0]
    assert y_clean.tolist() == [1.0, 3.0]


def test_a_joint_group_may_not_mix_logged_and_unlogged_targets() -> None:
    """TransformedTargetRegressor wraps the whole fit, so the transform is not a per-column choice."""
    from yg_eo_soilnet.trainers.sklearn_trainer import ModelTrainer

    trainer = ModelTrainer.__new__(ModelTrainer)
    trainer.columns_to_transform = ["target_a"]

    with pytest.raises(ValueError, match="disagree about the log transform"):
        trainer._guard_uniform_log_transform(["target_a", "target_b"])

    # Uniform either way is fine, and a single target has nothing to disagree with.
    trainer._guard_uniform_log_transform(["target_a"])
    trainer._guard_uniform_log_transform(["target_b", "target_c"])


def test_the_evaluation_frame_names_one_prediction_column_per_target() -> None:
    """Deliberately the convention the Lightning trainer emits, so one reader fans out either."""
    from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger

    X_test = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    y_test = pd.DataFrame({"target_a": [0.0, 1.0, 2.0], "target_b": [0.0, 10.0, 20.0]})
    predictions = np.column_stack([np.arange(3.0), np.arange(3.0) * 10.0])

    frame = ChildRunLogger._build_sklearn_evaluation_frame(
        predictions, X_test, y_test, ["target_a", "target_b"], "Ridge"
    )

    assert frame["prediction_target_a"].tolist() == [0.0, 1.0, 2.0]
    assert frame["prediction_target_b"].tolist() == [0.0, 10.0, 20.0]
    assert frame["target_names"].iloc[0] == "target_a__target_b"
    # No plain `prediction`: it used to alias target 0, so a reader got the first target's numbers
    # under a name that claims to be the run's.
    assert "prediction" not in frame.columns


def test_a_single_target_fit_still_writes_a_plain_prediction_column() -> None:
    from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger

    frame = ChildRunLogger._build_sklearn_evaluation_frame(
        np.arange(2, dtype=float),
        pd.DataFrame({"feature": [1.0, 2.0]}),
        pd.Series([0.0, 1.0], name="target_a"),
        ["target_a"],
        "Ridge",
    )
    assert frame["prediction"].tolist() == [0.0, 1.0]


# --- metrics ---------------------------------------------------------------


def test_scoring_two_targets_at_once_is_refused_rather_than_pooled() -> None:
    """The reshape(-1) this replaces mixed pH with g/kg into one RMSE, silently."""
    observed = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
    predicted = observed + 0.1

    with pytest.raises(ValueError, match="ONE target at a time"):
        regression_metrics(observed, predicted)

    # A column vector is one target and stays fine.
    assert regression_metrics(observed[:, :1], predicted[:, :1])["rmse_test"] == pytest.approx(0.1)


def test_per_target_keys_are_resolved_to_their_stem_s_space() -> None:
    from yg_eo_soilnet.metrics import ORIGINAL_UNITS, STANDARDIZED_LOG1P, metric_space_for

    spaces = metric_space_for(["rmse_test", "rmse_test_clay_pct", "val_r2", "val_r2_clay_pct", "made_up"])

    assert spaces["rmse_test"] == spaces["rmse_test_clay_pct"] == ORIGINAL_UNITS
    assert spaces["val_r2"] == spaces["val_r2_clay_pct"] == STANDARDIZED_LOG1P
    # Still a visible gap rather than a plausible-looking default.
    assert spaces["made_up"] == "unknown"


class _ToyRegressor:
    """The metric machinery on its own, without an architecture around it."""

    def __init__(self, target_dim, target_names):
        from yg_eo_soilnet.models.lightningmodules._regression_base import SoilRegressionLightningBase

        self.__class__ = type("_Toy", (SoilRegressionLightningBase,), {})
        super(type(self), self).__init__()
        self._init_regression_targets(
            target_dim=target_dim,
            target_names=target_names,
            target_mean=None,
            target_scale=None,
            target_transform=None,
            loss_name="mse",
            huber_delta=1.0,
            learning_rate=1e-3,
            optimizer_name="adamw",
            weight_decay=0.0,
            scheduler_type="none",
            scheduler_factor=0.5,
            scheduler_patience=1,
            scheduler_min_lr=0.0,
            scheduler_monitor="val_loss",
        )


def _epoch_metrics(target_dim, target_names, predictions, targets):
    model = _ToyRegressor(target_dim, target_names)
    logged: dict[str, float] = {}
    model.log = lambda name, value, **kwargs: logged.update({name: float(value)})
    model._accumulate_metrics("val", torch.as_tensor(predictions), torch.as_tensor(targets))
    model._log_epoch_metrics("val")
    return logged


def test_a_single_target_r2_is_unchanged() -> None:
    targets = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    logged = _epoch_metrics(1, ["only"], targets.clone(), targets)

    assert logged["val_r2"] == pytest.approx(1.0)
    # Nothing suffixed for one target: there is nothing to disambiguate.
    assert not any(key.startswith("val_r2_") for key in logged)


def test_a_collapsed_target_is_visible_instead_of_averaged_away() -> None:
    """The reason the pooled r2 was worth replacing.

    target_a is predicted perfectly and target_b is predicted as a constant. Pooling both into one
    flattened r2 reports a healthy-looking number for a run that has half failed.
    """
    targets = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    predictions = torch.tensor([[1.0, 2.5], [2.0, 2.5], [3.0, 2.5], [4.0, 2.5]])

    logged = _epoch_metrics(2, ["target_a", "target_b"], predictions, targets)

    assert logged["val_r2_target_a"] == pytest.approx(1.0)
    assert logged["val_r2_target_b"] < 0.01
    # The unsuffixed value is the MEAN of the two, which is what makes it comparable with a
    # per-target run's single number.
    assert logged["val_r2"] == pytest.approx(
        (logged["val_r2_target_a"] + logged["val_r2_target_b"]) / 2
    )
    # And the collapse shows in the spread ratio too.
    assert logged["val_pred_std_ratio_target_b"] == pytest.approx(0.0)


# --- the shared run tree ---------------------------------------------------
# One rule for both families: the tree follows the TARGET GROUP, not the framework. A group of one
# is a single flat run; a group of several is a model run holding the model, plus one child per
# target holding that target's evaluation. sklearn used to be flat whatever the target count, and
# Lightning grew a third level as soon as a second target appeared.


class _RecordingRuns:
    """Stands in for mlflow.start_run, remembering the run names it was asked to open."""

    def __init__(self):
        self.names: list[str] = []

    def __call__(self, run_name=None, nested=False, **kwargs):
        from contextlib import nullcontext

        self.names.append(run_name)
        return nullcontext(SimpleNamespace(info=SimpleNamespace(run_id="run")))


def _quiet_logger(monkeypatch, logger):
    """Silence everything that would touch a tracking server or the filesystem."""
    import yg_eo_soilnet.logger.mlflow_loggers as module

    for name in ("set_tags", "log_params", "log_metric", "log_artifact", "log_metrics"):
        monkeypatch.setattr(module.mlflow, name, MagicMock(), raising=False)
    monkeypatch.setattr("yg_eo_soilnet.artifacts.mlflow.log_artifact", MagicMock())
    for name in ("_log_plots", "_log_shap_artifacts", "_write_json_artifact", "_log_table_artifact",
                 "_write_split_summary", "_log_split_summary", "_promote_champion", "_log_cv_results",
                 "_log_checkpoint", "_tag_model_logging", "_log_pred_obs_artifact"):
        monkeypatch.setattr(logger, name, MagicMock(), raising=False)
    return logger


def test_a_joint_lightning_run_is_a_model_run_with_one_child_per_target(monkeypatch) -> None:
    import yg_eo_soilnet.logger.mlflow_loggers as module

    logger = _quiet_logger(monkeypatch, ChildRunLogger())
    runs = _RecordingRuns()
    monkeypatch.setattr(module.mlflow, "start_run", runs)
    serialize = MagicMock(return_value=True)
    monkeypatch.setattr(logger, "_log_lightning_serialized_model", serialize)

    logger.log_lightning_child_run(
        config=SimpleNamespace(EXPLAIN_ENABLED=False),
        target="target_a__target_b",
        model_name="soil_cnn",
        evaluation_df=pd.DataFrame(
            {
                "target_a": [1.0, 2.0, 3.0, 4.0],
                "target_b": [10.0, 20.0, 30.0, 40.0],
                "prediction_target_a": [1.1, 1.9, 3.2, 3.8],
                "prediction_target_b": [11.0, 19.0, 32.0, 38.0],
                "target_names": ["target_a__target_b"] * 4,
            }
        ),
        validation_metrics={"val_loss": 0.25},
        test_metrics={"test_loss": 0.5},
        model=SimpleNamespace(),
    )

    # The caller already opened the model run, so only the two children are opened here.
    assert runs.names == ["target_a_soil_cnn", "target_b_soil_cnn"]
    # The multi-output model is logged ONCE, under the group's label. It used to be logged inside
    # the per-target loop, putting the same checkpoint in the registry under one name per target -
    # each claiming to be about a single target while predicting them all.
    serialize.assert_called_once()
    assert serialize.call_args.kwargs["target"] == "target_a__target_b"


def test_a_single_target_lightning_run_opens_no_child_at_all(monkeypatch) -> None:
    import yg_eo_soilnet.logger.mlflow_loggers as module

    logger = _quiet_logger(monkeypatch, ChildRunLogger())
    runs = _RecordingRuns()
    monkeypatch.setattr(module.mlflow, "start_run", runs)
    monkeypatch.setattr(logger, "_log_lightning_serialized_model", MagicMock(return_value=True))

    logger.log_lightning_child_run(
        config=SimpleNamespace(EXPLAIN_ENABLED=False),
        target="target_a",
        model_name="soil_cnn",
        evaluation_df=pd.DataFrame({"target_a": [1.0, 2.0, 3.0], "prediction": [1.1, 1.9, 3.2]}),
        validation_metrics={},
        test_metrics={},
        model=SimpleNamespace(),
    )

    assert runs.names == []


def test_a_joint_sklearn_run_has_the_same_shape_as_a_joint_lightning_one(monkeypatch) -> None:
    import yg_eo_soilnet.logger.mlflow_loggers as module

    logger = _quiet_logger(monkeypatch, ChildRunLogger())
    runs = _RecordingRuns()
    monkeypatch.setattr(module.mlflow, "start_run", runs)
    log_model = MagicMock(return_value=SimpleNamespace(model_uri="models:/x/1", registered_model_version=1))
    monkeypatch.setattr(module.mlflow.sklearn, "log_model", log_model)
    monkeypatch.setattr(module, "infer_signature", MagicMock(return_value=None))
    monkeypatch.setattr(logger, "_evaluate_sklearn_target", MagicMock())

    class JointEstimator:
        def predict(self, X):
            return np.column_stack([np.arange(len(X), dtype=float), np.arange(len(X), dtype=float) * 10.0])

    X_test = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
    y_test = pd.DataFrame({"target_a": [0.0, 1.1, 2.2, 2.8], "target_b": [0.0, 11.0, 22.0, 28.0]})

    logger.log_child_run(
        config=SimpleNamespace(
            ENABLE_CLUSTERING=False, CLUSTERING_STRATEGY={}, MLFLOW_REGISTER_MODELS=False, EXPLAIN_ENABLED=False
        ),
        search=SimpleNamespace(best_params_={"model__alpha": 1.0}, best_index_=0),
        cv_results=pd.DataFrame({"params": [{"model__alpha": 1.0}], "mean_test_score": [-1.0]}),
        best_model=JointEstimator(),
        X_train=X_test,
        y_train=y_test,
        X_test=X_test,
        y_test=y_test,
        target="target_a__target_b",
        targets=["target_a", "target_b"],
        param_names=["model__alpha"],
        model_name="Ridge",
        plot_functions={},
    )

    assert runs.names == ["target_a_Ridge", "target_b_Ridge"]
    log_model.assert_called_once()
    assert log_model.call_args.kwargs["name"] == "target_a__target_b_Ridge"


def test_a_joint_run_still_reports_a_scalar_rmse_test(monkeypatch) -> None:
    """Champion promotion reads it as a scalar and promotes nothing when only suffixed keys exist."""
    logger = ChildRunLogger()
    frame = pd.DataFrame(
        {
            "target_a": [1.0, 2.0, 3.0, 4.0],
            "target_b": [10.0, 20.0, 30.0, 40.0],
            "prediction_target_a": [1.1, 1.9, 3.2, 3.8],
            "prediction_target_b": [11.0, 19.0, 32.0, 38.0],
            "target_names": ["target_a__target_b"] * 4,
        }
    )

    metrics = logger._per_target_metrics(frame, "target_a__target_b", "soil_cnn")

    assert "rmse_test_target_a" in metrics and "rmse_test_target_b" in metrics
    assert metrics["rmse_test"] == pytest.approx(
        (metrics["rmse_test_target_a"] + metrics["rmse_test_target_b"]) / 2
    )
    assert metrics["r2_test"] == pytest.approx(
        (metrics["r2_test_target_a"] + metrics["r2_test_target_b"]) / 2
    )


def test_the_leaderboard_reaches_a_joint_run_s_per_target_rows(monkeypatch) -> None:
    """A joint run keeps its comparable numbers one level deeper than a per-target run.

    Searching only direct children left a joint run off the board entirely: its model run carried
    no target tag and its per-target runs were grandchildren.
    """
    import yg_eo_soilnet.logger.mlflow_loggers as module
    from yg_eo_soilnet.logger.mlflow_loggers import ParentRunLogger

    def run(run_id, target=None, model=None, framework="lightning", **metrics):
        return SimpleNamespace(
            info=SimpleNamespace(run_id=run_id, experiment_id="exp"),
            data=SimpleNamespace(
                tags={"target": target, "model_name": model, "framework": framework},
                metrics=metrics,
            ),
        )

    tree = {
        "parent": [run("model_run", "a__b", "soil_cnn"), run("flat", "c", "Ridge", "sklearn", rmse_test=3.0)],
        "model_run": [run("a_run", "a", "soil_cnn", rmse_test=1.0), run("b_run", "b", "soil_cnn", rmse_test=2.0)],
    }

    client = SimpleNamespace(
        get_run=lambda run_id: run("parent"),
        search_runs=lambda experiment_ids, filter_string: tree.get(
            filter_string.split("'")[1], []
        ),
    )
    monkeypatch.setattr(module.mlflow.tracking, "MlflowClient", lambda: client)

    board = ParentRunLogger()._collect_leaderboard("parent")

    # The two targets of the joint run, plus the flat sklearn run - and NOT the model run, whose
    # metrics are means over its children and whose "a__b" label names no measurable target.
    assert sorted(board["target"].tolist()) == ["a", "b", "c"]
    assert board.set_index("target")["rmse_test"].to_dict() == {"a": 1.0, "b": 2.0, "c": 3.0}
