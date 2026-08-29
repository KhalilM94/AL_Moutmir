"""Inference with uncertainty: the sklearn round-trip, the pyfunc contract, and relog's refusal."""

import numpy as np
import pandas as pd
import pytest

import mlflow
from yg_eo_soilnet.uncertainty.conformal import ConformalCalibrator
from yg_eo_soilnet.uncertainty.predictors import EnsembleRegressor


def _fitted_ensemble(n_members: int = 3, target_names=("target_a",)):
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=60), "f2": rng.normal(size=60)})
    y = 2.0 * X["f1"] - X["f2"]

    members = []
    for seed in range(n_members):
        rows = np.random.default_rng(seed).integers(0, len(X), len(X))
        members.append(Ridge().fit(X.iloc[rows], y.iloc[rows]))

    calibrators = {name: ConformalCalibrator(q=1.9, alpha=0.05, n_calib=100) for name in target_names}
    return EnsembleRegressor(members, list(target_names), calibrators=calibrators), X


# --- the sklearn round-trip ------------------------------------------------


def test_the_logged_ensemble_reloads_and_still_predicts_the_mean():
    ensemble, X = _fitted_ensemble()
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            sk_model=ensemble,
            name="target_a_Ridge",
            skops_trusted_types=[
                "numpy.dtype",
                "yg_eo_soilnet.uncertainty.predictors.EnsembleRegressor",
                "yg_eo_soilnet.uncertainty.conformal.ConformalCalibrator",
            ],
        )

    restored = mlflow.sklearn.load_model(info.model_uri)
    assert np.allclose(restored.predict(X), ensemble.predict(X))


def test_the_reloaded_ensemble_still_carries_its_members_and_calibrators():
    # The property that makes batch inference with uncertainty possible at all: what comes back is
    # the EnsembleRegressor, not a bare estimator, so predict_frame is still there.
    ensemble, X = _fitted_ensemble()
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            sk_model=ensemble,
            name="target_a_Ridge",
            skops_trusted_types=[
                "numpy.dtype",
                "yg_eo_soilnet.uncertainty.predictors.EnsembleRegressor",
                "yg_eo_soilnet.uncertainty.conformal.ConformalCalibrator",
            ],
        )

    restored = mlflow.sklearn.load_model(info.model_uri)
    assert len(restored.members) == 3
    assert restored.calibrators["target_a"].q == pytest.approx(1.9)

    frame = restored.predict_frame(X)
    assert list(frame.columns) == [
        "target_a_pred",
        "target_a_std",
        "target_a_epistemic_std",
        "target_a_aleatoric_std",
        "target_a_lower",
        "target_a_upper",
    ]
    assert (frame["target_a_upper"] > frame["target_a_lower"]).all()


def test_the_reloaded_intervals_match_the_ones_computed_before_logging():
    ensemble, X = _fitted_ensemble()
    before = ensemble.predict_frame(X)
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            sk_model=ensemble,
            name="target_a_Ridge",
            skops_trusted_types=[
                "numpy.dtype",
                "yg_eo_soilnet.uncertainty.predictors.EnsembleRegressor",
                "yg_eo_soilnet.uncertainty.conformal.ConformalCalibrator",
            ],
        )
    after = mlflow.sklearn.load_model(info.model_uri).predict_frame(X)
    pd.testing.assert_frame_equal(before, after)


def test_the_mlflow_pyfunc_wrapper_returns_the_mean_and_ignores_params():
    # Pinning a real MLflow constraint rather than our own behaviour: _SklearnModelWrapper.predict
    # is `return self.sklearn_model.predict(data)` and drops params on the floor. Uncertainty from a
    # sklearn ensemble therefore comes from mlflow.sklearn.load_model(...).predict_frame(), NOT from
    # the pyfunc flavor. If a future MLflow forwards params, this test fails and the serving path
    # can be simplified.
    ensemble, X = _fitted_ensemble()
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            sk_model=ensemble,
            name="target_a_Ridge",
            skops_trusted_types=[
                "numpy.dtype",
                "yg_eo_soilnet.uncertainty.predictors.EnsembleRegressor",
                "yg_eo_soilnet.uncertainty.conformal.ConformalCalibrator",
            ],
        )

    served = mlflow.pyfunc.load_model(info.model_uri)
    result = np.asarray(served.predict(X, params={"uncertainty": True}))
    assert result.ndim == 1
    assert np.allclose(result, ensemble.predict(X))


# --- the Lightning pyfunc contract -----------------------------------------


class _FakePredictor:
    """Stands in for SoilSequencePredictor: the pyfunc only uses these three members."""

    def __init__(self, sigma=None):
        self.preprocessing_state = {"target_names": ["target_a", "target_b"]}
        self._sigma = sigma

    def predict_with_uncertainty(self, bundle):
        means = np.array([[1.0, 2.0], [3.0, 4.0]])
        return means, (np.full_like(means, self._sigma) if self._sigma is not None else None)


def _pyfunc(monkeypatch, sigma=None):
    from yg_eo_soilnet.serving import lightning_pyfunc as module

    served = module.SoilSequencePyfunc(model=object())
    served._predictor = _FakePredictor(sigma)
    monkeypatch.setattr(module, "bundle_from_frame", lambda frame, state: object())
    return served


def test_the_pyfunc_returns_only_the_targets_by_default(monkeypatch):
    served = _pyfunc(monkeypatch, sigma=0.5)
    output = served.predict(None, pd.DataFrame({"a": [1, 2]}))
    assert list(output.columns) == ["target_a", "target_b"]


def test_the_pyfunc_widens_its_output_when_uncertainty_is_requested(monkeypatch):
    served = _pyfunc(monkeypatch, sigma=0.5)
    output = served.predict(None, pd.DataFrame({"a": [1, 2]}), params={"uncertainty": True})
    assert list(output.columns) == ["target_a", "target_b", "target_a_std", "target_b_std"]
    assert (output["target_a_std"] == 0.5).all()


def test_asking_a_point_head_for_uncertainty_returns_the_ordinary_columns(monkeypatch):
    # A checkpoint from before the variance head existed has no sigma to give. That is not an error
    # at inference time, and raising would break every caller that passes the flag unconditionally.
    served = _pyfunc(monkeypatch, sigma=None)
    output = served.predict(None, pd.DataFrame({"a": [1, 2]}), params={"uncertainty": True})
    assert list(output.columns) == ["target_a", "target_b"]


# --- the predictor ---------------------------------------------------------


def test_the_predictor_unpacks_a_variance_heads_tuple():
    # The regression this guards: predict_step returns a tuple on a variance head, and calling
    # .detach() straight on it - which the predictor used to do - raises AttributeError and makes
    # the checkpoint unservable.
    import torch

    from yg_eo_soilnet.serving.sequence_predictor import SoilSequencePredictor

    class FakeModel:
        training = False
        target_dim = 1
        predict_variance = True

        def eval(self):
            return self

        def train(self):
            return self

        def predict_step(self, batch, index):
            rows = len(batch["indices"])
            return torch.ones(rows, 1), torch.full((rows, 1), 0.25)

    class FakeDataModule:
        def collate(self, indices):
            return {"indices": indices}

        def apply_preprocessing_state(self, state):
            return None

    predictor = SoilSequencePredictor.__new__(SoilSequencePredictor)
    predictor.model = FakeModel()
    predictor.preprocessing_state = {"target_names": ["target_a"]}
    predictor._datamodule = lambda bundle, **kwargs: FakeDataModule()

    bundle = type("B", (), {"num_points": 4, "point_ids": [0, 1, 2, 3]})()
    import yg_eo_soilnet.serving.sequence_predictor as predictor_module

    original = predictor_module.SoilSequenceBundle.from_mapping
    predictor_module.SoilSequenceBundle.from_mapping = staticmethod(lambda value: bundle)
    try:
        means, sigma = predictor.predict_with_uncertainty(bundle)
    finally:
        predictor_module.SoilSequenceBundle.from_mapping = original

    assert means.shape == (4, 1)
    assert sigma is not None and np.allclose(sigma, 0.25)


# --- relog -----------------------------------------------------------------


def test_relog_refuses_a_single_checkpoint_from_an_ensemble_run(monkeypatch, tmp_path):
    """One member is not the ensemble, and the run's metrics describe the ensemble."""
    import relog

    checkpoint = tmp_path / "best.ckpt"
    checkpoint.write_bytes(b"")

    class FakeRun:
        info = type("I", (), {"experiment_id": "0"})()
        data = type(
            "D",
            (),
            {
                "tags": {"model_name": "soil_tabular", "target": "target_a"},
                "params": {"uncertainty_n_members": "5"},
                "metrics": {},
            },
        )()

    monkeypatch.setattr(relog.mlflow, "MlflowClient", lambda: type("C", (), {"get_run": lambda self, _id: FakeRun()})())
    monkeypatch.setattr(relog.mlflow, "set_experiment", lambda **kwargs: None)
    monkeypatch.setattr(relog, "load_config", lambda path: object(), raising=False)

    with pytest.raises(SystemExit, match="trained an ensemble of 5 members"):
        relog.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--run-id",
                "abc",
                "--model-class",
                "yg_eo_soilnet.models.lightningmodules.soil_tabular_lightning_module.SoilTabularLightningModule",
            ]
        )
