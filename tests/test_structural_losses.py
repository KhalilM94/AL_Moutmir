"""The structure-aware multi-target losses: mahalanobis, correlation_penalty and cosine.

These read ACROSS targets, so what is worth pinning is not that they produce a number but that the
number responds to the thing they exist for - the plausibility of a predicted COMBINATION - and that
they degrade safely when the batch, the covariance or the target count cannot support them.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from yg_eo_soilnet.datamodules.sequence.sequence_bundle import SoilSequenceBundle
from yg_eo_soilnet.datamodules.sequence.sequence_datamodule import SoilSequenceDataModule
from yg_eo_soilnet.models.config_fatories.lightning_config_factory import LightningConfigFactory
from yg_eo_soilnet.models.lightningmodules.losses import (
    CorrelationPenaltyLoss,
    CosineStructureLoss,
    MahalanobisLoss,
    build_loss_fn,
)
from yg_eo_soilnet.models.lightningmodules.soil_tabular_lightning_module import (
    SoilTabularLightningModule,
)

# Two targets that move together and a third that is nearly independent - the shape the soil
# targets actually have, and enough structure for a "defiant" error direction to exist.
CORRELATED = np.array(
    [
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ]
)


def _pair(n_rows=64, target_dim=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    predictions = torch.randn(n_rows, target_dim, generator=generator)
    targets = torch.randn(n_rows, target_dim, generator=generator)
    return predictions, targets


def _module(**overrides) -> SoilTabularLightningModule:
    kwargs = dict(
        static_dim=3,
        target_dim=3,
        target_names=["target_a", "target_b", "target_c"],
        static_hidden_dims=[4],
        head_hidden_dims=[4],
        target_covariance=CORRELATED.tolist(),
    )
    kwargs.update(overrides)
    return SoilTabularLightningModule(**kwargs)


def _detach_logging(module):
    """Silence LightningModule.log, which warns when there is no Trainer attached.

    These tests exercise the step logic directly rather than through a Trainer, and the suite runs
    with filterwarnings = ["error"], so the warning would fail the test for the wrong reason.
    """
    module.log = lambda *args, **kwargs: None
    return module


# --- mahalanobis -----------------------------------------------------------


def test_mahalanobis_with_uncorrelated_targets_is_exactly_mse() -> None:
    """The anchor property. Without the /D normalization this is off by a factor of target_dim,
    and every threshold tuned against val_loss - the plateau factor, early stopping - silently
    changes meaning when the loss is switched."""
    predictions, targets = _pair()
    loss = MahalanobisLoss(np.eye(3), target_dim=3, shrinkage=0.0)
    assert torch.allclose(loss(predictions, targets), nn.MSELoss()(predictions, targets))


def test_mahalanobis_charges_more_for_an_error_that_defies_the_correlation() -> None:
    """The reason the loss exists: two error vectors of identical L2 norm, one moving the
    correlated pair together and one pulling them apart."""
    _, targets = _pair()
    loss = MahalanobisLoss(CORRELATED, target_dim=3, shrinkage=0.05)

    aligned = targets + torch.tensor([0.3, 0.3, 0.0])
    defiant = targets + torch.tensor([0.3, -0.3, 0.0])
    assert torch.allclose(
        (aligned - targets).norm(), (defiant - targets).norm()
    ), "the two errors must have the same magnitude for the comparison to mean anything"

    assert float(loss(defiant, targets)) > 5.0 * float(loss(aligned, targets))


def test_mahalanobis_survives_a_singular_covariance(caplog) -> None:
    """Two perfectly collinear targets make Sigma singular. Eigenvalue clipping is what keeps this
    from raising or - worse - returning a negative loss that only surfaces as a non-finite failure
    several steps later.

    Reported through the logger rather than ``warnings``: the suite runs with
    filterwarnings = ["error"], so a real warning here would fail whichever test happened to build
    an ill-conditioned covariance rather than reporting a data property to whoever reads the run.
    """
    predictions, targets = _pair()
    singular = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with caplog.at_level(logging.WARNING, logger="yg_eo_soilnet.models.lightningmodules.losses"):
        loss = MahalanobisLoss(singular, target_dim=3, shrinkage=0.0)
    assert "rank-deficient" in caplog.text
    value = loss(predictions, targets)
    assert torch.isfinite(value) and float(value) >= 0.0


def test_mahalanobis_shrinkage_reduces_the_condition_number() -> None:
    ill_conditioned = np.array([[1.0, 0.999, 0.0], [0.999, 1.0, 0.0], [0.0, 0.0, 1.0]])
    sharp = MahalanobisLoss(ill_conditioned, target_dim=3, shrinkage=0.0)
    shrunk = MahalanobisLoss(ill_conditioned, target_dim=3, shrinkage=0.2)
    assert shrunk.condition_number < sharp.condition_number


def test_mahalanobis_rejects_a_covariance_of_the_wrong_shape() -> None:
    with pytest.raises(ValueError, match="3x3 matrix"):
        MahalanobisLoss(np.eye(2), target_dim=3)


# --- correlation penalty ---------------------------------------------------


def test_correlation_penalty_vanishes_when_predictions_match_the_reference() -> None:
    """Predictions drawn to have exactly the reference correlation cost nothing beyond the base
    term, so any positive penalty elsewhere is real signal rather than a constant offset."""
    generator = torch.Generator().manual_seed(3)
    factor = torch.linalg.cholesky(torch.as_tensor(CORRELATED, dtype=torch.float32))
    predictions = torch.randn(4096, 3, generator=generator) @ factor.T
    targets = torch.zeros_like(predictions)

    loss = CorrelationPenaltyLoss(nn.MSELoss(), CORRELATED, target_dim=3, weight=1.0)
    loss(predictions, targets)
    assert loss.last_components["penalty"] < 1e-3


def test_correlation_penalty_charges_for_an_inverted_pair() -> None:
    """Flipping the sign of one of a correlated pair leaves every per-target error distribution
    intact and only changes the joint structure - which is precisely what MSE cannot see."""
    generator = torch.Generator().manual_seed(3)
    factor = torch.linalg.cholesky(torch.as_tensor(CORRELATED, dtype=torch.float32))
    faithful = torch.randn(512, 3, generator=generator) @ factor.T
    inverted = faithful * torch.tensor([1.0, -1.0, 1.0])
    targets = torch.zeros_like(faithful)

    loss = CorrelationPenaltyLoss(nn.MSELoss(), CORRELATED, target_dim=3, weight=1.0)
    loss(faithful, targets)
    faithful_penalty = loss.last_components["penalty"]
    loss(inverted, targets)
    assert loss.last_components["penalty"] > 100.0 * faithful_penalty


def test_correlation_penalty_is_skipped_below_the_minimum_batch() -> None:
    """A trailing validation batch of four rows cannot estimate a correlation coefficient. It must
    fall back to the base loss rather than contribute noise."""
    predictions, targets = _pair(n_rows=4)
    loss = CorrelationPenaltyLoss(nn.MSELoss(), CORRELATED, target_dim=3, weight=1.0, min_batch=16)
    assert torch.allclose(loss(predictions, targets), nn.MSELoss()(predictions, targets))
    assert loss.last_components["penalty"] == 0.0


def test_correlation_penalty_tolerates_a_collapsed_prediction_column() -> None:
    """Early in training a column can be constant, making its correlation undefined. The std floor
    has to hold, because this is a normal state to pass through rather than a failure."""
    predictions, targets = _pair()
    predictions[:, 1] = 0.0
    loss = CorrelationPenaltyLoss(nn.MSELoss(), CORRELATED, target_dim=3, weight=1.0)
    assert torch.isfinite(loss(predictions, targets))


def test_correlation_penalty_reports_both_halves() -> None:
    """The two components are logged as separate metrics; without them a lambda that does nothing
    and a lambda that has swamped the accuracy term look identical from val_loss alone."""
    predictions, targets = _pair()
    loss = CorrelationPenaltyLoss(nn.MSELoss(), CORRELATED, target_dim=3, weight=1.0)
    total = loss(predictions, targets)
    components = loss.last_components
    assert components["penalty"] > 0.0
    assert float(total) == pytest.approx(components["base"] + components["penalty"], rel=1e-5)


# --- cosine ----------------------------------------------------------------


def test_cosine_penalty_vanishes_for_a_proportional_prediction() -> None:
    """The whole claim of the loss: a prediction whose magnitude is wrong but whose RATIOS are
    right is not penalized by the structural term."""
    targets = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, 4.0]])
    loss = CosineStructureLoss(nn.MSELoss(), space="standardized", weight=1.0)
    loss(targets * 1.7, targets)
    assert loss.last_components["penalty"] == pytest.approx(0.0, abs=1e-6)


def test_cosine_penalty_charges_for_a_distorted_ratio() -> None:
    targets = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, 4.0]])
    distorted = targets * torch.tensor([3.0, 1.0, 0.3])
    loss = CosineStructureLoss(nn.MSELoss(), space="standardized", weight=1.0)
    loss(distorted, targets)
    assert loss.last_components["penalty"] > 0.01


def test_cosine_spaces_measure_different_angles() -> None:
    """`original` inverts log1p and standardization first, so it scores the ratio between physical
    quantities; `standardized` scores the shape of the joint z-score anomaly. Confusing the two is
    the easiest way to misread the loss, so they must not coincide."""
    predictions, targets = _pair()
    shared = dict(weight=1.0)
    original = CosineStructureLoss(
        nn.MSELoss(),
        space="original",
        target_mean=[1.0, 2.0, 3.0],
        target_scale=[1.0, 1.0, 1.0],
        target_transform="log1p",
        **shared,
    )
    standardized = CosineStructureLoss(nn.MSELoss(), space="standardized", **shared)
    original(predictions, targets)
    standardized(predictions, targets)
    assert original.last_components["penalty"] != pytest.approx(
        standardized.last_components["penalty"]
    )


def test_cosine_ignores_rows_sitting_at_the_target_mean() -> None:
    """In standardized space such a row is the zero vector: it has no direction to reproduce, and
    scoring its angle would feed pure noise into the gradient."""
    targets = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    predictions = torch.tensor([[1.0, 2.0, 3.0], [5.0, -4.0, 2.0]])
    loss = CosineStructureLoss(nn.MSELoss(), space="standardized", weight=1.0)
    loss(predictions, targets)
    assert loss.last_components["penalty"] == pytest.approx(0.0, abs=1e-6)


# --- selection and guards --------------------------------------------------


@pytest.mark.parametrize(
    "loss_name,expected",
    [
        ("mse", "MSELoss"),
        ("l2", "MSELoss"),
        ("huber", "HuberLoss"),
        ("smooth_l1", "SmoothL1Loss"),
        ("smoothl1", "SmoothL1Loss"),
        ("mahalanobis", "MahalanobisLoss"),
        ("correlation_penalty", "CorrelationPenaltyLoss"),
        ("cosine", "CosineStructureLoss"),
    ],
)
def test_loss_name_selects_the_criterion(loss_name, expected) -> None:
    built = build_loss_fn(loss_name, target_dim=3, target_covariance=CORRELATED)
    assert type(built).__name__ == expected


def test_loss_name_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unknown loss_name"):
        build_loss_fn("mape", target_dim=3, target_covariance=CORRELATED)


def test_loss_base_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unknown loss_base"):
        build_loss_fn("cosine", target_dim=3, loss_base="mape")


def test_cosine_space_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unknown cosine_space"):
        build_loss_fn("cosine", target_dim=3, cosine_space="polar")


@pytest.mark.parametrize("loss_name", ["mahalanobis", "correlation_penalty", "cosine"])
def test_a_structural_loss_refuses_a_single_target(loss_name) -> None:
    """Under per-target grouping these are vacuous. Failing loudly beats training MSE while the
    config and the MLflow params both say mahalanobis."""
    with pytest.raises(ValueError, match="at least 2"):
        build_loss_fn(loss_name, target_dim=1, target_covariance=np.eye(1))


@pytest.mark.parametrize("loss_name", ["mahalanobis", "correlation_penalty"])
def test_a_covariance_loss_refuses_to_build_without_one(loss_name) -> None:
    with pytest.raises(ValueError, match="covariance"):
        build_loss_fn(loss_name, target_dim=3)


def test_cosine_needs_no_covariance() -> None:
    """It is per-row, which is exactly why it is the option available at small batch sizes."""
    assert build_loss_fn("cosine", target_dim=3) is not None


# --- integration with the LightningModule ----------------------------------


@pytest.mark.parametrize("loss_name", ["mahalanobis", "correlation_penalty", "cosine"])
def test_the_module_builds_and_steps_with_a_structural_loss(loss_name) -> None:
    module = _detach_logging(_module(loss_name=loss_name))
    generator = torch.Generator().manual_seed(1)
    batch = {
        "x_static": torch.randn(32, 3, generator=generator),
        "y": torch.randn(32, 3, generator=generator),
    }
    loss = module._shared_step(batch, "train")
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_the_loss_buffers_follow_the_module_onto_a_device() -> None:
    """The loss is assigned to self.loss_fn, so it is a submodule and its buffers move with .to().
    A plain attribute would leave the whitening matrix on the CPU and fail only on a GPU run."""
    module = _module(loss_name="mahalanobis")
    assert any("loss_fn" in name for name, _ in module.named_modules())
    module.to(torch.float64)
    assert module.loss_fn.whitening.dtype == torch.float64


def test_a_structural_loss_refuses_a_heteroscedastic_head() -> None:
    """_shared_step routes to beta-NLL whenever the head emits a log variance and never consults
    loss_fn, so this combination would report a loss_name it never optimized."""
    with pytest.raises(ValueError, match="heteroscedastic"):
        _module(loss_name="mahalanobis", predict_variance=True)


def test_the_covariance_survives_save_hyperparameters() -> None:
    """It has to reach hyper_parameters as plain builtins: a numpy array there makes the checkpoint
    unloadable under torch.load's weights_only=True default."""
    module = _module(loss_name="mahalanobis")
    stored = module.hparams["target_covariance"]
    assert isinstance(stored, list)
    assert all(isinstance(value, float) for row in stored for value in row)


def test_a_point_loss_reports_no_components() -> None:
    """_shared_step logs whatever last_components holds, so a plain MSE run must not acquire two
    extra metrics that would then be missing from every comparison against an older run."""
    module = _module(loss_name="mse")
    assert getattr(module.loss_fn, "last_components", None) is None


# --- the whole chain -------------------------------------------------------


def _correlated_bundle(n_points: int = 64) -> SoilSequenceBundle:
    generator = np.random.default_rng(7)
    first = generator.standard_normal(n_points)
    second = 0.85 * first + 0.53 * generator.standard_normal(n_points)
    return SoilSequenceBundle(
        point_ids=list(range(n_points)),
        static_features=generator.standard_normal((n_points, 2)).astype(np.float32),
        static_feature_names=["f1", "f2"],
        targets=np.column_stack([first, second]).astype(np.float32),
        target_names=["target_a", "target_b"],
    )


@pytest.mark.parametrize("loss_name", ["mahalanobis", "correlation_penalty", "cosine"])
def test_datamodule_to_factory_to_trainer_end_to_end(loss_name) -> None:
    """The covariance is fitted in the datamodule, offered by the factory and consumed by the
    module. Each link is pinned on its own elsewhere; this is the one test that would notice if the
    three stopped agreeing about the name of the thing being passed."""
    from lightning.pytorch import Trainer

    datamodule = SoilSequenceDataModule(
        _correlated_bundle(), batch_size=16, val_size=0.25, test_size=0.25, seed=0
    )
    datamodule.setup("fit")
    assert datamodule.target_covariance_ is not None

    factory = LightningConfigFactory(registry={}, config=SimpleNamespace(TARGET_COLUMNS=[]))
    spec = {
        "import_path": (
            "yg_eo_soilnet.models.lightningmodules.soil_tabular_lightning_module"
            ".SoilTabularLightningModule"
        ),
        "init_args": {
            "static_hidden_dims": [4],
            "head_hidden_dims": [4],
            "loss_name": loss_name,
            "loss_min_batch": 8,
        },
    }
    module = factory._build_model(spec, datamodule)
    assert type(module.loss_fn).__name__ != "MSELoss"

    trainer = Trainer(
        max_epochs=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, datamodule=datamodule)
    assert torch.isfinite(torch.as_tensor(trainer.callback_metrics["train_loss"]))

    if loss_name != "mahalanobis":
        # The composite losses report their halves as their own metrics, which is what makes a
        # lambda that is doing nothing distinguishable from one that has swamped the accuracy term.
        assert "train_loss_base" in trainer.callback_metrics
        assert "train_loss_penalty" in trainer.callback_metrics
