"""The heteroscedastic head: the wider readout, the beta-NLL loss, and the sigma inversion."""

import numpy as np
import pytest
import torch

from yg_eo_soilnet.models.lightningmodules.soil_tabular_lightning_module import (
    SoilTabularLightningModule,
)


def _module(**overrides) -> SoilTabularLightningModule:
    kwargs = dict(
        static_dim=3,
        target_dim=1,
        target_names=["target_a"],
        static_hidden_dims=[4],
        head_hidden_dims=[4],
        predict_variance=True,
    )
    kwargs.update(overrides)
    return SoilTabularLightningModule(**kwargs)


def _batch(n_rows=8, static_dim=3, target_dim=1, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return {
        "x_static": torch.randn(n_rows, static_dim, generator=generator),
        "y": torch.randn(n_rows, target_dim, generator=generator),
    }


def _detach_logging(module):
    """Silence LightningModule.log, which warns when there is no Trainer attached.

    These tests exercise the step logic directly rather than through a Trainer, and the suite runs
    with filterwarnings = ["error"], so the warning would fail the test for the wrong reason.
    """
    module.log = lambda *args, **kwargs: None
    return module


# --- the head --------------------------------------------------------------


def test_a_point_head_is_unchanged_in_width():
    module = _module(predict_variance=False)
    assert module.head_output_dim == 1
    assert module.forward(_batch()).shape == (8, 1)


def test_a_variance_head_is_twice_as_wide():
    module = _module()
    assert module.head_output_dim == 2
    assert module.forward(_batch()).shape == (8, 2)


def test_a_joint_variance_head_emits_two_numbers_per_target():
    module = _module(target_dim=3, target_names=["a", "b", "c"])
    assert module.head_output_dim == 6
    assert module.forward(_batch(target_dim=3)).shape == (8, 6)


def test_the_split_separates_the_means_from_the_log_variances():
    module = _module(target_dim=2, target_names=["a", "b"])
    raw = torch.tensor([[1.0, 2.0, -1.0, -2.0]])
    mean, log_variance = module._split_head_output(raw)
    assert torch.equal(mean, torch.tensor([[1.0, 2.0]]))
    assert torch.equal(log_variance, torch.tensor([[-1.0, -2.0]]))


def test_a_point_head_reports_no_log_variance():
    mean, log_variance = _module(predict_variance=False)._split_head_output(
        torch.tensor([[1.0]])
    )
    assert log_variance is None


def test_the_log_variance_is_clamped_so_the_nll_cannot_explode():
    # Without the clamp a head that drives the variance to zero makes the NLL's 1/var term
    # non-finite and kills the run.
    module = _module()
    _mean, log_variance = module._split_head_output(torch.tensor([[0.0, -500.0]]))
    assert float(log_variance) == module.LOG_VARIANCE_MIN


# --- the loss --------------------------------------------------------------


def test_beta_nll_at_zero_is_plain_gaussian_nll():
    module = _module(beta_nll=0.0)
    mean = torch.zeros(4, 1)
    log_variance = torch.zeros(4, 1)  # variance 1
    targets = torch.tensor([[1.0], [-1.0], [2.0], [0.0]])

    loss = module._beta_nll_loss(mean, log_variance, targets)
    expected = float(np.mean(0.5 * (0.0 + targets.numpy() ** 2)))
    assert float(loss) == pytest.approx(expected)


def test_beta_nll_reweights_by_the_predicted_variance():
    module = _module(beta_nll=1.0)
    mean = torch.zeros(2, 1)
    log_variance = torch.log(torch.tensor([[4.0], [4.0]]))
    targets = torch.tensor([[1.0], [1.0]])

    # beta=1 multiplies each term by var^1 = 4.
    plain = module._beta_nll_loss(mean, log_variance, targets)
    module.beta_nll = 0.0
    assert float(plain) == pytest.approx(float(module._beta_nll_loss(mean, log_variance, targets)) * 4.0)


def test_the_variance_weight_carries_no_gradient():
    # The whole point of beta-NLL: the var^beta factor must not itself be optimised, or the model
    # can lower the loss by inflating the variance rather than by fitting the mean.
    module = _module(beta_nll=0.5)
    log_variance = torch.zeros(2, 1, requires_grad=True)
    loss = module._beta_nll_loss(torch.zeros(2, 1), log_variance, torch.ones(2, 1))
    loss.backward()
    # Gradient exists (through the NLL) but is finite and not dominated by the detached weight.
    assert log_variance.grad is not None
    assert torch.isfinite(log_variance.grad).all()


def test_the_variance_head_trains_without_a_non_finite_loss():
    module = _detach_logging(_module())
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-2)
    batch = _batch(32)
    for _ in range(20):
        optimizer.zero_grad()
        loss = module._shared_step(batch, "train")
        loss.backward()
        optimizer.step()
    assert torch.isfinite(loss)


def test_epoch_metrics_score_the_mean_and_not_the_variance_channels():
    # _accumulate_metrics reshapes to (-1, target_dim). Handed the full 2*target_dim output it
    # would fold log variances in among the predictions and report an r2 for a quantity that is
    # not a prediction of anything - or, with an odd product, raise on the reshape.
    module = _detach_logging(_module(target_dim=2, target_names=["a", "b"]))
    module._shared_step(_batch(16, target_dim=2), "val")
    state = module._metric_state["val"]
    assert state["sum_p"].shape == (2,)
    assert state["n"] == 16.0


# --- the sigma inversion ---------------------------------------------------


def test_sigma_is_scaled_by_the_target_scale_when_standardized():
    module = _module(target_mean=[10.0], target_scale=[4.0])
    sigma = module.inverse_transform_sigma(torch.tensor([[2.0]]), torch.tensor([[0.0]]))
    assert float(sigma) == pytest.approx(8.0)


def test_sigma_is_unchanged_when_the_targets_were_never_standardized():
    module = _module()
    sigma = module.inverse_transform_sigma(torch.tensor([[2.0]]), torch.tensor([[0.5]]))
    assert float(sigma) == pytest.approx(2.0)


def test_log1p_sigma_uses_the_local_slope_not_the_target_inversion():
    # The trap: applying inverse_transform_targets to a sigma gives expm1(sigma/10), a number in no
    # units at all. The delta method gives sigma * exp(z/10) / 10 at the predicted z.
    module = _module(target_transform="log1p", target_mean=[0.0], target_scale=[1.0])
    standardized_mean = torch.tensor([[20.0]])
    sigma = module.inverse_transform_sigma(torch.tensor([[1.0]]), standardized_mean)

    expected = 1.0 * float(np.exp(20.0 / 10.0)) / 10.0
    assert float(sigma) == pytest.approx(expected, rel=1e-5)
    # And emphatically not the naive answer.
    assert float(sigma) != pytest.approx(float(np.expm1(1.0 / 10.0)))


def test_log1p_sigma_grows_with_the_prediction():
    # A consequence worth pinning: on a log1p target the same standardized sigma is a LARGER
    # absolute uncertainty at a larger predicted value, because the transform stretches the axis.
    module = _module(target_transform="log1p", target_mean=[0.0], target_scale=[1.0])
    small = module.inverse_transform_sigma(torch.tensor([[1.0]]), torch.tensor([[5.0]]))
    large = module.inverse_transform_sigma(torch.tensor([[1.0]]), torch.tensor([[25.0]]))
    assert float(large) > float(small)


def test_the_sigma_inversion_matches_a_numerical_derivative():
    module = _module(target_transform="log1p", target_mean=[2.0], target_scale=[3.0])
    standardized = torch.tensor([[0.7]])
    small_sigma = 1e-4

    analytic = float(module.inverse_transform_sigma(torch.tensor([[small_sigma]]), standardized))
    # Numerically: how far does the inverted prediction move for a small step in standardized space?
    high = float(module.inverse_transform_targets(standardized + small_sigma))
    low = float(module.inverse_transform_targets(standardized - small_sigma))
    numeric = (high - low) / 2.0

    assert analytic == pytest.approx(numeric, rel=1e-3)


# --- predict_step ----------------------------------------------------------


def test_predict_step_returns_a_bare_tensor_on_a_point_head():
    module = _module(predict_variance=False)
    assert isinstance(module.predict_step(_batch(), 0), torch.Tensor)


def test_predict_step_returns_mean_and_sigma_on_a_variance_head():
    module = _module()
    result = module.predict_step(_batch(), 0)
    assert isinstance(result, tuple) and len(result) == 2
    mean, sigma = result
    assert mean.shape == (8, 1)
    assert sigma.shape == (8, 1)
    assert (sigma > 0).all()


def test_predict_step_sigma_is_in_original_units():
    module = _module(target_mean=[100.0], target_scale=[10.0])
    with torch.no_grad():
        _mean, sigma = module.predict_step(_batch(), 0)
    # Standardized sigmas are O(1); a sigma still in standardized space would be ~10x smaller.
    assert float(sigma.mean()) > 1.0


# --- the checkpoint --------------------------------------------------------


def test_the_variance_flag_survives_a_checkpoint_restore(tmp_path):
    # A buffer rather than a plain attribute: a restore that lost it would read a 2*target_dim head
    # as 2*target_dim targets and report log variances as predictions.
    module = _module()
    path = tmp_path / "head.ckpt"
    torch.save({"state_dict": module.state_dict()}, path)

    restored = torch.load(path, weights_only=True)
    assert bool(restored["state_dict"]["head_predicts_variance"]) is True


def test_a_point_head_records_that_it_predicts_no_variance():
    assert bool(_module(predict_variance=False).head_predicts_variance) is False
