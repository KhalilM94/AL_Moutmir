"""The ensemble mechanics: seeds, the bootstrap decision, and the variance decomposition."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from yg_eo_soilnet.uncertainty.ensemble import (
    aggregate,
    bootstrap_indices,
    member_seeds,
    should_bootstrap,
)


def test_member_seeds_are_spaced_by_the_stride_not_by_one():
    assert member_seeds(42, 3, stride=1000) == [42, 1042, 2042]


def test_member_seeds_are_reproducible_for_the_same_base_seed():
    assert member_seeds(7, 5) == member_seeds(7, 5)


def test_member_seeds_rejects_an_empty_ensemble():
    with pytest.raises(ValueError, match="at least 1"):
        member_seeds(42, 0)


def test_auto_bootstraps_every_sklearn_estimator():
    assert should_bootstrap(Ridge()) is True
    assert should_bootstrap(PLSRegression()) is True
    assert should_bootstrap(GradientBoostingRegressor()) is True


def test_exposing_a_random_state_does_not_make_an_estimator_stochastic():
    # The reason `auto` cannot use the cheap "does it expose random_state?" test, pinned here so
    # nobody reintroduces it. Ridge REPORTS a random_state but only its sag/saga solvers consult
    # one; under the default solver the fit is a closed-form solve and the seed is inert. Two
    # differently-seeded Ridges are therefore byte-identical, and a seed-only ensemble of them has
    # a standard deviation of exactly zero.
    first = Ridge().set_params(random_state=1).fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0])
    second = Ridge().set_params(random_state=999).fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0])
    assert "random_state" in Ridge().get_params(deep=False)
    assert np.array_equal(first.coef_, second.coef_)


def test_never_is_the_way_to_opt_out_of_bootstrapping():
    # The Lightning family: a different weight initialization already produces a different model,
    # so resampling would shrink the training set for no additional spread.
    assert should_bootstrap(GradientBoostingRegressor(), mode="never") is False
    assert should_bootstrap(Ridge(), mode="always") is True


def test_an_unknown_bootstrap_mode_is_refused_rather_than_treated_as_auto():
    with pytest.raises(ValueError, match="bootstrap must be one of"):
        should_bootstrap(Ridge(), mode="sometimes")


def test_a_non_estimator_is_bootstrapped_rather_than_inspected():
    assert should_bootstrap(object()) is True


def test_bootstrap_indices_are_reproducible_and_drawn_with_replacement():
    first = bootstrap_indices(100, seed=42)
    assert np.array_equal(first, bootstrap_indices(100, seed=42))
    assert first.shape == (100,)
    assert first.max() < 100
    # With replacement over 100 draws, seeing every row exactly once has probability ~1e-42.
    assert len(np.unique(first)) < 100


def test_bootstrap_indices_differ_between_seeds():
    assert not np.array_equal(bootstrap_indices(100, seed=1), bootstrap_indices(100, seed=2))


def test_aggregate_mean_is_the_mean_of_the_members():
    members = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
    result = aggregate(members)
    assert np.allclose(result.mean.reshape(-1), [3.0, 4.0])


def test_aggregate_epistemic_std_is_the_spread_across_members():
    members = [np.array([1.0]), np.array([3.0])]
    result = aggregate(members)
    assert np.allclose(result.epistemic_std.reshape(-1), [1.0])


def test_aggregate_without_sigmas_reports_no_aleatoric_component():
    result = aggregate([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
    assert np.allclose(result.aleatoric_std, 0.0)
    assert np.allclose(result.total_std, result.epistemic_std)


def test_aggregate_averages_variances_not_standard_deviations():
    # Members claiming sigma 3 and 4 average to sqrt((9+16)/2) = 3.5355, not to 3.5. Averaging the
    # standard deviations understates a mixture whose members disagree about the noise level.
    result = aggregate([np.array([0.0]), np.array([0.0])], [np.array([3.0]), np.array([4.0])])
    assert result.aleatoric_std.reshape(-1)[0] == pytest.approx(np.sqrt(12.5))


def test_total_std_adds_variances_under_the_root():
    # epistemic 3, aleatoric 4 -> total 5, not 7.
    result = aggregate([np.array([-3.0]), np.array([3.0])], [np.array([4.0]), np.array([4.0])])
    assert result.epistemic_std.reshape(-1)[0] == pytest.approx(3.0)
    assert result.aleatoric_std.reshape(-1)[0] == pytest.approx(4.0)
    assert result.total_std.reshape(-1)[0] == pytest.approx(5.0)


def test_the_two_components_sum_to_the_total_variance():
    rng = np.random.default_rng(0)
    members = [rng.normal(size=(20, 3)) for _ in range(5)]
    sigmas = [np.abs(rng.normal(size=(20, 3))) for _ in range(5)]
    result = aggregate(members, sigmas)
    assert np.allclose(
        result.total_std**2, result.epistemic_std**2 + result.aleatoric_std**2
    )


def test_aggregate_widens_a_single_target_member_to_two_dimensions():
    result = aggregate([np.array([1.0, 2.0, 3.0])])
    assert result.mean.shape == (3, 1)


def test_aggregate_preserves_the_target_axis_of_a_multi_target_member():
    result = aggregate([np.zeros((4, 3)), np.ones((4, 3))])
    assert result.mean.shape == (4, 3)


def test_aggregate_refuses_a_sigma_list_that_does_not_match_the_members():
    with pytest.raises(ValueError, match="correspond one to one"):
        aggregate([np.array([1.0]), np.array([2.0])], [np.array([1.0])])


def test_aggregate_refuses_an_empty_ensemble():
    with pytest.raises(ValueError, match="at least one member"):
        aggregate([])
