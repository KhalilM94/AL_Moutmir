"""The interval and distributional metrics, and their contract with yg_eo_soilnet.metrics."""

import numpy as np
import pytest
from scipy import stats

from yg_eo_soilnet.metrics import METRIC_DIRECTION, METRIC_SPACE, ORIGINAL_UNITS, metric_space_for
from yg_eo_soilnet.uncertainty.metrics import UNCERTAINTY_METRIC_STEMS, uncertainty_metrics


def test_picp_is_the_fraction_of_observations_inside_the_interval():
    observed = np.array([0.0, 1.0, 2.0, 30.0])
    predicted = np.zeros(4)
    metrics = uncertainty_metrics(
        observed, predicted, np.ones(4), lower=np.full(4, -5.0), upper=np.full(4, 5.0)
    )
    assert metrics["picp_test"] == pytest.approx(0.75)


def test_mpiw_is_the_mean_interval_width():
    metrics = uncertainty_metrics(
        np.zeros(4), np.zeros(4), np.ones(4),
        lower=np.array([-1.0, -2.0, -3.0, -4.0]),
        upper=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    assert metrics["mpiw_test"] == pytest.approx(5.0)


def test_coverage_error_is_signed_so_over_and_under_confidence_are_distinguishable():
    # 100% coverage at a nominal 95% is over-cautious: positive.
    over = uncertainty_metrics(
        np.zeros(4), np.zeros(4), np.ones(4), lower=np.full(4, -9.0), upper=np.full(4, 9.0),
        alpha=0.05,
    )
    assert over["coverage_error_test"] == pytest.approx(0.05)

    # 0% coverage is over-confident: negative.
    under = uncertainty_metrics(
        np.ones(4), np.zeros(4), np.ones(4), lower=np.full(4, -0.1), upper=np.full(4, 0.1),
        alpha=0.05,
    )
    assert under["coverage_error_test"] == pytest.approx(-0.95)


def test_the_interval_score_charges_for_width_and_for_misses():
    # One point inside a width-2 band: score is the width alone.
    inside = uncertainty_metrics(
        np.zeros(2), np.zeros(2), np.ones(2), lower=np.full(2, -1.0), upper=np.full(2, 1.0),
        alpha=0.1,
    )
    assert inside["interval_score_test"] == pytest.approx(2.0)

    # Same band, observations 1.0 outside it: width 2 plus (2/alpha) * 1.0 = 2 + 20.
    outside = uncertainty_metrics(
        np.full(2, 2.0), np.zeros(2), np.ones(2), lower=np.full(2, -1.0), upper=np.full(2, 1.0),
        alpha=0.1,
    )
    assert outside["interval_score_test"] == pytest.approx(22.0)


def test_the_interval_score_prefers_a_narrow_covering_band_over_a_vacuous_one():
    # The property that makes it the one rankable number: coverage alone would tie these at 100%.
    narrow = uncertainty_metrics(
        np.zeros(10), np.zeros(10), np.ones(10),
        lower=np.full(10, -1.0), upper=np.full(10, 1.0),
    )
    vacuous = uncertainty_metrics(
        np.zeros(10), np.zeros(10), np.ones(10),
        lower=np.full(10, -1000.0), upper=np.full(10, 1000.0),
    )
    assert narrow["picp_test"] == vacuous["picp_test"] == 1.0
    assert narrow["interval_score_test"] < vacuous["interval_score_test"]


def test_nll_matches_the_gaussian_log_density():
    residual, sigma = 1.5, 2.0
    observed = np.array([residual, residual])
    metrics = uncertainty_metrics(observed, np.zeros(2), np.full(2, sigma))
    expected = -float(np.mean(stats.norm.logpdf(observed, loc=0.0, scale=sigma)))
    assert metrics["nll_test"] == pytest.approx(expected)


def test_crps_of_a_perfect_prediction_matches_the_closed_form():
    # residual 0, sigma 1: CRPS = 2*phi(0) - 1/sqrt(pi) = 0.7979 - 0.5642.
    metrics = uncertainty_metrics(np.zeros(2), np.zeros(2), np.ones(2))
    expected = 2.0 * stats.norm.pdf(0.0) - 1.0 / np.sqrt(np.pi)
    assert metrics["crps_test"] == pytest.approx(expected)


def test_crps_scales_with_sigma_when_the_prediction_is_exact():
    tight = uncertainty_metrics(np.zeros(2), np.zeros(2), np.ones(2))["crps_test"]
    loose = uncertainty_metrics(np.zeros(2), np.zeros(2), np.full(2, 2.0))["crps_test"]
    assert loose == pytest.approx(2.0 * tight)


def test_ence_is_near_zero_when_sigma_matches_the_realised_error():
    rng = np.random.default_rng(0)
    sigma = rng.uniform(0.5, 3.0, size=4000)
    observed = rng.normal(scale=sigma)
    metrics = uncertainty_metrics(observed, np.zeros(4000), sigma)
    assert metrics["ence_test"] < 0.15


def test_ence_is_large_when_sigma_is_uniformly_wrong():
    rng = np.random.default_rng(1)
    sigma = rng.uniform(0.5, 3.0, size=4000)
    observed = rng.normal(scale=sigma)
    # Claim a tenth of the true noise everywhere.
    metrics = uncertainty_metrics(observed, np.zeros(4000), sigma * 0.1)
    assert metrics["ence_test"] > 1.0


def test_sigma_error_corr_detects_uncertainty_that_tracks_the_error():
    rng = np.random.default_rng(2)
    sigma = rng.uniform(0.1, 5.0, size=2000)
    observed = rng.normal(scale=sigma)
    metrics = uncertainty_metrics(observed, np.zeros(2000), sigma)
    assert metrics["sigma_error_corr_test"] > 0.4


def test_sigma_error_corr_is_near_zero_when_sigma_ranks_points_at_random():
    # The failure global coverage cannot see: perfectly calibrated on average, no per-point
    # information at all.
    rng = np.random.default_rng(3)
    observed = rng.normal(size=2000)
    sigma = rng.uniform(0.5, 1.5, size=2000)
    metrics = uncertainty_metrics(observed, np.zeros(2000), sigma)
    assert abs(metrics["sigma_error_corr_test"]) < 0.1


def test_a_constant_sigma_omits_the_rank_correlation_rather_than_reporting_zero():
    metrics = uncertainty_metrics(np.arange(100.0), np.zeros(100), np.ones(100))
    assert "sigma_error_corr_test" not in metrics


def test_omitting_the_interval_skips_the_interval_metrics_only():
    metrics = uncertainty_metrics(np.arange(100.0), np.zeros(100), np.ones(100))
    assert "picp_test" not in metrics
    assert "mpiw_test" not in metrics
    assert "nll_test" in metrics
    assert "crps_test" in metrics


def test_a_degenerate_sigma_omits_the_distributional_metrics_only():
    metrics = uncertainty_metrics(
        np.arange(100.0), np.zeros(100), np.zeros(100),
        lower=np.full(100, -200.0), upper=np.full(100, 200.0),
    )
    assert metrics["picp_test"] == 1.0
    assert "nll_test" not in metrics
    assert "crps_test" not in metrics


def test_suffixed_keys_are_produced_for_a_multi_target_run():
    metrics = uncertainty_metrics(
        np.zeros(10), np.zeros(10), np.ones(10),
        lower=np.full(10, -1.0), upper=np.full(10, 1.0),
        suffix="_clay_pct",
    )
    assert "picp_test_clay_pct" in metrics
    assert "picp_test" not in metrics


def test_fewer_than_two_finite_rows_returns_an_empty_dict_not_nans():
    # Same policy as regression_metrics: NaN metrics would poison the leaderboard.
    assert uncertainty_metrics([1.0], [0.0], [1.0]) == {}
    assert uncertainty_metrics([np.nan, np.nan], [0.0, 0.0], [1.0, 1.0]) == {}


def test_rows_non_finite_in_any_column_are_dropped_together():
    observed = np.array([0.0, 0.0, 0.0, 0.0])
    sigma = np.array([1.0, np.nan, 1.0, 1.0])
    metrics = uncertainty_metrics(
        observed, np.zeros(4), sigma,
        lower=np.full(4, -1.0), upper=np.full(4, 1.0),
    )
    assert metrics["picp_test"] == 1.0
    assert metrics["mean_sigma_test"] == pytest.approx(1.0)


def test_a_negative_sigma_is_dropped_rather_than_taken_as_its_absolute_value():
    # A negative sigma means a variance was handed over where a standard deviation was expected.
    # Dropping keeps the mistake visible instead of quietly halving the reported uncertainty.
    metrics = uncertainty_metrics(
        np.zeros(4), np.zeros(4), np.array([1.0, -1.0, 1.0, 1.0]),
    )
    assert metrics["mean_sigma_test"] == pytest.approx(1.0)


def test_a_multi_column_input_is_refused_the_way_regression_metrics_refuses_one():
    with pytest.raises(ValueError, match="scores ONE target at a time"):
        uncertainty_metrics(np.zeros((10, 2)), np.zeros((10, 2)), np.ones((10, 2)))


def test_mismatched_input_lengths_are_refused():
    with pytest.raises(ValueError, match="mismatched lengths"):
        uncertainty_metrics(np.zeros(10), np.zeros(9), np.ones(10))


def test_every_uncertainty_stem_is_registered_in_the_shared_direction_table():
    # The rule the metrics module exists to enforce: a metric name means one thing, and which way is
    # better is recorded in one place rather than inferred from a sign.
    for stem in UNCERTAINTY_METRIC_STEMS:
        assert stem in METRIC_DIRECTION


def test_every_uncertainty_metric_is_declared_to_be_in_original_units():
    for stem in UNCERTAINTY_METRIC_STEMS:
        assert METRIC_SPACE[f"{stem}_test"] == ORIGINAL_UNITS


def test_metric_space_resolves_a_suffixed_uncertainty_metric():
    spaces = metric_space_for(["picp_test_clay_pct", "mpiw_test_clay_pct"])
    assert spaces == {
        "picp_test_clay_pct": ORIGINAL_UNITS,
        "mpiw_test_clay_pct": ORIGINAL_UNITS,
    }


def test_picp_has_no_ranking_direction_because_maximising_it_is_wrong():
    # A band spanning the whole target range covers 100%. `coverage_error` is the rankable form.
    assert METRIC_DIRECTION["picp"] is None
    assert METRIC_DIRECTION["coverage_error"] == "zero"
