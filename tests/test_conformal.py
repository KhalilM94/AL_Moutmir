"""Split-conformal calibration: the order statistic, the guarantee, and the degenerate cases."""

import logging
import math

import numpy as np
import pytest

from yg_eo_soilnet.uncertainty.conformal import (
    MIN_RELIABLE_CALIBRATION_ROWS,
    ConformalCalibrator,
    fit_conformal,
)


def _gaussian_calibration(n_rows: int, sigma: float = 1.0, seed: int = 0):
    """`n_rows` points whose residuals really are Normal(0, sigma), with sigma known."""
    rng = np.random.default_rng(seed)
    mean = rng.normal(size=n_rows) * 10.0
    observed = mean + rng.normal(scale=sigma, size=n_rows)
    return observed, mean, np.full(n_rows, sigma)


def test_the_quantile_is_the_finite_sample_corrected_order_statistic():
    # 19 scores, alpha=0.05: ceil(20 * 0.95) = 19, which is rank 19 of 19 -> saturates at the max.
    # 99 scores: ceil(100 * 0.95) = 95, i.e. the 95th of 99 rather than np.quantile's interpolation.
    scores = np.arange(1.0, 100.0)
    observed, mean = scores, np.zeros_like(scores)
    calibrator = fit_conformal(observed, mean, np.ones_like(scores), alpha=0.05)
    rank = math.ceil((99 + 1) * 0.95)
    assert calibrator.q == pytest.approx(float(np.sort(np.abs(scores))[rank - 1]))


def test_the_corrected_quantile_is_at_least_the_plain_one():
    # The correction can only widen the interval. A version that under-covers would be strictly
    # narrower here, and always in the same direction, which is the failure it exists to prevent.
    observed, mean, sigma = _gaussian_calibration(200, seed=3)
    calibrator = fit_conformal(observed, mean, sigma, alpha=0.05)
    scores = np.abs(observed - mean) / sigma
    assert calibrator.q >= float(np.quantile(scores, 0.95))


def test_coverage_on_held_out_data_reaches_the_nominal_level():
    calibration = _gaussian_calibration(2000, seed=1)
    calibrator = fit_conformal(*calibration, alpha=0.05)

    observed, mean, sigma = _gaussian_calibration(4000, seed=2)
    lower, upper = calibrator.intervals(mean, sigma)
    coverage = float(np.mean((observed >= lower) & (observed <= upper)))
    assert coverage == pytest.approx(0.95, abs=0.02)


def test_coverage_holds_even_when_the_model_sigma_is_badly_wrong():
    # The point of conformal: the guarantee needs no assumption that sigma is right. Here sigma is
    # understated tenfold, which would make a naive mu +- 1.96*sigma interval cover ~15%.
    rng = np.random.default_rng(5)
    mean = rng.normal(size=3000) * 10.0
    observed = mean + rng.normal(scale=1.0, size=3000)
    claimed_sigma = np.full(3000, 0.1)

    calibrator = fit_conformal(observed[:1500], mean[:1500], claimed_sigma[:1500], alpha=0.1)
    lower, upper = calibrator.intervals(mean[1500:], claimed_sigma[1500:])
    coverage = float(np.mean((observed[1500:] >= lower) & (observed[1500:] <= upper)))
    assert coverage == pytest.approx(0.90, abs=0.03)


def test_a_heteroscedastic_sigma_produces_intervals_that_vary_per_point():
    rng = np.random.default_rng(7)
    sigma = np.abs(rng.normal(size=1000)) + 0.1
    mean = np.zeros(1000)
    observed = rng.normal(scale=sigma)

    calibrator = fit_conformal(observed, mean, sigma, alpha=0.05)
    lower, upper = calibrator.intervals(mean, sigma)
    widths = upper - lower
    assert calibrator.normalized is True
    assert widths.std() > 0.0


def test_a_degenerate_sigma_falls_back_to_a_constant_width_band():
    # A deterministic ensemble reports sigma == 0 everywhere. Dividing by it would be meaningless,
    # so q becomes an absolute residual quantile and the flag says so.
    rng = np.random.default_rng(11)
    observed = rng.normal(size=500)
    mean = np.zeros(500)

    calibrator = fit_conformal(observed, mean, np.zeros(500), alpha=0.05)
    assert calibrator.normalized is False

    lower, upper = calibrator.intervals(mean, np.zeros(500))
    widths = upper - lower
    assert np.allclose(widths, widths[0])
    assert widths[0] > 0.0


def test_the_constant_width_fallback_still_covers():
    rng = np.random.default_rng(13)
    observed = rng.normal(size=4000)
    mean = np.zeros(4000)

    calibrator = fit_conformal(observed[:2000], mean[:2000], None, alpha=0.05)
    lower, upper = calibrator.intervals(mean[2000:], np.zeros(2000))
    coverage = float(np.mean((observed[2000:] >= lower) & (observed[2000:] <= upper)))
    assert coverage == pytest.approx(0.95, abs=0.02)


def test_omitting_sigma_entirely_is_the_same_as_passing_zeros():
    observed = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
    mean = np.zeros(5)
    assert fit_conformal(observed, mean, None).q == fit_conformal(observed, mean, np.zeros(5)).q


def test_non_finite_calibration_rows_are_dropped_not_propagated():
    observed = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    mean = np.zeros(5)
    calibrator = fit_conformal(observed, mean, np.ones(5), alpha=0.2)
    assert calibrator.n_calib == 4
    assert np.isfinite(calibrator.q)


def test_a_small_calibration_set_warns_rather_than_failing_silently(caplog):
    observed, mean, sigma = _gaussian_calibration(MIN_RELIABLE_CALIBRATION_ROWS - 10, seed=17)
    logger = logging.getLogger("conformal-test")
    with caplog.at_level(logging.WARNING, logger="conformal-test"):
        calibrator = fit_conformal(observed, mean, sigma, alpha=0.05, logger=logger)
    assert calibrator.n_calib == MIN_RELIABLE_CALIBRATION_ROWS - 10
    assert "coverage guarantee" in caplog.text


def test_too_few_calibration_rows_is_an_error_not_a_silent_wrong_answer():
    with pytest.raises(ValueError, match="at least 2 finite calibration rows"):
        fit_conformal(np.array([1.0]), np.array([0.0]), np.array([1.0]))


def test_mismatched_calibration_lengths_are_refused():
    with pytest.raises(ValueError, match="pairs them row for row"):
        fit_conformal(np.zeros(5), np.zeros(4), np.ones(5))


def test_an_alpha_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError, match="alpha must be in"):
        fit_conformal(np.zeros(10), np.zeros(10), np.ones(10), alpha=1.0)


def test_the_calibrator_describes_itself_in_loggable_scalars():
    payload = ConformalCalibrator(q=1.5, alpha=0.05, n_calib=100).to_dict()
    assert payload == {
        "conformal_q": 1.5,
        "conformal_alpha": 0.05,
        "conformal_n_calib": 100,
        "conformal_normalized": True,
    }
    assert all(isinstance(value, (int, float, bool)) for value in payload.values())


def test_a_lower_alpha_gives_a_wider_interval():
    observed, mean, sigma = _gaussian_calibration(1000, seed=19)
    tight = fit_conformal(observed, mean, sigma, alpha=0.20)
    wide = fit_conformal(observed, mean, sigma, alpha=0.01)
    assert wide.q > tight.q
