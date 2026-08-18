"""The unified metric contract.

The bug this module exists to prevent: `mean_test_score` used to mean a POSITIVE cross-validated
RMSE on the sklearn path and a NEGATIVE `-test_loss` on the Lightning path, and the parent
leaderboard plotted both on one axis. The tests here pin the two properties that keep that from
coming back - no metric is negative by convention, and both families produce the same number for
the same predictions.
"""

import numpy as np
import pandas as pd
import pytest

from yg_eo_soilnet.metrics import (
    METRIC_DIRECTION,
    METRIC_SPACE,
    METRIC_STEMS,
    cv_rmse_from_search,
    metric_space_for,
    regression_metrics,
)


def test_regression_metrics_match_manual_calculation() -> None:
    observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted = np.array([1.2, 1.9, 3.4, 3.6, 5.3])

    metrics = regression_metrics(observed, predicted)

    residuals = predicted - observed
    assert np.isclose(metrics["rmse_test"], np.sqrt(np.mean(residuals**2)))
    assert np.isclose(metrics["mae_test"], np.mean(np.abs(residuals)))
    assert np.isclose(metrics["bias_test"], np.mean(residuals))
    assert metrics["n_test"] == 5.0


def test_rpiq_uses_the_spread_of_the_observations_not_the_predictions() -> None:
    """rpd/rpiq take (predictions, targets); swapping them measures the wrong spread.

    Predictions are systematically under-dispersed, so the swapped call under-reports the score.
    This pins the argument order at the one place both families now go through.
    """
    observed = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    predicted = np.array([2.0, 2.5, 3.0, 3.5, 4.0])  # deliberately narrow

    metrics = regression_metrics(observed, predicted)

    rmse = np.sqrt(np.mean((predicted - observed) ** 2))
    expected_rpiq = (np.percentile(observed, 75) - np.percentile(observed, 25)) / rmse
    assert np.isclose(metrics["rpiq_test"], expected_rpiq)
    assert np.isclose(metrics["rpd_test"], np.std(observed, ddof=1) / rmse)


def test_no_unified_metric_is_negative_for_a_reasonable_fit() -> None:
    """The whole point of the schema: nothing is negated to express 'higher is better'."""
    observed = np.linspace(0.0, 10.0, 50)
    predicted = observed + np.sin(observed) * 0.2

    metrics = regression_metrics(observed, predicted)

    for name, value in metrics.items():
        if name.startswith("bias"):
            continue  # genuinely signed: it says which way the model is off
        assert value >= 0.0, f"{name} came back negative ({value})"


def test_r2_can_be_negative_because_the_quantity_genuinely_is() -> None:
    observed = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([10.0, -5.0, 12.0, -8.0])

    assert regression_metrics(observed, predicted)["r2_test"] < 0.0


def test_metrics_drop_non_finite_pairs_together() -> None:
    observed = np.array([1.0, np.nan, 3.0, 4.0])
    predicted = np.array([1.1, 2.0, np.inf, 4.2])

    metrics = regression_metrics(observed, predicted)

    # Only positions 0 and 3 have finite values on BOTH sides.
    assert metrics["n_test"] == 2.0
    assert np.isclose(metrics["rmse_test"], np.sqrt(np.mean([0.1**2, 0.2**2])))


def test_metrics_are_empty_rather_than_nan_when_there_is_nothing_to_score() -> None:
    assert regression_metrics([], []) == {}
    assert regression_metrics([1.0], [1.1]) == {}


def test_degenerate_denominators_omit_their_metric_instead_of_emitting_inf() -> None:
    constant = np.array([2.0, 2.0, 2.0, 2.0])
    metrics = regression_metrics(constant, constant + 0.5)
    assert "r2_test" not in metrics  # zero target variance

    perfect = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = regression_metrics(perfect, perfect)
    assert "rpd_test" not in metrics and "rpiq_test" not in metrics  # zero rmse


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="value"):
        regression_metrics([1.0, 2.0], [1.0])


def test_split_and_suffix_shape_the_metric_names() -> None:
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.1, 2.1, 2.9], split="test", suffix="_clay_pct")
    assert "rmse_test_clay_pct" in metrics
    assert "r2_test_clay_pct" in metrics


def test_cv_rmse_flips_the_neg_scorer_sign_exactly_once() -> None:
    """GridSearchCV runs neg_root_mean_squared_error, so cv_results_ holds negative RMSE."""
    cv_results = pd.DataFrame(
        {
            "mean_test_score": [-3.0, -2.0, -5.0],
            "std_test_score": [0.5, 0.4, 0.9],
            "mean_train_score": [-1.5, -1.0, -2.5],
        }
    )

    metrics = cv_rmse_from_search(cv_results, best_index=1)

    assert metrics["rmse_cv_mean"] == 2.0
    assert metrics["rmse_cv_train_mean"] == 1.0
    # NOT flipped: negating every fold score leaves their standard deviation unchanged, so a flip
    # here would report a negative spread.
    assert metrics["rmse_cv_std"] == 0.4


def test_cv_rmse_accepts_the_raw_cv_results_dict() -> None:
    metrics = cv_rmse_from_search({"mean_test_score": np.array([-4.0])}, best_index=0)
    assert metrics == {"rmse_cv_mean": 4.0}


def test_cv_rmse_skips_columns_it_does_not_have() -> None:
    assert cv_rmse_from_search(pd.DataFrame({"params": [{}]}), best_index=0) == {}


def test_every_stem_has_a_declared_direction_and_space() -> None:
    for stem in METRIC_STEMS:
        assert stem in METRIC_DIRECTION
        assert f"{stem}_test" in METRIC_SPACE


def test_metric_space_separates_original_units_from_standardized_log1p() -> None:
    spaces = metric_space_for(["rmse_test", "r2_test", "test_loss", "test_r2"])

    assert spaces["rmse_test"] == "original_units"
    assert spaces["r2_test"] == "original_units"
    # The LightningModule computes these against the transformed target, and they keep their names
    # because early stopping, checkpointing and the HPO objective all reference them.
    assert spaces["test_loss"] == "standardized_log1p"
    assert spaces["test_r2"] == "standardized_log1p"


def test_unknown_metric_names_are_reported_rather_than_dropped() -> None:
    assert metric_space_for(["something_new"]) == {"something_new": "unknown"}
