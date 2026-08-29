"""How sigma becomes an interval, and the three answers that are not interchangeable.

The sigma columns are the same whichever method is chosen; what differs is the claim the
lower/upper pair makes. These tests pin that each method makes the claim it says it does, and that
the two things which silently depend on the choice - whether the sklearn family gives up its val
split, and what nominal level the metrics grade against - follow it.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

from yg_eo_soilnet.uncertainty.conformal import ConformalCalibrator
from yg_eo_soilnet.uncertainty.intervals import (
    GaussianInterval,
    SigmaInterval,
    build_interval_estimators,
    describe,
    effective_alpha,
    needs_calibration_set,
    normalize_method,
)


# --- the estimators --------------------------------------------------------


def test_a_gaussian_interval_is_the_normal_multiple_of_sigma():
    lower, upper = GaussianInterval(alpha=0.05).intervals(np.zeros(3), np.ones(3))
    assert np.allclose(upper, 1.959964, atol=1e-5)
    assert np.allclose(lower, -1.959964, atol=1e-5)


def test_a_sigma_interval_is_k_standard_deviations():
    lower, upper = SigmaInterval(k=2.0).intervals(np.array([10.0]), np.array([0.5]))
    assert lower[0] == pytest.approx(9.0) and upper[0] == pytest.approx(11.0)


def test_a_conformal_interval_scales_sigma_by_its_fitted_q():
    lower, upper = ConformalCalibrator(q=3.0, alpha=0.05, n_calib=100).intervals(
        np.array([10.0]), np.array([0.5])
    )
    assert lower[0] == pytest.approx(8.5) and upper[0] == pytest.approx(11.5)


def test_the_three_methods_disagree_on_the_same_sigma():
    """The point of the whole feature: same sigma, three different bands."""
    mean, sigma = np.array([0.0]), np.array([1.0])
    conformal = ConformalCalibrator(q=50.3, alpha=0.05, n_calib=800).intervals(mean, sigma)[1][0]
    gaussian = GaussianInterval(alpha=0.05).intervals(mean, sigma)[1][0]
    one_sigma = SigmaInterval(k=1.0).intervals(mean, sigma)[1][0]
    assert one_sigma < gaussian < conformal


# --- nominal coverage ------------------------------------------------------


def test_conformal_and_gaussian_claim_one_minus_alpha():
    assert ConformalCalibrator(q=1.0, alpha=0.05, n_calib=10).nominal_coverage == pytest.approx(0.95)
    assert GaussianInterval(alpha=0.05).nominal_coverage == pytest.approx(0.95)
    assert GaussianInterval(alpha=0.20).nominal_coverage == pytest.approx(0.80)


def test_a_sigma_band_claims_what_k_implies_not_one_minus_alpha():
    # This is why nominal_coverage is a property rather than a config lookup.
    assert SigmaInterval(k=1.0).nominal_coverage == pytest.approx(0.6827, abs=1e-4)
    assert SigmaInterval(k=2.0).nominal_coverage == pytest.approx(0.9545, abs=1e-4)


def test_the_effective_alpha_follows_the_method():
    assert effective_alpha("conformal", alpha=0.05, k=1.0) == pytest.approx(0.05)
    assert effective_alpha("gaussian", alpha=0.10, k=1.0) == pytest.approx(0.10)
    # A +-1 sigma band misses ~31.7% of the time, not 5%.
    assert effective_alpha("sigma", alpha=0.05, k=1.0) == pytest.approx(0.3173, abs=1e-4)


def test_grading_a_sigma_band_against_alpha_would_have_been_wrong():
    """The concrete failure the effective alpha prevents: a perfect ±1σ band, mis-graded.

    Its coverage_error against 1-alpha would read -0.27, which looks like a badly broken model.
    Against what k actually claims it reads ~0.
    """
    from yg_eo_soilnet.uncertainty.metrics import uncertainty_metrics

    rng = np.random.default_rng(0)
    observed = rng.normal(size=20000)
    predicted, sigma = np.zeros(20000), np.ones(20000)
    lower, upper = SigmaInterval(k=1.0).intervals(predicted, sigma)

    naive = uncertainty_metrics(observed, predicted, sigma, lower, upper, alpha=0.05)
    correct = uncertainty_metrics(
        observed, predicted, sigma, lower, upper, alpha=effective_alpha("sigma", k=1.0)
    )
    assert naive["coverage_error_test"] == pytest.approx(-0.27, abs=0.02)
    assert correct["coverage_error_test"] == pytest.approx(0.0, abs=0.02)


# --- method resolution -----------------------------------------------------


def test_the_legacy_split_conformal_spelling_still_resolves():
    assert normalize_method("split_conformal") == "conformal"
    assert normalize_method("CONFORMAL") == "conformal"
    assert normalize_method(None) == "conformal"


def test_an_unknown_method_is_refused_by_name():
    with pytest.raises(ValueError, match="Unknown uncertainty.interval.method"):
        normalize_method("bootstrap")


def test_only_conformal_needs_held_out_rows():
    assert needs_calibration_set("conformal") is True
    assert needs_calibration_set("split_conformal") is True
    assert needs_calibration_set("gaussian") is False
    assert needs_calibration_set("sigma") is False
    assert needs_calibration_set("none") is False


# --- the dispatch ----------------------------------------------------------


def test_gaussian_and_sigma_are_built_without_any_data():
    for method, kind in (("gaussian", GaussianInterval), ("sigma", SigmaInterval)):
        built = build_interval_estimators(method, ["clay_pct", "sand_pct"], alpha=0.05, k=1.0)
        assert set(built) == {"clay_pct", "sand_pct"}
        assert all(isinstance(v, kind) for v in built.values())


def test_none_produces_no_estimators_and_therefore_no_interval_columns():
    assert build_interval_estimators("none", ["clay_pct"]) == {}


def test_conformal_without_a_calibration_set_says_what_to_do_instead():
    with pytest.raises(ValueError, match="gaussian|sigma"):
        build_interval_estimators("conformal", ["clay_pct"])


def test_the_bar_label_names_the_claim_being_made():
    assert describe(SigmaInterval(k=1.0)) == "±1σ"
    assert describe(SigmaInterval(k=2.5)) == "±2.5σ"
    assert describe(GaussianInterval(alpha=0.05)) == "gaussian 95%"
    assert describe(ConformalCalibrator(q=2.0, alpha=0.05, n_calib=10)) == "conformal 95%"
    assert describe(None) == ""


# --- the fit-pool consequence ----------------------------------------------


def _trainer(method, **overrides):
    import logging

    from yg_eo_soilnet.trainers import ModelTrainer

    base = dict(
        UNCERTAINTY_ENABLED=True,
        UNCERTAINTY_INTERVAL_METHOD=method,
        UNCERTAINTY_CALIBRATION_SOURCE="val",
        CATEGORICAL_FEATURES=[],
        CLUSTERING_STRATEGY={"enabled": False, "params": {}},
    )
    base.update(overrides)
    logger = logging.getLogger("interval-test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return ModelTrainer(config=SimpleNamespace(**base), logger=logger)


def _split_dict():
    import pandas as pd

    return {
        "X_train": pd.DataFrame({"f": range(100)}),
        "y_train": pd.DataFrame({"t": range(100)}),
        "X_train_only": pd.DataFrame({"f": range(80)}),
        "y_train_only": pd.DataFrame({"t": range(80)}),
        "X_val": pd.DataFrame({"f": range(20)}),
        "y_val": pd.DataFrame({"t": range(20)}),
    }


def test_conformal_still_reserves_the_val_split():
    fit_pool, calibration = _trainer("conformal")._resolve_fit_pool(_split_dict())
    assert len(fit_pool["X"]) == 80 and calibration is not None


@pytest.mark.parametrize("method", ["gaussian", "sigma", "none"])
def test_a_method_needing_no_calibration_keeps_the_whole_fit_pool(method):
    """Reserving val for a band that is pure arithmetic would cost ~15% of the rows for nothing."""
    fit_pool, calibration = _trainer(method)._resolve_fit_pool(_split_dict())
    assert len(fit_pool["X"]) == 100
    assert calibration is None


# --- config ----------------------------------------------------------------


def _config_with(tmp_path, uncertainty_yaml: str):
    """A real Config over a minimal tree, reusing test_config.py's fixture content.

    Building the whole tree matters: Config resolves its sub-configs relative to the main file, so
    a one-key stub raises before it ever reaches the uncertainty block.
    """
    from config import Config
    from tests.test_config import (
        BASE_DATA_SPEC_CONTENT,
        BASE_LIGHTNING_CONFIG_CONTENT,
        BASE_SKLEARN_CONFIG_CONTENT,
        MOCK_LIGHTNING_REGISTRY,
    )

    (tmp_path / "data_spec.yml").write_text(BASE_DATA_SPEC_CONTENT.strip())
    (tmp_path / "sklearn.yml").write_text(BASE_SKLEARN_CONFIG_CONTENT.strip())
    (tmp_path / "lightning.yml").write_text(BASE_LIGHTNING_CONFIG_CONTENT.strip())
    (tmp_path / "registry.yml").write_text("enabled: true\n")
    (tmp_path / "lightning_registry.yml").write_text(MOCK_LIGHTNING_REGISTRY)

    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "common:\n"
        "    DATA_SPEC_PATH: data_spec.yml\n"
        "    SKLEARN_CONFIG_PATH: sklearn.yml\n"
        "    LIGHTNING_CONFIG_PATH: lightning.yml\n"
        "    SKLEARN_REGISTRY_PATH: registry.yml\n"
        "    LIGHTNING_REGISTRY_PATH: lightning_registry.yml\n"
        + uncertainty_yaml
    )
    return Config(
        config_path=str(config_path),
        registry_path=str(tmp_path / "registry.yml"),
        lightning_registry_path=str(tmp_path / "lightning_registry.yml"),
    )


def test_the_interval_block_is_read_from_the_config(tmp_path):
    config = _config_with(tmp_path, """    uncertainty:
        interval:
            method: sigma
            k: 2.0
""")
    assert config.UNCERTAINTY_INTERVAL_METHOD == "sigma"
    assert config.UNCERTAINTY_INTERVAL_K == 2.0


def test_a_config_predating_the_interval_block_still_works(tmp_path):
    """`calibration.method` is what configs in the wild set; it must keep resolving."""
    config = _config_with(tmp_path, """    uncertainty:
        calibration:
            method: split_conformal
            alpha: 0.10
""")
    assert normalize_method(config.UNCERTAINTY_INTERVAL_METHOD) == "conformal"
    assert config.UNCERTAINTY_ALPHA == pytest.approx(0.10)


def test_the_legacy_none_still_means_none(tmp_path):
    config = _config_with(tmp_path, """    uncertainty:
        calibration:
            method: none
""")
    assert normalize_method(config.UNCERTAINTY_INTERVAL_METHOD) == "none"


def test_the_interval_block_wins_over_the_legacy_key(tmp_path):
    config = _config_with(tmp_path, """    uncertainty:
        interval:
            method: gaussian
        calibration:
            method: split_conformal
""")
    assert normalize_method(config.UNCERTAINTY_INTERVAL_METHOD) == "gaussian"


def test_the_default_is_conformal_when_nothing_is_configured(tmp_path):
    config = _config_with(tmp_path, "")
    assert normalize_method(config.UNCERTAINTY_INTERVAL_METHOD) == "conformal"
