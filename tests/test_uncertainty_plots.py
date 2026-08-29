"""The uncertainty column contract, and the plots that read it."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from yg_eo_soilnet.plot_utils import (  # noqa: E402
    _resolve_prediction_column,
    create_parent_pred_obs,
    create_pred_obs_plot,
)
from yg_eo_soilnet.uncertainty import attach_uncertainty_columns  # noqa: E402
from yg_eo_soilnet.uncertainty.columns import (  # noqa: E402
    column_name,
    interval_columns,
    is_prediction_column,
    sigma_column,
)
from yg_eo_soilnet.uncertainty.conformal import ConformalCalibrator  # noqa: E402
from yg_eo_soilnet.uncertainty.ensemble import aggregate  # noqa: E402
from yg_eo_soilnet.uncertainty.plots import reliability_curve, sigma_vs_error  # noqa: E402


def _eval_frame(n_rows: int = 60, with_uncertainty: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    observed = rng.normal(loc=20.0, scale=5.0, size=n_rows)
    predicted = observed + rng.normal(scale=2.0, size=n_rows)
    frame = pd.DataFrame({"target": observed, "prediction": predicted})
    if with_uncertainty:
        sigma = np.abs(rng.normal(loc=2.0, scale=0.5, size=n_rows))
        frame["prediction_std"] = sigma
        frame["prediction_lower"] = predicted - 2.0 * sigma
        frame["prediction_upper"] = predicted + 2.0 * sigma
    return frame


# --- the column contract ---------------------------------------------------


def test_a_single_target_run_writes_unsuffixed_columns():
    assert column_name("prediction_std", "clay_pct", multi_target=False) == "prediction_std"


def test_a_multi_target_run_suffixes_every_column_with_its_target():
    assert column_name("prediction_std", "clay_pct", multi_target=True) == "prediction_std_clay_pct"


def test_an_uncertainty_column_is_not_mistaken_for_a_prediction_column():
    assert is_prediction_column("prediction") is True
    assert is_prediction_column("prediction_clay_pct") is True
    assert is_prediction_column("prediction_std") is False
    assert is_prediction_column("prediction_std_clay_pct") is False
    assert is_prediction_column("prediction_lower_clay_pct") is False
    assert is_prediction_column("prediction_epistemic_std_clay_pct") is False
    assert is_prediction_column("clay_pct") is False


def test_the_prediction_column_fallback_never_selects_a_standard_deviation():
    # The specific trap: the positional fallback in _resolve_prediction_column takes the first
    # column starting with "prediction_", and on a joint uncertainty frame that can be a sigma.
    # Plotting it would put standard deviations on the predicted axis and look entirely plausible.
    frame = pd.DataFrame(
        {
            "prediction_epistemic_std_clay_pct": [9.0],
            "prediction_std_clay_pct": [9.0],
            "prediction_clay_pct": [1.0],
        }
    )
    assert _resolve_prediction_column(frame, target_name=None, target_index=0) == "prediction_clay_pct"


def test_interval_columns_prefers_the_suffixed_pair_on_a_joint_frame():
    frame = pd.DataFrame(
        {
            "prediction_lower_clay_pct": [1.0],
            "prediction_upper_clay_pct": [2.0],
            "prediction_lower_sand_pct": [3.0],
            "prediction_upper_sand_pct": [4.0],
        }
    )
    lower, upper = interval_columns(frame, "sand_pct")
    assert lower.iloc[0] == 3.0 and upper.iloc[0] == 4.0


def test_interval_columns_returns_none_for_a_frame_from_a_run_without_uncertainty():
    assert interval_columns(_eval_frame(with_uncertainty=False)) is None
    assert sigma_column(_eval_frame(with_uncertainty=False)) is None


def test_a_half_written_interval_is_treated_as_absent_rather_than_half_used():
    frame = _eval_frame()
    frame = frame.drop(columns=["prediction_upper"])
    assert interval_columns(frame) is None


def test_attach_writes_unsuffixed_columns_for_one_target():
    frame = pd.DataFrame({"target": np.zeros(5), "prediction": np.zeros(5)})
    prediction = aggregate([np.zeros(5), np.ones(5)])
    calibrator = ConformalCalibrator(q=2.0, alpha=0.05, n_calib=100)

    attach_uncertainty_columns(frame, prediction, ["clay_pct"], {"clay_pct": calibrator})

    assert "prediction_std" in frame.columns
    assert "prediction_std_clay_pct" not in frame.columns
    # Members 0 and 1: ensemble mean 0.5, epistemic std 0.5, no aleatoric part, so the interval is
    # 0.5 -+ q * 0.5 with q = 2.
    assert frame["prediction_epistemic_std"].iloc[0] == pytest.approx(0.5)
    assert frame["prediction_aleatoric_std"].iloc[0] == pytest.approx(0.0)
    assert frame["prediction_lower"].iloc[0] == pytest.approx(-0.5)
    assert frame["prediction_upper"].iloc[0] == pytest.approx(1.5)


def test_attach_suffixes_every_column_for_a_joint_group():
    frame = pd.DataFrame({"a": np.zeros(5), "b": np.zeros(5)})
    prediction = aggregate([np.zeros((5, 2)), np.ones((5, 2))])

    attach_uncertainty_columns(frame, prediction, ["a", "b"])

    for stem in ("prediction_std", "prediction_epistemic_std", "prediction_aleatoric_std"):
        assert f"{stem}_a" in frame.columns
        assert f"{stem}_b" in frame.columns
        assert stem not in frame.columns


def test_attach_writes_no_interval_when_there_is_no_calibrator():
    frame = pd.DataFrame({"target": np.zeros(5)})
    attach_uncertainty_columns(frame, aggregate([np.zeros(5), np.ones(5)]), ["clay_pct"])
    assert "prediction_std" in frame.columns
    assert "prediction_lower" not in frame.columns


def test_a_single_target_frame_with_uncertainty_still_reads_as_single_target():
    """The uncertainty columns must not make a one-target frame look like a joint one.

    _iter_target_eval_frames decided that by scanning for `prediction_*`, which the sigma and
    interval columns now also match. A single-target uncertainty frame then found no
    prediction_<target>, yielded nothing, and the run logged neither rmse_test nor picp_test - with
    no error anywhere, because an empty metric dict is indistinguishable from a metric-free run.
    """
    from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger

    frame = _eval_frame()
    frame["clay_pct"] = frame["target"]

    frames = list(
        ChildRunLogger()._iter_target_eval_frames(frame, target="clay_pct", model_name="Ridge")
    )
    assert len(frames) == 1
    _yielded, target_name, prediction_column = frames[0]
    assert target_name == "clay_pct"
    assert prediction_column == "prediction"


def test_the_parent_collector_never_plots_a_standard_deviation_as_a_prediction(tmp_path):
    """`_collect_eval_dfs` had the same `prediction_*` bug as `_iter_target_eval_frames`.

    A single-target uncertainty frame took the multi-target branch, and the fallback there picked
    the first `prediction_*` column - which is `prediction_std`, since it sorts right after
    `prediction`. That value was then assigned to `prediction`, so the parent's pred_error_plot.png
    plotted standard deviations on the predicted axis and looked entirely plausible.
    """
    import mlflow

    from yg_eo_soilnet.logger.mlflow_loggers import ParentRunLogger

    frame = _eval_frame(40)
    frame["clay_pct"] = frame["target"]
    # Sigma is deliberately far from the prediction, so picking the wrong column is unmissable.
    frame["prediction_std"] = 999.0
    frame["target_names"] = "clay_pct"

    with mlflow.start_run() as parent:
        parent_id = parent.info.run_id
        with mlflow.start_run(nested=True) as child:
            mlflow.set_tags({"target": "clay_pct", "model_name": "Ridge"})
            path = tmp_path / "eval_results.csv"
            frame.to_csv(path, index=False)
            mlflow.log_artifact(str(path), artifact_path="eval_results")
            child.info.run_id

    collected = ParentRunLogger()._collect_eval_dfs(parent_id)
    assert len(collected) == 1
    assert not (collected[0]["prediction"] == 999.0).any()
    assert np.allclose(collected[0]["prediction"], frame["prediction"])


def test_a_joint_frame_with_uncertainty_still_fans_out_per_target():
    from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger

    frame = pd.DataFrame(
        {
            "clay_pct": [1.0, 2.0],
            "sand_pct": [3.0, 4.0],
            "prediction_clay_pct": [1.1, 2.1],
            "prediction_sand_pct": [3.1, 4.1],
            "prediction_std_clay_pct": [0.5, 0.5],
            "prediction_std_sand_pct": [0.5, 0.5],
            "target_names": ["clay_pct__sand_pct"] * 2,
        }
    )
    frames = list(
        ChildRunLogger()._iter_target_eval_frames(
            frame, target="clay_pct__sand_pct", model_name="Ridge"
        )
    )
    assert [name for _f, name, _c in frames] == ["clay_pct", "sand_pct"]
    assert [col for _f, _n, col in frames] == ["prediction_clay_pct", "prediction_sand_pct"]


# --- the plots -------------------------------------------------------------


def test_the_pred_obs_plot_still_renders_without_any_uncertainty_columns(tmp_path):
    # The majority case: every frame from a run with uncertainty disabled.
    artifacts = create_pred_obs_plot(_eval_frame(with_uncertainty=False), {}, str(tmp_path))
    assert (tmp_path / "obs_pred_and_residual_plot.png").exists()
    assert "obs_pred_and_residual_plot" in artifacts


def test_the_pred_obs_plot_renders_with_uncertainty_columns(tmp_path):
    artifacts = create_pred_obs_plot(_eval_frame(), {}, str(tmp_path))
    assert (tmp_path / "obs_pred_and_residual_plot.png").exists()
    assert "obs_pred_and_residual_plot" in artifacts


def test_the_axis_is_framed_on_the_points_not_on_the_widest_bar():
    """One very uncertain point must not squash every prediction into an unreadable band."""
    import matplotlib.pyplot as plt

    from yg_eo_soilnet.plot_utils import _frame_on_data

    figure, axis = plt.subplots()
    try:
        axis.errorbar([0, 1], [0, 1], yerr=[500, 500], fmt="none")
        _frame_on_data(axis, np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        low, high = axis.get_ylim()
    finally:
        plt.close(figure)

    assert high < 10.0 and low > -10.0


def test_the_pred_obs_panel_shares_one_range_across_both_axes():
    """A pred-vs-obs scatter only reads correctly when its identity line is a true diagonal."""
    from yg_eo_soilnet.plot_utils import _square_limits

    low, high = _square_limits(np.array([0.0, 100.0]), np.array([40.0, 60.0]))
    assert low < 0.0 and high > 100.0


def test_the_square_range_reaches_the_bar_ends():
    from yg_eo_soilnet.plot_utils import _square_limits

    observed = np.array([0.0, 100.0])
    predicted = np.array([20.0, 80.0])
    interval = (np.array([-20.0, -10.0]), np.array([110.0, 120.0]))

    plain_low, plain_high = _square_limits(observed, predicted)
    low, high = _square_limits(observed, predicted, interval)
    assert low < plain_low and high > plain_high


def test_a_heavy_tailed_sigma_cannot_squash_the_scatter():
    """The percentile clip alone is not enough when the far tail is still very far out.

    Real case: with sigma right-skewed, the 98th percentile of the bar ends sat about twice the
    target range beyond the data and left the points in the middle fifth of the panel.
    """
    from yg_eo_soilnet.plot_utils import MAX_INTERVAL_EXTENSION, _square_limits

    observed = np.linspace(0.0, 100.0, 500)
    predicted = observed.copy()
    # Every bar end is far out, so no percentile can drop them - only the extension ceiling can.
    interval = (np.full(500, -500.0), np.full(500, 600.0))

    low, high = _square_limits(observed, predicted, interval)
    data_span = 100.0
    assert low >= -(MAX_INTERVAL_EXTENSION * data_span) * 1.1 - 10.0
    assert high <= 100.0 + (MAX_INTERVAL_EXTENSION * data_span) * 1.1 + 10.0
    # The points still occupy a usable share of the panel.
    assert data_span / (high - low) > 0.5


def test_one_enormous_bar_does_not_decide_the_whole_axis():
    # The 790-wide case. min/max framing would put the axis at roughly -400..400 and collapse the
    # scatter into a band; the percentile clip ignores the outer 2% at each end.
    from yg_eo_soilnet.plot_utils import _square_limits

    observed = np.linspace(0.0, 100.0, 200)
    predicted = observed.copy()
    lower = np.full(200, -10.0)
    upper = np.full(200, 110.0)
    lower[0], upper[0] = -400.0, 400.0

    low, high = _square_limits(observed, predicted, (lower, upper))
    assert low > -100.0 and high < 200.0


def test_the_square_range_survives_a_frame_with_no_interval():
    from yg_eo_soilnet.plot_utils import _square_limits

    low, high = _square_limits(np.array([1.0, 9.0]), np.array([2.0, 8.0]), None)
    assert low < 1.0 and high > 9.0


def test_frame_square_leaves_the_panel_at_a_one_to_one_aspect():
    import matplotlib.pyplot as plt

    from yg_eo_soilnet.plot_utils import _frame_square

    figure, axis = plt.subplots()
    try:
        _frame_square(axis, -5.0, 105.0)
        assert axis.get_aspect() == 1.0
        assert axis.get_xlim() == axis.get_ylim() == (-5.0, 105.0)
    finally:
        plt.close(figure)


def test_the_error_bars_are_red_with_ends_you_can_actually_see():
    """Caps and verticals are styled separately; one alpha on the container cannot do both."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from yg_eo_soilnet.plot_utils import ERROR_BAR_COLOR, _draw_error_bars

    figure, axis = plt.subplots()
    try:
        y = np.array([1.0, 2.0, 3.0])
        caplines, barlinecols = _draw_error_bars(
            axis, np.arange(3.0), y, y - 1.0, y + 1.0, np.arange(3)
        )
        assert caplines, "no caps drawn, so the interval ends are invisible"
        expected = mcolors.to_rgba(ERROR_BAR_COLOR)[:3]
        for cap in caplines:
            assert mcolors.to_rgba(cap.get_color())[:3] == expected
            assert cap.get_markeredgewidth() > 1.0
        # The ends have to stand out from the verticals, not fade with them.
        assert caplines[0].get_alpha() > barlinecols[0].get_alpha()
    finally:
        plt.close(figure)


def test_the_bars_are_thinned_but_the_metrics_still_cover_every_point():
    from yg_eo_soilnet.plot_utils import MAX_ERROR_BARS, _error_bar_positions

    positions = _error_bar_positions(np.arange(1000.0))
    assert len(positions) == MAX_ERROR_BARS
    # Spread across the whole range rather than clustered where the data is dense.
    assert positions[0] < 10 and positions[-1] > 990
    # Deterministic: the same run always draws the same picture.
    assert np.array_equal(positions, _error_bar_positions(np.arange(1000.0)))


def test_a_small_split_keeps_a_bar_on_every_point():
    from yg_eo_soilnet.plot_utils import _error_bar_positions

    assert len(_error_bar_positions(np.arange(50.0))) == 50


def test_the_plot_closes_every_figure_it_opens(tmp_path):
    # Called once per target inside a training loop, so a leak here accumulates until matplotlib
    # warns - and with warnings as errors that fails a run several targets later.
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    for frame in (_eval_frame(with_uncertainty=False), _eval_frame()):
        create_pred_obs_plot(frame, {}, str(tmp_path))
    assert set(plt.get_fignums()) == before


def test_the_plot_survives_a_model_that_fits_its_test_split_exactly(tmp_path):
    # RPIQ and RPD divide by rmse. A degenerate estimator makes that a divide-by-zero, which used to
    # take the plot - and the artifact logging around it - down with it.
    exact = pd.DataFrame({"target": [1.0, 2.0, 3.0, 4.0], "prediction": [1.0, 2.0, 3.0, 4.0]})
    assert create_pred_obs_plot(exact, {}, str(tmp_path))


def test_the_parent_overlay_renders_with_and_without_intervals():
    frame = _eval_frame()
    frame["target_name"] = "clay_pct"
    frame["clay_pct"] = frame["target"]
    frame["prediction_clay_pct"] = frame["prediction"]
    frame["prediction_lower_clay_pct"] = frame["prediction_lower"]
    frame["prediction_upper_clay_pct"] = frame["prediction_upper"]
    frame["model_name"] = "Ridge"

    assert create_parent_pred_obs([frame]) is not None

    bare = _eval_frame(with_uncertainty=False)
    bare["target_name"] = "clay_pct"
    bare["clay_pct"] = bare["target"]
    bare["prediction_clay_pct"] = bare["prediction"]
    bare["model_name"] = "Ridge"
    assert create_parent_pred_obs([bare]) is not None


def test_the_reliability_curve_returns_a_figure_the_caller_saves():
    frame = _eval_frame(200)
    import matplotlib.pyplot as plt

    figure = reliability_curve(
        frame["target"], frame["prediction"], frame["prediction_std"], target_name="clay_pct"
    )
    try:
        assert figure is not None
        assert figure.axes[0].get_xlabel() == "Nominal coverage"
    finally:
        plt.close(figure)


def test_the_reliability_curve_draws_into_a_supplied_axis_and_returns_none():
    import matplotlib.pyplot as plt

    frame = _eval_frame(200)
    figure, axis = plt.subplots()
    try:
        assert reliability_curve(
            frame["target"], frame["prediction"], frame["prediction_std"], axis=axis
        ) is None
    finally:
        plt.close(figure)


def test_sigma_vs_error_bins_by_equal_count_and_returns_a_figure():
    frame = _eval_frame(200)
    import matplotlib.pyplot as plt

    figure = sigma_vs_error(
        frame["target"], frame["prediction"], frame["prediction_std"], target_name="clay_pct"
    )
    try:
        assert figure is not None
        assert figure.axes[0].get_xlabel().startswith("Predicted")
    finally:
        plt.close(figure)


def test_sigma_vs_error_degrades_gracefully_on_a_split_too_small_to_bin():
    import matplotlib.pyplot as plt

    figure = sigma_vs_error(np.zeros(5), np.zeros(5), np.ones(5))
    try:
        assert figure is not None
    finally:
        plt.close(figure)


# --- the parent overlay: one frame per target, on its own panel ------------


def _target_frame(target, observed, predicted, first_column=None):
    """A per-target eval frame shaped like ParentRunLogger._collect_eval_dfs yields.

    `first_column` stands in for the feature that happens to come first in the frame - the column
    the old positional fallback would have seized on.
    """
    frame = pd.DataFrame()
    if first_column is not None:
        frame["landform_class"] = first_column
    frame[target] = observed
    frame["prediction"] = predicted
    frame["target_name"] = target
    frame["model_name"] = "TabICL"
    return frame


def test_a_categorical_first_column_no_longer_crashes_the_parent_overlay():
    """The reported failure: run 13a8925b, five per-target children, landform_class first.

    np.isfinite on 'upper_slope_flat' raised a TypeError from inside the ufunc and took the whole
    run down after every model had already trained.
    """
    frames = [
        _target_frame("clay_pct", [10.0, 20.0], [11.0, 21.0], ["upper_slope_flat", "lower_slope"]),
        _target_frame("ph_water", [7.0, 8.0], [7.1, 8.1], ["upper_slope_flat", "lower_slope"]),
    ]
    figure = create_parent_pred_obs(frames)
    try:
        assert figure is not None
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def _panel_x(figure, title):
    import numpy as np

    for axis in figure.axes:
        if axis.get_title() == title:
            chunks = [c.get_offsets()[:, 0] for c in axis.collections if len(c.get_offsets())]
            return np.concatenate(chunks) if chunks else np.array([])
    raise AssertionError(f"no panel titled {title!r}")


def test_a_frame_for_one_target_stays_off_another_targets_panel():
    """The root cause, asserted on the DRAWN data rather than on not raising.

    The two targets are given disjoint ranges, so a panel that borrowed the other frame's rows
    would show it immediately.
    """
    import matplotlib.pyplot as plt

    frames = [
        _target_frame("clay_pct", [10.0, 12.0], [10.5, 12.5]),
        _target_frame("ph_water", [900.0, 950.0], [905.0, 955.0]),
    ]
    figure = create_parent_pred_obs(frames)
    try:
        clay = _panel_x(figure, "clay_pct")
        ph = _panel_x(figure, "ph_water")
        assert len(clay) == 2 and clay.max() < 100
        assert len(ph) == 2 and ph.min() > 100
    finally:
        plt.close(figure)


def test_the_silent_case_is_fixed_too_not_just_the_crash():
    """An all-numeric first column never raised - it plotted a reflectance band as Observed.

    Run 7d09124 did exactly this and produced a pred_error_plot nobody could tell was wrong, which
    is why the fix is a target check and not a try/except.
    """
    import matplotlib.pyplot as plt

    frames = [
        _target_frame("clay_pct", [10.0, 12.0], [10.5, 12.5], [0.031, 0.032]),
        _target_frame("ph_water", [7.0, 8.0], [7.1, 8.1], [0.041, 0.042]),
    ]
    figure = create_parent_pred_obs(frames)
    try:
        clay = _panel_x(figure, "clay_pct")
        # Two points, not four, and none of them a reflectance value near 0.03.
        assert len(clay) == 2
        assert clay.min() > 1.0
    finally:
        plt.close(figure)


def test_a_frame_without_a_target_name_still_plots():
    """The legacy single-target path, which the target check must not break."""
    import matplotlib.pyplot as plt

    frame = pd.DataFrame({"target": [1.0, 2.0], "prediction": [1.1, 2.1], "model_name": "Ridge"})
    figure = create_parent_pred_obs([frame])
    try:
        assert figure is not None
    finally:
        plt.close(figure)


def test_a_frame_with_no_numeric_column_is_skipped_rather_than_guessed_at():
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(
        {"landform_class": ["a", "b"], "prediction": [1.0, 2.0], "model_name": "Ridge"}
    )
    figure = create_parent_pred_obs([frame])
    try:
        assert figure is not None  # renders an empty panel rather than inventing an x axis
    finally:
        plt.close(figure)


def test_numbers_stored_as_strings_still_plot():
    """Coercion rather than trust: a numeric-looking object column is usable, not fatal."""
    import matplotlib.pyplot as plt

    frame = _target_frame("clay_pct", ["10.0", "12.0"], [10.5, 12.5])
    figure = create_parent_pred_obs([frame])
    try:
        assert len(_panel_x(figure, "clay_pct")) == 2
    finally:
        plt.close(figure)
