from yg_eo_soilnet.utils import rpiq_score
from yg_eo_soilnet.uncertainty.columns import interval_columns, is_prediction_column, sigma_column
from sklearn.metrics import r2_score, root_mean_squared_error
import numpy as np
import pandas as pd
import seaborn as sns
def _normalise_target_names(values):
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    try:
        return [value for value in values if value is not None and str(value) != ""]
    except TypeError:
        return [values]


def _resolve_prediction_column(df, target_name=None, target_index=None):
    candidates = []
    if target_name:
        candidates.extend(
            [
                f"prediction_{target_name}",
                f"prediction_{str(target_name).replace(' ', '_')}",
                f"pred_{target_name}",
            ]
        )
    if target_index is not None:
        candidates.append(f"prediction_{target_index}")
    candidates.append("prediction")

    for column_name in candidates:
        if column_name in df.columns:
            return column_name

    # is_prediction_column, not `startswith("prediction_")`. An eval frame from an uncertainty run
    # also carries prediction_std_<t>, prediction_lower_<t> and friends, and the positional fallback
    # below would happily return one of those - drawing standard deviations on the predicted axis,
    # with a plot that looks plausible and is wrong.
    prediction_columns = [
        column_name for column_name in df.columns if is_prediction_column(column_name)
    ]
    if target_index is not None and target_index < len(prediction_columns):
        return prediction_columns[target_index]
    if prediction_columns:
        return prediction_columns[0]
    return None


def _create_parent_pred_obs_multitarget(eval_dfs):
    if not eval_dfs:
        return None

    prepared_frames = []
    target_names = []

    for eval_df in eval_dfs:
        if eval_df is None or eval_df.empty:
            continue
        frame = eval_df.copy()
        frame_targets = _normalise_target_names(frame["target_name"].dropna().unique()) if "target_name" in frame.columns else []
        if not frame_targets:
            frame_targets = [None]
        frame["_resolved_target_name"] = frame["target_name"] if "target_name" in frame.columns else None
        prepared_frames.append((frame, frame_targets))
        for target_name in frame_targets:
            if target_name is not None and target_name not in target_names:
                target_names.append(target_name)

    if not prepared_frames:
        return None

    if not target_names:
        target_names = [None]

    import math
    import matplotlib.pyplot as plt
    import numpy as np

    n_targets = len(target_names)
    n_cols = min(2, n_targets) if n_targets > 1 else 1
    n_rows = math.ceil(n_targets / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    for target_index, target_name in enumerate(target_names):
        axis = axes[target_index // n_cols][target_index % n_cols]
        axis.set_title(str(target_name) if target_name is not None else "Predicted vs observed")
        axis.set_xlabel("Observed")
        axis.set_ylabel("Predicted")

        all_values = []
        for frame, frame_targets in prepared_frames:
            # Only frames that actually scored THIS target belong on this panel. A run whose targets
            # each got their own child - which is what happens whenever an estimator cannot fit a
            # 2-D y, so `MULTI_TARGET_MODE: joint` falls back to one model per target - produces one
            # frame per target, and every one of them carries a plain `prediction` column. Without
            # this check the panel for target A also picks up target B's frame, fails to find an
            # `A` column in it, and invents one from whatever column happens to come first: a
            # reflectance band on a numeric dataset, a categorical on this one. The former plotted
            # silently wrong for months; the latter is the isfinite TypeError that finally surfaced
            # it.
            #
            # Both halves of `covers` earn their place. `frame_targets != [None]` keeps a frame that
            # carries no target_name at all on its legacy path, and the column test means a frame
            # that genuinely holds this target's observations is never dropped over its label.
            covers = target_name in frame_targets or target_name in frame.columns
            if target_name is not None and frame_targets != [None] and not covers:
                continue

            resolved_target_name = target_name if target_name is not None else (frame_targets[0] if frame_targets else None)
            prediction_column = _resolve_prediction_column(frame, resolved_target_name, target_index)
            if prediction_column is None:
                continue

            observed_column = None
            for candidate in (
                resolved_target_name,
                "target",
                "y_true",
                "obs",
                "observation",
                "actual",
            ):
                if candidate is not None and candidate in frame.columns:
                    observed_column = candidate
                    break

            if observed_column is None:
                # Numeric by DTYPE, not merely by not being one of two known label columns - the
                # old test was a name check under a name that promised otherwise. Skipping when
                # nothing qualifies beats plotting an arbitrary column against the predictions.
                numeric_candidates = [
                    column_name
                    for column_name in frame.columns
                    if column_name not in {"model_name", "target_name", "target_names"}
                    and pd.api.types.is_numeric_dtype(frame[column_name])
                ]
                if not numeric_candidates:
                    continue
                observed_column = numeric_candidates[0]

            # Coerced rather than trusted, the same way metrics._finite_pairs does it: a column of
            # numbers-as-strings still plots, and anything genuinely non-numeric becomes NaN and is
            # masked out below instead of raising from inside a ufunc.
            x_values = pd.to_numeric(frame[observed_column], errors="coerce").to_numpy(dtype=float)
            y_values = pd.to_numeric(frame[prediction_column], errors="coerce").to_numpy(dtype=float)
            valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
            if not valid_mask.any():
                continue

            model_name = frame["model_name"].iloc[0] if "model_name" in frame.columns and not frame["model_name"].empty else "model"
            # Bars before the points, under them, and in the series' own colour so two models
            # overlaid on one axis stay distinguishable. Thinner and fainter than on the per-run
            # plot: this axis carries every model at once, so several models' bars overlap here
            # where only one model's do there.
            interval = interval_columns(frame, resolved_target_name)
            series_color = None
            if interval is not None:
                lower = interval[0].to_numpy()[valid_mask]
                upper = interval[1].to_numpy()[valid_mask]
                bars = axis.errorbar(
                    x_values[valid_mask],
                    y_values[valid_mask],
                    yerr=[
                        np.maximum(y_values[valid_mask] - lower, 0.0),
                        np.maximum(upper - y_values[valid_mask], 0.0),
                    ],
                    fmt="none",
                    elinewidth=0.6,
                    alpha=0.2,
                    capsize=0,
                    zorder=1,
                )
                series_color = bars.lines[2][0].get_color()[0] if bars.lines[2] else None
            axis.scatter(
                x_values[valid_mask],
                y_values[valid_mask],
                alpha=0.5,
                s=18,
                label=str(model_name),
                color=series_color,
                zorder=3,
            )
            all_values.extend([x_values[valid_mask], y_values[valid_mask]])

        if all_values:
            combined = np.concatenate(all_values)
            min_value = np.nanmin(combined)
            max_value = np.nanmax(combined)
            axis.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black", linewidth=1)
        axis.legend(loc="best")

    for axis in axes.flatten()[n_targets:]:
        axis.set_visible(False)

    fig.tight_layout()
    return fig


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

import os

def cv_val_curve(cv_results, scoring: str = "neg_root_mean_squared_error"):
    """
    Plot mean train/test CV scores with std bands, and highlight the best parameter.

    Args:
        cv_results (dict): The cv_results_ attribute from GridSearchCV
        param_name (str): The hyperparameter name (e.g. 'model__n_components')
        scoring (str): The scoring metric used in GridSearchCV.
                       If it's a "neg_*" metric, values will be flipped.

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    
    param_key, = [str(col) for col in cv_results.columns if str(col).startswith("param_")]
    param_name = param_key.rsplit("__", 1)[-1]
    param_values = np.array(cv_results[param_key], dtype=object)

    #param_name = param_key[0].rsplit("__", 1)[-1]
    #param_values = np.array(cv_results[param_key], dtype=object)

    # Handle categorical (non-numeric) params
    if not np.issubdtype(param_values.dtype, np.number):
        param_values = param_values.astype(str)

    # Flip scores if it's a neg_* metric
    def process_scores(scores):
        return -scores if scoring.startswith("neg_") else scores

    mean_train = process_scores(np.array(cv_results["mean_train_score"], dtype=float))
    std_train = np.array(cv_results["std_train_score"], dtype=float)

    mean_test = process_scores(np.array(cv_results["mean_test_score"], dtype=float))
    std_test = np.array(cv_results["std_test_score"], dtype=float)

    # Best param index
    best_idx = np.argmax(mean_test) if not scoring.startswith("neg_") else np.argmin(mean_test)
    best_param = param_values[best_idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mean train with std band
    ax.plot(param_values, mean_train, color="blue", marker="o", linewidth=2.5, label="Mean Train")
    if np.issubdtype(mean_train.dtype, np.number):
        ax.fill_between(param_values, mean_train - std_train, mean_train + std_train,
                        color="blue", alpha=0.15, label="CV_Train")

    # Plot mean test with std band
    ax.plot(param_values, mean_test, color="red", marker="o", linewidth=2.5, label="Mean Test")
    if np.issubdtype(mean_test.dtype, np.number):
        ax.fill_between(param_values, mean_test - std_test, mean_test + std_test,
                        color="red", alpha=0.15, label="CV_Test")

    # Highlight best param point
    ax.scatter([best_param], [mean_train[best_idx]], color="blue", s=120, marker="*", label="Best Train", edgecolor="black")
    ax.scatter([best_param], [mean_test[best_idx]], color="red", s=120, marker="*", label="Best Test", edgecolor="black")

    # Labels
    ylabel = scoring if not scoring.startswith("neg_") else scoring.replace("neg_", "")
    ax.set_xlabel(param_name)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Best {param_name} = {best_param}")
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig

def cv_parallel_coordinates(cv_results):
    cv_results['mean_test_score'] = cv_results['mean_test_score'].abs()
    best_index = cv_results['mean_test_score'].idxmin()

    param_cols = [col for col in cv_results.columns if col.startswith('param_')]
    ynames = [col.rsplit('__', 1)[-1] for col in param_cols] + ['Mean Test Score']
    ys = cv_results[param_cols + ['mean_test_score']].values
    parallels = ys.shape[0]

    # Scaling
    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)
    dys = ymaxs - ymins
    dys = np.where(dys == 0, 1, dys)
    ymins -= dys*0.02
    ymaxs += dys*0.02
    dys = ymaxs - ymins
    zs = np.zeros_like(ys)
    zs[:,0] = ys[:,0]
    zs[:,1:] = (ys[:,1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    # Main axes
    fig, host = plt.subplots(figsize=(12,6))

    axes = [host] + [host.twinx() for _ in range(ys.shape[1]-1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i/(ys.shape[1]-1)))

    # Colormap
    norm = mcolors.Normalize(vmin=cv_results['mean_test_score'].min(), vmax=cv_results['mean_test_score'].max())
    cmap = matplotlib.colormaps['viridis']
    # Draw other lines
    for j in range(parallels):
        verts = list(zip(np.linspace(0,len(ys[0])-1,len(ys[0])*3-2),
                             np.repeat(zs[j,:],3)[1:-1]))
        codes = [Path.MOVETO]+[Path.CURVE4]*(len(verts)-1)
        path = Path(verts, codes)
        if j!=best_index:
            patch = patches.PathPatch(path, facecolor='none', lw=0.5,
                                      edgecolor=cmap(norm(cv_results['mean_test_score'].iloc[j])))
        else:
            patch = patches.PathPatch(path, facecolor='none', lw=3, edgecolor='crimson', label = 'Best Model')
        host.add_patch(patch)
    legend_line = Line2D([0], [0], color='crimson', lw=3, label='Best Model')
    host.legend(handles=[legend_line], loc="lower left", bbox_to_anchor=(1.0, -0.1))
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=host, anchor=(0.2, 0.5))

    best_params = cv_results.loc[best_index][param_cols].to_dict()
    best_params_str = ", ".join([f"{str(k).rsplit('__', 1)[-1]}={v}" for k, v in best_params.items()])

    host.set_xlim(0, ys.shape[1]-1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(ynames, rotation=45, ha='right', fontsize=8)
    host.tick_params(axis='x', which='major', pad=7)
    host.grid(True, which='major', axis='y')
    host.spines['top'].set_visible(True)
    host.spines['bottom'].set_visible(True)
    host.set_axisbelow(True)
    host.set_title(f'Best Model: {best_params_str}', fontsize=10)
    return fig

# Most bars a pred-vs-obs panel will draw. Past roughly this many, neighbouring bars are less than
# a pixel apart and merge into a solid grey curtain that hides the scatter underneath - which is
# worse than showing fewer bars, because it hides the very thing the panel is about. The metrics and
# the CSV always cover every point; only the picture is thinned.
MAX_ERROR_BARS = 200


def _error_bar_positions(x_values, cap=MAX_ERROR_BARS):
    """Row positions to draw bars for: every row, or an even spread across the x range.

    Evenly spaced through the x-SORTED order rather than a random sample, so the drawn bars span the
    whole range of the axis instead of clustering wherever the data is dense. Deterministic, so the
    same run always produces the same picture.
    """
    import numpy as np

    values = np.asarray(x_values, dtype=float)
    if values.size <= cap:
        return np.arange(values.size)
    order = np.argsort(values)
    return np.sort(order[np.linspace(0, values.size - 1, cap).astype(int)])


# How much of the interval extent the predicted-vs-observed panel is allowed to ignore, per end.
# The bars have to be reachable for their caps to be visible, but framing on their true extent is
# not an option: on a linear model the mean interval already exceeds the target's own range and the
# widest is several times it, so the scatter would collapse into a band. Clipping the outer 2% at
# each end puts most caps on-screen and leaves a handful running off the edge, which is the honest
# reading for a point the model is genuinely unsure about.
INTERVAL_CLIP_PERCENTILE = 2.0

# Hard ceiling on how far past the data the axis may stretch to accommodate bars, as a fraction of
# the data's own span, per end. The percentile clip alone is not enough: sigma has a long right
# tail, so on a real run the 98th percentile of the bar ends still sat about twice the target range
# away and squeezed the scatter into the middle fifth of the panel. This bounds the compression
# directly - the points always keep at least ~1/(1 + 2*0.35) of the axis.
MAX_INTERVAL_EXTENSION = 0.35

# The error bars and their caps. Red because grey at low alpha was invisible against the scatter.
ERROR_BAR_COLOR = "#d62728"
ERROR_BAR_LINE_ALPHA = 0.35
ERROR_BAR_CAP_ALPHA = 0.85


def _square_limits(observed, predicted, interval=None, percentile=INTERVAL_CLIP_PERCENTILE, margin=0.05):
    """One ``(low, high)`` range for BOTH axes of a predicted-vs-observed panel.

    A pred-vs-obs scatter is only readable when the identity line is a true 45 degree diagonal, and
    that needs the two axes to share a range as well as an aspect - otherwise the cloud is stretched
    along whichever axis happens to span less.

    ``interval``, when given, widens the range toward the bar ends, but by a PERCENTILE rather than
    by their min and max. That distinction is the whole point of this function: a single very
    uncertain point has an interval several times the target's range, and letting it set the limits
    is exactly the blow-out that framing on the data alone was introduced to avoid.
    """
    import numpy as np

    candidates = [np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)]
    finite = np.concatenate([values[np.isfinite(values)] for values in candidates])
    if finite.size == 0:
        return 0.0, 1.0

    low, high = _extend_range(
        float(finite.min()), float(finite.max()), interval, percentile=percentile
    )
    pad = (high - low) * margin or 1.0
    return low - pad, high + pad


def _extend_range(low, high, interval, percentile=INTERVAL_CLIP_PERCENTILE):
    """Widen ``(low, high)`` toward the interval ends, under two independent limits.

    The percentile drops the few pathological bars. The extension ceiling handles the case the
    percentile cannot - a heavy-tailed sigma, where even the 98th percentile is far enough out to
    squash the data into the middle of the panel. Whichever binds first wins.

    Shared by the predicted-vs-observed panel and the residual panel so their bars are clipped by
    the same rule; without it one panel shows its caps and the other does not.
    """
    import numpy as np

    if interval is None:
        return low, high

    headroom = MAX_INTERVAL_EXTENSION * ((high - low) or 1.0)
    lower, upper = (np.asarray(bound, dtype=float) for bound in interval)
    if np.isfinite(lower).any():
        clipped = float(np.percentile(lower[np.isfinite(lower)], percentile))
        low = min(low, max(clipped, low - headroom))
    if np.isfinite(upper).any():
        clipped = float(np.percentile(upper[np.isfinite(upper)], 100.0 - percentile))
        high = max(high, min(clipped, high + headroom))
    return low, high


def _frame_square(ax, low, high):
    """Give an axes one shared range and a 1:1 aspect, so its diagonal is a real diagonal.

    ``adjustable="box"`` reshapes the axes box rather than the data limits, which is what keeps the
    range exactly as asked. Call this AFTER every plotting call on the panel - seaborn autoscales on
    draw and would otherwise overwrite the limits set here.
    """
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")


def _draw_error_bars(ax, x_values, y_values, lower, upper, positions):
    """Vertical prediction intervals with visible ends.

    The verticals and the caps are styled SEPARATELY, which a single ``alpha=`` on the errorbar call
    cannot do - it fades both by the same amount, and the setting that makes a few hundred
    overlapping verticals bearable is far too faint for the caps that mark where each interval
    actually stops. Faint red lines, solid red caps.

    ``yerr`` takes the two half-widths rather than half of ``upper - lower``: a conformal interval is
    only symmetric when its calibrator is, and halving the width would bake in an assumption that
    need not hold.
    """
    import numpy as np

    _plotline, caplines, barlinecols = ax.errorbar(
        np.asarray(x_values)[positions],
        np.asarray(y_values)[positions],
        yerr=[
            np.maximum(np.asarray(y_values - lower)[positions], 0.0),
            np.maximum(np.asarray(upper - y_values)[positions], 0.0),
        ],
        fmt="none",
        ecolor=ERROR_BAR_COLOR,
        elinewidth=0.6,
        capsize=3,
        zorder=1,
    )
    for bar in barlinecols:
        bar.set_alpha(ERROR_BAR_LINE_ALPHA)
    for cap in caplines:
        cap.set_alpha(ERROR_BAR_CAP_ALPHA)
        cap.set_markeredgewidth(1.2)
        cap.set_color(ERROR_BAR_COLOR)
    return caplines, barlinecols


def _frame_on_data(ax, x_values, y_values, x_margin=0.05, y_margin=0.05, y_interval=None):
    """Set the axis limits from the POINTS, ignoring how far the error bars reach.

    Matplotlib autoscales to include every artist, so one point with a very wide interval decides
    the whole y-axis and squashes the scatter into an unreadable band. Framing on the data keeps the
    picture about the predictions; the long bars run off the edge, which is the right reading.

    Used by the RESIDUAL panel, whose axes are predicted against residual - two different
    quantities, so it takes an independent range per axis and no aspect. The predicted-vs-observed
    panel uses _square_limits and _frame_square instead.

    ``y_interval`` widens the y range toward the bar ends under the same bounded rule the square
    panel uses. Without it the residual axis stays tight on the residuals while the bars are as wide
    as the whole target range, so every bar spans the full panel and not one cap is on-screen.
    """
    import numpy as np

    for setter, values, margin, interval in (
        (ax.set_xlim, np.asarray(x_values, dtype=float), x_margin, None),
        (ax.set_ylim, np.asarray(y_values, dtype=float), y_margin, y_interval),
    ):
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        low, high = _extend_range(float(finite.min()), float(finite.max()), interval)
        pad = (high - low) * margin or 1.0
        setter(low - pad, high + pad)


def create_pred_obs_plot(eval_df, builtin_metrics, artifacts_dir):
    """
    Create a 3- or 4-panel plot:
      1. Scatter plot of Predicted vs Actual (with uncertainty bars when the frame has them)
      2. Residuals vs Predicted
      3. KDE density of Predicted vs Actual
      4. Reliability curve — only when the frame carries a calibrated interval

    The uncertainty columns are OPTIONAL. Most frames reaching this function come from runs with
    uncertainty disabled and must render exactly as they always have, so every addition below is
    guarded on the column being present rather than on a flag the caller would have to pass - which
    also keeps the signature that MLflow's custom-artifact contract depends on.

    Args:
        eval_df (DataFrame): must carry `target` and `prediction`; may carry `prediction_std`
            and the `prediction_lower`/`prediction_upper` pair.
        builtin_metrics (dict): supplied by MLflow's evaluator; unused.
        artifacts_dir (str): directory this function SAVES into.

    Returns:
        dict: {artifact_name: path}, MLflow's custom-artifact contract.
    """
    y_test = eval_df["target"]
    y_pred = eval_df["prediction"]
    residuals = y_test - y_pred

    interval = interval_columns(eval_df)
    sigma = sigma_column(eval_df)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel 1: Predicted vs Actual ---
    ax = axes[0]
    bar_positions = _error_bar_positions(y_test) if interval is not None else None
    if interval is not None:
        # Drawn FIRST and at zorder 1 so the bars sit under the points: bars painted on top hide the
        # very structure the plot exists to show.
        _draw_error_bars(ax, y_test, y_pred, interval[0], interval[1], bar_positions)
    sns.regplot(x=y_test, y=y_pred, ax=ax,
                scatter_kws={'alpha': 0.6, 'edgecolor': 'k', 'zorder': 3},
                line_kws={'color': 'blue'}
                )
    if sigma is not None:
        # Colour by sigma over the bars. At this point count the bars overlap and stop being
        # readable per point, while the colour survives - so "where is this model uncertain?" stays
        # answerable from the picture rather than only from the CSV.
        scatter = ax.scatter(
            y_test, y_pred, c=sigma, cmap="viridis", s=22, alpha=0.85,
            edgecolor="k", linewidth=0.3, zorder=4,
        )
        # An INSET axes, not `fig.colorbar(..., ax=ax)`. The `ax=` form makes room for the colorbar
        # by shrinking the axes it is given, which left this panel about 9% narrower than the two
        # beside it - one panel in a three-panel strip visibly out of proportion. inset_axes
        # positions in axes-fraction coordinates and leaves the box alone, so the panel keeps the
        # width and the 1:1 aspect set below.
        colorbar = fig.colorbar(scatter, cax=ax.inset_axes([1.02, 0.0, 0.035, 1.0]))
        colorbar.set_label("Predictive σ")

    # One shared range for both axes, so the identity line below is a true 45 degree diagonal and
    # the cloud is not stretched along whichever axis spans less. Applied whether or not the run
    # carried uncertainty: the panel had never set an aspect, and a pred-vs-obs scatter that is not
    # square misreads at a glance.
    low, high = _square_limits(y_test, y_pred, interval)

    # Identity line across the WHOLE panel rather than the observed range, so it runs corner to
    # corner instead of stopping short inside a wider frame.
    ax.plot([low, high], [low, high], 'r--', lw=2, zorder=5)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{y_test.name}\nPredicted vs Actual")
    ax.grid(True)
    # After every plotting call: seaborn autoscales on draw and would overwrite these limits.
    _frame_square(ax, low, high)

    rmse = root_mean_squared_error(y_test, y_pred)
    annotation = f"RMSE={rmse:.2f}\nR²={r2_score(y_test, y_pred):.2f}"
    if rmse > 0.0:
        # Skipped at rmse == 0 rather than divided anyway. RPIQ and RPD are ratios with rmse in the
        # denominator, so a model that fits its test split exactly - a degenerate estimator, or a
        # target that leaked into the features - makes them infinite. This is the same policy
        # regression_metrics applies; without it the annotation raises a divide-by-zero and takes
        # the whole plot, and the artifact logging around it, down with it.
        #
        # (predictions, targets), in that order: the IQR in the numerator is read off the
        # SECOND argument. Passing (y_test, y_pred) measures the spread of the predictions,
        # which under-reports RPIQ because predictions are systematically under-dispersed.
        annotation += f"\nRPIQ={rpiq_score(y_pred, y_test):.2f}"
    if interval is not None:
        lower, upper = interval
        covered = float(np.mean((y_test >= lower) & (y_test <= upper)))
        # The coverage is annotated beside the bars ON PURPOSE. A prediction interval drawn without
        # the fraction it actually captured is a decoration; printed together, the picture states a
        # claim and the number checks it.
        # Computed over EVERY point even when only a subset is drawn, so the number never describes
        # a different population from the metric of the same name in the run.
        annotation += (
            f"\nPICP={covered:.3f}"
            f"\nMPIW={float(np.mean(upper - lower)):.2f}"
        )
        # What KIND of bar this is, when the caller said. The same picture means different things
        # under conformal, gaussian and sigma - a reader cannot tell them apart by looking, and the
        # PICP beside it is only interpretable once you know which claim is being made. Carried on
        # the frame's `attrs` because MLflow's custom-artifact contract fixes this signature.
        label = eval_df.attrs.get("interval_label") if hasattr(eval_df, "attrs") else None
        if label:
            annotation += f"\nbar: {label}"
        if bar_positions is not None and len(bar_positions) < len(y_test):
            annotation += f"\nbars: {len(bar_positions)} of {len(y_test)}"
    ax.text(0.05, 0.95, annotation,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    # --- Panel 2: Residuals vs Predicted ---
    ax = axes[1]
    if interval is not None:
        lower, upper = interval
        # Subsampled on the PREDICTED axis, which is this panel's x, so the bars spread across it
        # rather than inheriting a selection made for a different axis. The interval is centred on
        # the prediction, so around a residual it becomes `residual -+ the same half-widths`.
        _draw_error_bars(
            ax,
            y_pred,
            residuals,
            residuals - (y_pred - lower),
            residuals + (upper - y_pred),
            _error_bar_positions(y_pred),
        )
    sns.scatterplot(
        x=y_pred,
        y=residuals,
        ax=ax,
        alpha=0.6,
        edgecolor='k',
        zorder=3,
    )
    ax.axhline(0, color="red", linestyle="--", lw=2)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title(f"{y_test.name}\nResiduals vs Predicted")
    ax.grid(True)
    ax.set_axisbelow(True)
    if interval is not None:
        # The y range reaches toward the bar ends here too, so the caps are on-screen for most
        # points. That turns the panel into a coverage picture: a bar that does not reach the zero
        # line is a point the interval MISSED, which is the ~5% the PICP beside it is counting.
        lower, upper = interval
        _frame_on_data(
            ax,
            y_pred,
            residuals,
            x_margin=0.05,
            y_interval=(residuals - (y_pred - lower), residuals + (upper - y_pred)),
        )
    # --- Panel 3: KDE density plot ---
    ax = axes[2]
    # np.isclose, not `!= 1`. A perfectly-fitting model gives a correlation of 0.9999999999999998
    # rather than exactly 1, which slips past an equality test - and then seaborn cannot estimate a
    # KDE over a degenerate covariance and warns, which under this project's warnings-as-errors
    # setting takes the plot down. The guard is meant to catch that case; it just has to do it with
    # a tolerance.
    correlation = np.corrcoef(y_test, y_pred)[0, 1]
    can_estimate_density = (
        np.std(y_test) > 0
        and np.std(y_pred) > 0
        and np.isfinite(correlation)
        and not np.isclose(abs(correlation), 1.0)
    )
    if can_estimate_density:
        sns.kdeplot(x=y_test, y=y_pred, fill=True, cmap="gnuplot2", thresh=0.005, levels=200, ax=ax)
    else:
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{y_test.name}\nDensity KDE of Pred vs Actual")
    ax.grid(True)

    # No reliability panel here on purpose. Grading the interval needs the conformal calibrator, and
    # this function is handed a frame and nothing else - MLflow's custom-artifact contract fixes its
    # signature. A version that swept Gaussian z-multiples of the RAW sigma instead produced a curve
    # far below the diagonal sitting next to an annotation reporting PICP=0.96, because those two
    # grade different things. The reliability curve lives in uncertainty/reliability.png, which is
    # written by log_uncertainty_artifacts and does have the calibrator.

    # Closed in a finally, and by identity rather than `plt.close()`'s "whatever is current". This
    # function is called once per target inside a training loop, so a figure left open on an error
    # path accumulates until matplotlib warns about it - and this suite turns warnings into errors,
    # which turns one bad plot into a failed run several targets later.
    try:
        fig.tight_layout()
        plot_path = os.path.join(artifacts_dir, "obs_pred_and_residual_plot.png")
        fig.savefig(plot_path, bbox_inches="tight", dpi=100)
    finally:
        plt.close(fig)
    return {"obs_pred_and_residual_plot": plot_path}

def create_parent_pred_obs(eval_dfs):
    return _create_parent_pred_obs_multitarget(eval_dfs)

def plot_leaderboard_scatter(leaderboard_df, metric_x="rmse_test", metric_y="r2_test",
                                        label_col="model", hue_col="target"):
    """
    Create a scatter subplot for each target showing model performance,
    with average RMSE and R² lines per target.
    """
    if leaderboard_df is None or leaderboard_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No leaderboard rows available", ha="center", va="center")
        ax.axis("off")
        return fig

    if metric_x not in leaderboard_df.columns or metric_y not in leaderboard_df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            f"Skipping leaderboard scatter: missing metric columns '{metric_x}' or '{metric_y}'",
            ha="center",
            va="center",
        )
        ax.axis("off")
        return fig

    # Defensive checks
    if leaderboard_df is None or len(leaderboard_df) == 0:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.text(0.5, 0.5, "No leaderboard data available", ha='center', va='center')
        ax.axis('off')
        return fig

    # Fallback if expected hue column missing
    if hue_col not in leaderboard_df.columns:
        # Try common alternate names
        if 'target_name' in leaderboard_df.columns:
            hue_col = 'target_name'
        else:
            # Create a pseudo target column
            hue_col = '_target_tmp_'
            leaderboard_df = leaderboard_df.copy()
            leaderboard_df[hue_col] = 'All'

    if label_col not in leaderboard_df.columns:
        # Try alternative naming
        if 'model_name' in leaderboard_df.columns:
            label_col = 'model_name'
        else:
            label_col = '_model_tmp_'
            leaderboard_df = leaderboard_df.copy()
            leaderboard_df[label_col] = range(len(leaderboard_df))

    targets = leaderboard_df[hue_col].dropna().unique()
    models = leaderboard_df[label_col].dropna().unique()
    if len(models) == 0:
        models = ['models']
    n_targets = max(1, len(targets))
    n_cols = max(1, len(models))
    n_rows = int(np.ceil(n_targets / n_cols)) or 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, target in zip(axes, targets):
        df_target = leaderboard_df[leaderboard_df[hue_col] == target]

        # Compute averages for this target
        avg_rmse = df_target[metric_x].mean()
        avg_r2 = df_target[metric_y].mean()

        sns.scatterplot(data=df_target, x=metric_x,y=metric_y,
                        hue=hue_col, s=100, ax=ax,legend=False
        )

        # Annotate points
        for _, row in df_target.iterrows():
            ax.text(row[metric_x], row[metric_y], str(row[label_col]),
                    horizontalalignment='left', size='small', color='black', weight='normal')

        # Add average lines per target
        ax.axvline(avg_rmse, color="blue", linestyle="--", label="Avg RMSE")
        ax.axhline(avg_r2, color="red", linestyle="--", label="Avg R²")

        ax.set_title(f"Target: {target}")
        ax.set_xlabel(metric_x.upper())
        ax.set_ylabel(metric_y.upper())

    # Remove empty subplots
    for j in range(len(targets), len(axes)):
        if j < len(axes):
            fig.delaxes(axes[j])

    plt.suptitle("Leaderboard: R² vs RMSE per Target", fontsize=16)
    plt.tight_layout()
    return fig