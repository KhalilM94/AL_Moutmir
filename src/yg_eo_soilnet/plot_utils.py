from yg_eo_soilnet.utils import rpiq_score
from sklearn.metrics import r2_score, root_mean_squared_error
import numpy as np
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

    prediction_columns = [column_name for column_name in df.columns if column_name.startswith("prediction_")]
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
                numeric_candidates = [column_name for column_name in frame.columns if column_name not in {"model_name", "target_name"}]
                if numeric_candidates:
                    observed_column = numeric_candidates[0]
                else:
                    continue

            x_values = frame[observed_column].to_numpy()
            y_values = frame[prediction_column].to_numpy()
            valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
            if not valid_mask.any():
                continue

            model_name = frame["model_name"].iloc[0] if "model_name" in frame.columns and not frame["model_name"].empty else "model"
            axis.scatter(x_values[valid_mask], y_values[valid_mask], alpha=0.5, s=18, label=str(model_name))
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

def create_pred_obs_plot(eval_df, builtin_metrics, artifacts_dir):
    """
    Create a 2-panel plot:
      1. Scatter plot of Predicted vs Actual
      2. Residuals vs Predicted

    Args:
        y_pred (array-like): Predicted values.
        y_test (array-like): True values.
        model_name (str): Name of the model (for title).

    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    y_test = eval_df["target"]
    y_pred = eval_df["prediction"]
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel 1: Predicted vs Actual ---
    ax = axes[0]
    sns.regplot(x=y_test, y=y_pred, ax=ax, 
                scatter_kws={'alpha': 0.6, 'edgecolor': 'k'},
                line_kws={'color': 'blue'}
                )
    # Identity line
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],'r--', lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{y_test.name}\nPredicted vs Actual")
    ax.grid(True)

    ax.text(0.05, 0.95,
            f"RMSE={root_mean_squared_error(y_test, y_pred):.2f}\n"
            f"R²={r2_score(y_test, y_pred):.2f}\n"
            f"RPIQ={rpiq_score(y_test, y_pred):.2f}",
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    # --- Panel 2: Residuals vs Predicted ---
    ax = axes[1]
    sns.scatterplot(
        x=y_pred,
        y=residuals,
        ax=ax,
        alpha=0.6,
        edgecolor='k'
    )
    ax.axhline(0, color="red", linestyle="--", lw=2)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title(f"{y_test.name}\nResiduals vs Predicted")
    ax.grid(True)
    ax.set_axisbelow(True)
    # --- Panel 3: KDE density plot ---
    ax = axes[2]
    if np.std(y_test) > 0 and np.std(y_pred) > 0 and np.corrcoef(y_test, y_pred)[0, 1] != 1:
        sns.kdeplot(x=y_test, y=y_pred, fill=True, cmap="gnuplot2", thresh=0.005, levels=200, ax=ax)
    else:
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{y_test.name}\nDensity KDE of Pred vs Actual")
    ax.grid(True)

    fig.tight_layout()
    plot_path = os.path.join(artifacts_dir, "obs_pred_and_residual_plot.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=100)
    plt.close()
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