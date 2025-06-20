import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score, mean_squared_error

def fuzzy_map_suffix(suffix, label_map):
    """Match suffix to human label by substring matching any key."""
    for key, label in label_map.items():
        if key in suffix:
            return label
    return suffix  # fallback to raw suffix if no match
# Function to prepare group labels from dummy-coded columns
def prepare_group_labels(X, valid_mask, prefix, label_map):
    """Extract dummy-encoded group labels and map to human-readable values."""
    dummy_cols = [col for col in X.columns if col.startswith(prefix)]
    if not dummy_cols:
        return None, None, None

    group_dummies = X.loc[valid_mask, dummy_cols]
    suffixes = group_dummies.idxmax(axis=1).str.replace(prefix, '', regex=False)
    labels = suffixes.map(lambda s: fuzzy_map_suffix(s, label_map))
    cat = pd.Categorical(labels)
    
    # Use get_cmap properly and avoid deprecated usage
    cmap = plt.cm.viridis  # directly use the colormap (viridis is a default colormap in matplotlib)
    discrete_colors = cmap(np.linspace(0, 1, len(cat.categories)))  # Generate discrete colors
    color_map = ListedColormap(discrete_colors)
    
    return cat.codes, cat.categories, color_map

def prepare_numeric_groups(X, valid_mask, numeric_column, n_bins=5, cmap_name="viridis"):
    """Discretize a numeric column and return group codes, labels, and colormap."""
    values = X.loc[valid_mask, numeric_column]
    bins = pd.qcut(values, q=n_bins, duplicates='drop')  # quantile-based bins
    labels = bins.astype(str)
    cat = pd.Categorical(labels)

    cmap = plt.colormaps[cmap_name]
    discrete_colors = cmap(np.linspace(0, 1, len(cat.categories)))
    color_map = ListedColormap(discrete_colors)

    return cat.codes, cat.categories, color_map

def plot_observed_vs_predicted(
    X_test,
    y_test_dict,
    model_pipelines,
    target_columns,
    columns_to_transform,
    model_dir="final_models",
    sup_title="Test set Observed vs Predicted",
    group_prefix=None,
    group_label_map=None,
    group_numeric_column=None,
    log_transformer=None
):

    model_names = list(model_pipelines.keys())
    n_targets = len(target_columns)
    n_models = len(model_names)

    # Create the plot grid
    fig, axes = plt.subplots(n_targets, n_models, figsize=(5 * n_models, 5 * n_targets))
    fig.suptitle(sup_title, fontsize=18)
    axes = np.atleast_2d(axes)

    # Loop through targets and models to generate the scatter plots
    for i, target in enumerate(target_columns):
        y_test_full = y_test_dict[target]
        is_log = target in columns_to_transform

        for j, model_name in enumerate(model_names):
            ax = axes[i, j]
            model_file = os.path.join(model_dir, f"{target.replace('/', '_')}_{model_name}.pkl")

            # Skip missing models
            if not os.path.exists(model_file):
                ax.set_title(f"{target} - {model_name} (Missing)")
                ax.axis("off")
                continue

            model = joblib.load(model_file)
            y_pred_raw = model.predict(X_test)

            # Align actual and predicted values
            y_test = pd.Series(y_test_full, index=X_test.index)
            y_pred = pd.Series(y_pred_raw, index=X_test.index)

            # Inverse transform for log-transformed targets
            if is_log and log_transformer:
                y_pred = log_transformer.inverse_transform(y_pred)

            # Filter out NaNs
            valid_mask = y_test.notna() & y_pred.notna()
            y_test_clean = y_test.loc[valid_mask]
            y_pred_clean = y_pred.loc[valid_mask]

            # Group label coloring (optional)
            group_codes, group_labels, cmap = None, None, None
            if group_prefix and group_label_map:
                group_codes, group_labels, cmap = prepare_group_labels(X_test, valid_mask, group_prefix, group_label_map)
            elif group_numeric_column:
                group_codes, group_labels, cmap = prepare_numeric_groups(X_test, valid_mask, group_numeric_column)
            
            if group_codes is not None and group_labels is not None and cmap is not None:
                ax.scatter(y_test_clean, y_pred_clean, c=group_codes, cmap=cmap,
                                     alpha=0.7, edgecolor='k', s=40)
                handles = [
                    Line2D([0], [0], marker='o', color='w',
                           label=label,
                           markerfacecolor=cmap(i),
                           markeredgecolor='k',
                           markersize=6)
                    for i, label in enumerate(group_labels)
                ]
                ax.legend(handles=handles, title="Group", loc="lower right", fontsize=8)
            else:
                ax.scatter(y_test_clean, y_pred_clean, alpha=0.5, s=40)

            # Identity line (1:1 line) and formatting
            min_val = min(y_test_clean.min(), y_pred_clean.min())
            max_val = max(y_test_clean.max(), y_pred_clean.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
            ax.set_title(f"{target} - {model_name}")
            ax.set_xlabel("Observed")
            ax.set_ylabel("Predicted")
            ax.set_aspect('equal', 'box')

            # Plot regression line
            coef = np.polyfit(y_test_clean, y_pred_clean, 1)
            reg_line = np.poly1d(coef)
            x_vals = np.linspace(min_val, max_val, 100)
            ax.plot(x_vals, reg_line(x_vals), 'k--', lw=1)

            # Annotate metrics
            r2 = r2_score(y_test_clean, y_pred_clean)
            rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred_clean))
            bias = np.mean(y_pred_clean - y_test_clean)

            ax.text(0.05, 0.95,
                    f"R²: {r2:.2f}\nRMSE: {rmse:.2f}\nBias: {bias:.2f}",
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # Tighten layout and display plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return plt

def plot_cv_folds_observed_vs_predicted(
    fold_preds,
    target,
    sup_title="CV Folds Observed vs Predicted",
    group_prefix=None,
    group_label_map=None,
    group_numeric_column=None,
    columns_to_transform=None,
    n_bins=5,
    cmap_name="viridis"
):
    """
    fold_preds: list of dicts with keys 'fold', 'y_val', 'y_pred', 'target', 'model', and optionally 'val_groups', 'X_val'
    Plots a grid: rows = models, columns = folds. Each subplot is a scatter for a model/fold.
    Supports group coloring (no log-transform applied).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from collections import defaultdict
    # Helper functions from this module
    from .plot_utils import prepare_group_labels, prepare_numeric_groups

    model_groups = defaultdict(list)
    for d in fold_preds:
        model_groups[d['model']].append(d)
    model_names = list(model_groups.keys())
    n_models = len(model_names)
    n_folds = max(len(v) for v in model_groups.values()) if n_models > 0 else 1
    fig, axes = plt.subplots(n_models, n_folds, figsize=(5 * n_folds, 5 * n_models))
    if n_models == 1:
        axes = np.atleast_2d(axes)
    elif n_folds == 1:
        axes = np.atleast_2d(axes).T
    fig.suptitle(f"{sup_title}\nTarget: {target}", fontsize=18)
    for i, model_name in enumerate(model_names):
        model_folds = sorted(model_groups[model_name], key=lambda d: d['fold'])
        for j, fold_pred in enumerate(model_folds):
            ax = axes[i, j]
            y_val = fold_pred['y_val']
            y_pred = fold_pred['y_pred']
            X_val = fold_pred.get('X_val', None)
            val_groups = fold_pred.get('val_groups', None)
            # Convert to numpy arrays for safety
            y_val = np.array(y_val)
            y_pred = np.array(y_pred)
            # No log-transform applied here
            # Filter out NaNs
            valid_mask = (~pd.isna(y_val)) & (~pd.isna(y_pred))
            y_val_clean = y_val[valid_mask]
            y_pred_clean = y_pred[valid_mask]
            # Group label coloring (optional)
            group_codes, group_labels, cmap = None, None, None
            if X_val is not None and valid_mask.sum() > 0:
                X_val_df = pd.DataFrame(X_val).reset_index(drop=True)
                valid_idx = np.where(valid_mask)[0]
                X_val_valid = X_val_df.iloc[valid_idx]
                if group_prefix and group_label_map:
                    # Use a mask that selects all rows in X_val_valid
                    group_codes, group_labels, cmap = prepare_group_labels(X_val_valid, np.ones(len(X_val_valid), dtype=bool), group_prefix, group_label_map)
                elif group_numeric_column and group_numeric_column in X_val_valid.columns:
                    group_codes, group_labels, cmap = prepare_numeric_groups(X_val_valid, X_val_valid.index, group_numeric_column, n_bins=n_bins, cmap_name=cmap_name)
            if len(y_val_clean) == 0 or len(y_pred_clean) == 0:
                ax.set_title(f"{model_name} - Fold {fold_pred['fold']}: No valid data")
                ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=12)
                ax.axis('off')
                continue
            if group_codes is not None and group_labels is not None and cmap is not None:
                ax.scatter(y_val_clean, y_pred_clean, c=group_codes, cmap=cmap, alpha=0.7, edgecolor='k', s=40)
                handles = [
                    Line2D([0], [0], marker='o', color='w',
                           label=label,
                           markerfacecolor=cmap(i),
                           markeredgecolor='k',
                           markersize=6)
                    for i, label in enumerate(group_labels)
                ]
                ax.legend(handles=handles, title="Group", loc="lower right", fontsize=8)
            else:
                ax.scatter(y_val_clean, y_pred_clean, alpha=0.5, s=40)
            min_val = min(np.min(y_val_clean), np.min(y_pred_clean))
            max_val = max(np.max(y_val_clean), np.max(y_pred_clean))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
            # Title with model, fold, and val groups
            if val_groups is not None:
                val_groups_str = ', '.join(str(int(g)) if isinstance(g, (int, float)) and float(g).is_integer() else str(g) for g in val_groups)
                ax.set_title(f"{model_name} - Fold {fold_pred['fold']}: Val set ({val_groups_str})")
            else:
                ax.set_title(f"{model_name} - Fold {fold_pred['fold']}")
            ax.set_xlabel("Observed")
            ax.set_ylabel("Predicted")
            ax.set_aspect('equal', 'box')
            # Regression line
            if len(y_val_clean) > 1 and len(y_pred_clean) > 1:
                coef = np.polyfit(y_val_clean, y_pred_clean, 1)
                reg_line = np.poly1d(coef)
                x_vals = np.linspace(min_val, max_val, 100)
                ax.plot(x_vals, reg_line(x_vals), 'k--', lw=1)
            # Metrics
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(y_val_clean, y_pred_clean)
            rmse = np.sqrt(mean_squared_error(y_val_clean, y_pred_clean))
            bias = np.mean(y_pred_clean - y_val_clean)
            ax.text(0.05, 0.95, f"R²: {r2:.2f}\nRMSE: {rmse:.2f}\nBias: {bias:.2f}",
                    transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    # Hide unused axes
    for i in range(n_models):
        for j in range(len(model_groups[model_names[i]]), n_folds):
            axes[i, j].axis('off')
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig