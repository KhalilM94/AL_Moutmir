import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from .plotters import prepare_group_labels, prepare_numeric_groups

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
            
            if group_codes is not None:
                ax.scatter(y_test_clean, y_pred_clean, c=group_codes, cmap=cmap,
                                     alpha=0.7, edgecolor='k', s=40)
                handles = [
                    plt.Line2D([0], [0], marker='o', color='w',
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