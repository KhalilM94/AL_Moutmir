from collections import defaultdict
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import contextily as ctx
import geopandas as gpd
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from collections import defaultdict
from matplotlib.colors import ListedColormap, Normalize
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
    dummy_column_names = [col for col in X.columns if col.startswith(prefix)]
    if not dummy_column_names:
        return None, None, None

    dummy_columns = X.loc[valid_mask, dummy_column_names]
    suffixes = dummy_columns.idxmax(axis=1).str.replace(prefix, '', regex=False)
    labels = suffixes.map(lambda s: fuzzy_map_suffix(s, label_map))
    cat = pd.Categorical(labels)
    
    # Use get_cmap properly and avoid deprecated usage
    cmap = plt.get_cmap('viridis')  # directly use the colormap (viridis is a default colormap in matplotlib)
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
    # Ensure axes is always 2D: shape (n_targets, n_models)
    if n_targets == 1 and n_models == 1:
        axes = np.array([[axes]])
    elif n_targets == 1:
        axes = axes[np.newaxis, :]
    elif n_models == 1:
        axes = axes[:, np.newaxis]

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
            y_pred_full = model.predict(X_test)

            # Align actual and predicted values
            y_test = pd.Series(y_test_full, index=X_test.index)
            y_pred = pd.Series(y_pred_full, index=X_test.index)

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
                ax.legend(handles=handles, title= group_prefix, loc="lower right", fontsize=8)
            else:
                ax.scatter(y_test_clean, y_pred_clean, alpha=0.7, s=40)

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
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return plt

def _get_global_min_max_cv_folds(fold_preds, 
                                 target, 
                                 columns_to_transform=None, 
                                 log_transformer=None, 
                                 model_dir="final_models"):
    """
    Compute global min and max for observed and predicted values across all folds and models.
    """
    all_y_vals = []
    all_y_preds = []
    for fold_pred in fold_preds:
        model_file = Path(model_dir) / f"{fold_pred['target'].replace('/', '_')}_{fold_pred['model']}_fold_{fold_pred['fold']}.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file {model_file} not found.")
        model = joblib.load(model_file)
        X_val = fold_pred['X_val']
        y_val = np.array(fold_pred['y_val'])
        y_pred = np.array(model.predict(X_val))
        is_log = columns_to_transform and target in columns_to_transform
        if is_log and log_transformer is not None:
            y_pred = log_transformer.inverse_transform(y_pred)
        valid_mask = (~pd.isna(y_val)) & (~pd.isna(y_pred))
        all_y_vals.append(y_val[valid_mask])
        all_y_preds.append(y_pred[valid_mask])
    if all_y_vals and all_y_preds:
        global_min = min(np.min(np.concatenate(all_y_vals)), np.min(np.concatenate(all_y_preds)))
        global_max = max(np.max(np.concatenate(all_y_vals)), np.max(np.concatenate(all_y_preds)))
    else:
        global_min, global_max = 0, 1
    return global_min, global_max


def _prepare_global_group_labels(fold_preds, group_prefix=None, group_label_map=None, group_numeric_column=None, n_bins=5, cmap_name="viridis"):
    """
    Compute global group labels and color map dict for all folds, for either dummy-coded or numeric group columns.
    Returns (global_group_labels, global_color_map_dict)
    """
    global_group_labels = None
    global_color_map_dict = None
    if group_prefix and group_label_map:
        all_labels = set()
        for d in fold_preds:
            X_val_df = pd.DataFrame(d['X_val'])
            dummy_cols = [col for col in X_val_df.columns if col.startswith(group_prefix)]
            if dummy_cols:
                suffixes = X_val_df[dummy_cols].idxmax(axis=1).str.replace(group_prefix, '', regex=False)
                labels = suffixes.map(lambda s: fuzzy_map_suffix(s, group_label_map))
                all_labels.update(labels.dropna().unique())
        if all_labels:
            global_group_labels = sorted(list(all_labels))
            cmap_obj = plt.cm.get_cmap(cmap_name, len(global_group_labels))
            global_color_map_dict = {label: cmap_obj(i) for i, label in enumerate(global_group_labels)}
    elif group_numeric_column:
        all_values = pd.concat([
            pd.DataFrame(d['X_val'])[group_numeric_column]
            for d in fold_preds if group_numeric_column in pd.DataFrame(d['X_val']).columns
        ]).dropna()
        if not all_values.empty:
            if isinstance(all_values, pd.DataFrame):
                all_values = all_values.iloc[:, 0]
            bins = pd.qcut(all_values, q=n_bins, duplicates='drop', labels=False, retbins=True)[1]
            bin_labels = [f'({bins[i]:.2f}, {bins[i+1]:.2f}]' for i in range(len(bins)-1)]
            global_group_labels = sorted(bin_labels)
            cmap_obj = plt.cm.get_cmap(cmap_name, len(global_group_labels))
            global_color_map_dict = {label: cmap_obj(i) for i, label in enumerate(global_group_labels)}
    return global_group_labels, global_color_map_dict

def plot_cv_folds_observed_vs_predicted(
    fold_preds,
    target,
    sup_title="CV Folds Observed vs Predicted",
    group_prefix=None,
    group_label_map=None,
    group_numeric_column=None,
    columns_to_transform=None,
    n_bins=5,
    cmap_name="viridis",
    log_transformer=None,
    model_dir="final_models"
):
    """
    fold_preds: list of dicts with keys 'fold', 'X_val', 'target', 'model', and optionally 'val_groups'.
    Plots a grid: rows = models, columns = folds. Each subplot is a scatter for a model/fold.
    Supports group coloring and log-transform inversion.
    """
    # --- Global group and color mapping ---
    global_group_labels, global_color_map_dict = _prepare_global_group_labels(
        fold_preds,
        group_prefix=group_prefix,
        group_label_map=group_label_map,
        group_numeric_column=group_numeric_column,
        n_bins=n_bins,
        cmap_name=cmap_name
    )
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

    # --- Compute global min/max for all folds for consistent axis scaling ---
    global_min, global_max = _get_global_min_max_cv_folds(
        fold_preds, target, columns_to_transform=columns_to_transform, log_transformer=log_transformer, model_dir=model_dir
    )

    for i, model_name in enumerate(model_names):
        model_folds = sorted(model_groups[model_name], key=lambda d: d['fold'])
        for j, fold_pred in enumerate(model_folds):
            ax = axes[i, j]
            model_file = Path(model_dir) / f"{fold_pred['target']}_{fold_pred['model']}_fold_{fold_pred['fold']}.pkl"
            model = joblib.load(model_file)
            X_val = fold_pred['X_val']
            y_val = np.array(fold_pred['y_val'])
            y_pred = np.array(model.predict(X_val))
            is_log = columns_to_transform and target in columns_to_transform
            if is_log and log_transformer is not None:
                y_pred = log_transformer.inverse_transform(y_pred)
            valid_mask = (~pd.isna(y_val)) & (~pd.isna(y_pred))
            y_val_clean = y_val[valid_mask]
            y_pred_clean = y_pred[valid_mask]
            
            # Group label coloring (optional)
            group_colors = None
            legend_handles = None
            labels = None

            if (group_prefix and group_label_map) or group_numeric_column:
                if X_val is not None and valid_mask.sum() > 0 and global_color_map_dict:
                    X_val_df = pd.DataFrame(X_val).reset_index(drop=True)
                    valid_idx = np.where(valid_mask)[0]
                    X_val_valid = X_val_df.iloc[valid_idx]

                    if group_prefix:
                        dummy_cols = [col for col in X_val_valid.columns if col.startswith(group_prefix)]
                        if dummy_cols:
                            suffixes = X_val_valid[dummy_cols].idxmax(axis=1).str.replace(group_prefix, '', regex=False)
                            labels = suffixes.map(lambda s: fuzzy_map_suffix(s, group_label_map))
                            group_colors = labels.map(global_color_map_dict).values
                    
                    elif group_numeric_column and group_numeric_column in X_val_valid.columns and global_group_labels:
                        values = X_val_valid[group_numeric_column]
                        # Find which global bin each value belongs to
                        interval_bins = pd.IntervalIndex.from_tuples([(float(c.strip('()[]').split(', ')[0]), float(c.strip('()[]').split(', ')[1])) for c in global_group_labels], closed='right')
                        bins = pd.cut(values, bins=interval_bins, right=True)
                        labels = bins.astype(str)
                        group_colors = labels.map(global_color_map_dict).values

                    if labels is not None and global_group_labels is not None:
                        present_labels = pd.Series(labels).dropna().unique()
                        legend_handles = [
                            Line2D([0], [0], marker='o', color='w', label=label,
                                   markerfacecolor=global_color_map_dict[label],
                                   markeredgecolor='k', markersize=6)
                            for label in global_group_labels if label in present_labels
                        ]

            if len(y_val_clean) == 0 or len(y_pred_clean) == 0:
                ax.set_title(f"{model_name} - Fold {fold_pred['fold']}: No valid data")
                ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=12)
                ax.axis('off')
                continue

            if group_colors is not None and legend_handles:
                # Filter out points where color could not be determined
                valid_color_mask = ~pd.isna(group_colors)
                ax.scatter(y_val_clean[valid_color_mask], y_pred_clean[valid_color_mask], c=group_colors[valid_color_mask], alpha=0.7, edgecolor='k', s=40)
                if legend_handles:
                    ax.legend(handles=legend_handles, title="Group", loc="lower right", fontsize=8)
            else:
                ax.scatter(y_val_clean, y_pred_clean, alpha=0.5, s=40)

            # Use global min/max for all folds
            ax.plot([global_min, global_max], [global_min, global_max], 'r--', lw=1)
            ax.set_xlim(global_min, global_max)
            ax.set_ylim(global_min, global_max)
            # Title with model, fold, and val groups
            val_groups = fold_pred.get('val_groups', None)
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
                x_vals = np.linspace(global_min, global_max, 100)
                ax.plot(x_vals, reg_line(x_vals), 'k--', lw=1)
            # Metrics
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
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    return fig

def plot_residuals(
    X_train,
    y_train_dict,
    lat_train,
    lon_train,
    X_test,
    y_test_dict,
    lat_test,
    lon_test,
    model_pipelines,
    target_columns,
    columns_to_transform,
    model_dir="final_models",
    sup_title="Residuals Analysis",
    log_transformer=None
):
    """
    Plots residual analysis for each model and target for both train and test sets.
    For each, it generates a figure with two subplots:
    1. Residuals vs. Predicted for train and test sets.
    2. A spatial map of residuals for train and test sets with a convex hull for the test set.
    """
    model_names = list(model_pipelines.keys())
    figs = []

    for target in target_columns:
        y_train_full = y_train_dict[target]
        y_test_full = y_test_dict[target]
        is_log = target in columns_to_transform

        for model_name in model_names:
            model_file = os.path.join(model_dir, f"{target.replace('/', '_')}_{model_name}.pkl")
            if not os.path.exists(model_file):
                continue

            model = joblib.load(model_file)

            # Process train data
            y_pred_train_raw = model.predict(X_train)
            y_train = pd.Series(y_train_full, index=X_train.index)
            y_pred_train = pd.Series(y_pred_train_raw, index=X_train.index)
            if is_log and log_transformer:
                y_pred_train = log_transformer.inverse_transform(y_pred_train)
            
            valid_mask_train = y_train.notna() & y_pred_train.notna()
            y_train_clean = y_train[valid_mask_train]
            y_pred_train_clean = y_pred_train[valid_mask_train]
            lat_train_clean = lat_train[valid_mask_train]
            lon_train_clean = lon_train[valid_mask_train]
            residuals_train = y_pred_train_clean - y_train_clean

            # Process test data
            y_pred_test_raw = model.predict(X_test)
            y_test = pd.Series(y_test_full, index=X_test.index)
            y_pred_test = pd.Series(y_pred_test_raw, index=X_test.index)
            if is_log and log_transformer:
                y_pred_test = log_transformer.inverse_transform(y_pred_test)

            valid_mask_test = y_test.notna() & y_pred_test.notna()
            y_test_clean = y_test[valid_mask_test]
            y_pred_test_clean = y_pred_test[valid_mask_test]
            lat_test_clean = lat_test[valid_mask_test]
            lon_test_clean = lon_test[valid_mask_test]
            residuals_test = y_pred_test_clean - y_test_clean

            if len(y_test_clean) == 0 or len(y_train_clean) == 0:
                continue

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            fig.suptitle(f"{sup_title}\nTarget: {target} - Model: {model_name}", fontsize=16)

            # Subplot 1: Residuals vs. Predicted
            ax1.scatter(y_pred_train_clean, residuals_train, alpha=0.5, edgecolor='none', s=30, label='Train')
            ax1.scatter(y_pred_test_clean, residuals_test, alpha=0.7, s=40, marker='^', edgecolor='none', label='Test')
            ax1.axhline(0, color='red', linestyle='--')
            ax1.set_xlabel("Predicted Values")
            ax1.set_ylabel("Residuals")
            ax1.set_title("Residuals vs. Predicted")
            ax1.legend()
            ax1.grid(True)

            # Subplot 2: Spatial Map of Residuals
            gdf_train = gpd.GeoDataFrame(
                {'residuals': residuals_train},
                geometry=gpd.points_from_xy(lon_train_clean, lat_train_clean),
                crs="EPSG:4326"
            ).to_crs(epsg=3857)
            
            gdf_test = gpd.GeoDataFrame(
                {'residuals': residuals_test},
                geometry=gpd.points_from_xy(lon_test_clean, lat_test_clean),
                crs="EPSG:4326"
            ).to_crs(epsg=3857)

            # Make color bar symmetrical around zero
            max_abs_residual = pd.concat([residuals_train, residuals_test]).abs().max()
            vmin = -max_abs_residual
            vmax = max_abs_residual
            norm = Normalize(vmin=vmin, vmax=vmax)

            # Plotting
            gdf_train.plot(ax=ax2, column='residuals', cmap='coolwarm', 
                           norm=norm, legend=False, s=30, 
                           marker='o', edgecolor='k', linewidth=0.2)
            gdf_test.plot(ax=ax2, column='residuals', cmap='coolwarm', 
                          norm=norm, legend=False, s=50, 
                          marker='^', edgecolor='k', linewidth=0.2)
            
            # Add convex hull for test set
            if len(lon_test_clean) >= 3:
                points = np.column_stack((lon_test_clean, lat_test_clean))
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                polygon = Polygon(hull_points)
                
                gdf_hull = gpd.GeoDataFrame([1], geometry=[polygon], crs="EPSG:4326").to_crs(epsg=3857)
                gdf_hull.plot(ax=ax2, facecolor='none', edgecolor='yellow', lw=2)

            ctx.add_basemap(ax2)
            ax2.set_title("Spatial Distribution of Residuals")
            ax2.set_xlabel("Longitude")
            ax2.set_ylabel("Latitude")
            
            # Create a shared colorbar
            sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax2, label="Residual Value")

            # Create legend for markers
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Train', markerfacecolor='gray', markersize=10),
                Line2D([0], [0], marker='^', color='w', label='Test', markerfacecolor='gray', markersize=10),
                Line2D([0], [0], color='yellow', lw=2, label='Test Area')
            ]
            ax2.legend(handles=legend_elements)
            
            plt.tight_layout(rect=(0, 0, 1, 0.95))
            figs.append((fig, target, model_name))

    return figs

def plot_feature_importances(model_file, X, bands_csv_path=None, worldclim_csv_path=None, top_n=20, title=None):
    """
    Plot and return a matplotlib figure for feature importances for a fitted model pipeline.
    - model_file: path to the saved model (joblib)
    - X: DataFrame of features (columns must match those used in training)
    - bands_csv_path: path to bandsList.csv for friendly band names (optional)
    - worldclim_csv_path: path to worldclim_names.csv for 'bio' feature names (optional)
    - top_n: number of top features to plot
    - title: plot title (optional)
    Returns: matplotlib figure
    """
    # Load model
    model = joblib.load(model_file)
    # Get feature importances (works for tree-based models)
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        importances = model.named_steps['model'].feature_importances_
    else:
        raise ValueError("Model does not have feature_importances_ attribute.")
    feature_names = X.columns
    # Load band dictionary if provided
    bands_dict = None
    if bands_csv_path is not None:
        bands_df = pd.read_csv(bands_csv_path, header=None)
        bands_dict = dict(zip(bands_df.index + 1, bands_df[1]))
    worldclim_dict = None
    if worldclim_csv_path is not None:
        wc_df = pd.read_csv(worldclim_csv_path, header=None)
        worldclim_dict = dict(zip(wc_df[0].astype(str), wc_df[1]))
    def get_friendly_name(feature):
        if bands_dict and feature.startswith("Band_"):
            try:
                band_num = int(feature.split("_")[1])
                wavelength = bands_dict.get(band_num, f"Band_{band_num}")
                return f"{float(wavelength):.2f} nm" if isinstance(wavelength, (int, float, np.floating)) else str(wavelength)
            except Exception:
                return feature
        elif worldclim_dict and feature.startswith("bio"):
            return worldclim_dict.get(feature, feature)
        else:
            return feature
    friendly_feature_names = [get_friendly_name(f) for f in feature_names]
    # Sort and select top N
    sorted_idx = np.argsort(importances)[::-1]
    top_idx = sorted_idx[:top_n]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(top_n), importances[top_idx][::-1], align='center')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([friendly_feature_names[i] for i in top_idx][::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title(title or f"Top {top_n} Important Features")
    fig.tight_layout()
    return fig

def calculate_vip(plsr_model, X):
    # Standard VIP calculation for PLSRegression
    t = plsr_model.x_scores_
    w = plsr_model.x_weights_
    q = plsr_model.y_loadings_
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    Wnorm2 = (w ** 2).sum(axis=0)
    vip = np.zeros((p,))
    for i in range(p):
        weight = np.array([(w[i, j] / np.sqrt(Wnorm2[j])) ** 2 for j in range(h)]).flatten()
        vip[i] = np.sqrt(p * (s.T @ weight) / s.sum())
    return vip

def plot_vip_bar(ax, plsr_model, X, top_n=20, friendly_names=None):
    vip_scores = calculate_vip(plsr_model, X)
    vip_series = pd.Series(vip_scores, index=X.columns)
    if friendly_names is not None:
        vip_series.index = friendly_names
    # Sort by VIP and plot in descending order
    top_vip = vip_series.sort_values(ascending=False).head(top_n)
    top_vip.plot(kind="bar", color="steelblue", ax=ax)
    ax.set_title(f"Top {top_n} PLSR Features by VIP Score")
    ax.set_ylabel("VIP Score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

def plot_plsr_biplot(ax, plsr_model, X, top_n=20, friendly_names=None):
    scores = plsr_model.x_scores_
    weights = plsr_model.x_weights_
    pc1, pc2 = 0, 1
    feature_names = np.array(friendly_names) if friendly_names is not None else np.array(X.columns)
    vip_scores = calculate_vip(plsr_model, X)
    # Get indices of top N VIP features
    top_idx = np.argsort(vip_scores)[::-1][:top_n]
    # Normalize sample projections to [-1, 1]
    scores_norm = (scores[:, [pc1, pc2]] - scores[:, [pc1, pc2]].min(axis=0)) / (scores[:, [pc1, pc2]].ptp(axis=0)) * 2 - 1
    ax.scatter(scores_norm[:, 0], scores_norm[:, 1], c='lightgray', edgecolor='k', alpha=0.6, label='Samples')
    # Normalize vectors to [-1, 1] for both PC1 and PC2, but only for top features
    w1s = weights[top_idx, pc1]
    w2s = weights[top_idx, pc2]
    w1s_norm = w1s / np.max(np.abs(w1s)) if np.max(np.abs(w1s)) != 0 else w1s
    w2s_norm = w2s / np.max(np.abs(w2s)) if np.max(np.abs(w2s)) != 0 else w2s
    for i, idx in enumerate(top_idx):
        ax.arrow(0, 0, w1s_norm[i], w2s_norm[i],
                 color='crimson', alpha=0.8, head_width=0.05, length_includes_head=True)
        ax.text(w1s_norm[i]*1.15, w2s_norm[i]*1.15,
                feature_names[idx], color='darkred', fontsize=9, ha='center', va='center')
    ax.set_xlabel("PLS Component 1")
    ax.set_ylabel("PLS Component 2")
    ax.set_title(f"PLSR Biplot with Top {top_n} Features")
    ax.grid(True)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

def plot_plsr_vip_and_biplot(model_file, X, bands_csv_path=None, worldclim_csv_path=None, top_n=20, title=None):
    plsr_model = joblib.load(model_file).named_steps['model']
    # Prepare friendly names
    bands_dict = None
    if bands_csv_path is not None:
        bands_df = pd.read_csv(bands_csv_path, header=None)
        bands_dict = dict(zip(bands_df.index + 1, bands_df[1]))
    worldclim_dict = None
    if worldclim_csv_path is not None:
        wc_df = pd.read_csv(worldclim_csv_path, header=None)
        worldclim_dict = dict(zip(wc_df[0].astype(str), wc_df[1]))
    def get_friendly_name(feature):
        if bands_dict and feature.startswith("Band_"):
            try:
                band_num = int(feature.split("_")[1])
                wavelength = bands_dict.get(band_num, f"Band_{band_num}")
                return f"{float(wavelength):.2f} nm" if isinstance(wavelength, (int, float, np.floating)) else str(wavelength)
            except Exception:
                return feature
        elif worldclim_dict and feature.startswith("bio"):
            return worldclim_dict.get(feature, feature)
        else:
            return feature
    friendly_feature_names = [get_friendly_name(f) for f in X.columns]
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    plot_vip_bar(axs[0], plsr_model, X, top_n, friendly_names=friendly_feature_names)
    plot_plsr_biplot(axs[1], plsr_model, X, top_n, friendly_names=friendly_feature_names)
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig

def plot_train_test_histograms(y_train, y_test, target_columns, plots_dir, bins=30, filename=None, orientation='horizontal'):
    """
    Plot overlayed histograms of train and test splits for each target column in a single figure.
    Each subplot is a target, with train and test histograms overlayed.
    Bins are normalized between train and test. Orientation can be 'horizontal' or 'vertical'.
    """
    n_targets = len(target_columns)
    # Determine subplot arrangement
    if orientation == 'vertical':
        fig, axes = plt.subplots(n_targets, 1, figsize=(7, 4 * n_targets), squeeze=False)
        axes = axes[:, 0]
    else:
        fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5), squeeze=False)
        axes = axes[0]
    for i, target in enumerate(target_columns):
        ax = axes[i]
        train_data = y_train[target].dropna()
        test_data = y_test[target].dropna()
        # Normalize bins between train and test
        combined = pd.concat([train_data, test_data])
        bin_edges = np.histogram_bin_edges(combined, bins=bins)
        ax.hist(train_data, bins=bin_edges, color='tab:blue', alpha=0.5, label='Train', density=True)
        ax.hist(test_data, bins=bin_edges, color='tab:orange', alpha=0.5, label='Test', density=True)
        ax.set_title(f"Train/Test Histogram: {target}")
        ax.set_xlabel(target)
        ax.set_ylabel("Density")
        ax.legend()
    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    if filename:
        out_path = os.path.join(plots_dir, filename)
    else:
        out_path = os.path.join(plots_dir, "train_test_histograms.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path