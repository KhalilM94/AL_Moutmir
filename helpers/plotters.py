import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns

def plot_density_heatmap(
    df,
    lon_col='Longitude_X',
    lat_col='Latitude_Y',
    variables=None,
    bins=(100, 100),
    cmap='viridis',
    figsize=(12, 8),
    suptitle='Point Density Heatmap',
    single_plot=False,
    colorbar_mode='continuous',  # 'continuous' or 'categorical'
    categories=None,             # List of category labels (if categorical)
    ):
    """
    Plots 2D heatmaps of point density or category-based maps over geo coordinates using Cartopy.
    
    Parameters:
    - df: pandas DataFrame
    - lon_col: name of longitude column
    - lat_col: name of latitude column
    - variables: list of subplot labels (or None for single plot)
    - bins: tuple for histogram resolution
    - cmap: colormap name or list of colors
    - figsize: tuple for figure size
    - suptitle: main title for multi-plot
    - single_plot: plot in one panel
    - colorbar_mode: 'continuous' or 'categorical'
    - categories: optional list of category names if using categorical colorbar
    """
    if single_plot or not variables:
        variables = [None]

    n = len(variables)
    cols = min(n, 2)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize,
                             subplot_kw={'projection': ccrs.PlateCarree()},
                             squeeze=False)
    fig.patch.set_facecolor('white')
    fig.suptitle(suptitle, fontsize=16)

    for i, var in enumerate(variables):
        row, col = divmod(i, cols)
        ax = axes[row][col]

        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black')
        ax.add_feature(cfeature.OCEAN)
        ax.set_title(var or 'Point Density')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        x = df[lon_col]
        y = df[lat_col]

        # Create histogram
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
        masked_heatmap = np.ma.masked_where(heatmap == 0, heatmap)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        # Color handling
        if colorbar_mode == 'categorical':
            # Use discrete colors
            if categories is None:
                categories = range(int(masked_heatmap.max()) + 1)
            num_cats = len(categories)
            cmap = mcolors.ListedColormap(plt.get_cmap(cmap).colors[:num_cats])
            bounds = np.arange(num_cats + 1) - 0.5
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            img = ax.imshow(masked_heatmap.T, extent=extent, origin='lower',
                            cmap=cmap, norm=norm, alpha=0.8, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(img, ax=ax, orientation='vertical', ticks=np.arange(num_cats),
                                shrink=0.7)
            cbar.ax.set_yticklabels(categories)
            cbar.set_label("Category")
        else:
            # Continuous colorbar
            img = ax.imshow(masked_heatmap.T, extent=extent, origin='lower',
                            cmap=cmap, alpha=0.7, transform=ccrs.PlateCarree())
            cbar = fig.colorbar(img, ax=ax, orientation='vertical', shrink=0.7)
            cbar.set_label('Density')

    # Hide unused subplots
    for j in range(i + 1, rows * cols):
        row, col = divmod(j, cols)
        fig.delaxes(axes[row][col])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_correlogram(df, columns, title='Correlogram', remove_outliers=False, alpha=0.2, sample_size=1000):
    """
    Plots a lower-triangle correlogram using Seaborn's pairplot.
    
    Parameters:
    - df: DataFrame containing the data.
    - columns: List of column names to include in the correlogram.
    - title: Title of the plot.
    - remove_outliers: Whether to apply IQR filtering.
    - alpha: Transparency for scatter plots.
    - sample_size: Max number of samples to use for plotting.
    """
    subset_df = df[columns].copy()

    # Optional: Remove outliers using IQR
    if remove_outliers:
        subset_df = remove_outliers(subset_df, columns)

    # Optional: Sample to speed up rendering
    if sample_size and subset_df.shape[0] > sample_size:
        subset_df = subset_df.sample(sample_size, random_state=42)

    # Pairplot
    g = sns.pairplot(subset_df, diag_kind='kde', plot_kws={'alpha': alpha})

    # Hide upper triangle
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_visible(False)

    g.figure.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df, columns, title='Correlation Matrix'):
    # Calculate correlation matrix
    corr = df[columns].corr()

    # Mask upper triangle and diagonal
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1, square=True,
                mask=mask, cbar_kws={'label': 'Correlation'})
    
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_multiple_datasets(dfs, columns, dataset_labels=None, kind='box', bins=30, ncols=4, figsize=(20, 5), violin=False):
    """
    Plots boxplots or histograms from one or multiple datasets in a grid layout.

    Parameters:
    - dfs: list of DataFrames (or a single DataFrame)
    - columns: list of column names to plot
    - dataset_labels: optional list of dataset labels (for legend)
    - kind: 'box' or 'hist'
    - bins: number of bins for histograms
    - ncols: number of columns in the plot grid
    - figsize: tuple (width, height per row)
    - violin: if True, plot violin plots instead of boxplots
    """
    # Ensure dfs is a list
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]

    num_datasets = len(dfs)
    if dataset_labels is None:
        dataset_labels = [f'Dataset {i+1}' for i in range(num_datasets)]

    # Filter columns that exist in all datasets
    valid_columns = [col for col in columns if all(col in df.columns for df in dfs)]
    num_rows = math.ceil(len(valid_columns) / ncols)

    fig, axes = plt.subplots(num_rows, ncols, figsize=(figsize[0], figsize[1] * num_rows))
    axes = axes.flatten()

    for i, col in enumerate(valid_columns):
        ax = axes[i]

        if kind == 'hist':
            # Use shared bin edges
            combined = pd.concat([df[col].dropna() for df in dfs])
            bin_edges = np.histogram_bin_edges(combined, bins=bins)

            for df, label in zip(dfs, dataset_labels):
                ax.hist(df[col].dropna(), bins=bin_edges, density=(num_datasets > 1),
                        alpha=0.6, label=label)

            ax.set_title(f'Histogram of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Density' if num_datasets > 1 else 'Frequency')
            if num_datasets > 1:
                ax.legend()

        elif kind == 'box':
            if violin:
                sns.violinplot(data=pd.concat(dfs, keys=dataset_labels), x='level_0', y=col, ax=ax)
                ax.set_title(f'Violin Plot of {col}')
            else:
                combined_df = pd.concat(
                    [df[[col]].assign(dataset=label) for df, label in zip(dfs, dataset_labels)]
                )
                sns.boxplot(x='dataset', y=col, data=combined_df, ax=ax)
                ax.set_title(f'Boxplot of {col}')

            ax.set_xlabel('')
            ax.set_ylabel('')

    # Remove unused subplots
    for j in range(len(valid_columns), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_soil_maps(df, columns, group_colormap=True, num_cols=4, figsize=(20, 5), 
                   discrete_bins=5, cmap='viridis', title='', coord_cols=['Longitude_X', 'Latitude_Y'],
                   show_cluster_labels=True):
    """
    Plot spatial maps (scatter) of one or more soil property columns, including KMeans clustering.

    Parameters:
    - df: DataFrame with spatial and value columns
    - columns: list of column names to plot
    - group_colormap: if True, group values into discrete color bins
    - num_cols: number of columns in subplot grid
    - figsize: base figure size (width, height per row)
    - discrete_bins: number of bins or classes (for grouped colormap)
    - cmap: Matplotlib colormap (string or object)
    - title: global or single-title override
    - coord_cols: [longitude_column, latitude_column]
    - show_cluster_labels: if True, annotate cluster centers with their label
    """
    lon_col, lat_col = coord_cols

    if 'KMeans_Cluster' in columns:
        cluster_col = 'KMeans_Cluster'
        unique_clusters = sorted(df[cluster_col].dropna().unique())
        best_k = len(unique_clusters)

        # Use ListedColormap with k distinct colors
        base_cmap = cmap if isinstance(cmap, mcolors.Colormap) else plt.get_cmap(cmap, best_k)
        cmap_listed = ListedColormap([base_cmap(i) for i in range(best_k)])

        # Map cluster values to index positions for color consistency
        cluster_to_index = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
        df['_cluster_idx'] = df[cluster_col].map(cluster_to_index)

        fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black')
        ax.add_feature(cfeature.OCEAN)

        # Normalize using the number of clusters
        norm = BoundaryNorm(boundaries=np.arange(-0.5, best_k + 0.5), ncolors=best_k)

        # Scatter plot with correct colors
        scatter = ax.scatter(df[lon_col], df[lat_col], c=df['_cluster_idx'],
                             cmap=cmap_listed, norm=norm, alpha=0.7, transform=ccrs.PlateCarree())

        # Optional cluster number labels at cluster centers
        show_labels = True
        if show_cluster_labels:
            for cluster_id in unique_clusters:
                cluster_data = df[df[cluster_col] == cluster_id]
                center_lon = cluster_data[lon_col].mean()
                center_lat = cluster_data[lat_col].mean()
                ax.text(center_lon, center_lat, str(cluster_id),
                        fontsize=12, fontweight='bold', color='black',
                        ha='center', va='center', transform=ccrs.PlateCarree(),
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle,pad=0.3', alpha=0.7))

        # Add proper colorbar
        sm = ScalarMappable(cmap=cmap_listed, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='Cluster')
        cbar.set_ticks(np.arange(best_k))
        cbar.set_ticklabels([str(c) for c in unique_clusters])

        # Title and axis labels
        plot_title = title or f'KMeans Clustering Map (k = {best_k})'
        ax.set_title(plot_title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        plt.tight_layout()
        plt.show()

        # Drop temp column
        df.drop(columns=['_cluster_idx'], inplace=True)
        return



    # Standard soil property plotting
    num_rows = math.ceil(len(columns) / num_cols)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols,
                             figsize=(figsize[0], figsize[1] * num_rows),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black')
        ax.add_feature(cfeature.OCEAN)

        values = df[col].dropna()
        lons = df[lon_col]
        lats = df[lat_col]

        if group_colormap:
            bounds = np.linspace(values.min(), values.max(), discrete_bins + 1)
            norm = BoundaryNorm(boundaries=bounds, ncolors=256)
            scatter = ax.scatter(lons, lats, c=df[col], cmap=cmap, norm=norm,
                                 alpha=0.7, transform=ccrs.PlateCarree())

            cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', 
                                shrink=0.5, pad=0.05)
            cbar.set_label(col)
            cbar.set_ticks(bounds[:-1] + np.diff(bounds) / 2)
            cbar.set_ticklabels([f'{bounds[i]:.1f}–{bounds[i+1]:.1f}' for i in range(len(bounds) - 1)])
        else:
            scatter = ax.scatter(lons, lats, c=df[col], cmap=cmap, alpha=0.7,
                                 transform=ccrs.PlateCarree())
            scatter.set_clim(values.min(), values.max())
            cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.05)
            cbar.set_label(col)

        map_title = title if title and len(columns) == 1 else f'{col} Distribution'
        ax.set_title(map_title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    # Remove unused axes
    for j in range(len(columns), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_spectra(x_values, y_values, titles=None, xlabel='Wavelength (nm)', ylabel='Reflectance', 
                 alpha=0.1, color='blue', ncols=2, figsize=(16, 10)):
    """
    Plots one or multiple spectra DataFrames in a grid of subplots.

    Parameters:
    - x_values: pandas.Series or list
        The x-axis values (e.g., wavelengths).
    - y_values: pandas.DataFrame or list of pandas.DataFrame
        Single DataFrame or list of DataFrames containing reflectance values.
    - titles: str or list of str, optional
        Title(s) for the plot(s). If a single string is provided with multiple DataFrames, it's broadcasted.
    - xlabel: str
        Label for the x-axis.
    - ylabel: str
        Label for the y-axis.
    - alpha: float
        Transparency of the lines.
    - color: str
        Line color.
    - ncols: int
        Number of columns in the subplot grid.
    - figsize: tuple
        Overall figure size.
    """
    # Normalize y_values to list
    if isinstance(y_values, pd.DataFrame):
        y_values = [y_values]

    # Normalize titles to list
    if isinstance(titles, str):
        titles = [titles] * len(y_values)
    elif titles is None:
        titles = [f"Spectra {i+1}" for i in range(len(y_values))]

    n = len(y_values)
    nrows = math.ceil(n / ncols)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, df in enumerate(y_values):
        ax = axes[i]
        for _, row in df.iterrows():
            ax.plot(x_values, row, alpha=alpha, color=color)
        
        ax.set_title(titles[i])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    # Hide any unused subplots
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()