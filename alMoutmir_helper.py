import math
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import pandas as pd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import uuid
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns


def remove_outliers(df, columns, method='iqr', multiplier=1.5, z_threshold=3):
    """
    Remove outliers from a DataFrame using IQR or Z-score method.

    Parameters:
    - df: Input DataFrame.
    - columns: List of column names to check for outliers.
    - method: 'iqr' or 'zscore'.
    - multiplier: IQR multiplier for 'iqr' method.
    - z_threshold: Threshold for 'zscore' method.

    Returns:
    - Filtered DataFrame with outliers removed.
    """
    try:
        # Force numeric conversion and keep only numeric data
        numeric_df = df[columns].apply(pd.to_numeric, errors='coerce')
        valid_columns = numeric_df.select_dtypes(include=[np.number]).columns.tolist()

        if not valid_columns:
            raise ValueError("No numeric columns found for outlier detection.")

        # Use only rows where valid columns are not NaN
        subset = numeric_df[valid_columns].dropna()

        if method == 'iqr':
            Q1 = subset.quantile(0.25)
            Q3 = subset.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR
            mask = ~((subset < lower) | (subset > upper)).any(axis=1)

        elif method == 'zscore':
            z_scores = np.abs(zscore(subset))
            mask = (z_scores < z_threshold).all(axis=1)

        else:
            raise ValueError("Invalid method: choose either 'iqr' or 'zscore'.")

        # Match mask index to original df
        clean_indices = subset[mask].index
        return df.loc[clean_indices]

    except ValueError as ve:
        print(f"[ValueError] {ve}")
        return df

    except Exception as e:
        print(f"[Unexpected Error] {e}")
        return df
    

def safe_convert(df, cols, dtype):
    """
    Safely convert columns in a DataFrame to specified data types.
    
    Parameters:
    - df: pandas DataFrame
    - cols: list of column names to convert
    - dtype: one of ['float', 'int', 'datetime', 'string']
    """
    existing_cols = [col for col in cols if col in df.columns]
    
    for col in existing_cols:
        if dtype == 'float':
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif dtype == 'int':
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        elif dtype == 'datetime':
            df[col] = pd.to_datetime(df[col], errors='coerce')
        elif dtype == 'string':
            df[col] = df[col].astype('string')


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

def drop_junk_spectra(df, bands_df, ranges):
    """
    Modify the spectra dataframe by setting specific bands to NaN based on given ranges.

    Parameters:
        df (pd.DataFrame): The original spectra dataframe.
        bands_df (pd.DataFrame): DataFrame containing band information with columns [0, 1].
                                 Column 0 contains band names, and column 1 contains band values.
        ranges (list of tuple): List of ranges to filter bands. Each tuple is (lower, upper).

    Returns:
        pd.DataFrame: Modified spectra dataframe with specified bands set to NaN.
    """
    # Duplicate the dataframe
    modified_spectra = df.copy()

    # Extract the band values
    band_values = bands_df[1].astype(float)

    # Identify the bands that fall within the specified ranges
    bands_to_nan = []
    for lower, upper in ranges:
        if lower is None:
            bands_to_nan.extend(bands_df[0][band_values <= upper].tolist())
        elif upper is None:
            bands_to_nan.extend(bands_df[0][band_values >= lower].tolist())
        else:
            bands_to_nan.extend(bands_df[0][(band_values >= lower) & (band_values <= upper)].tolist())

    # Set the identified band columns to NaN
    modified_spectra[bands_to_nan] = np.nan

    return modified_spectra

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


def kmeans_clustering_map(df, feature_cols, coord_cols=['Longitude_X', 'Latitude_Y'], k_range=(3, 20),
                          cmap='Set1', random_state=42, title='KMeans Clustering Map',
                          show_map=True, return_df=False, show_cluster_labels=True):
    """
    Perform KMeans clustering with silhouette-based tuning and optionally plot clusters on a map.

    Parameters:
    - df: DataFrame
    - feature_cols: list of features used for clustering
    - coord_cols: [lon_col, lat_col]
    - k_range: (min_k, max_k) range for K
    - cmap: Matplotlib colormap name
    - random_state: seed
    - title: plot title
    - show_map: if True, plots cluster map
    - return_df: if True, returns the DataFrame with clusters

    Returns:
    - DataFrame with KMeans_Cluster column (only if return_df=True)
    """
    # Drop missing values for selected columns
    clean_df = df.dropna(subset=feature_cols + coord_cols).copy()
    X = clean_df[feature_cols].values

    # Find best K using silhouette score
    best_k, best_score = None, -1
    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k = k
            best_score = score

    # Final KMeans
    kmeans = KMeans(n_clusters=best_k, random_state=random_state)
    final_labels = kmeans.fit_predict(X)
    clean_df['KMeans_Cluster'] = final_labels + 1  # 1-based cluster IDs

    if show_map:
        # Use the integrated soil map plotter
        plot_soil_maps(clean_df, columns=['KMeans_Cluster'], group_colormap=True,
                       discrete_bins=best_k, cmap=plt.get_cmap(cmap, best_k),
                       title=f'{title} (Optimal k={best_k})',
                       coord_cols=coord_cols, show_cluster_labels= show_cluster_labels)

    if return_df:
        return clean_df  # Return only when explicitly requested
    

def extract_spectral_values(
    image_path,
    df,
    lat_col='Latitude_Y',
    lon_col='Longitude_X',
    nodata_value=None,
    mask_path=None,
    verbose=True
):
    """
    Extract hyperspectral pixel values at geographic coordinates from a DataFrame,
    with optional mask support.

    Parameters:
    - image_path (str): Path to the raster (e.g., hyperspectral image).
    - df (pd.DataFrame): DataFrame containing lat/lon coordinates.
    - lat_col (str): Column name for latitude (default: 'Latitude_Y').
    - lon_col (str): Column name for longitude (default: 'Longitude_X').
    - nodata_value (float or int or None): If set, pixels with this value will be skipped.
    - mask_path (str or None): Optional. Path to a mask raster aligned with image_path. Values > 0 are considered valid.
    - verbose (bool): Whether to show progress bar.

    Returns:
    - extracted_df (pd.DataFrame): DataFrame with uuid, coordinates, and spectral band values.
    - df_with_uuid (pd.DataFrame): Original DataFrame with 'uuid' column added.
    """

    df = df.copy()
    df['uuid'] = [str(uuid.uuid4()) for _ in range(len(df))]
    results = []

    with rasterio.open(image_path) as src:
        need_reprojection = src.crs.to_epsg() != 4326
        reader = WarpedVRT(src, crs='EPSG:4326', resampling=Resampling.nearest) if need_reprojection else src

        # Open the mask if provided
        if mask_path:
            mask_src = rasterio.open(mask_path)
            if mask_src.crs.to_epsg() != 4326:
                mask_reader = WarpedVRT(mask_src, crs='EPSG:4326', resampling=Resampling.nearest)
            else:
                mask_reader = mask_src
        else:
            mask_reader = None

        with reader as vrt:
            iterable = tqdm(df.iterrows(), total=len(df), desc="Extracting") if verbose else df.iterrows()
            
            for _, row in iterable:
                lat, lon = row[lat_col], row[lon_col]
                try:
                    x, y = vrt.index(lon, lat)
                    if 0 <= x < vrt.width and 0 <= y < vrt.height:
                        if mask_reader:
                            mx, my = mask_reader.index(lon, lat)
                            if 0 <= mx < mask_reader.width and 0 <= my < mask_reader.height:
                                mask_val = mask_reader.read(1, window=((my, my+1), (mx, mx+1)))[0, 0]
                                if mask_val <= 0 or np.isnan(mask_val):
                                    continue  # skip if masked out
                            else:
                                continue  # outside mask bounds
                        
                        pixel = vrt.read(window=((y, y + 1), (x, x + 1)))[:, 0, 0]
                        if nodata_value is not None and all(val == nodata_value for val in pixel):
                            continue
                        
                        record = {
                            'uuid': row['uuid'],
                            lat_col: lat,
                            lon_col: lon
                        }
                        for i, val in enumerate(pixel):
                            record[f'Band_{i+1}'] = val
                        results.append(record)
                except:
                    continue

        if mask_path:
            mask_src.close()
            if isinstance(mask_reader, WarpedVRT):
                mask_reader.close()

    extracted_df = pd.DataFrame(results)
    return extracted_df, df


