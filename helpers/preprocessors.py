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
from helpers.plotters import plot_soil_maps

def load_dict_from_file(file_path):
    # Create an empty dictionary
    result_dict = {}

    # Open the file
    with open(file_path, 'r') as f:
        # Loop through each line in the file
        for line in f:
            # Strip leading/trailing whitespace
            line = line.strip()
            
            # Skip empty lines or lines that don't contain a colon
            if not line or ':' not in line:
                continue
            
            # Split the line into key and value
            key, value = line.split(":", 1)
            
            # Strip any extra whitespace and add to dictionary
            result_dict[key.strip()] = value.strip()

    return result_dict

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