from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
from sklearn.cluster import KMeans
from yg_eo_soilnet.utils import assign_grid_ids

import matplotlib.pyplot as plt
import tempfile
import os
import mlflow
import numpy as np

class BaseSpatialClusterStrategy(ABC):
    """
    Abstract base class for spatial clustering strategies.
    """

    @abstractmethod
    def cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clusters the input DataFrame spatially and adds a 'cluster' column.
        """
        pass

    def plot_train_test(
        self,
        df: pd.DataFrame,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        lon_col: str = "lon",
        lat_col: str = "lat",
        title: str = "Train/Test Split",
        artifact_path: str = "splits_plots",
        filename: str = "train_test_split.png",
        show: bool = False,
    ):
        """
        Plot clustered points with train/test coloring, save to temp file,
        and log to MLflow as an artifact.
        """

        fig, ax = plt.subplots(figsize=(10, 10))

        # plot train points
        ax.scatter(
            df.loc[train_idx, lon_col],
            df.loc[train_idx, lat_col],
            c="blue",
            s=12,
            alpha=0.6,
            label="Train",
            zorder=2,
        )

        # plot test points
        ax.scatter(
            df.loc[test_idx, lon_col],
            df.loc[test_idx, lat_col],
            c="red",
            s=20,
            alpha=0.8,
            label="Test",
            marker="x",
            zorder=3,
        )

        # overlay grid boundaries if available on this strategy
        grid_gdf = getattr(self, "grid_gdf_", None)
        if grid_gdf is not None:
            grid_gdf.boundary.plot(ax=ax, color="lightgray", linewidth=0.5, zorder=1)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()

        # save to temp file and log to MLflow
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, filename)
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            mlflow.log_artifact(filepath, artifact_path=artifact_path)

        if show:
            plt.show()

        plt.close(fig)

@dataclass
class KMeansClusterStrategy(BaseSpatialClusterStrategy):
    """
    KMeans-based spatial clustering.

    Parameters:
    -----------
    n_clusters : int, default=12
        Number of spatial clusters to form.
    lat_col : str, default='lat'
        Name of the latitude column in the DataFrame.
    lon_col : str, default='lon'
        Name of the longitude column in the DataFrame.
    random_state : int, default=42
        Random seed for reproducibility of clustering.
    """
    n_clusters: int = 12
    lat_col: str = 'lat'
    lon_col: str = 'lon'
    random_state: int = 42

    def cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        coords = df[[self.lat_col, self.lon_col]].dropna()
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        labels = kmeans.fit_predict(coords) + 1  # 1-indexed

        df = df.copy()
        df.loc[coords.index, 'cluster'] = labels.astype(int)
        return df.dropna(subset=['cluster'])

@dataclass
class SpatialGridClusterStrategy(BaseSpatialClusterStrategy):
    """
    Regular grid-based spatial clustering.

    Parameters
    ----------
    cell_size_m : int
        Size of each grid cell in meters.
    lat_col : str, default='lat'
        Latitude column in the DataFrame.
    lon_col : str, default='lon'
        Longitude column in the DataFrame.
    """

    cell_size_m: int
    lat_col: str = "lat"
    lon_col: str = "lon"
    random_state: int = 42

    def cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        grid_ids, grid_gdf = assign_grid_ids(
            df, cell_size_m=self.cell_size_m, lon_col=self.lon_col, lat_col=self.lat_col
        )
        df["cluster"] = grid_ids.astype(int)
        self.grid_gdf_ = grid_gdf  # save polygons for later plotting
        return df
