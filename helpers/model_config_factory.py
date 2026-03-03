from abc import ABC, abstractmethod
import importlib
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from sklearn.cluster import KMeans
from .misc_utils import assign_grid_ids

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

@dataclass
class ModelConfigFactory:
    """Factory class to build model configurations dynamically based on a registry."""
    registry:dict
    random_state:int = 42
    
    @staticmethod
    def _dynamic_import(import_path):
        module_path, class_name = import_path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(f"Failed to import module '{module_path}': {e}")
        return getattr(module, class_name)
    

    def build_model_configs(self, num_features):
        """Build dynamic model configurations from self.config.MODEL_REGISTRY."""
        model_configs = {}

        for name, spec in self.registry.items():
            if not spec.get("enabled", False):
                continue
            try:
                ModelClass = self._dynamic_import(spec["import_path"])
            except Exception as e:
                print(f"[Warning] Failed to import {name}: {e}")
                continue

            init_args = spec.get("init_args", {})
            custom_model_builder = spec.get("custom_model_builder", None)

            # Handle Keras or other wrappers with custom model builders
            if custom_model_builder:
                builder_func = self._dynamic_import(custom_model_builder)
                model_instance = ModelClass(build_fn=lambda: builder_func(num_features))
            else:
                if 'input_dim' in init_args:
                    init_args['input_dim'] = num_features
                model_instance = ModelClass(**init_args)

            model_configs[name] = {
                "model": model_instance,
                "params": spec.get("params", {}),
                "modeltype": spec.get("modeltype", "ml")
            }

        return model_configs
    
    def load_splitter_from_config(self) -> Optional[BaseSpatialClusterStrategy]:
        """Load and return the splitter configuration from the registry."""
        if self.registry.get("enabled", True) is True:
            class_path = self.registry["class_path"]
            params = self.registry.get("params", {})
            params.setdefault("random_state", self.random_state)

            SplitterClass = self._dynamic_import(class_path)
            return SplitterClass(**params)
        else:
            print("[Warning] Splitter configuration not found or disabled in the registry.")
            return None