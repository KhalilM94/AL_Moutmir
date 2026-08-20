from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from yg_eo_soilnet.data_manager import DataManager
from yg_eo_soilnet.datamodules.lightning.lightning_graph import SingleNodeGraphDataModule
from yg_eo_soilnet.datamodules.lightning.spatiotemporal_graph_builder import SpatiotemporalGraphBuilder
from yg_eo_soilnet.models.lightningmodules.soil_graph_lightning_module import SoilGraphLightningModule


def _write_graph_csvs(tmp_path: Path):
    static_df = pd.DataFrame(
        {
            "point_id": [1, 2, 3],
            "lat": [0.0, 0.5, 1.0],
            "lon": [0.0, 0.5, 1.0],
            "target_a": [1.0, 2.0, 3.0],
            "static_1": [10.0, 11.0, 12.0],
            "static_2": [20.0, 21.0, 22.0],
        }
    )
    timeseries_df = pd.DataFrame(
        {
            "point_id": [1, 1, 2, 2, 3, 3],
            "month": ["2020-01", "2020-02", "2020-01", "2020-02", "2020-01", "2020-02"],
            "S1_vv": [0.1, 0.0, 0.3, 0.4, 0.5, 0.6],
            "S2_b2": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            "MODIS_ndvi": [2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
        }
    )

    static_path = tmp_path / "static.csv"
    timeseries_path = tmp_path / "timeseries.csv"
    static_df.to_csv(static_path, index=False)
    timeseries_df.to_csv(timeseries_path, index=False)
    return static_path, timeseries_path


def test_data_manager_builds_single_node_spatiotemporal_graph(tmp_path: Path, logger) -> None:
    pytest.importorskip("torch")
    static_path, timeseries_path = _write_graph_csvs(tmp_path)
    config = SimpleNamespace(
        DATA_FOLDER=str(tmp_path),
        DATA_FILE="static.csv",
        STATIC_CSV_PATH=str(static_path),
        TIMESERIES_CSV_PATH=str(timeseries_path),
        POINT_ID_COLUMN="point_id",
        LAT_COLUMN="lat",
        LON_COLUMN="lon",
        TIME_COLUMN="month",
        TEMPORAL_FEATURES_ENABLED=True,
        MODALITY_PREFIX_MAP={"S1": "S1_", "S2": "S2_", "MODIS": "MODIS_"},
        S1_COLUMNS=[],
        S2_COLUMNS=[],
        MODIS_COLUMNS=[],
        TARGET_COLUMNS=["target_a"],
        ELIMINATED_FEATURES=[],
        TEST_SIZE=0.33,
        LIGHTNING_VAL_SIZE=0.33,
        RANDOM_SEED=42,
        SPATIAL_RADIUS=0.1,
        BASELINE_METHOD="knn",
        BASELINE_K_NEIGHBORS=1,
    )

    manager = DataManager(config=config, logger=logger)
    bundle = SpatiotemporalGraphBuilder(manager.config, manager.logger, manager).build(graph_data_args={"spatial_radius": 2.0, "baseline_k_neighbors": 1})

    assert bundle["static_features"].shape == (3, 2)
    assert bundle["targets"].shape == (3, 1)
    assert bundle["temporal_enabled"] is True
    assert bundle["temporal_features"]["s1"].shape == (3, 2, 1)
    assert bundle["temporal_features"]["s2"].shape == (3, 2, 1)
    assert bundle["temporal_features"]["modis"].shape == (3, 2, 1)
    assert bundle["temporal_lengths"]["s1"].shape == (3,)
    assert bundle["temporal_masks"]["s1"].shape == (3, 2)
    assert bundle["temporal_lengths"]["s1"].tolist() == [2, 2, 2]
    assert bundle["edge_index"].shape[0] == 2
    assert bundle["edge_attr"].shape[1] == 1


def test_data_manager_skips_spatial_graph_construction_when_disabled(tmp_path: Path, logger, monkeypatch) -> None:
    pytest.importorskip("torch")
    static_path, timeseries_path = _write_graph_csvs(tmp_path)
    config = SimpleNamespace(
        DATA_FOLDER=str(tmp_path),
        DATA_FILE="static.csv",
        STATIC_CSV_PATH=str(static_path),
        TIMESERIES_CSV_PATH=str(timeseries_path),
        POINT_ID_COLUMN="point_id",
        LAT_COLUMN="lat",
        LON_COLUMN="lon",
        TIME_COLUMN="month",
        TEMPORAL_FEATURES_ENABLED=True,
        MODALITY_PREFIX_MAP={"S1": "S1_", "S2": "S2_", "MODIS": "MODIS_"},
        S1_COLUMNS=[],
        S2_COLUMNS=[],
        MODIS_COLUMNS=[],
        TARGET_COLUMNS=["target_a"],
        ELIMINATED_FEATURES=[],
        TEST_SIZE=0.33,
        LIGHTNING_VAL_SIZE=0.33,
        RANDOM_SEED=42,
        SPATIAL_RADIUS=0.1,
        BASELINE_METHOD="knn",
        BASELINE_K_NEIGHBORS=1,
    )

    manager = DataManager(config=config, logger=logger)
    builder = SpatiotemporalGraphBuilder(manager.config, manager.logger, manager)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("spatial graph construction should be skipped when disabled")

    monkeypatch.setattr(builder, "compute_baseline_residuals", _fail_if_called)
    monkeypatch.setattr(builder, "build_edge_index", _fail_if_called)

    bundle = builder.build(graph_data_args={"spatial_graph_enabled": False})

    assert bundle["spatial_graph_enabled"] is False
    assert bundle["edge_index"].shape == (2, 0)
    assert bundle["edge_attr"].shape == (0, 1)
    assert bundle["residuals"].shape == (0, 1)
    assert bundle["temporal_enabled"] is True


def test_attention_pooling_handles_zero_length_temporal_sequences() -> None:
    pytest.importorskip("torch")

    model = SoilGraphLightningModule(
        static_dim=2,
        target_dim=1,
        hidden_dim=4,
        temporal_hidden_dim=3,
        modality_dims={"s1": 1},
        temporal_enabled=True,
        temporal_pooling="attention",
        spatial_graph_enabled=False,
    )
    batch = {
        "x_static": torch.randn(2, 2),
        "temporal_features": {"s1": torch.randn(2, 3, 1)},
        "temporal_lengths": {"s1": torch.tensor([0, 2], dtype=torch.long)},
        "temporal_masks": {"s1": torch.tensor([[False, False, False], [True, True, False]], dtype=torch.bool)},
        "y": torch.zeros(2, 1),
    }

    embedding = model._encode_temporal(batch, device=batch["x_static"].device)

    assert embedding is not None
    assert torch.isfinite(embedding).all()
    assert torch.allclose(embedding[0], torch.zeros_like(embedding[0]))


def test_data_manager_converts_non_wgs84_coordinates_to_utm(tmp_path: Path, logger) -> None:
    pytest.importorskip("pyproj")
    from pyproj import CRS, Transformer

    source_crs = CRS.from_epsg(3857)
    wgs84_crs = CRS.from_epsg(4326)
    to_source = Transformer.from_crs(wgs84_crs, source_crs, always_xy=True)
    to_utm = Transformer.from_crs(wgs84_crs, CRS.from_epsg(32631), always_xy=True)

    lon_values = [0.0, 0.01]
    lat_values = [0.0, 0.01]
    source_x, source_y = to_source.transform(lon_values, lat_values)
    expected_utm_x, expected_utm_y = to_utm.transform(lon_values, lat_values)

    static_df = pd.DataFrame(
        {
            "point_id": [1, 2],
            "lat": source_y,
            "lon": source_x,
            "target_a": [1.0, 2.0],
            "static_1": [10.0, 11.0],
        }
    )
    static_path = tmp_path / "static.csv"
    static_df.to_csv(static_path, index=False)

    config = SimpleNamespace(
        DATA_FOLDER=str(tmp_path),
        DATA_FILE="static.csv",
        STATIC_CSV_PATH=str(static_path),
        TIMESERIES_CSV_PATH=None,
        POINT_ID_COLUMN="point_id",
        LAT_COLUMN="lat",
        LON_COLUMN="lon",
        TIME_COLUMN="month",
        TEMPORAL_FEATURES_ENABLED=False,
        TARGET_COLUMNS=["target_a"],
        ELIMINATED_FEATURES=[],
        TEST_SIZE=0.5,
        LIGHTNING_VAL_SIZE=0.5,
        RANDOM_SEED=42,
        SPATIAL_RADIUS=5000.0,
        BASELINE_METHOD="knn",
        BASELINE_K_NEIGHBORS=1,
    )

    manager = DataManager(config=config, logger=logger)
    bundle = SpatiotemporalGraphBuilder(manager.config, manager.logger, manager).build(
        graph_data_args={
            "convert_coordinates_to_utm": True,
            "coordinate_crs": "EPSG:3857",
            "spatial_graph_enabled": False,
        }
    )

    assert bundle["coords_crs"] == "EPSG:32631"
    np.testing.assert_allclose(bundle["coords"][:, 0], np.asarray(expected_utm_x, dtype=np.float32), rtol=1e-5, atol=1e-1)
    np.testing.assert_allclose(bundle["coords"][:, 1], np.asarray(expected_utm_y, dtype=np.float32), rtol=1e-5, atol=1e-1)


def test_data_manager_cleans_non_finite_graph_rows_before_build(tmp_path: Path, logger) -> None:
    pytest.importorskip("torch")
    static_df = pd.DataFrame(
        {
            "point_id": [1, 2, 3],
            "lat": [0.0, 0.5, 1.0],
            "lon": [0.0, 0.5, 1.0],
            "target_a": [1.0, 2.0, 3.0],
            "static_1": [10.0, 11.0, 12.0],
            "static_2": [20.0, np.nan, 22.0],
        }
    )
    # One NaN in three rows is 33% of the column, which the sparsity gate rejects outright. This
    # test is about the graph path still DROPPING such a row, so the gate is opened for it.
    timeseries_df = pd.DataFrame(
        {
            "point_id": [1, 1, 2, 2, 3, 3],
            "month": ["2020-01", "2020-02", "2020-01", "2020-02", "2020-01", "2020-02"],
            "S1_vv": [0.1, 0.0, 0.3, 0.4, 0.5, np.nan],
            "S2_b2": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            "MODIS_ndvi": [2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
        }
    )
    static_path = tmp_path / "static.csv"
    timeseries_path = tmp_path / "timeseries.csv"
    static_df.to_csv(static_path, index=False)
    timeseries_df.to_csv(timeseries_path, index=False)

    config = SimpleNamespace(
        DATA_FOLDER=str(tmp_path),
        DATA_FILE="static.csv",
        STATIC_CSV_PATH=str(static_path),
        TIMESERIES_CSV_PATH=str(timeseries_path),
        POINT_ID_COLUMN="point_id",
        LAT_COLUMN="lat",
        LON_COLUMN="lon",
        TIME_COLUMN="month",
        TEMPORAL_FEATURES_ENABLED=True,
        MODALITY_PREFIX_MAP={"S1": "S1_", "S2": "S2_", "MODIS": "MODIS_"},
        S1_COLUMNS=[],
        S2_COLUMNS=[],
        MODIS_COLUMNS=[],
        TARGET_COLUMNS=["target_a"],
        ELIMINATED_FEATURES=[],
        TEST_SIZE=0.33,
        LIGHTNING_VAL_SIZE=0.33,
        RANDOM_SEED=42,
        SPATIAL_RADIUS=2.0,
        BASELINE_METHOD="knn",
        BASELINE_K_NEIGHBORS=1,
        MAX_MISSING_COLUMN_RATIO=1.0,
    )

    manager = DataManager(config=config, logger=logger)
    bundle = SpatiotemporalGraphBuilder(manager.config, manager.logger, manager).build(graph_data_args={"spatial_graph_enabled": False})

    assert bundle["point_ids"] == [1, 3]
    assert bundle["static_features"].shape[0] == 2
    assert bundle["temporal_features"]["s1"].shape[0] == 2
    assert np.isfinite(bundle["static_features"]).all()
    assert np.isfinite(bundle["temporal_features"]["s1"]).all()


def test_build_edge_index_uses_sparse_radius_neighbors(monkeypatch) -> None:
    from sklearn.neighbors import NearestNeighbors

    calls = {}

    def fake_radius_neighbors(self, X=None, return_distance=True, sort_results=False):
        calls["called"] = True
        calls["shape"] = None if X is None else np.asarray(X).shape
        return [np.asarray([0, 1]), np.asarray([0, 1])], [np.asarray([0, 1]), np.asarray([1, 0])]

    monkeypatch.setattr(NearestNeighbors, "radius_neighbors", fake_radius_neighbors)

    builder = SpatiotemporalGraphBuilder(SimpleNamespace(), SimpleNamespace(), None)
    edge_index, edge_attr = builder.build_edge_index(
        coords=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        radius=2.0,
        residuals=np.asarray([[0.1], [0.2]], dtype=np.float32),
    )

    assert calls["called"] is True
    assert calls["shape"] == (2, 2)
    assert edge_index.shape == (2, 2)
    assert edge_attr.shape == (2, 1)
    assert edge_index.tolist() == [[1, 0], [0, 1]]


def test_single_node_graph_datamodule_returns_graph_batches(tmp_path: Path, logger) -> None:
    torch = pytest.importorskip("torch")
    static_path, timeseries_path = _write_graph_csvs(tmp_path)
    config = SimpleNamespace(
        DATA_FOLDER=str(tmp_path),
        DATA_FILE="static.csv",
        STATIC_CSV_PATH=str(static_path),
        TIMESERIES_CSV_PATH=str(timeseries_path),
        POINT_ID_COLUMN="point_id",
        LAT_COLUMN="lat",
        LON_COLUMN="lon",
        TIME_COLUMN="month",
        TEMPORAL_FEATURES_ENABLED=True,
        MODALITY_PREFIX_MAP={"S1": "S1_", "S2": "S2_", "MODIS": "MODIS_"},
        S1_COLUMNS=[],
        S2_COLUMNS=[],
        MODIS_COLUMNS=[],
        TARGET_COLUMNS=["target_a"],
        ELIMINATED_FEATURES=[],
        TEST_SIZE=0.33,
        LIGHTNING_VAL_SIZE=0.33,
        RANDOM_SEED=42,
        SPATIAL_RADIUS=2.0,
        BASELINE_K_NEIGHBORS=1,
    )

    manager = DataManager(config=config, logger=logger)
    bundle = SpatiotemporalGraphBuilder(manager.config, manager.logger, manager).build()

    datamodule = SingleNodeGraphDataModule(spatiotemporal_graph=bundle, batch_size=1, val_size=0.33, num_workers=0, pin_memory=False, persistent_workers=False, seed=42)
    datamodule.setup("fit")

    train_batch = next(iter(datamodule.train_dataloader()))
    val_batch = next(iter(datamodule.val_dataloader()))
    test_batch = next(iter(datamodule.test_dataloader()))

    assert datamodule.static_dim == 2
    assert datamodule.target_dim == 1
    assert datamodule.modality_dims == {"s1": 1, "s2": 1, "modis": 1}
    assert datamodule.temporal_steps == 2
    assert train_batch["x_static"].shape[1] == 2
    assert train_batch["y"].shape[1] == 1
    assert train_batch["edge_index"].shape[0] == 2
    assert train_batch["temporal_lengths"]["s1"].ndim == 1
    assert train_batch["temporal_masks"]["s1"].ndim == 2
    assert isinstance(train_batch["x_static"], torch.Tensor)
    assert val_batch["x_static"].shape[1] == 2
    assert test_batch["x_static"].shape[1] == 2
    assert list(datamodule.X_test_frame_.columns) == ["static_1", "static_2"]
    assert list(datamodule.y_test_frame_.columns) == ["target_a"]


def test_single_node_graph_datamodule_reindexes_edges_within_each_split() -> None:
    """Subgraph edge_index must be renumbered to local node ids, not left at global ids.

    The stale lightning_graph copy masked edges without reindexing, leaving global indices that
    are out of range for a subgraph. The batch sizes here are chosen so edges actually survive
    batching - the `assert numel() > 0` below is what stops this test silently going vacuous, as
    an earlier `if numel() == 0: continue` version of it did.
    """
    torch = pytest.importorskip("torch")
    n = 200
    # A dense-ish chain plus long-range edges, so any contiguous batch retains some of them.
    sources = np.concatenate([np.arange(n - 1), np.arange(n - 2)])
    targets = np.concatenate([np.arange(1, n), np.arange(2, n)])
    graph = {
        "point_ids": list(range(n)),
        "static_features": np.random.default_rng(0).normal(size=(n, 3)).astype(np.float32),
        "targets": np.random.default_rng(1).normal(size=(n, 1)).astype(np.float32),
        "coords": np.zeros((n, 2), dtype=np.float32),
        "edge_index": np.stack([sources, targets]).astype(np.int64),
        "edge_attr": np.zeros((sources.size, 1), dtype=np.float32),
        "spatial_graph_enabled": True,
    }

    datamodule = SingleNodeGraphDataModule(
        spatiotemporal_graph=graph, batch_size=64, test_size=0.25, val_size=0.25,
        num_workers=0, seed=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # the batching guard fires by design here
        datamodule.setup("fit")

    checked = 0
    for name, loader in (
        ("train", datamodule.train_dataloader()),
        ("val", datamodule.val_dataloader()),
        ("test", datamodule.test_dataloader()),
    ):
        for batch in loader:
            num_nodes = batch["x_static"].shape[0]
            edge_index = batch["edge_index"]
            if edge_index.numel() == 0:
                continue
            assert int(edge_index.min()) >= 0, f"{name}: negative node id after reindexing"
            assert int(edge_index.max()) < num_nodes, (
                f"{name}: edge_index max {int(edge_index.max())} >= {num_nodes} nodes - "
                "indices were not remapped to the subgraph"
            )
            checked += 1

    assert checked > 0, "fixture retained no edges in any batch; the test would assert nothing"


def test_single_node_graph_datamodule_exposes_predict_dataloader(tmp_path: Path, logger) -> None:
    pytest.importorskip("torch")
    static_path, timeseries_path = _write_graph_csvs(tmp_path)
    config = SimpleNamespace(
        DATA_FOLDER=str(tmp_path),
        DATA_FILE="static.csv",
        STATIC_CSV_PATH=str(static_path),
        TIMESERIES_CSV_PATH=str(timeseries_path),
        POINT_ID_COLUMN="point_id",
        LAT_COLUMN="lat",
        LON_COLUMN="lon",
        TIME_COLUMN="month",
        TEMPORAL_FEATURES_ENABLED=True,
        MODALITY_PREFIX_MAP={"S1": "S1_", "S2": "S2_", "MODIS": "MODIS_"},
        S1_COLUMNS=[],
        S2_COLUMNS=[],
        MODIS_COLUMNS=[],
        TARGET_COLUMNS=["target_a"],
        ELIMINATED_FEATURES=[],
        TEST_SIZE=0.33,
        LIGHTNING_VAL_SIZE=0.33,
        RANDOM_SEED=42,
        SPATIAL_RADIUS=2.0,
        BASELINE_K_NEIGHBORS=1,
    )

    manager = DataManager(config=config, logger=logger)
    bundle = SpatiotemporalGraphBuilder(manager.config, manager.logger, manager).build()

    datamodule = SingleNodeGraphDataModule(
        spatiotemporal_graph=bundle,
        batch_size=1,
        val_size=0.33,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        seed=42,
    )
    datamodule.setup("fit")

    predict_batch = next(iter(datamodule.predict_dataloader()))

    assert predict_batch["x_static"].shape[1] == 2
    assert predict_batch["y"].shape[1] == 1


def test_soil_graph_lightning_module_forward_pass_runs() -> None:
    pytest.importorskip("torch")

    model = SoilGraphLightningModule(
        static_dim=2,
        target_dim=1,
        hidden_dim=8,
        temporal_hidden_dim=4,
        modality_dims={},
        temporal_steps=None,
        edge_attr_dim=1,
        num_graph_layers=1,
        dropout=0.0,
        learning_rate=1e-3,
        temporal_enabled=False,
    )

    batch = {
        "x_static": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32),
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        "edge_attr": torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32),
        "y": torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32),
        "temporal_features": {},
    }

    predictions = model.forward(batch)

    assert predictions.shape == (3, 1)


def test_soil_graph_lightning_module_configures_adamw_with_plateau_scheduler() -> None:
    pytest.importorskip("torch")

    model = SoilGraphLightningModule(
        static_dim=2,
        target_dim=1,
        hidden_dim=8,
        temporal_hidden_dim=4,
        modality_dims={},
        temporal_steps=None,
        edge_attr_dim=1,
        num_graph_layers=1,
        dropout=0.0,
        learning_rate=1e-3,
        weight_decay=1e-4,
        optimizer_name="adamw",
        scheduler_type="plateau",
        scheduler_factor=0.5,
        scheduler_patience=2,
        scheduler_min_lr=1e-6,
        scheduler_monitor="val_loss",
        temporal_enabled=False,
    )

    configured = model.configure_optimizers()

    assert isinstance(configured, dict)
    optimizer = configured["optimizer"]
    scheduler_config = configured["lr_scheduler"]

    assert optimizer.__class__.__name__ == "AdamW"
    assert optimizer.defaults["lr"] == pytest.approx(1e-3)
    assert optimizer.defaults["weight_decay"] == pytest.approx(1e-4)
    assert scheduler_config["monitor"] == "val_loss"
    assert scheduler_config["interval"] == "epoch"
    assert scheduler_config["frequency"] == 1
    assert scheduler_config["scheduler"].__class__.__name__ == "ReduceLROnPlateau"


def test_soil_graph_lightning_module_can_disable_spatial_graph() -> None:
    torch = pytest.importorskip("torch")

    class _FailingGraphBlock(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("spatial graph block should not run when disabled")

    model = SoilGraphLightningModule(
        static_dim=2,
        target_dim=1,
        hidden_dim=8,
        temporal_hidden_dim=4,
        modality_dims={},
        temporal_steps=None,
        edge_attr_dim=1,
        num_graph_layers=1,
        dropout=0.0,
        learning_rate=1e-3,
        temporal_enabled=False,
        spatial_graph_enabled=False,
    )
    model.graph_blocks = torch.nn.ModuleList([_FailingGraphBlock()])

    batch = {
        "x_static": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_attr": torch.tensor([[0.1], [0.2]], dtype=torch.float32),
        "y": torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        "temporal_features": {},
    }

    predictions = model.forward(batch)

    assert predictions.shape == (2, 1)


def test_soil_graph_lightning_module_uses_temporal_lengths_and_masks() -> None:
    torch = pytest.importorskip("torch")

    model = SoilGraphLightningModule(
        static_dim=2,
        target_dim=1,
        hidden_dim=8,
        temporal_hidden_dim=4,
        temporal_lstm_hidden_dim=4,
        temporal_lstm_num_layers=2,
        temporal_lstm_dropout=0.1,
        temporal_lstm_bidirectional=False,
        temporal_pooling="last",
        modality_dims={"s1": 2, "s2": 1, "modis": 3},
        temporal_steps=4,
        edge_attr_dim=1,
        num_graph_layers=1,
        dropout=0.0,
        learning_rate=1e-3,
        temporal_enabled=True,
    )

    batch = {
        "x_static": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_attr": torch.tensor([[0.1], [0.2]], dtype=torch.float32),
        "y": torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        "temporal_features": {
            "s1": torch.tensor(
                [
                    [[1.0, 0.0], [2.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    [[3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [0.0, 0.0]],
                ],
                dtype=torch.float32,
            ),
            "s2": torch.tensor(
                [
                    [[1.0], [0.0], [0.0], [0.0]],
                    [[2.0], [3.0], [0.0], [0.0]],
                ],
                dtype=torch.float32,
            ),
            "modis": torch.tensor(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[7.0, 8.0, 9.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ],
                dtype=torch.float32,
            ),
        },
        "temporal_lengths": {
            "s1": torch.tensor([2, 3], dtype=torch.long),
            "s2": torch.tensor([1, 2], dtype=torch.long),
            "modis": torch.tensor([2, 1], dtype=torch.long),
        },
        "temporal_masks": {
            "s1": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.long),
            "s2": torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]], dtype=torch.long),
            "modis": torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.long),
        },
    }

    predictions = model.forward(batch)

    assert predictions.shape == (2, 1)


def test_soil_graph_lightning_module_validation_fails_on_nan_targets() -> None:
    torch = pytest.importorskip("torch")

    model = SoilGraphLightningModule(
        static_dim=2,
        target_dim=1,
        hidden_dim=8,
        temporal_hidden_dim=4,
        modality_dims={},
        temporal_steps=None,
        edge_attr_dim=1,
        num_graph_layers=1,
        dropout=0.0,
        learning_rate=1e-3,
        temporal_enabled=False,
    )

    batch = {
        "x_static": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_attr": torch.tensor([[0.1], [0.2]], dtype=torch.float32),
        "y": torch.tensor([[1.0], [float("nan")]], dtype=torch.float32),
        "temporal_features": {},
    }

    with pytest.raises(ValueError, match="Non-finite target values encountered during val step"):
        model.validation_step(batch, 0)


def test_data_manager_parses_generic_temporal_prefixes_and_model_uses_them(tmp_path: Path, logger) -> None:
    torch = pytest.importorskip("torch")

    static_df = pd.DataFrame(
        {
            "point_id": [1, 2],
            "lat": [0.0, 1.0],
            "lon": [0.0, 1.0],
            "target_a": [1.0, 2.0],
            "static_1": [10.0, 11.0],
        }
    )
    timeseries_df = pd.DataFrame(
        {
            "point_id": [1, 1, 2, 2],
            "month": ["2020-01", "2020-02", "2020-01", "2020-02"],
            "RADAR_vh": [0.1, 0.2, 0.3, 0.4],
            "OPT_ndvi": [1.1, 1.2, 1.3, 1.4],
        }
    )

    static_path = tmp_path / "static.csv"
    timeseries_path = tmp_path / "timeseries.csv"
    static_df.to_csv(static_path, index=False)
    timeseries_df.to_csv(timeseries_path, index=False)

    config = SimpleNamespace(
        DATA_FOLDER=str(tmp_path),
        DATA_FILE="static.csv",
        STATIC_CSV_PATH=str(static_path),
        TIMESERIES_CSV_PATH=str(timeseries_path),
        POINT_ID_COLUMN="point_id",
        LAT_COLUMN="lat",
        LON_COLUMN="lon",
        TIME_COLUMN="month",
        TEMPORAL_FEATURES={"enabled": True, "time_column": "month", "modality_prefix_map": {"radar": "RADAR_", "optical": "OPT_"}},
        MODALITY_PREFIX_MAP={},
        TARGET_COLUMNS=["target_a"],
        ELIMINATED_FEATURES=[],
        TEST_SIZE=0.5,
        LIGHTNING_VAL_SIZE=0.5,
        RANDOM_SEED=42,
        SPATIAL_RADIUS=2.0,
        BASELINE_METHOD="knn",
        BASELINE_K_NEIGHBORS=1,
    )

    manager = DataManager(config=config, logger=logger)
    bundle = SpatiotemporalGraphBuilder(manager.config, manager.logger, manager).build()

    datamodule = SingleNodeGraphDataModule(
        spatiotemporal_graph=bundle,
        batch_size=1,
        val_size=0.5,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        seed=42,
    )
    datamodule.setup("fit")

    assert set(bundle["temporal_features"]) == {"radar", "optical"}
    assert bundle["temporal_features"]["radar"].shape == (2, 2, 1)
    assert datamodule.modality_dims == {"radar": 1, "optical": 1}

    model = SoilGraphLightningModule(
        static_dim=2,
        target_dim=1,
        hidden_dim=8,
        temporal_hidden_dim=4,
        modality_dims=datamodule.modality_dims,
        temporal_steps=2,
        edge_attr_dim=1,
        num_graph_layers=1,
        dropout=0.0,
        learning_rate=1e-3,
        temporal_enabled=True,
    )

    batch = {
        "x_static": torch.tensor([[10.0, 20.0], [11.0, 21.0]], dtype=torch.float32),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_attr": torch.tensor([[0.1], [0.2]], dtype=torch.float32),
        "y": torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        "temporal_features": {
            "radar": torch.tensor([[[0.1], [0.2]], [[0.3], [0.4]]], dtype=torch.float32),
            "optical": torch.tensor([[[1.1], [1.2]], [[1.3], [1.4]]], dtype=torch.float32),
        },
    }

    predictions = model.forward(batch)

    assert predictions.shape == (2, 1)

def test_graph_datamodule_split_uses_test_size_and_val_size_independently() -> None:
    """_split_indices used to pass val_size as the test fraction, ignoring TEST_SIZE entirely."""
    pytest.importorskip("torch")
    num_nodes = 100
    graph = {
        "point_ids": list(range(num_nodes)),
        "static_features": np.zeros((num_nodes, 2), dtype=np.float32),
        "targets": np.zeros((num_nodes, 1), dtype=np.float32),
        "coords": np.zeros((num_nodes, 2), dtype=np.float32),
        "edge_index": np.zeros((2, 0), dtype=np.int64),
        "edge_attr": np.zeros((0, 1), dtype=np.float32),
    }

    datamodule = SingleNodeGraphDataModule(
        spatiotemporal_graph=graph, batch_size=1, test_size=0.5, val_size=0.1, seed=42
    )
    datamodule.setup("fit")

    train_n = len(datamodule.X_train_frame_)
    val_n = len(datamodule.X_val_frame_)
    test_n = len(datamodule.X_test_frame_)

    assert test_n == 50  # TEST_SIZE, not val_size
    assert val_n == 5  # 10% of the remaining 50
    assert train_n == 45
    assert train_n + val_n + test_n == num_nodes


def test_graph_datamodule_splits_without_indices_in_the_bundle() -> None:
    """Splitting is the datamodule's job now; the bundle carries no train/val/test indices."""
    pytest.importorskip("torch")
    num_nodes = 10
    graph = {
        "point_ids": list(range(num_nodes)),
        "static_features": np.zeros((num_nodes, 2), dtype=np.float32),
        "targets": np.zeros((num_nodes, 1), dtype=np.float32),
        "coords": np.zeros((num_nodes, 2), dtype=np.float32),
        "edge_index": np.zeros((2, 0), dtype=np.int64),
        "edge_attr": np.zeros((0, 1), dtype=np.float32),
    }

    datamodule = SingleNodeGraphDataModule(spatiotemporal_graph=graph, batch_size=1, test_size=0.2, val_size=0.25)
    datamodule.setup("fit")

    assert not hasattr(datamodule.spatiotemporal_graph, "train_idx")
    assert len(datamodule.X_test_frame_) == 2
    assert len(datamodule.X_val_frame_) == 2
    assert len(datamodule.X_train_frame_) == 6


# --- per-branch sizing, head depth, normalization --------------------------


def _model(**overrides):
    pytest.importorskip("torch")
    from yg_eo_soilnet.models.lightningmodules.soil_graph_lightning_module import SoilGraphLightningModule
    kwargs = dict(
        static_dim=11, target_dim=1,
        modality_dims={"s2": 10, "s1": 8, "soil": 3},
        temporal_steps=12, temporal_enabled=True, spatial_graph_enabled=False,
    )
    kwargs.update(overrides)
    return SoilGraphLightningModule(**kwargs)


def test_scalar_temporal_hidden_dim_applies_one_width_to_every_modality() -> None:
    model = _model(temporal_lstm_hidden_dim=64, static_hidden_dims=[64])

    assert {name: enc.hidden_size for name, enc in model.temporal_encoders.items()} == {
        "s2": 64, "s1": 64, "soil": 64
    }
    assert model.fusion_input_dim == 64 + 3 * 64


def test_dict_temporal_hidden_dim_sizes_each_modality_independently() -> None:
    model = _model(
        static_hidden_dims=[24],
        temporal_lstm_hidden_dim={"s2": 48, "s1": 32, "soil": 16},
    )

    assert {name: enc.hidden_size for name, enc in model.temporal_encoders.items()} == {
        "s2": 48, "s1": 32, "soil": 16
    }
    assert model.fusion_input_dim == 24 + 48 + 32 + 16
    # the static branch projects from its own width, not the shared hidden_dim
    assert model.static_encoder[0].out_features == 24


def test_output_head_is_a_bare_linear_by_default() -> None:
    torch = pytest.importorskip("torch")

    model = _model()

    assert isinstance(model.output_head, torch.nn.Linear)
    assert model.output_head.out_features == 1


def test_output_head_becomes_a_tapering_mlp_when_enabled() -> None:
    torch = pytest.importorskip("torch")

    model = _model(head_hidden_dims=[64, 32], static_hidden_dims=[24],
                   temporal_lstm_hidden_dim={"s2": 48, "s1": 32, "soil": 16})

    assert isinstance(model.output_head, torch.nn.Sequential)
    widths = [layer.out_features for layer in model.output_head if isinstance(layer, torch.nn.Linear)]
    assert widths == [64, 32, 1]  # taper, then the readout

    batch = {
        "x_static": torch.zeros(5, 11),
        "edge_index": torch.zeros((2, 0), dtype=torch.long),
        "edge_attr": torch.zeros((0, 1)),
        "temporal_features": {name: torch.zeros(5, 12, dim) for name, dim in (("s2", 10), ("s1", 8), ("soil", 3))},
        "temporal_lengths": {name: torch.full((5,), 12, dtype=torch.long) for name in ("s2", "s1", "soil")},
        "temporal_masks": {name: torch.ones(5, 12, dtype=torch.long) for name in ("s2", "s1", "soil")},
        "temporal_enabled": True,
    }
    assert model(batch).shape == (5, 1)


def test_use_layer_norm_false_reproduces_the_unnormalized_tree() -> None:
    torch = pytest.importorskip("torch")

    assert sum(1 for m in _model(use_layer_norm=True).modules() if isinstance(m, torch.nn.LayerNorm)) > 0
    assert sum(1 for m in _model(use_layer_norm=False).modules() if isinstance(m, torch.nn.LayerNorm)) == 0


def test_residual_graph_block_passes_its_input_through_when_the_update_is_zero() -> None:
    """Proves the skip connection exists - the block used to return only the update."""
    torch = pytest.importorskip("torch")
    from yg_eo_soilnet.models.lightningmodules.soil_graph_lightning_module import ResidualGraphBlock

    block = ResidualGraphBlock(hidden_dim=4, edge_attr_dim=0, dropout=0.0, use_layer_norm=False)
    for parameter in block.node_update.parameters():
        torch.nn.init.zeros_(parameter)

    node_features = torch.randn(3, 4)
    out = block(node_features, edge_index=torch.zeros((2, 0), dtype=torch.long))

    torch.testing.assert_close(out, node_features)


def test_static_fallback_width_matches_the_fusion_dim_when_there_are_no_static_features() -> None:
    """static_dim=0 used to emit a hidden_dim-wide vector while fusion expected static_hidden_dim."""
    torch = pytest.importorskip("torch")

    model = _model(
        static_dim=0, static_hidden_dims=[24], hidden_dim=64,
        temporal_lstm_hidden_dim={"s2": 48, "s1": 32, "soil": 16},
    )
    assert model.fusion_input_dim == 24 + 48 + 32 + 16

    batch = {
        "x_static": torch.zeros(5, 0),
        "edge_index": torch.zeros((2, 0), dtype=torch.long),
        "edge_attr": torch.zeros((0, 1)),
        "temporal_features": {n: torch.zeros(5, 12, d) for n, d in (("s2", 10), ("s1", 8), ("soil", 3))},
        "temporal_lengths": {n: torch.full((5,), 12, dtype=torch.long) for n in ("s2", "s1", "soil")},
        "temporal_masks": {n: torch.ones(5, 12, dtype=torch.long) for n in ("s2", "s1", "soil")},
        "temporal_enabled": True,
    }
    assert model(batch).shape == (5, 1)


def test_head_widths_are_built_exactly_as_listed() -> None:
    """No shape rule hides in the model: the config names every width the head has.

    The taper used to be implicit - head_num_layers + a halving hidden_dim floored at a minimum -
    and every registry comment describing one of these heads had drifted from what it built. The
    pyramid is now drawn in hpo/constraints.py and arrives here as an ordinary list.
    """
    torch = pytest.importorskip("torch")

    model = _model(
        head_hidden_dims=[128, 64, 32, 16, 16],
        static_hidden_dims=[24], temporal_lstm_hidden_dim={"s2": 48, "s1": 32, "soil": 16},
    )

    widths = [layer.out_features for layer in model.output_head if isinstance(layer, torch.nn.Linear)]
    assert widths == [128, 64, 32, 16, 16, 1]


def test_unsized_modality_raises_instead_of_silently_inheriting_a_default() -> None:
    pytest.importorskip("torch")

    with pytest.raises(ValueError, match="'soil' has no width in temporal_lstm_hidden_dim"):
        _model(temporal_lstm_hidden_dim={"s2": 48, "s1": 32})   # 'soil' omitted


def test_batching_guard_warns_when_the_graph_is_on_but_edges_do_not_survive() -> None:
    """Mini-batching keeps only edges with both endpoints in the same batch."""
    pytest.importorskip("torch")
    n = 200
    graph = {
        "point_ids": list(range(n)),
        "static_features": np.zeros((n, 2), dtype=np.float32),
        "targets": np.zeros((n, 1), dtype=np.float32),
        "coords": np.zeros((n, 2), dtype=np.float32),
        "edge_index": np.stack([np.arange(n - 1), np.arange(1, n)]).astype(np.int64),
        "edge_attr": np.zeros((n - 1, 1), dtype=np.float32),
        "spatial_graph_enabled": True,
    }

    datamodule = SingleNodeGraphDataModule(spatiotemporal_graph=graph, batch_size=8, test_size=0.2, val_size=0.2)
    with pytest.warns(RuntimeWarning, match="message passing is effectively inert"):
        datamodule.setup("fit")


def test_batching_guard_is_silent_when_the_spatial_graph_is_off() -> None:
    pytest.importorskip("torch")
    import warnings

    n = 200
    graph = {
        "point_ids": list(range(n)),
        "static_features": np.zeros((n, 2), dtype=np.float32),
        "targets": np.zeros((n, 1), dtype=np.float32),
        "coords": np.zeros((n, 2), dtype=np.float32),
        "edge_index": np.stack([np.arange(n - 1), np.arange(1, n)]).astype(np.int64),
        "edge_attr": np.zeros((n - 1, 1), dtype=np.float32),
        "spatial_graph_enabled": False,
    }

    datamodule = SingleNodeGraphDataModule(spatiotemporal_graph=graph, batch_size=8, test_size=0.2, val_size=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        datamodule.setup("fit")


# --- mean-collapse regressions ---------------------------------------------


def _temporal_graph(num_nodes: int = 10, time_steps: int = 4, observed=(0, 1), value: float = 10.0):
    """A graph whose observations sit only at `observed` steps; every other step is padding."""
    features = np.zeros((num_nodes, time_steps, 1), dtype=np.float32)
    mask = np.zeros((num_nodes, time_steps), dtype=bool)
    for step in observed:
        features[:, step, 0] = value
        mask[:, step] = True
    return {
        "point_ids": list(range(num_nodes)),
        "static_features": np.zeros((num_nodes, 2), dtype=np.float32),
        "targets": np.arange(num_nodes, dtype=np.float32).reshape(-1, 1),
        "coords": np.zeros((num_nodes, 2), dtype=np.float32),
        "edge_index": np.zeros((2, 0), dtype=np.int64),
        "edge_attr": np.zeros((0, 1), dtype=np.float32),
        "temporal_enabled": True,
        "temporal_features": {"s1": features},
        "temporal_masks": {"s1": mask},
        "temporal_lengths": {"s1": mask.sum(axis=1).astype(np.int64)},
    }


def test_temporal_stats_ignore_padding_when_the_train_split_is_a_subset() -> None:
    """The mask spans every node while the values are already subset to train.

    Comparing the two raw shapes is always False once a split exists, so the statistics used to be
    fitted over the zero padding - dragging every channel mean toward 0 and inflating its scale.
    """
    pytest.importorskip("torch")
    datamodule = SingleNodeGraphDataModule(
        spatiotemporal_graph=_temporal_graph(num_nodes=10, observed=(0, 1), value=10.0),
        batch_size=2,
        test_size=0.2,
        val_size=0.25,
        seed=42,
    )
    datamodule.setup("fit")

    assert len(datamodule.X_train_frame_) < 10  # the split must be a strict subset
    # Observed cells only: mean 10.0, sd 0. Counting the padding would give 5.0 and sd 5.0.
    assert datamodule.temporal_mean_["s1"] == pytest.approx([10.0])
    assert datamodule.temporal_scale_["s1"] == pytest.approx([1.0])  # zero sd is floored to 1.0


def test_unobserved_timesteps_standardize_to_zero_not_a_spike() -> None:
    """A gap is a raw 0, which standardizes to -mean/scale - a multi-sigma square wave."""
    pytest.importorskip("torch")
    datamodule = SingleNodeGraphDataModule(
        spatiotemporal_graph=_temporal_graph(num_nodes=10, observed=(0, 1), value=10.0),
        batch_size=2,
        test_size=0.2,
        val_size=0.25,
        seed=42,
    )
    datamodule.setup("fit")

    sample = datamodule._build_graph_sample(np.array([0, 1], dtype=np.int64))
    values = sample.temporal_features["s1"]

    assert values[:, 2, :].abs().max().item() == 0.0
    assert values[:, 3, :].abs().max().item() == 0.0


def test_temporal_encoder_reads_observations_past_the_observed_count() -> None:
    """temporal_lengths is a COUNT, not a length.

    Observations are written at absolute positions on the shared date axis, so packing to the count
    truncated every point whose readings were not a dense prefix - discarding the later years.
    """
    torch = pytest.importorskip("torch")

    model = SoilGraphLightningModule(
        static_dim=2,
        target_dim=1,
        modality_dims={"s1": 1},
        temporal_steps=4,
        temporal_enabled=True,
        spatial_graph_enabled=False,
        temporal_pooling="attention",
        temporal_lstm_hidden_dim=4,
        head_hidden_dims=[],
        dropout=0.0,
    )
    model.eval()

    # The two rows are identical over steps 0-1 and differ only at steps 2-3, which is exactly the
    # region a count-as-length encoder never reaches (count == 2).
    mask = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.long)
    batch = {
        "x_static": torch.zeros(2, 2),
        "edge_index": torch.zeros((2, 0), dtype=torch.long),
        "edge_attr": torch.zeros((0, 1)),
        "temporal_features": {
            "s1": torch.tensor(
                [[[0.0], [0.0], [7.0], [7.0]], [[0.0], [0.0], [-7.0], [-7.0]]],
                dtype=torch.float32,
            )
        },
        "temporal_masks": {"s1": mask},
        "temporal_lengths": {"s1": mask.sum(dim=1)},
        "temporal_enabled": True,
    }

    with torch.no_grad():
        predictions = model.forward(batch)

    assert not torch.allclose(predictions[0], predictions[1])


def test_last_pooling_reads_the_last_observed_step_not_the_end_of_the_sequence() -> None:
    torch = pytest.importorskip("torch")

    model = SoilGraphLightningModule(
        static_dim=1,
        target_dim=1,
        modality_dims={"s1": 1},
        temporal_steps=5,
        temporal_enabled=True,
        spatial_graph_enabled=False,
        temporal_pooling="last",
        temporal_lstm_hidden_dim=3,
        dropout=0.0,
    )
    model.eval()

    step_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)
    features = torch.tensor([[[1.0], [2.0], [3.0], [0.0], [0.0]]], dtype=torch.float32)
    with torch.no_grad():
        outputs, hidden = model._run_lstm(model.temporal_encoders["s1"], features, step_mask.bool(), None)
        pooled = model._last_hidden_state(model.temporal_encoders["s1"], hidden, outputs, step_mask.bool())

    assert torch.allclose(pooled, outputs[:, 2, :])  # index 2 is the last observed step
    assert not torch.allclose(pooled, outputs[:, -1, :])


def test_builder_warns_when_observations_are_not_a_dense_prefix() -> None:
    observed_mask = np.array([[True, False, True, True], [True, True, False, False]], dtype=bool)

    with pytest.warns(RuntimeWarning, match="not a dense prefix"):
        SpatiotemporalGraphBuilder._warn_if_not_prefix_observed("s1", observed_mask)


def test_builder_is_silent_when_observations_are_a_dense_prefix() -> None:
    import warnings

    observed_mask = np.array([[True, True, False, False], [True, False, False, False]], dtype=bool)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        SpatiotemporalGraphBuilder._warn_if_not_prefix_observed("s1", observed_mask)


def test_output_head_leaves_the_readout_input_unnormalized() -> None:
    """LayerNorm before the readout strips the magnitude a regressor needs to reach the tails."""
    torch = pytest.importorskip("torch")

    head = _model(head_hidden_dims=[64, 32, 32], use_layer_norm=True).output_head
    modules = list(head)
    last_hidden_linear = max(i for i, m in enumerate(modules[:-1]) if isinstance(m, torch.nn.Linear))
    tail = modules[last_hidden_linear + 1 : -1]

    assert not any(isinstance(m, (torch.nn.LayerNorm, torch.nn.Dropout)) for m in tail)
    assert any(isinstance(m, torch.nn.LayerNorm) for m in modules[:last_hidden_linear])


@pytest.mark.parametrize(
    "norm_type,expected",
    [("batch", "BatchNorm1d"), ("layer", "LayerNorm"), ("none", "Identity")],
)
def test_fusion_norm_type_selects_the_normalization(norm_type, expected) -> None:
    pytest.importorskip("torch")

    assert type(_model(fusion_norm_type=norm_type).fusion_norm).__name__ == expected


def test_fusion_norm_type_rejects_unknown_values() -> None:
    pytest.importorskip("torch")

    with pytest.raises(ValueError, match="fusion_norm_type"):
        _model(fusion_norm_type="instance")


@pytest.mark.parametrize(
    "loss_name,expected",
    [("mse", "MSELoss"), ("huber", "HuberLoss"), ("smooth_l1", "SmoothL1Loss")],
)
def test_loss_name_selects_the_criterion(loss_name, expected) -> None:
    pytest.importorskip("torch")

    assert type(_model(loss_name=loss_name).loss_fn).__name__ == expected


def test_loss_name_rejects_unknown_values() -> None:
    pytest.importorskip("torch")

    with pytest.raises(ValueError, match="Unknown loss_name"):
        _model(loss_name="mape")


def test_log1p_target_transform_round_trips_to_original_units() -> None:
    torch = pytest.importorskip("torch")

    raw_targets = np.array([[0.5], [2.0], [4.0], [9.0], [1.0], [3.0], [6.0], [7.0]], dtype=np.float32)
    graph = {
        "point_ids": list(range(len(raw_targets))),
        "static_features": np.zeros((len(raw_targets), 2), dtype=np.float32),
        "targets": raw_targets,
        "coords": np.zeros((len(raw_targets), 2), dtype=np.float32),
        "edge_index": np.zeros((2, 0), dtype=np.int64),
        "edge_attr": np.zeros((0, 1), dtype=np.float32),
    }

    datamodule = SingleNodeGraphDataModule(
        spatiotemporal_graph=graph, batch_size=2, test_size=0.25, val_size=0.25, target_transform="log1p"
    )
    datamodule.setup("fit")

    model = SoilGraphLightningModule(
        static_dim=2,
        target_dim=1,
        temporal_enabled=False,
        spatial_graph_enabled=False,
        target_mean=[float(datamodule.target_mean_[0])],
        target_scale=[float(datamodule.target_scale_[0])],
        target_transform="log1p",
    )

    standardized = datamodule._standardize_targets(raw_targets)
    recovered = model.inverse_transform_targets(torch.as_tensor(standardized))

    assert recovered.numpy() == pytest.approx(raw_targets, rel=1e-4)
    # The standardization must be fitted in transformed space, not raw space.
    assert float(datamodule.target_mean_[0]) == pytest.approx(
        float(np.mean(10.0 * np.log1p(raw_targets[datamodule.train_idx_]))), rel=1e-5
    )


def test_target_transform_defaults_to_a_plain_standardization_round_trip() -> None:
    torch = pytest.importorskip("torch")

    model = SoilGraphLightningModule(
        static_dim=2,
        target_dim=1,
        temporal_enabled=False,
        spatial_graph_enabled=False,
        target_mean=[2.5],
        target_scale=[1.5],
    )

    assert not bool(model.targets_are_log1p)
    recovered = model.inverse_transform_targets(torch.tensor([[0.0], [2.0]])).numpy()
    assert recovered.ravel() == pytest.approx([2.5, 5.5])


def test_shared_step_logs_the_real_batch_size() -> None:
    """Lightning weights the epoch mean by batch_size; a constant 1 let the tail batch dominate."""
    torch = pytest.importorskip("torch")

    model = _model(head_hidden_dims=[], dropout=0.0)
    model.eval()
    recorded = {}
    model.log = lambda name, value, **kwargs: recorded.setdefault(name, kwargs)

    batch = {
        "x_static": torch.zeros(5, 11),
        "edge_index": torch.zeros((2, 0), dtype=torch.long),
        "edge_attr": torch.zeros((0, 1)),
        "y": torch.zeros(5, 1),
        "temporal_features": {name: torch.zeros(5, 12, dim) for name, dim in (("s2", 10), ("s1", 8), ("soil", 3))},
        "temporal_masks": {name: torch.ones(5, 12, dtype=torch.long) for name in ("s2", "s1", "soil")},
        "temporal_enabled": True,
    }

    model._shared_step(batch, "val")

    assert recorded["val_loss"]["batch_size"] == 5


def test_train_loader_drops_a_trailing_partial_batch_but_predict_keeps_every_row() -> None:
    """BatchNorm1d raises on a batch of one, and predictions are aligned to y_test_frame_ by row."""
    pytest.importorskip("torch")
    num_nodes = 101
    graph = {
        "point_ids": list(range(num_nodes)),
        "static_features": np.zeros((num_nodes, 2), dtype=np.float32),
        "targets": np.zeros((num_nodes, 1), dtype=np.float32),
        "coords": np.zeros((num_nodes, 2), dtype=np.float32),
        "edge_index": np.zeros((2, 0), dtype=np.int64),
        "edge_attr": np.zeros((0, 1), dtype=np.float32),
    }

    datamodule = SingleNodeGraphDataModule(
        spatiotemporal_graph=graph, batch_size=16, test_size=0.2, val_size=0.25, seed=42
    )
    datamodule.setup("fit")

    assert datamodule.train_dataloader().drop_last is True
    assert datamodule.predict_dataloader().drop_last is False
    predicted_rows = sum(batch.x_static.shape[0] for batch in datamodule.predict_dataloader())
    assert predicted_rows == len(datamodule.y_test_frame_)


# --- the shared split plan -------------------------------------------------


def _graph_bundle(tmp_path: Path, logger):
    static_path, timeseries_path = _write_graph_csvs(tmp_path)
    config = SimpleNamespace(
        DATA_FOLDER=str(tmp_path),
        DATA_FILE="static.csv",
        STATIC_CSV_PATH=str(static_path),
        TIMESERIES_CSV_PATH=str(timeseries_path),
        POINT_ID_COLUMN="point_id",
        LAT_COLUMN="lat",
        LON_COLUMN="lon",
        TIME_COLUMN="month",
        TEMPORAL_FEATURES_ENABLED=True,
        MODALITY_PREFIX_MAP={"S1": "S1_", "S2": "S2_", "MODIS": "MODIS_"},
        S1_COLUMNS=[],
        S2_COLUMNS=[],
        MODIS_COLUMNS=[],
        TARGET_COLUMNS=["target_a"],
        ELIMINATED_FEATURES=[],
        TEST_SIZE=0.33,
        LIGHTNING_VAL_SIZE=0.33,
        RANDOM_SEED=42,
        SPATIAL_RADIUS=2.0,
        BASELINE_K_NEIGHBORS=1,
    )
    manager = DataManager(config=config, logger=logger)
    return SpatiotemporalGraphBuilder(manager.config, manager.logger, manager).build()


def test_a_zero_val_size_no_longer_raises(tmp_path: Path, logger) -> None:
    """This module called train_test_split directly, which rejects test_size=0 outright.

    The sequence datamodule has always handled it; the graph one raised, so the two families could
    not even be configured the same way.
    """
    pytest.importorskip("torch")
    bundle = _graph_bundle(tmp_path, logger)

    datamodule = SingleNodeGraphDataModule(
        spatiotemporal_graph=bundle, batch_size=len(bundle.point_ids), val_size=0.0, test_size=0.0, seed=42
    )
    datamodule.setup("fit")

    assert datamodule.val_idx_.size == 0
    assert datamodule.test_idx_.size == 0
    assert datamodule.train_idx_.size == len(bundle.point_ids)


def test_a_split_plan_decides_the_graph_holdout(tmp_path: Path, logger) -> None:
    """The graph family resolves the same shared plan as sklearn and the sequence family."""
    pytest.importorskip("torch")
    from yg_eo_soilnet.datamodules.splitting import SplitPlan

    bundle = _graph_bundle(tmp_path, logger)
    point_ids = list(bundle.point_ids)
    assignments = pd.Series("train", index=pd.Index(point_ids, name="point_id"), dtype=object)
    assignments.iloc[0] = "test"
    plan = SplitPlan(
        assignments=assignments,
        strategy="random",
        test_size=0.33,
        val_size=0.0,
        seed=42,
        population_policy="intersect",
    )

    datamodule = SingleNodeGraphDataModule(
        spatiotemporal_graph=bundle, batch_size=len(point_ids), val_size=0.33, test_size=0.33, split_plan=plan
    )
    datamodule.setup("fit")

    assert {point_ids[i] for i in datamodule.test_idx_} == {point_ids[0]}
    assert datamodule.train_idx_.size == len(point_ids) - 1
