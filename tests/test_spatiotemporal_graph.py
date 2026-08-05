"""The typed bundle owns temporal alignment and numeric validation.

These used to be implemented twice - once in the builder, once in the graph datamodule - with the
datamodule's copy silently re-doing work for builder-produced bundles.
"""

import numpy as np
import pytest

from yg_eo_soilnet.datamodules.lightning.spatiotemporal_graph import SpatiotemporalGraph


def _graph(**overrides) -> SpatiotemporalGraph:
    defaults = dict(
        point_ids=[100, 200, 300],
        static_features=np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        targets=np.asarray([[0.1], [0.2], [0.3]], dtype=np.float32),
        coords=np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
    )
    defaults.update(overrides)
    return SpatiotemporalGraph(**defaults)


def test_temporal_arrays_are_aligned_to_static_point_order_at_construction() -> None:
    graph = _graph(
        temporal_features={"s1": np.asarray([[[0.1]], [[0.2]]], dtype=np.float32)},
        temporal_lengths={"s1": np.asarray([1, 1], dtype=np.int64)},
        temporal_masks={"s1": np.asarray([[True], [True]], dtype=np.bool_)},
        temporal_metadata={"point_ids": [200, 300]},  # temporal order, missing point 100
    )

    assert graph.temporal_features["s1"].shape == (3, 1, 1)
    np.testing.assert_allclose(graph.temporal_features["s1"][0], np.asarray([[0.0]], dtype=np.float32))
    np.testing.assert_allclose(graph.temporal_features["s1"][1], np.asarray([[0.1]], dtype=np.float32))
    np.testing.assert_allclose(graph.temporal_features["s1"][2], np.asarray([[0.2]], dtype=np.float32))
    np.testing.assert_array_equal(graph.temporal_lengths["s1"], np.asarray([0, 1, 1], dtype=np.int64))
    # metadata now describes the aligned ordering
    assert graph.temporal_metadata["point_ids"] == [100, 200, 300]


def test_alignment_is_a_noop_when_orders_already_match() -> None:
    features = np.asarray([[[1.0]], [[2.0]], [[3.0]]], dtype=np.float32)
    graph = _graph(
        temporal_features={"s1": features},
        temporal_metadata={"point_ids": [100, 200, 300]},
    )

    np.testing.assert_allclose(graph.temporal_features["s1"], features)


def test_alignment_is_skipped_without_temporal_metadata() -> None:
    features = np.asarray([[[1.0]], [[2.0]]], dtype=np.float32)
    graph = _graph(temporal_features={"s1": features})

    np.testing.assert_allclose(graph.temporal_features["s1"], features)


def test_from_mapping_coerces_and_ignores_unknown_keys() -> None:
    graph = SpatiotemporalGraph.from_mapping(
        {
            "point_ids": [1, 2],
            "static_features": np.zeros((2, 3), dtype=np.float32),
            "train_idx": np.asarray([0, 1]),  # split keys are no longer part of the bundle
            "not_a_field": "ignored",
        }
    )

    assert graph.point_ids == [1, 2]
    assert graph.static_features.shape == (2, 3)
    assert not hasattr(graph, "train_idx")
    assert not hasattr(graph, "not_a_field")


def test_from_mapping_passes_an_existing_graph_through() -> None:
    graph = _graph()

    assert SpatiotemporalGraph.from_mapping(graph) is graph


def test_mapping_style_access_still_works() -> None:
    graph = _graph()

    assert graph["point_ids"] == [100, 200, 300]
    assert graph.get("target_names") == []
    assert graph.get("missing", "fallback") == "fallback"
    assert "static_features" in graph.keys()


def test_validate_names_the_offending_row_and_column() -> None:
    graph = _graph(
        static_features=np.asarray([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]], dtype=np.float32),
        static_feature_names=["feat_a", "feat_b"],
    )

    with pytest.raises(ValueError, match=r"static_features.*row 200.*column 'feat_b'"):
        graph.validate()


def test_validate_passes_for_a_finite_graph() -> None:
    _graph().validate()
