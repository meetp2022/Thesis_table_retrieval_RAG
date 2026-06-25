"""
Unit tests for the feature extraction module (src/graph/feature_extraction.py).

These tests use a lightweight mock for the sentence-transformer model
to keep tests fast and avoid downloading large models in CI.
"""

import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock

import networkx as nx
from torch_geometric.data import Data

from src.data.dataset_loader import TableQARecord
from src.graph.table_to_graph import table_to_graph
from src.graph.feature_extraction import (
    GraphFeatureExtractor,
    NODE_TYPE_MAP,
    EDGE_TYPE_MAP,
    NUM_NODE_TYPES,
)


# ────────────────────────────────────────────────
#  Fixtures
# ────────────────────────────────────────────────

@pytest.fixture
def simple_record():
    return TableQARecord(
        id="feat-001",
        question="What is the population of France?",
        answers=["67 million"],
        table_header=["Country", "Population"],
        table_rows=[
            ["France", "67000000"],
            ["Germany", "83000000"],
        ],
        table_title="European Countries",
        dataset="test",
        domain="general",
    )


@pytest.fixture
def simple_graph(simple_record):
    return table_to_graph(simple_record)


@pytest.fixture
def empty_graph():
    """Graph with no nodes."""
    G = nx.DiGraph()
    G.graph["record_id"] = "empty"
    G.graph["dataset"] = "test"
    return G


@pytest.fixture
def mock_extractor():
    """
    Extractor with a mocked sentence-transformer model that returns
    random 768d vectors (deterministic with fixed seed).
    """
    extractor = GraphFeatureExtractor(
        embedding_model_name="mock-model",
        embedding_dim=768,
    )

    def _mock_embed(texts, **kwargs):
        np.random.seed(42)
        return np.random.randn(len(texts), 768).astype(np.float32)

    extractor._embed_texts = _mock_embed
    return extractor


# ────────────────────────────────────────────────
#  Constants
# ────────────────────────────────────────────────

class TestConstants:
    def test_node_type_map_has_three_types(self):
        assert len(NODE_TYPE_MAP) == 3
        assert set(NODE_TYPE_MAP.keys()) == {"header", "data_cell", "table_metadata"}

    def test_edge_type_map_has_four_types(self):
        assert len(EDGE_TYPE_MAP) == 4
        assert set(EDGE_TYPE_MAP.keys()) == {
            "header_to_cell",
            "row_adjacency",
            "column_adjacency",
            "table_to_header",
        }


# ────────────────────────────────────────────────
#  Single graph conversion
# ────────────────────────────────────────────────

class TestConvert:
    def test_returns_pyg_data(self, mock_extractor, simple_graph):
        data = mock_extractor.convert(simple_graph)
        assert isinstance(data, Data)

    def test_feature_shape(self, mock_extractor, simple_graph):
        data = mock_extractor.convert(simple_graph)
        num_nodes = simple_graph.number_of_nodes()
        # 768 text + 2 position + 3 type + 1 numeric = 774
        assert data.x.shape == (num_nodes, 774)

    def test_edge_index_shape(self, mock_extractor, simple_graph):
        data = mock_extractor.convert(simple_graph)
        num_edges = simple_graph.number_of_edges()
        assert data.edge_index.shape == (2, num_edges)

    def test_edge_type_shape(self, mock_extractor, simple_graph):
        data = mock_extractor.convert(simple_graph)
        num_edges = simple_graph.number_of_edges()
        assert data.edge_type.shape == (num_edges,)

    def test_edge_type_values_valid(self, mock_extractor, simple_graph):
        data = mock_extractor.convert(simple_graph)
        valid_types = set(EDGE_TYPE_MAP.values())
        for et in data.edge_type.tolist():
            assert et in valid_types

    def test_metadata_stored(self, mock_extractor, simple_graph):
        data = mock_extractor.convert(simple_graph)
        assert data.record_id == "feat-001"
        assert data.dataset == "test"

    def test_node_texts_stored(self, mock_extractor, simple_graph):
        data = mock_extractor.convert(simple_graph)
        assert len(data.node_texts) == simple_graph.number_of_nodes()
        assert "France" in data.node_texts

    def test_node_types_stored(self, mock_extractor, simple_graph):
        data = mock_extractor.convert(simple_graph)
        assert "header" in data.node_types
        assert "data_cell" in data.node_types
        assert "table_metadata" in data.node_types

    def test_features_are_float32(self, mock_extractor, simple_graph):
        data = mock_extractor.convert(simple_graph)
        assert data.x.dtype == torch.float32

    def test_edge_index_is_long(self, mock_extractor, simple_graph):
        data = mock_extractor.convert(simple_graph)
        assert data.edge_index.dtype == torch.long


# ────────────────────────────────────────────────
#  Feature components
# ────────────────────────────────────────────────

class TestFeatureComponents:
    def test_position_encoding_normalised(self, mock_extractor):
        # row=50, col=25 with defaults max_rows=100, max_cols=50
        pos = mock_extractor._position_features(50, 25)
        assert pos[0] == pytest.approx(0.5)
        assert pos[1] == pytest.approx(0.5)

    def test_position_negative_row_clipped(self, mock_extractor):
        # Headers have row=-1, should be clipped to 0
        pos = mock_extractor._position_features(-1, 3)
        assert pos[0] == 0.0

    def test_cell_type_onehot_header(self):
        vec = GraphFeatureExtractor._cell_type_onehot("header")
        assert vec.sum() == 1.0
        assert vec[NODE_TYPE_MAP["header"]] == 1.0

    def test_cell_type_onehot_data_cell(self):
        vec = GraphFeatureExtractor._cell_type_onehot("data_cell")
        assert vec[NODE_TYPE_MAP["data_cell"]] == 1.0

    def test_cell_type_onehot_metadata(self):
        vec = GraphFeatureExtractor._cell_type_onehot("table_metadata")
        assert vec[NODE_TYPE_MAP["table_metadata"]] == 1.0

    def test_numeric_flag_true(self):
        assert GraphFeatureExtractor._numeric_flag(True)[0] == 1.0

    def test_numeric_flag_false(self):
        assert GraphFeatureExtractor._numeric_flag(False)[0] == 0.0


# ────────────────────────────────────────────────
#  Edge cases
# ────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_graph(self, mock_extractor, empty_graph):
        data = mock_extractor.convert(empty_graph)
        assert data.x.shape == (0, 774)
        assert data.edge_index.shape == (2, 0)
        assert data.node_texts == []

    def test_single_node_graph(self, mock_extractor):
        """A graph with one header, no data rows, no edges."""
        rec = TableQARecord(
            id="single",
            question="?",
            answers=["x"],
            table_header=["Only"],
            table_rows=[],
            dataset="test",
        )
        G = table_to_graph(rec)
        data = mock_extractor.convert(G)
        assert data.x.shape == (1, 774)
        assert data.edge_index.shape == (2, 0)


# ────────────────────────────────────────────────
#  Batch conversion
# ────────────────────────────────────────────────

class TestBatchConversion:
    def test_batch_returns_list(self, mock_extractor, simple_graph):
        graphs = [simple_graph, simple_graph]
        data_list = mock_extractor.convert_batch(graphs, show_progress=False)
        assert len(data_list) == 2
        assert all(isinstance(d, Data) for d in data_list)

    def test_batch_preserves_record_ids(self, mock_extractor, simple_record):
        rec2 = TableQARecord(
            id="feat-002",
            question="?",
            answers=["x"],
            table_header=["A"],
            table_rows=[["1"]],
            dataset="test",
        )
        g1 = table_to_graph(simple_record)
        g2 = table_to_graph(rec2)
        data_list = mock_extractor.convert_batch([g1, g2], show_progress=False)
        assert data_list[0].record_id == "feat-001"
        assert data_list[1].record_id == "feat-002"

    def test_batch_empty_list(self, mock_extractor):
        assert mock_extractor.convert_batch([], show_progress=False) == []


# ────────────────────────────────────────────────
#  Config factory
# ────────────────────────────────────────────────

class TestFromConfig:
    def test_from_config_defaults(self):
        config = {}
        ext = GraphFeatureExtractor.from_config(config)
        assert ext.embedding_dim == 768
        assert ext.max_rows == 100
        assert ext.max_cols == 50

    def test_from_config_custom(self):
        config = {
            "embedding": {"model": "custom/model", "dimension": 384},
            "table_parsing": {"max_rows": 50, "max_cols": 20},
        }
        ext = GraphFeatureExtractor.from_config(config)
        assert ext.embedding_model_name == "custom/model"
        assert ext.embedding_dim == 384
        assert ext.max_rows == 50
        assert ext.max_cols == 20
