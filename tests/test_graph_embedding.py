"""
Unit tests for the GraphSAGE embedding model (src/graph/graph_embedding.py).

Uses small random-feature graphs to test model architecture, shapes, and
normalisation without requiring real text embeddings.
"""

import pytest
import torch
import numpy as np

from torch_geometric.data import Data, Batch

from src.graph.graph_embedding import TableGraphEncoder


# ────────────────────────────────────────────────
#  Fixtures
# ────────────────────────────────────────────────

@pytest.fixture
def tiny_data():
    """A minimal PyG Data: 4 nodes, 3 edges, 774-dim features."""
    torch.manual_seed(42)
    x = torch.randn(4, 774)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 2], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    data.record_id = "tiny-001"
    return data


@pytest.fixture
def isolated_data():
    """A graph with 2 nodes and no edges."""
    torch.manual_seed(0)
    x = torch.randn(2, 774)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_type = torch.zeros(0, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    data.record_id = "iso-001"
    return data


@pytest.fixture
def encoder():
    """Default encoder with standard config dimensions."""
    return TableGraphEncoder(
        input_dim=774,
        hidden_dim=256,
        output_dim=768,
        num_layers=2,
        dropout=0.1,
        normalize_output=True,
        pool_method="mean",
    )


@pytest.fixture
def small_encoder():
    """Smaller encoder for fast testing."""
    return TableGraphEncoder(
        input_dim=774,
        hidden_dim=32,
        output_dim=64,
        num_layers=2,
        dropout=0.0,
        normalize_output=True,
    )


# ────────────────────────────────────────────────
#  Architecture
# ────────────────────────────────────────────────

class TestArchitecture:
    def test_num_conv_layers(self, encoder):
        assert len(encoder.convs) == 2

    def test_single_layer_encoder(self):
        enc = TableGraphEncoder(input_dim=774, hidden_dim=256, output_dim=64, num_layers=1)
        assert len(enc.convs) == 1

    def test_three_layer_encoder(self):
        enc = TableGraphEncoder(input_dim=774, hidden_dim=128, output_dim=64, num_layers=3)
        assert len(enc.convs) == 3

    def test_config_attributes_stored(self, encoder):
        assert encoder.input_dim == 774
        assert encoder.hidden_dim == 256
        assert encoder.output_dim == 768
        assert encoder.normalize_output is True


# ────────────────────────────────────────────────
#  Forward pass shapes
# ────────────────────────────────────────────────

class TestForwardPass:
    def test_node_embedding_shape(self, small_encoder, tiny_data):
        node_emb, _ = small_encoder(tiny_data)
        assert node_emb.shape == (4, 64)

    def test_graph_embedding_shape(self, small_encoder, tiny_data):
        _, graph_emb = small_encoder(tiny_data)
        assert graph_emb.shape == (1, 64)

    def test_output_is_normalised(self, small_encoder, tiny_data):
        node_emb, graph_emb = small_encoder(tiny_data)
        # Check L2 norms are ~1.0
        norms = torch.norm(node_emb, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

        graph_norm = torch.norm(graph_emb, dim=-1)
        assert torch.allclose(graph_norm, torch.ones_like(graph_norm), atol=1e-5)

    def test_no_normalisation(self, tiny_data):
        enc = TableGraphEncoder(
            input_dim=774, hidden_dim=32, output_dim=64,
            normalize_output=False, num_layers=2,
        )
        node_emb, graph_emb = enc(tiny_data)
        # Norms should NOT all be 1.0
        norms = torch.norm(node_emb, dim=-1)
        assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_isolated_nodes(self, small_encoder, isolated_data):
        """Graphs with no edges should still produce valid embeddings."""
        node_emb, graph_emb = small_encoder(isolated_data)
        assert node_emb.shape == (2, 64)
        assert graph_emb.shape == (1, 64)
        assert not torch.isnan(node_emb).any()

    def test_eval_mode_deterministic(self, small_encoder, tiny_data):
        small_encoder.eval()
        _, g1 = small_encoder(tiny_data)
        _, g2 = small_encoder(tiny_data)
        assert torch.allclose(g1, g2)


# ────────────────────────────────────────────────
#  Batched forward
# ────────────────────────────────────────────────

class TestBatchedForward:
    def test_batched_graph_embedding_shape(self, small_encoder, tiny_data, isolated_data):
        batched = Batch.from_data_list([tiny_data, isolated_data])
        _, graph_emb = small_encoder(batched)
        assert graph_emb.shape == (2, 64)  # 2 graphs

    def test_batched_node_embedding_total(self, small_encoder, tiny_data, isolated_data):
        batched = Batch.from_data_list([tiny_data, isolated_data])
        node_emb, _ = small_encoder(batched)
        assert node_emb.shape == (6, 64)  # 4 + 2 nodes


# ────────────────────────────────────────────────
#  encode_batch (inference helper)
# ────────────────────────────────────────────────

class TestEncodeBatch:
    def test_encode_batch_returns_correct_types(self, small_encoder, tiny_data):
        node_embs, graph_embs = small_encoder.encode_batch([tiny_data, tiny_data])
        assert isinstance(node_embs, list)
        assert len(node_embs) == 2
        assert isinstance(graph_embs, torch.Tensor)
        assert graph_embs.shape == (2, 64)

    def test_encode_batch_node_shapes(self, small_encoder, tiny_data, isolated_data):
        node_embs, _ = small_encoder.encode_batch([tiny_data, isolated_data])
        assert node_embs[0].shape == (4, 64)
        assert node_embs[1].shape == (2, 64)

    def test_encode_batch_empty_list(self, small_encoder):
        node_embs, graph_embs = small_encoder.encode_batch([])
        assert node_embs == []
        assert graph_embs.shape == (0, 64)

    def test_encode_batch_on_cpu(self, small_encoder, tiny_data):
        _, graph_embs = small_encoder.encode_batch([tiny_data])
        assert graph_embs.device == torch.device("cpu")


# ────────────────────────────────────────────────
#  Pooling methods
# ────────────────────────────────────────────────

class TestPooling:
    def test_max_pool(self, tiny_data):
        enc = TableGraphEncoder(
            input_dim=774, hidden_dim=32, output_dim=64,
            pool_method="max", num_layers=2,
        )
        _, graph_emb = enc(tiny_data)
        assert graph_emb.shape == (1, 64)

    def test_invalid_pool_raises(self, tiny_data):
        enc = TableGraphEncoder(
            input_dim=774, hidden_dim=32, output_dim=64,
            pool_method="invalid",
        )
        with pytest.raises(ValueError, match="Unknown pool method"):
            enc(tiny_data)


# ────────────────────────────────────────────────
#  Config factory
# ────────────────────────────────────────────────

class TestFromConfig:
    def test_from_default_config(self):
        config = {
            "embedding": {"dimension": 768},
            "graph": {
                "node_features": {
                    "text_embedding": True,
                    "position_encoding": True,
                    "cell_type": True,
                    "numeric_flag": True,
                }
            },
            "graph_embedding": {
                "hidden_dim": 256,
                "output_dim": 768,
                "num_layers": 2,
                "dropout": 0.1,
                "normalize_output": True,
            },
            "indexing": {"pool_graph_embedding": "mean"},
        }
        enc = TableGraphEncoder.from_config(config)
        assert enc.input_dim == 774
        assert enc.hidden_dim == 256
        assert enc.output_dim == 768
        assert enc.pool_method == "mean"

    def test_from_config_without_numeric(self):
        config = {
            "embedding": {"dimension": 768},
            "graph": {
                "node_features": {
                    "text_embedding": True,
                    "position_encoding": True,
                    "cell_type": True,
                    "numeric_flag": False,
                }
            },
            "graph_embedding": {"hidden_dim": 128, "output_dim": 384},
        }
        enc = TableGraphEncoder.from_config(config)
        assert enc.input_dim == 773  # 768 + 2 + 3


# ────────────────────────────────────────────────
#  Save / Load
# ────────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load(self, small_encoder, tiny_data, tmp_path):
        # Get original output
        small_encoder.eval()
        _, orig_emb = small_encoder(tiny_data)

        # Save
        model_path = str(tmp_path / "encoder.pt")
        small_encoder.save(model_path)

        # Load into a fresh encoder
        new_encoder = TableGraphEncoder(
            input_dim=774, hidden_dim=32, output_dim=64, num_layers=2,
        )
        new_encoder.load(model_path)
        new_encoder.eval()
        _, loaded_emb = new_encoder(tiny_data)

        assert torch.allclose(orig_emb, loaded_emb, atol=1e-6)
