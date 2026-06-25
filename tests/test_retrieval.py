"""
Unit tests for the shared retrieval pipeline:
    - FaissVectorStore  (vector_store.py)
    - QueryEncoder      (query_encoder.py)
    - TableRetriever    (retriever.py)

Uses synthetic embeddings (no model downloads) for fast, deterministic tests.
"""

import numpy as np
import pytest

from src.pipelines.shared.vector_store import FaissVectorStore
from src.pipelines.shared.query_encoder import QueryEncoder
from src.pipelines.shared.retriever import TableRetriever, RetrievalResult


# ────────────────────────────────────────────────
#  Fixtures
# ────────────────────────────────────────────────

@pytest.fixture
def dim():
    return 64


@pytest.fixture
def store(dim):
    return FaissVectorStore(embedding_dim=dim, index_type="IndexFlatIP", normalize=True)


@pytest.fixture
def sample_embeddings(dim):
    """5 normalised random vectors."""
    np.random.seed(42)
    vecs = np.random.randn(5, dim).astype(np.float32)
    # L2-normalise
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def sample_ids():
    return ["rec-0", "rec-1", "rec-2", "rec-3", "rec-4"]


@pytest.fixture
def sample_metadata():
    return [
        {"dataset": "wikitq", "domain": "general"},
        {"dataset": "wikitq", "domain": "general"},
        {"dataset": "tatqa", "domain": "finance"},
        {"dataset": "tatqa", "domain": "finance"},
        {"dataset": "finqa", "domain": "finance"},
    ]


@pytest.fixture
def populated_store(store, sample_embeddings, sample_ids, sample_metadata):
    store.add(sample_embeddings, sample_ids, sample_metadata)
    return store


@pytest.fixture
def mock_query_encoder(dim):
    """QueryEncoder with mocked encode method (no model loading)."""
    enc = QueryEncoder.__new__(QueryEncoder)
    enc.model_name = "mock"
    enc.dimension = dim
    enc.normalize = True
    enc.device = "cpu"
    enc.query_prefix = ""
    enc._model = None

    def _mock_encode(queries, batch_size=64):
        if isinstance(queries, str):
            queries = [queries]
        np.random.seed(hash(queries[0]) % 2**31)
        vecs = np.random.randn(len(queries), dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    enc.encode = _mock_encode
    return enc


# ════════════════════════════════════════════════
#  FaissVectorStore Tests
# ════════════════════════════════════════════════

class TestFaissVectorStoreInit:
    def test_empty_on_creation(self, store):
        assert len(store) == 0

    def test_repr(self, store, dim):
        r = repr(store)
        assert str(dim) in r
        assert "IndexFlatIP" in r

    def test_unsupported_index_type(self, dim):
        with pytest.raises(ValueError, match="Unsupported index type"):
            FaissVectorStore(embedding_dim=dim, index_type="IndexIVFFlat")


class TestFaissVectorStoreAdd:
    def test_add_vectors(self, store, sample_embeddings, sample_ids):
        store.add(sample_embeddings, sample_ids)
        assert len(store) == 5

    def test_add_with_metadata(self, populated_store):
        meta = populated_store.get_metadata("rec-2")
        assert meta["dataset"] == "tatqa"
        assert meta["domain"] == "finance"

    def test_add_wrong_dim_raises(self, store):
        bad = np.random.randn(3, 32).astype(np.float32)
        with pytest.raises(ValueError, match="Expected embeddings"):
            store.add(bad, ["a", "b", "c"])

    def test_add_mismatched_ids_raises(self, store, dim):
        vecs = np.random.randn(3, dim).astype(np.float32)
        with pytest.raises(ValueError, match="does not match"):
            store.add(vecs, ["a", "b"])

    def test_add_duplicates_skipped(self, populated_store, sample_embeddings, dim):
        """Adding the same record IDs again should skip them."""
        populated_store.add(
            sample_embeddings[:2],
            ["rec-0", "rec-1"],
        )
        assert len(populated_store) == 5  # no new entries

    def test_incremental_add(self, store, dim):
        v1 = np.random.randn(2, dim).astype(np.float32)
        v2 = np.random.randn(3, dim).astype(np.float32)
        store.add(v1, ["a", "b"])
        store.add(v2, ["c", "d", "e"])
        assert len(store) == 5


class TestFaissVectorStoreSearch:
    def test_search_returns_correct_count(self, populated_store, sample_embeddings):
        scores, ids = populated_store.search(sample_embeddings[:1], top_k=3)
        assert scores.shape == (1, 3)
        assert len(ids) == 1
        assert len(ids[0]) == 3

    def test_search_self_is_top_match(self, populated_store, sample_embeddings):
        """Searching with a vector should return itself as the best match."""
        scores, ids = populated_store.search(sample_embeddings[:1], top_k=1)
        assert ids[0][0] == "rec-0"

    def test_search_scores_descending(self, populated_store, sample_embeddings):
        scores, _ = populated_store.search(sample_embeddings[:1], top_k=5)
        for i in range(len(scores[0]) - 1):
            assert scores[0][i] >= scores[0][i + 1]

    def test_search_batch_queries(self, populated_store, sample_embeddings):
        scores, ids = populated_store.search(sample_embeddings[:3], top_k=2)
        assert scores.shape == (3, 2)
        assert len(ids) == 3

    def test_search_1d_query(self, populated_store, sample_embeddings):
        """A 1D vector should be reshaped automatically."""
        scores, ids = populated_store.search(sample_embeddings[0], top_k=2)
        assert scores.shape == (1, 2)

    def test_search_wrong_dim_raises(self, populated_store):
        bad = np.random.randn(1, 32).astype(np.float32)
        with pytest.raises(ValueError, match="Query dim"):
            populated_store.search(bad)

    def test_search_empty_store(self, store, dim):
        q = np.random.randn(1, dim).astype(np.float32)
        scores, ids = store.search(q, top_k=5)
        assert scores.shape[1] == 0
        assert ids == [[]]

    def test_top_k_clamped_to_index_size(self, populated_store, sample_embeddings):
        """Requesting more results than vectors should not error."""
        scores, ids = populated_store.search(sample_embeddings[:1], top_k=100)
        assert scores.shape == (1, 5)  # only 5 vectors in index


class TestFaissVectorStorePersistence:
    def test_save_and_load(self, populated_store, sample_embeddings, tmp_path):
        save_dir = str(tmp_path / "store")
        populated_store.save(save_dir)

        loaded = FaissVectorStore.load(save_dir)
        assert len(loaded) == 5
        assert loaded.embedding_dim == populated_store.embedding_dim

        # Verify search consistency
        orig_scores, orig_ids = populated_store.search(sample_embeddings[:1], top_k=3)
        load_scores, load_ids = loaded.search(sample_embeddings[:1], top_k=3)

        np.testing.assert_array_almost_equal(orig_scores, load_scores)
        assert orig_ids == load_ids

    def test_metadata_survives_save_load(self, populated_store, tmp_path):
        save_dir = str(tmp_path / "store")
        populated_store.save(save_dir)
        loaded = FaissVectorStore.load(save_dir)
        assert loaded.get_metadata("rec-4") == {"dataset": "finqa", "domain": "finance"}


class TestFaissVectorStoreReset:
    def test_reset_clears_everything(self, populated_store):
        populated_store.reset()
        assert len(populated_store) == 0
        assert populated_store.get_metadata("rec-0") == {}


class TestFaissVectorStoreFromConfig:
    def test_from_config(self):
        config = {
            "embedding": {"dimension": 384, "normalize": True},
            "vector_store": {"index_type": "IndexFlatIP"},
        }
        store = FaissVectorStore.from_config(config)
        assert store.embedding_dim == 384
        assert store.index_type == "IndexFlatIP"

    def test_from_config_defaults(self):
        store = FaissVectorStore.from_config({})
        assert store.embedding_dim == 768


# ════════════════════════════════════════════════
#  QueryEncoder Tests
# ════════════════════════════════════════════════

class TestQueryEncoder:
    def test_encode_single_string(self, mock_query_encoder, dim):
        vec = mock_query_encoder.encode("test query")
        assert vec.shape == (1, dim)

    def test_encode_list(self, mock_query_encoder, dim):
        vecs = mock_query_encoder.encode(["q1", "q2", "q3"])
        assert vecs.shape == (3, dim)

    def test_encode_returns_float32(self, mock_query_encoder):
        vec = mock_query_encoder.encode("test")
        assert vec.dtype == np.float32

    def test_repr(self, mock_query_encoder):
        r = repr(mock_query_encoder)
        assert "mock" in r

    def test_from_config(self):
        config = {
            "embedding": {
                "model": "custom/model",
                "dimension": 384,
                "normalize": False,
            }
        }
        enc = QueryEncoder.from_config(config)
        assert enc.model_name == "custom/model"
        assert enc.dimension == 384
        assert enc.normalize is False


# ════════════════════════════════════════════════
#  TableRetriever Tests
# ════════════════════════════════════════════════

class TestTableRetriever:
    def test_retrieve_returns_results(
        self, populated_store, mock_query_encoder
    ):
        retriever = TableRetriever(populated_store, mock_query_encoder, top_k=3)
        results = retriever.retrieve("some question")
        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_results_have_correct_ranks(
        self, populated_store, mock_query_encoder
    ):
        retriever = TableRetriever(populated_store, mock_query_encoder, top_k=5)
        results = retriever.retrieve("test question")
        ranks = [r.rank for r in results]
        assert ranks == [1, 2, 3, 4, 5]

    def test_results_scores_descending(
        self, populated_store, mock_query_encoder
    ):
        retriever = TableRetriever(populated_store, mock_query_encoder, top_k=5)
        results = retriever.retrieve("test question")
        scores = [r.score for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_results_have_metadata(
        self, populated_store, mock_query_encoder
    ):
        retriever = TableRetriever(populated_store, mock_query_encoder, top_k=5)
        results = retriever.retrieve("test")
        for r in results:
            assert "dataset" in r.metadata

    def test_retrieve_with_override_top_k(
        self, populated_store, mock_query_encoder
    ):
        retriever = TableRetriever(populated_store, mock_query_encoder, top_k=5)
        results = retriever.retrieve("test", top_k=2)
        assert len(results) == 2

    def test_retrieve_batch(
        self, populated_store, mock_query_encoder
    ):
        retriever = TableRetriever(populated_store, mock_query_encoder, top_k=3)
        batch_results = retriever.retrieve_batch(["q1", "q2"])
        assert len(batch_results) == 2
        assert all(len(r) == 3 for r in batch_results)

    def test_retrieve_batch_empty(
        self, populated_store, mock_query_encoder
    ):
        retriever = TableRetriever(populated_store, mock_query_encoder)
        assert retriever.retrieve_batch([]) == []


class TestRecallAtK:
    def test_recall_perfect(self, populated_store, mock_query_encoder, sample_embeddings):
        """When query = indexed vector, recall@5 should be 1.0."""
        # Override encode to return the actual indexed vectors
        def _return_indexed(queries, batch_size=64):
            if isinstance(queries, str):
                queries = [queries]
            # Map query text to index
            idx_map = {"q0": 0, "q1": 1, "q2": 2}
            indices = [idx_map.get(q, 0) for q in queries]
            return sample_embeddings[indices]

        mock_query_encoder.encode = _return_indexed

        retriever = TableRetriever(populated_store, mock_query_encoder, top_k=5)
        recall = retriever.recall_at_k(
            queries=["q0", "q1", "q2"],
            gold_record_ids=["rec-0", "rec-1", "rec-2"],
            k=5,
        )
        assert recall == 1.0

    def test_recall_empty_queries(self, populated_store, mock_query_encoder):
        retriever = TableRetriever(populated_store, mock_query_encoder)
        assert retriever.recall_at_k([], [], k=5) == 0.0


class TestRetrievalResult:
    def test_dataclass_fields(self):
        r = RetrievalResult(
            record_id="test-1",
            score=0.95,
            rank=1,
            metadata={"dataset": "wikitq"},
        )
        assert r.record_id == "test-1"
        assert r.score == 0.95
        assert r.rank == 1
        assert r.metadata == {"dataset": "wikitq"}

    def test_default_metadata(self):
        r = RetrievalResult(record_id="x", score=0.5, rank=1)
        assert r.metadata == {}


class TestRetrieverFromConfig:
    def test_from_config(self, populated_store, mock_query_encoder):
        config = {"retrieval": {"top_k": 3}}
        retriever = TableRetriever.from_config(
            config, populated_store, mock_query_encoder
        )
        assert retriever.top_k == 3

    def test_repr(self, populated_store, mock_query_encoder):
        retriever = TableRetriever(populated_store, mock_query_encoder, top_k=5)
        r = repr(retriever)
        assert "top_k=5" in r
        assert "index_size=5" in r
