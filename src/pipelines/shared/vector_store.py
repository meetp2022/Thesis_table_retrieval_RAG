"""
FAISS vector store wrapper for table embedding indexing and retrieval.

Provides a pipeline-agnostic interface for:
    - Adding embeddings with associated record IDs and metadata
    - Top-k similarity search (cosine via IndexFlatIP with normalised vectors)
    - Persisting / loading the index to / from disk

Works with any embedding source (text, image, or graph-pooled vectors).

Usage:
    >>> from src.pipelines.shared.vector_store import FaissVectorStore
    >>> store = FaissVectorStore(embedding_dim=768)
    >>> store.add(embeddings, record_ids)
    >>> scores, ids = store.search(query_vectors, top_k=5)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    import faiss
except ImportError:
    raise ImportError(
        "faiss-cpu is required for the vector store. "
        "Install with: pip install faiss-cpu"
    )


class FaissVectorStore:
    """
    FAISS-backed vector store with record ID mapping.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the vectors to index (e.g. 768 for BGE / GraphSAGE).
    index_type : str
        FAISS index type. Supported: 'IndexFlatIP' (inner product / cosine),
        'IndexFlatL2' (Euclidean distance).
    normalize : bool
        If True, L2-normalise vectors before adding / searching.
        Required for cosine similarity with IndexFlatIP.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        index_type: str = "IndexFlatIP",
        normalize: bool = True,
    ):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.normalize = normalize

        # Build FAISS index
        self.index = self._build_index(index_type, embedding_dim)

        # ID mapping: FAISS row position ↔ record_id
        self._idx_to_id: List[str] = []
        self._id_to_idx: Dict[str, int] = {}

        # Optional per-record metadata
        self._metadata: Dict[str, Dict[str, Any]] = {}

        logger.debug(
            f"FaissVectorStore initialised: dim={embedding_dim}, "
            f"index={index_type}, normalize={normalize}"
        )

    # ── Index construction ───────────────────────

    @staticmethod
    def _build_index(index_type: str, dim: int) -> faiss.Index:
        if index_type == "IndexFlatIP":
            return faiss.IndexFlatIP(dim)
        elif index_type == "IndexFlatL2":
            return faiss.IndexFlatL2(dim)
        else:
            raise ValueError(
                f"Unsupported index type: {index_type}. "
                f"Use 'IndexFlatIP' or 'IndexFlatL2'."
            )

    # ── Add embeddings ───────────────────────────

    def add(
        self,
        embeddings: np.ndarray,
        record_ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add embeddings to the index.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (n, embedding_dim). Will be cast to float32.
        record_ids : list[str]
            One record ID per embedding. Must match length of embeddings.
        metadata : list[dict], optional
            Per-record metadata dicts (dataset, domain, etc.).
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embeddings of shape (n, {self.embedding_dim}), "
                f"got {embeddings.shape}"
            )
        if len(record_ids) != embeddings.shape[0]:
            raise ValueError(
                f"record_ids length ({len(record_ids)}) does not match "
                f"embeddings count ({embeddings.shape[0]})"
            )

        # Check for duplicate IDs
        existing = set(self._id_to_idx.keys())
        dupes = existing.intersection(record_ids)
        if dupes:
            logger.warning(
                f"Skipping {len(dupes)} duplicate record_ids already in index: "
                f"{list(dupes)[:5]}..."
            )
            # Filter out duplicates
            mask = [rid not in existing for rid in record_ids]
            embeddings = embeddings[mask]
            record_ids = [rid for rid, keep in zip(record_ids, mask) if keep]
            if metadata:
                metadata = [m for m, keep in zip(metadata, mask) if keep]

        if len(record_ids) == 0:
            return

        # Normalise
        if self.normalize:
            faiss.normalize_L2(embeddings)

        # Add to FAISS
        start_idx = self.index.ntotal
        self.index.add(embeddings)

        # Update ID mapping
        for i, rid in enumerate(record_ids):
            idx = start_idx + i
            self._idx_to_id.append(rid)
            self._id_to_idx[rid] = idx

        # Store metadata
        if metadata:
            for rid, meta in zip(record_ids, metadata):
                self._metadata[rid] = meta

        logger.debug(
            f"Added {len(record_ids)} vectors to index "
            f"(total: {self.index.ntotal})"
        )

    # ── Search ───────────────────────────────────

    def search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, List[List[str]]]:
        """
        Search the index for nearest neighbours.

        Parameters
        ----------
        query_vectors : np.ndarray
            Shape (num_queries, embedding_dim).
        top_k : int
            Number of results per query.

        Returns
        -------
        scores : np.ndarray
            Shape (num_queries, top_k). Similarity scores.
        record_ids : list[list[str]]
            Nested list of record IDs, one inner list per query.
        """
        query_vectors = np.asarray(query_vectors, dtype=np.float32)

        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)

        if query_vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query dim ({query_vectors.shape[1]}) != "
                f"index dim ({self.embedding_dim})"
            )

        if self.normalize:
            faiss.normalize_L2(query_vectors)

        # Clamp top_k to index size
        effective_k = min(top_k, self.index.ntotal)
        if effective_k == 0:
            num_q = query_vectors.shape[0]
            return np.zeros((num_q, 0), dtype=np.float32), [[] for _ in range(num_q)]

        scores, indices = self.index.search(query_vectors, effective_k)

        # Map FAISS indices → record IDs
        record_id_results = []
        for row in indices:
            ids = []
            for idx in row:
                if 0 <= idx < len(self._idx_to_id):
                    ids.append(self._idx_to_id[idx])
                else:
                    ids.append("")  # shouldn't happen with valid index
            record_id_results.append(ids)

        return scores, record_id_results

    # ── Metadata access ──────────────────────────

    def get_metadata(self, record_id: str) -> Dict[str, Any]:
        """Retrieve stored metadata for a record ID."""
        return self._metadata.get(record_id, {})

    # ── Persistence ──────────────────────────────

    def save(self, directory: str) -> None:
        """
        Save the FAISS index and ID mappings to disk.

        Creates:
            directory/index.faiss   — the FAISS index binary
            directory/mappings.json — ID mapping + metadata
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(dir_path / "index.faiss"))

        # Save mappings
        mappings = {
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "normalize": self.normalize,
            "idx_to_id": self._idx_to_id,
            "metadata": self._metadata,
        }
        with open(dir_path / "mappings.json", "w") as f:
            json.dump(mappings, f, indent=2)

        logger.info(f"Saved vector store ({self.index.ntotal} vectors) to {directory}")

    @classmethod
    def load(cls, directory: str) -> "FaissVectorStore":
        """Load a saved vector store from disk."""
        dir_path = Path(directory)

        # Load mappings first to get config
        with open(dir_path / "mappings.json", "r") as f:
            mappings = json.load(f)

        store = cls(
            embedding_dim=mappings["embedding_dim"],
            index_type=mappings["index_type"],
            normalize=mappings["normalize"],
        )

        # Replace the empty index with the saved one
        store.index = faiss.read_index(str(dir_path / "index.faiss"))

        # Restore ID mappings
        store._idx_to_id = mappings["idx_to_id"]
        store._id_to_idx = {rid: i for i, rid in enumerate(store._idx_to_id)}
        store._metadata = mappings.get("metadata", {})

        logger.info(
            f"Loaded vector store ({store.index.ntotal} vectors) from {directory}"
        )
        return store

    # ── Utilities ────────────────────────────────

    def reset(self) -> None:
        """Clear the index and all mappings."""
        self.index = self._build_index(self.index_type, self.embedding_dim)
        self._idx_to_id = []
        self._id_to_idx = {}
        self._metadata = {}
        logger.debug("Vector store reset")

    def __len__(self) -> int:
        return self.index.ntotal

    def __repr__(self) -> str:
        return (
            f"FaissVectorStore(dim={self.embedding_dim}, "
            f"index={self.index_type}, n={self.index.ntotal})"
        )

    # ── Config-driven factory ────────────────────

    @classmethod
    def from_config(cls, config: dict) -> "FaissVectorStore":
        """
        Create a vector store from the merged pipeline config.

        Reads:
        - config["embedding"]["dimension"]
        - config["vector_store"]["index_type"]
        - config["embedding"]["normalize"]
        """
        emb = config.get("embedding", {})
        vs = config.get("vector_store", {})

        return cls(
            embedding_dim=emb.get("dimension", 768),
            index_type=vs.get("index_type", "IndexFlatIP"),
            normalize=emb.get("normalize", True),
        )
