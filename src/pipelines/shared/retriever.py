"""
Table retriever — orchestrates query encoding + FAISS search to return
ranked table results.

Ties together QueryEncoder and FaissVectorStore into a clean retrieval API
that the LLM answer generation phase consumes.

Usage:
    >>> from src.pipelines.shared.retriever import TableRetriever
    >>> retriever = TableRetriever(vector_store, query_encoder, top_k=5)
    >>> results = retriever.retrieve("What was the revenue in 2024?")
    >>> for r in results:
    ...     print(r.rank, r.record_id, r.score)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from src.pipelines.shared.vector_store import FaissVectorStore
from src.pipelines.shared.query_encoder import QueryEncoder


# ────────────────────────────────────────────────
#  Result dataclass
# ────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    A single retrieval result.

    Attributes
    ----------
    record_id : str
        The ID of the retrieved table record.
    score : float
        Similarity score from FAISS (higher = more similar for IP index).
    rank : int
        1-based rank in the result list.
    metadata : dict
        Any metadata stored alongside the embedding (dataset, domain, etc.).
    """
    record_id: str
    score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# ────────────────────────────────────────────────
#  Table retriever
# ────────────────────────────────────────────────

class TableRetriever:
    """
    End-to-end table retriever: question → top-k ranked tables.

    Parameters
    ----------
    vector_store : FaissVectorStore
        Indexed table embeddings.
    query_encoder : QueryEncoder
        Encodes query strings into the search space.
    top_k : int
        Default number of results to return.
    """

    def __init__(
        self,
        vector_store: FaissVectorStore,
        query_encoder: QueryEncoder,
        top_k: int = 5,
    ):
        self.vector_store = vector_store
        self.query_encoder = query_encoder
        self.top_k = top_k

    # ── Single query ─────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k tables for a single query.

        Parameters
        ----------
        query : str
            Natural-language question.
        top_k : int, optional
            Override the default top_k.

        Returns
        -------
        list[RetrievalResult]
            Ranked results, best first.
        """
        k = top_k or self.top_k

        # Encode query
        query_vec = self.query_encoder.encode(query)  # (1, dim)

        # Search
        scores, record_ids = self.vector_store.search(query_vec, top_k=k)

        # Build results
        results = []
        for rank_idx, (rid, score) in enumerate(
            zip(record_ids[0], scores[0])
        ):
            if not rid:  # skip empty IDs (shouldn't happen)
                continue
            results.append(
                RetrievalResult(
                    record_id=rid,
                    score=float(score),
                    rank=rank_idx + 1,
                    metadata=self.vector_store.get_metadata(rid),
                )
            )

        logger.debug(
            f"Retrieved {len(results)} tables for query: "
            f"{query[:60]!r}..."
        )
        return results

    # ── Batch retrieval ──────────────────────────

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
    ) -> List[List[RetrievalResult]]:
        """
        Retrieve top-k tables for a batch of queries.

        Parameters
        ----------
        queries : list[str]
            List of natural-language questions.
        top_k : int, optional
            Override the default top_k.

        Returns
        -------
        list[list[RetrievalResult]]
            One result list per query.
        """
        if not queries:
            return []

        k = top_k or self.top_k

        # Batch encode
        query_vecs = self.query_encoder.encode(queries)  # (n, dim)

        # Batch search
        scores, record_ids = self.vector_store.search(query_vecs, top_k=k)

        # Build results per query
        all_results = []
        for q_idx in range(len(queries)):
            results = []
            for rank_idx, (rid, score) in enumerate(
                zip(record_ids[q_idx], scores[q_idx])
            ):
                if not rid:
                    continue
                results.append(
                    RetrievalResult(
                        record_id=rid,
                        score=float(score),
                        rank=rank_idx + 1,
                        metadata=self.vector_store.get_metadata(rid),
                    )
                )
            all_results.append(results)

        logger.info(f"Batch retrieved tables for {len(queries)} queries")
        return all_results

    # ── Evaluation helper ────────────────────────

    def recall_at_k(
        self,
        queries: List[str],
        gold_record_ids: List[str],
        k: int = 5,
    ) -> float:
        """
        Compute Recall@k: fraction of queries where the gold table
        appears in the top-k retrieved results.

        Parameters
        ----------
        queries : list[str]
            Questions to evaluate.
        gold_record_ids : list[str]
            The correct record ID for each query.
        k : int
            Cutoff for retrieval.

        Returns
        -------
        float
            Recall@k score between 0.0 and 1.0.
        """
        if not queries:
            return 0.0

        batch_results = self.retrieve_batch(queries, top_k=k)

        hits = 0
        for gold_id, results in zip(gold_record_ids, batch_results):
            retrieved_ids = {r.record_id for r in results}
            if gold_id in retrieved_ids:
                hits += 1

        recall = hits / len(queries)
        logger.info(f"Recall@{k}: {recall:.4f} ({hits}/{len(queries)})")
        return recall

    # ── Config-driven factory ────────────────────

    @classmethod
    def from_config(
        cls,
        config: dict,
        vector_store: FaissVectorStore,
        query_encoder: QueryEncoder,
    ) -> "TableRetriever":
        """
        Create a retriever from the merged pipeline config.

        Reads config["retrieval"]["top_k"].
        """
        ret = config.get("retrieval", {})
        return cls(
            vector_store=vector_store,
            query_encoder=query_encoder,
            top_k=ret.get("top_k", 5),
        )

    def __repr__(self) -> str:
        return (
            f"TableRetriever(top_k={self.top_k}, "
            f"index_size={len(self.vector_store)})"
        )
