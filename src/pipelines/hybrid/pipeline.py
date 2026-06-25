"""
Hybrid retrieval pipeline — late fusion of text baseline and graph-augmented scores.

Improvements over naive fusion:
- **Score normalisation**: min-max normalises each pipeline's scores to [0, 1]
  before fusing, so different score scales don't skew the blend.
- **Larger candidate pool**: fetches up to 50 candidates per pipeline (was 20)
  to give fusion more material to work with.
- **Cross-encoder re-ranking** (optional): after fusion selects the top-N
  candidates, a lightweight cross-encoder (MiniLM) re-ranks them using
  joint (query, table) relevance scores.

Fusion strategy:
    score_hybrid = alpha * norm(score_text) + (1 - alpha) * norm(score_graph)

Usage:
    >>> pipeline = HybridPipeline.from_config(text_cfg, graph_cfg)
    >>> pipeline.index(records)
    >>> results = pipeline.retriever.retrieve("What was the revenue?", top_k=5)
    # With cross-encoder re-ranking:
    >>> results = pipeline.retriever.retrieve("...", top_k=5, rerank=True)
"""

import time
from typing import Dict, List, Optional

from loguru import logger

from src.data.dataset_loader import TableQARecord
from src.pipelines.shared.retriever import RetrievalResult
from src.pipelines.shared.query_encoder import QueryEncoder
from src.pipelines.text_baseline.pipeline import TextBaselinePipeline, linearise_table
from src.pipelines.graph_augmented.pipeline import GraphAugmentedPipeline
from src.pipelines.shared.cross_encoder import CrossEncoderReranker
from src.pipelines.shared.llm_client import create_llm_client


def _minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalise a score dict to [0, 1].  No-op if all values are equal."""
    if not scores:
        return scores
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {k: 1.0 for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


class HybridRetriever:
    """
    Fuses retrieval results from text and graph pipelines.

    For a given query:
    1. Fetch up to ``fetch_k`` candidates from each pipeline.
    2. Min-max normalise both score sets independently.
    3. Fuse with linear interpolation: alpha * text + (1-alpha) * graph.
    4. (Optional) Re-rank top candidates with a cross-encoder.
    """

    def __init__(
        self,
        text_pipeline: TextBaselinePipeline,
        graph_pipeline: GraphAugmentedPipeline,
        query_encoder: QueryEncoder,
        alpha: float = 0.5,
        top_k: int = 5,
        reranker: Optional[CrossEncoderReranker] = None,
        record_lookup: Optional[Dict[str, TableQARecord]] = None,
        text_config: Optional[dict] = None,
    ):
        self.text_pipeline = text_pipeline
        self.graph_pipeline = graph_pipeline
        self.query_encoder = query_encoder
        self.alpha = alpha
        self.top_k = top_k
        self.reranker = reranker
        self.record_lookup = record_lookup  # shared reference, populated during index()
        self.text_config = text_config or {}

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank: bool = False,
    ) -> List[RetrievalResult]:
        """
        Retrieve tables using normalised, fused text + graph scores.

        Parameters
        ----------
        query : str
            Natural-language question.
        top_k : int, optional
            Number of final results to return.
        rerank : bool
            Apply cross-encoder re-ranking after fusion (requires
            ``reranker`` to be set and ``record_lookup`` populated).

        Returns
        -------
        list[RetrievalResult]
            Ranked results by fused (and optionally re-ranked) score.
        """
        k = top_k or self.top_k

        # Pull a generous candidate pool from each pipeline
        # More candidates → fusion has more signal to work with
        fetch_k = min(k * 5, 50)

        text_results = self.text_pipeline.retriever.retrieve(query, top_k=fetch_k)
        graph_results = self.graph_pipeline.retriever.retrieve(query, top_k=fetch_k)

        # Collect raw scores per record_id
        text_scores_raw: Dict[str, float] = {r.record_id: r.score for r in text_results}
        graph_scores_raw: Dict[str, float] = {r.record_id: r.score for r in graph_results}
        all_ids = set(text_scores_raw) | set(graph_scores_raw)

        # Min-max normalise each pipeline independently before fusion
        text_norm = _minmax_normalize(text_scores_raw)
        graph_norm = _minmax_normalize(graph_scores_raw)

        # Fuse: alpha * text + (1 - alpha) * graph
        # Missing entries default to 0 (not in that pipeline's top-k)
        fused = []
        for rid in all_ids:
            ts = text_norm.get(rid, 0.0)
            gs = graph_norm.get(rid, 0.0)
            fused_score = self.alpha * ts + (1 - self.alpha) * gs
            fused.append((rid, fused_score, text_scores_raw.get(rid, 0.0), graph_scores_raw.get(rid, 0.0)))

        fused.sort(key=lambda x: x[1], reverse=True)

        # Cross-encoder re-ranking (optional)
        if rerank and self.reranker and self.record_lookup:
            # Take more candidates to give reranker room to reorder
            rerank_pool = fused[:max(k * 3, 15)]
            candidates = []
            for rid, *_ in rerank_pool:
                rec = self.record_lookup.get(rid)
                if rec:
                    table_text = linearise_table(rec, self.text_config)
                    candidates.append((rid, table_text))

            if candidates:
                reranked = self.reranker.rerank(query, candidates, top_k=k)
                # Build a score lookup from reranked positions
                rerank_score_map = {rid: score for rid, score in reranked}
                # Keep original fused scores as fallback
                fused_score_map = {rid: score for rid, score, *_ in fused}
                raw_ts_map = {rid: ts for rid, _, ts, _ in fused}
                raw_gs_map = {rid: gs for rid, _, _, gs in fused}

                results = []
                for rank, (rid, ce_score) in enumerate(reranked, 1):
                    metadata = (
                        self.text_pipeline.vector_store.get_metadata(rid)
                        or self.graph_pipeline.vector_store.get_metadata(rid)
                        or {}
                    )
                    metadata["text_score"] = raw_ts_map.get(rid, 0.0)
                    metadata["graph_score"] = raw_gs_map.get(rid, 0.0)
                    metadata["fused_score"] = fused_score_map.get(rid, 0.0)
                    metadata["cross_encoder_score"] = ce_score
                    metadata["pipeline"] = "hybrid+rerank"
                    results.append(RetrievalResult(
                        record_id=rid,
                        score=ce_score,
                        rank=rank,
                        metadata=metadata,
                    ))
                return results

        # Standard fusion output (no reranking)
        results = []
        for rank, (rid, score, ts, gs) in enumerate(fused[:k], 1):
            metadata = (
                self.text_pipeline.vector_store.get_metadata(rid)
                or self.graph_pipeline.vector_store.get_metadata(rid)
                or {}
            )
            metadata["text_score"] = ts
            metadata["graph_score"] = gs
            metadata["pipeline"] = "hybrid"
            results.append(RetrievalResult(
                record_id=rid,
                score=score,
                rank=rank,
                metadata=metadata,
            ))

        return results


class HybridPipeline:
    """
    End-to-end hybrid pipeline combining text and graph retrieval.

    Indexes tables in both pipelines, then uses HybridRetriever
    for normalised, fused retrieval at query time.
    """

    def __init__(
        self,
        text_pipeline: TextBaselinePipeline,
        graph_pipeline: GraphAugmentedPipeline,
        retriever: HybridRetriever,
        alpha: float = 0.5,
    ):
        self.text_pipeline = text_pipeline
        self.graph_pipeline = graph_pipeline
        self.retriever = retriever
        self.alpha = alpha
        self._record_lookup: Dict[str, TableQARecord] = {}

    def index(self, records: List[TableQARecord]) -> None:
        """Index tables in both text and graph pipelines."""
        start = time.perf_counter()

        for rec in records:
            self._record_lookup[rec.id] = rec

        logger.info(f"Hybrid indexing: {len(records)} records into both pipelines...")

        t0 = time.perf_counter()
        self.text_pipeline.index(records)
        text_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.graph_pipeline.index(records)
        graph_time = time.perf_counter() - t0

        total = time.perf_counter() - start
        logger.info(
            f"Hybrid indexing complete in {total:.1f}s "
            f"(text: {text_time:.1f}s, graph: {graph_time:.1f}s)"
        )

    @property
    def vector_store(self):
        """Return text vector store for compatibility."""
        return self.text_pipeline.vector_store

    @classmethod
    def from_config(
        cls,
        text_config: dict,
        graph_config: dict,
        alpha: float = 0.5,
        fallback_to_mock: bool = True,
        use_reranker: bool = False,
    ) -> "HybridPipeline":
        """
        Build hybrid pipeline from both pipeline configs.

        Parameters
        ----------
        text_config : dict
            Merged config for text baseline pipeline.
        graph_config : dict
            Merged config for graph-augmented pipeline.
        alpha : float
            Fusion weight: alpha * norm(text) + (1-alpha) * norm(graph).
        fallback_to_mock : bool
            Use mock LLM if Ollama unavailable.
        use_reranker : bool
            Load and attach the cross-encoder re-ranker.
        """
        logger.info(f"Building Hybrid Pipeline (alpha={alpha}, reranker={use_reranker})...")

        text_pipe = TextBaselinePipeline.from_config(
            text_config, fallback_to_mock=fallback_to_mock
        )
        graph_pipe = GraphAugmentedPipeline.from_config(
            graph_config, fallback_to_mock=fallback_to_mock
        )

        # Shared query encoder saves memory
        query_encoder = text_pipe.query_encoder

        reranker = CrossEncoderReranker() if use_reranker else None

        # record_lookup is a mutable dict — HybridPipeline.index() populates it
        # and HybridRetriever holds a reference so it stays in sync
        record_lookup: Dict[str, TableQARecord] = {}

        retriever = HybridRetriever(
            text_pipeline=text_pipe,
            graph_pipeline=graph_pipe,
            query_encoder=query_encoder,
            alpha=alpha,
            reranker=reranker,
            record_lookup=record_lookup,
            text_config=text_config,
        )

        pipeline = cls(
            text_pipeline=text_pipe,
            graph_pipeline=graph_pipe,
            retriever=retriever,
            alpha=alpha,
        )

        # Wire the shared record_lookup so the retriever sees indexed records
        pipeline._record_lookup = record_lookup
        retriever.record_lookup = pipeline._record_lookup

        logger.info(f"Hybrid Pipeline ready (alpha={alpha})")
        return pipeline
