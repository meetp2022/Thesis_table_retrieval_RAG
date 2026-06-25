"""
Optional cross-encoder re-ranker for improved retrieval precision.

A cross-encoder jointly encodes (query, document) pairs and produces a
relevance score, which is significantly more accurate than bi-encoder
dot-product similarity — at the cost of higher latency.

Usage (post-retrieval):
    reranker = CrossEncoderReranker()
    candidates = [(rid, table_text), ...]
    ranked = reranker.rerank(query, candidates, top_k=5)

References / attribution
------------------------
- Cross-encoder re-ranking after bi-encoder retrieval ("retrieve then
  re-rank") is a standard pattern documented by sentence-transformers:
  https://www.sbert.net/examples/applications/retrieve_rerank/README.html
- Re-ranker model cross-encoder/ms-marco-MiniLM-L-6-v2 is a public
  checkpoint trained on MS MARCO (Nguyen et al., 2016).
- The wrapper class, candidate handling and integration with the hybrid
  pipeline in this file are this project's own code.
"""

from typing import List, Optional, Tuple

from loguru import logger


class CrossEncoderReranker:
    """
    Re-ranks retrieval candidates using a cross-encoder model.

    Uses lazy loading so the model is only downloaded/loaded on first use.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.  Default is a lightweight MiniLM model
        (22 MB) fine-tuned on MS-MARCO passage ranking — works well for
        table retrieval too.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    # ── Lazy model loading ───────────────────────

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder ready")
        return self._model

    # ── Reranking ────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        Re-rank candidates using cross-encoder relevance scores.

        Parameters
        ----------
        query : str
            The user's question.
        candidates : list of (record_id, table_text)
            Candidate tables to re-rank.  Pass more than ``top_k`` so
            the cross-encoder has room to reorder.
        top_k : int
            How many results to return.

        Returns
        -------
        list of (record_id, score)
            Sorted by cross-encoder score, descending.
        """
        if not candidates:
            return []

        pairs = [(query, text) for _, text in candidates]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip([rid for rid, _ in candidates], scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def __repr__(self) -> str:
        loaded = self._model is not None
        return f"CrossEncoderReranker(model={self.model_name!r}, loaded={loaded})"
