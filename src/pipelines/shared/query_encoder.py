"""
Query encoder — embeds natural-language questions into the same vector space
used for table indexing, enabling FAISS similarity search.

Uses the same BAAI/bge-base-en-v1.5 model from the pipeline config.
Reusable across all three pipelines (text baseline uses this for both
indexing and querying; graph pipeline uses it for query-side only).

Usage:
    >>> from src.pipelines.shared.query_encoder import QueryEncoder
    >>> encoder = QueryEncoder()
    >>> query_vec = encoder.encode("What is the revenue for 2024?")
    >>> batch_vecs = encoder.encode(["Q1", "Q2", "Q3"])
"""

from typing import List, Optional, Union

import numpy as np
from loguru import logger


class QueryEncoder:
    """
    Embeds query strings using a sentence-transformer model.

    Parameters
    ----------
    model_name : str
        HuggingFace sentence-transformer model name.
    dimension : int
        Expected output embedding dimension.
    normalize : bool
        L2-normalise embeddings (required for cosine similarity with FAISS IP).
    device : str or None
        Device for inference ('cpu', 'cuda'). Auto-detects if None.
    query_prefix : str
        Optional prefix prepended to queries (some models like BGE
        benefit from "Represent this sentence: " for asymmetric retrieval).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        dimension: int = 768,
        normalize: bool = True,
        device: Optional[str] = None,
        query_prefix: str = "",
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.normalize = normalize
        self.query_prefix = query_prefix

        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model = None  # lazy-loaded

    # ── Lazy model loading ───────────────────────

    @property
    def model(self):
        """Lazy-load the sentence-transformer model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(
                f"Loading query encoder: {self.model_name} "
                f"(device={self.device})"
            )
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
        return self._model

    # ── Encoding ─────────────────────────────────

    def encode(
        self,
        queries: Union[str, List[str]],
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Encode one or more query strings into embeddings.

        Parameters
        ----------
        queries : str or list[str]
            A single query string or a list of queries.
        batch_size : int
            Batch size for the sentence-transformer.

        Returns
        -------
        np.ndarray
            Shape (n, dimension) where n is the number of queries.
            Always returns a 2D array, even for a single query.
        """
        # Handle single string
        single = isinstance(queries, str)
        if single:
            queries = [queries]

        # Prepend query prefix if configured
        if self.query_prefix:
            queries = [self.query_prefix + q for q in queries]

        embeddings = self.model.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )

        embeddings = embeddings.astype(np.float32)

        logger.debug(f"Encoded {len(queries)} queries → shape {embeddings.shape}")
        return embeddings

    # ── Config-driven factory ────────────────────

    @classmethod
    def from_config(cls, config: dict) -> "QueryEncoder":
        """
        Create a query encoder from the merged pipeline config.

        Reads:
        - config["embedding"]["model"]
        - config["embedding"]["dimension"]
        - config["embedding"]["normalize"]
        """
        emb = config.get("embedding", {})

        return cls(
            model_name=emb.get("model", "BAAI/bge-base-en-v1.5"),
            dimension=emb.get("dimension", 768),
            normalize=emb.get("normalize", True),
        )

    def __repr__(self) -> str:
        return (
            f"QueryEncoder(model={self.model_name!r}, "
            f"dim={self.dimension}, device={self.device})"
        )
