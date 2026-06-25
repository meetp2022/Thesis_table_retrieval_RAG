"""
Image encoder for the image retrieval baseline (Pipeline 2).

Primary backend: **CLIP** (``openai/clip-vit-base-patch32``)
    - Joint text-image embedding space: 512-dim, single vector per item
    - CPU-friendly, compatible with the existing FaissVectorStore
    - Serves as the industry-standard visual retrieval baseline

Optional backend: **ColPali** (``vidore/colpali-v1.2``) — *GPU recommended*
    - Multi-vector (ColBERT-style) visual document retrieval
    - SOTA on ViDoRe benchmark; pooled to a single vector here for
      drop-in compatibility with the existing FAISS indexing code

Both backends expose the same :meth:`encode_images` and
:meth:`encode_queries` interface so ``ImageBaselinePipeline`` can switch
between them via config.
"""

from typing import List, Optional, Union

import numpy as np
from loguru import logger
from PIL import Image


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


class CLIPImageEncoder:
    """
    CLIP-based image + query encoder producing single-vector embeddings.

    The same model encodes both images (table screenshots) and text queries
    into a shared 512-dim space, enabling cross-modal retrieval via cosine
    similarity.

    Parameters
    ----------
    model_name : str
        HuggingFace CLIP model id (default: ``openai/clip-vit-base-patch32``).
    device : str, optional
        ``"cuda"`` or ``"cpu"``. Auto-detected if None.
    normalize : bool
        L2-normalise output embeddings (required for cosine sim with FAISS IP).
    """

    backend = "clip"
    embedding_dim = 512

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.normalize = normalize

        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._processor = None

    def _load(self):
        from transformers import CLIPModel, CLIPProcessor

        logger.info(f"Loading CLIP: {self.model_name} (device={self.device})")
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        self._processor = CLIPProcessor.from_pretrained(self.model_name)

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            self._load()
        return self._processor

    def encode_images(
        self,
        images: List[Image.Image],
        batch_size: int = 4,
    ) -> np.ndarray:
        import gc
        import torch

        all_embs: List[np.ndarray] = []
        total = len(images)
        for i in range(0, total, batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
            # Handle both tensor and ModelOutput return types
            if hasattr(feats, "image_embeds"):
                feats = feats.image_embeds
            elif hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            elif hasattr(feats, "last_hidden_state"):
                # Pool CLS token if raw hidden states returned
                feats = feats.last_hidden_state[:, 0, :]
            all_embs.append(feats.cpu().numpy().astype(np.float32))
            del inputs, feats
            if (i // batch_size) % 10 == 0 and total > 20:
                gc.collect()
            if total > 50 and (i + batch_size) % 50 == 0:
                logger.info(f"    encoded {min(i + batch_size, total)}/{total}")

        embeddings = np.concatenate(all_embs, axis=0)
        return _l2_normalize(embeddings) if self.normalize else embeddings

    def encode_queries(
        self,
        queries: Union[str, List[str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        import torch

        if isinstance(queries, str):
            queries = [queries]

        all_embs: List[np.ndarray] = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self.device)
            with torch.no_grad():
                feats = self.model.get_text_features(**inputs)
            if hasattr(feats, "text_embeds"):
                feats = feats.text_embeds
            elif hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            elif hasattr(feats, "last_hidden_state"):
                feats = feats.last_hidden_state[:, 0, :]
            all_embs.append(feats.cpu().numpy().astype(np.float32))

        embeddings = np.concatenate(all_embs, axis=0)
        return _l2_normalize(embeddings) if self.normalize else embeddings

    @classmethod
    def from_config(cls, config: dict) -> "CLIPImageEncoder":
        ie = config.get("image_encoding", {})
        return cls(
            model_name=ie.get("model", "openai/clip-vit-base-patch32"),
            normalize=config.get("indexing", {}).get("normalize_embeddings", True),
        )

    def __repr__(self) -> str:
        return f"CLIPImageEncoder(model={self.model_name!r}, device={self.device})"


class ColPaliImageEncoder:
    """
    ColPali-based image + query encoder (multi-vector → pooled single vector).

    Requires the ``colpali_engine`` package and a GPU for reasonable speed.
    For the thesis laptop path, prefer :class:`CLIPImageEncoder`. This class
    is available for an optional Kaggle run to produce a stronger image-side
    number for Chapter 5.

    The native ColPali output is multi-vector (ColBERT-style). For drop-in
    FAISS compatibility we mean-pool across patch tokens, yielding a single
    vector per image / query. This is a deliberate simplification, noted in
    the thesis limitations.
    """

    backend = "colpali"
    embedding_dim = 128  # ColPali patch-token dim

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.2",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.normalize = normalize

        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._processor = None

    def _load(self):
        try:
            from colpali_engine.models import ColPali, ColPaliProcessor
        except ImportError as e:
            raise ImportError(
                "colpali_engine is required for the ColPali backend. "
                "Install with: pip install colpali-engine"
            ) from e

        logger.info(f"Loading ColPali: {self.model_name} (device={self.device})")
        self._model = (
            ColPali.from_pretrained(self.model_name)
            .to(self.device)
            .eval()
        )
        self._processor = ColPaliProcessor.from_pretrained(self.model_name)

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            self._load()
        return self._processor

    def _pool(self, multi_vec: np.ndarray) -> np.ndarray:
        """Mean-pool token dimension → single vector per item."""
        if multi_vec.ndim == 3:  # (batch, tokens, dim)
            return multi_vec.mean(axis=1)
        return multi_vec

    def encode_images(
        self,
        images: List[Image.Image],
        batch_size: int = 4,
    ) -> np.ndarray:
        import torch

        all_embs: List[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor.process_images(batch).to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
            feats = out.cpu().numpy().astype(np.float32)
            all_embs.append(self._pool(feats))

        embeddings = np.concatenate(all_embs, axis=0)
        return _l2_normalize(embeddings) if self.normalize else embeddings

    def encode_queries(
        self,
        queries: Union[str, List[str]],
        batch_size: int = 16,
    ) -> np.ndarray:
        import torch

        if isinstance(queries, str):
            queries = [queries]

        all_embs: List[np.ndarray] = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            inputs = self.processor.process_queries(batch).to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
            feats = out.cpu().numpy().astype(np.float32)
            all_embs.append(self._pool(feats))

        embeddings = np.concatenate(all_embs, axis=0)
        return _l2_normalize(embeddings) if self.normalize else embeddings

    @classmethod
    def from_config(cls, config: dict) -> "ColPaliImageEncoder":
        ie = config.get("image_encoding", {})
        return cls(
            model_name=ie.get("model", "vidore/colpali-v1.2"),
            normalize=config.get("indexing", {}).get("normalize_embeddings", True),
        )


def build_image_encoder(config: dict):
    """Factory: pick CLIP or ColPali based on ``image_encoding.backend``."""
    backend = config.get("image_encoding", {}).get("backend", "clip").lower()
    if backend == "colpali":
        return ColPaliImageEncoder.from_config(config)
    return CLIPImageEncoder.from_config(config)
