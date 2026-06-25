"""
Image Baseline Pipeline (Pipeline 2).

Pipeline flow:
    1. Render each :class:`TableQARecord` to a PNG image (matplotlib)
    2. Encode the image with a vision-language model (CLIP by default,
       ColPali optionally) → single L2-normalised vector
    3. Index vectors in FAISS (``IndexFlatIP`` = cosine similarity)
    4. At query time, encode the natural-language question with the
       **same** VLM's text tower and retrieve nearest neighbours

This completes the three-way encoding comparison claimed in the thesis
contribution statement:

    - Pipeline 1: OCR/string encoding (text baseline, BGE)
    - Pipeline 2: **image-based encoding** (this pipeline, CLIP)
    - Pipeline 3: graph-augmented encoding (GNN on table graph)

The image baseline mirrors how enterprise RAG systems actually retrieve
tables from scanned PDFs today — our hypothesis is that the graph-augmented
pipeline preserves cell/header relationships better than either flat
encoding and therefore outperforms both at retrieval.
"""

import gc
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from PIL import Image

from src.data.dataset_loader import TableQARecord
from src.pipelines.shared.image_encoder import build_image_encoder
from src.pipelines.shared.table_renderer import render_table_to_image
from src.pipelines.shared.vector_store import FaissVectorStore


# ────────────────────────────────────────────────
#  Retriever
# ────────────────────────────────────────────────

class ImageRetriever:
    """
    Retriever that encodes text queries with the image pipeline's VLM text
    tower and searches the image-side FAISS index.
    """

    def __init__(self, vector_store: FaissVectorStore, encoder):
        self.vector_store = vector_store
        self.encoder = encoder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        q_emb = self.encoder.encode_queries([query])
        scores, ids = self.vector_store.search(q_emb, top_k=top_k)
        return list(zip(ids[0], scores[0].tolist()))

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = 5,
    ) -> List[List[Tuple[str, float]]]:
        q_emb = self.encoder.encode_queries(queries)
        scores, ids = self.vector_store.search(q_emb, top_k=top_k)
        return [
            list(zip(row_ids, row_scores.tolist()))
            for row_ids, row_scores in zip(ids, scores)
        ]


# ────────────────────────────────────────────────
#  Pipeline
# ────────────────────────────────────────────────

class ImageBaselinePipeline:
    """Pipeline 2: render → encode → FAISS → retrieve."""

    def __init__(
        self,
        config: Dict[str, Any],
        vector_store: FaissVectorStore,
        encoder,
        retriever: ImageRetriever,
    ):
        self.config = config
        self.vector_store = vector_store
        self.encoder = encoder
        self.retriever = retriever
        self._record_lookup: Dict[str, TableQARecord] = {}

    # ── Indexing phase ───────────────────────────

    def index(self, records: List[TableQARecord]) -> None:
        start = time.perf_counter()

        # Dedup by ID
        seen_ids: set = set()
        unique_records: List[TableQARecord] = []
        for rec in records:
            if rec.id not in seen_ids:
                seen_ids.add(rec.id)
                unique_records.append(rec)
            self._record_lookup[rec.id] = rec

        logger.info(
            f"Indexing {len(unique_records)} unique tables via image encoding "
            f"(from {len(records)} records)..."
        )

        # Step 1: Render tables → PNG files (disk-backed to avoid OOM/segfault
        # when holding hundreds of PIL objects + CLIP weights simultaneously)
        render_cfg = self.config.get("table_rendering", {})
        emb_cfg = self.config.get("embedding", {})
        batch_size = emb_cfg.get("batch_size", 4)

        t0 = time.perf_counter()
        tmpdir = tempfile.mkdtemp(prefix="image_baseline_")
        image_paths: List[str] = []
        for i, rec in enumerate(unique_records):
            try:
                img = render_table_to_image(
                    rec,
                    max_rows=render_cfg.get("max_rows", 30),
                    max_cols=render_cfg.get("max_cols", 12),
                    dpi=render_cfg.get("dpi", 100),
                    font_size=render_cfg.get("font_size", 9),
                    include_title=render_cfg.get("include_title", True),
                    include_context=render_cfg.get("include_context", False),
                )
            except Exception as e:
                logger.warning(f"Render failed for {rec.id}: {e}; using blank placeholder")
                img = Image.new("RGB", (640, 320), color="white")

            path = f"{tmpdir}/{i:06d}.png"
            img.save(path, format="PNG", optimize=False)
            image_paths.append(path)
            img.close()
            del img
            if (i + 1) % 50 == 0:
                gc.collect()
                logger.info(f"  rendered {i+1}/{len(unique_records)}")
        logger.info(f"  Rendering:        {time.perf_counter() - t0:.2f}s")

        # Release matplotlib font cache + any leaked figures before CLIP load
        try:
            import matplotlib.pyplot as _plt
            _plt.close("all")
        except Exception:
            pass
        gc.collect()

        # Step 2: Encode images in small chunks from disk
        t0 = time.perf_counter()
        embeddings_chunks: List[np.ndarray] = []
        chunk = batch_size * 4  # load 16 images, encode in sub-batches of 4
        for start_i in range(0, len(image_paths), chunk):
            paths_batch = image_paths[start_i : start_i + chunk]
            loaded = [Image.open(p).convert("RGB") for p in paths_batch]
            embs = self.encoder.encode_images(loaded, batch_size=batch_size)
            embeddings_chunks.append(embs)
            for im in loaded:
                im.close()
            del loaded
            gc.collect()

        table_embeddings = np.concatenate(embeddings_chunks, axis=0)
        logger.info(
            f"  Image encoding:   {time.perf_counter() - t0:.2f}s  "
            f"({table_embeddings.shape})"
        )

        # Step 3: Index in FAISS
        record_ids = [rec.id for rec in unique_records]
        metadata = [
            {
                "dataset": rec.dataset,
                "num_rows": rec.num_rows,
                "num_cols": rec.num_cols,
                "pipeline": "image_baseline",
            }
            for rec in unique_records
        ]

        self.vector_store.reset()
        self.vector_store.add(table_embeddings, record_ids, metadata)

        total = time.perf_counter() - start
        logger.info(
            f"Indexing complete: {len(self.vector_store)} tables indexed "
            f"in {total:.2f}s"
        )

    # ── Persistence ──────────────────────────────

    def save(self, directory: str) -> None:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save(str(dir_path / "vector_store"))
        logger.info(f"ImageBaselinePipeline index saved to {directory}")

    def load_index(self, directory: str) -> None:
        dir_path = Path(directory)
        loaded = FaissVectorStore.load(str(dir_path / "vector_store"))
        self.vector_store.index = loaded.index
        self.vector_store._idx_to_id = loaded._idx_to_id
        self.vector_store._id_to_idx = loaded._id_to_idx
        self.vector_store._metadata = loaded._metadata
        logger.info(f"ImageBaselinePipeline index loaded from {directory}")

    # ── Config-driven factory ────────────────────

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImageBaselinePipeline":
        encoder = build_image_encoder(config)

        # VLM output dim overrides global embedding.dimension
        vlm_dim = config.get("image_encoding", {}).get(
            "embedding_dimension", getattr(encoder, "embedding_dim", 512)
        )
        vs = FaissVectorStore(
            embedding_dim=vlm_dim,
            index_type=config.get("vector_store", {}).get("index_type", "IndexFlatIP"),
            normalize=config.get("indexing", {}).get("normalize_embeddings", True),
        )
        retriever = ImageRetriever(vector_store=vs, encoder=encoder)

        return cls(
            config=config,
            vector_store=vs,
            encoder=encoder,
            retriever=retriever,
        )
