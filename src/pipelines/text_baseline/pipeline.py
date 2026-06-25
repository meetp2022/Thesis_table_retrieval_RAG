"""
Text Baseline Pipeline (Pipeline 1).

The industry-standard approach for table retrieval:
    1. Linearise each table → Markdown string
    2. Embed with BAAI/bge-base-en-v1.5 (same model as query side)
    3. Index embeddings in FAISS (IndexFlatIP, cosine similarity)
    4. Retrieve top-k tables for a query via ANN search
    5. Generate answer with Mistral via Ollama (or mock)
    6. Evaluate with EM, F1, R@1, R@5

This pipeline is the **comparison baseline** for the thesis.  The graph
pipeline (Pipeline 3) should outperform it by capturing structural
relationships that plain text linearisation loses.

Usage:
    >>> from src.pipelines.text_baseline.pipeline import TextBaselinePipeline
    >>> pipeline = TextBaselinePipeline.from_config(config)
    >>> pipeline.index(records)
    >>> results = pipeline.run(records)
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from src.data.dataset_loader import TableQARecord
from src.pipelines.shared.vector_store import FaissVectorStore
from src.pipelines.shared.query_encoder import QueryEncoder
from src.pipelines.shared.retriever import TableRetriever
from src.pipelines.shared.llm_client import create_llm_client
from src.pipelines.shared.answer_generator import AnswerGenerator
from src.evaluation.metrics import evaluate_predictions, EvaluationResult


# ────────────────────────────────────────────────
#  Lineariser helper
# ────────────────────────────────────────────────

def linearise_table(record: TableQARecord, config: dict) -> str:
    """
    Convert a TableQARecord to a plain-text string for embedding.

    The output format depends on the linearisation config:

        [Title]
        | col1 | col2 | … |
        | --- | --- | … |
        | v1  | v2  | … |
        …

    For hybrid datasets that include paragraph context (TAT-QA, FinQA),
    the context is appended after the table.

    Parameters
    ----------
    record : TableQARecord
        The table record to linearise.
    config : dict
        Merged pipeline config (reads from config["linearisation"]).

    Returns
    -------
    str
        The linearised table string ready for embedding.
    """
    lin_cfg = config.get("linearisation", {})
    include_title = lin_cfg.get("include_table_title", True)
    include_context = config.get("generation", {}).get("include_context_text", True)
    max_rows = config.get("table_parsing", {}).get("max_rows", 100)

    parts: List[str] = []

    # Title / topic header
    if include_title and record.table_title:
        parts.append(f"Table: {record.table_title}")

    # Markdown table (use record's built-in method, then optionally truncate)
    # We re-build here so we can respect max_rows
    header_line = "| " + " | ".join(record.table_header) + " |"
    sep_line = "| " + " | ".join(["---"] * len(record.table_header)) + " |"
    parts.append(header_line)
    parts.append(sep_line)

    rows_to_use = record.table_rows[:max_rows]
    for row in rows_to_use:
        padded = list(row) + [""] * (len(record.table_header) - len(row))
        parts.append("| " + " | ".join(padded[:len(record.table_header)]) + " |")

    if len(record.table_rows) > max_rows:
        parts.append(f"[... {len(record.table_rows) - max_rows} more rows truncated ...]")

    # Optional paragraph context (TAT-QA / FinQA)
    if include_context and record.context_text:
        parts.append("")
        parts.append(f"Context: {record.context_text}")

    return "\n".join(parts)


# ────────────────────────────────────────────────
#  Pipeline
# ────────────────────────────────────────────────

class TextBaselinePipeline:
    """
    Pipeline 1: Text Baseline (Table Linearisation + BGE + FAISS).

    Parameters
    ----------
    config : dict
        Merged pipeline config (base_config + pipeline1_text).
    vector_store : FaissVectorStore
        FAISS index for table text embeddings.
    query_encoder : QueryEncoder
        Encodes query strings; also used for table-side indexing.
    retriever : TableRetriever
        Retrieves top-k tables for a query.
    answer_generator : AnswerGenerator
        Generates answers from retrieved table context.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        vector_store: FaissVectorStore,
        query_encoder: QueryEncoder,
        retriever: TableRetriever,
        answer_generator: AnswerGenerator,
    ):
        self.config = config
        self.vector_store = vector_store
        self.query_encoder = query_encoder
        self.retriever = retriever
        self.answer_generator = answer_generator

        # Record lookup built during indexing
        self._record_lookup: Dict[str, TableQARecord] = {}

    # ── Indexing phase ───────────────────────────

    def index(self, records: List[TableQARecord]) -> None:
        """
        Linearise each table, embed with BGE, and add to FAISS.

        Steps:
            1. Deduplicate records by ID
            2. Linearise each table to a text string
            3. Batch-embed with the query encoder model
            4. Add normalised vectors to FAISS

        Parameters
        ----------
        records : list[TableQARecord]
            Dataset records to index.
        """
        start = time.perf_counter()

        # Build record lookup (dedup by ID for indexing)
        seen_ids: set = set()
        unique_records: List[TableQARecord] = []
        for rec in records:
            if rec.id not in seen_ids:
                seen_ids.add(rec.id)
                unique_records.append(rec)
            self._record_lookup[rec.id] = rec

        logger.info(
            f"Indexing {len(unique_records)} unique tables "
            f"(from {len(records)} records)..."
        )

        # Step 1: Linearise tables → text strings
        t0 = time.perf_counter()
        texts = [linearise_table(rec, self.config) for rec in unique_records]
        logger.info(f"  Linearisation:   {time.perf_counter() - t0:.2f}s")

        # Step 2: Embed all table texts in one batch
        t0 = time.perf_counter()
        emb_cfg = self.config.get("embedding", {})
        batch_size = emb_cfg.get("batch_size", 32)
        table_embeddings = self.query_encoder.encode(texts, batch_size=batch_size)
        logger.info(
            f"  Text embedding:  {time.perf_counter() - t0:.2f}s  "
            f"({table_embeddings.shape})"
        )

        # Step 3: Index in FAISS
        record_ids = [rec.id for rec in unique_records]
        metadata = [
            {
                "dataset": rec.dataset,
                "num_rows": rec.num_rows,
                "num_cols": rec.num_cols,
                "pipeline": "text_baseline",
            }
            for rec in unique_records
        ]

        self.vector_store.reset()
        self.vector_store.add(
            table_embeddings,
            record_ids,
            metadata,
        )

        total = time.perf_counter() - start
        logger.info(
            f"Indexing complete: {len(self.vector_store)} tables indexed "
            f"in {total:.2f}s"
        )

    # ── Inference phase ──────────────────────────

    def run(
        self,
        records: List[TableQARecord],
        max_samples: Optional[int] = None,
    ) -> EvaluationResult:
        """
        Run end-to-end: retrieve + generate + evaluate.

        Parameters
        ----------
        records : list[TableQARecord]
            Records with questions to answer. The index must already be
            built via :meth:`index`.
        max_samples : int, optional
            Limit evaluation to first N samples (for quick testing).

        Returns
        -------
        EvaluationResult
            Aggregated metrics (EM, F1, R@1, R@5, latency).
        """
        if len(self.vector_store) == 0:
            raise RuntimeError(
                "Vector store is empty. Call pipeline.index(records) first."
            )

        eval_records = records[:max_samples] if max_samples else records
        logger.info(f"Running evaluation on {len(eval_records)} samples...")

        # Batch retrieve
        questions = [r.question for r in eval_records]
        batch_results = self.retriever.retrieve_batch(questions)

        # Batch generate
        gen_results = self.answer_generator.generate_batch(
            questions,
            batch_results,
            self._record_lookup,
        )

        # Evaluate
        gold_answers = [r.answers for r in eval_records]
        gold_ids = [r.id for r in eval_records]

        eval_result = evaluate_predictions(
            gen_results,
            gold_answers,
            gold_record_ids=gold_ids,
        )

        return eval_result

    # ── Persistence ──────────────────────────────

    def save(self, directory: str) -> None:
        """Save the FAISS index to disk."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save(str(dir_path / "vector_store"))
        logger.info(f"TextBaselinePipeline index saved to {directory}")

    def load_index(self, directory: str) -> None:
        """Load a previously saved index from disk."""
        dir_path = Path(directory)
        loaded_store = FaissVectorStore.load(str(dir_path / "vector_store"))
        self.vector_store.index = loaded_store.index
        self.vector_store._idx_to_id = loaded_store._idx_to_id
        self.vector_store._id_to_idx = loaded_store._id_to_idx
        self.vector_store._metadata = loaded_store._metadata
        logger.info(f"TextBaselinePipeline index loaded from {directory}")

    # ── Config-driven factory ────────────────────

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        fallback_to_mock: bool = True,
    ) -> "TextBaselinePipeline":
        """
        Build the full pipeline from a merged config dict.

        Parameters
        ----------
        config : dict
            Merged config from load_config(pipeline="text").
        fallback_to_mock : bool
            If True, use MockLLMClient when Ollama is unavailable.
        """
        logger.info("Building Text Baseline Pipeline from config...")

        vector_store = FaissVectorStore.from_config(config)
        query_encoder = QueryEncoder.from_config(config)
        retriever = TableRetriever.from_config(config, vector_store, query_encoder)
        llm_client = create_llm_client(config, fallback_to_mock=fallback_to_mock)
        answer_generator = AnswerGenerator.from_config(config, llm_client)

        logger.info(
            f"TextBaselinePipeline ready "
            f"(LLM: {llm_client.__class__.__name__})"
        )

        return cls(
            config=config,
            vector_store=vector_store,
            query_encoder=query_encoder,
            retriever=retriever,
            answer_generator=answer_generator,
        )

    def __len__(self) -> int:
        return len(self.vector_store)

    def __repr__(self) -> str:
        return (
            f"TextBaselinePipeline("
            f"index_size={len(self.vector_store)}, "
            f"encoder={self.query_encoder.model_name!r})"
        )
