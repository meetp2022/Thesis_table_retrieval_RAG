"""
Graph-Augmented Table Retrieval Pipeline (Pipeline 3).

End-to-end orchestrator that wires all components:
    1. Load dataset → TableQARecord list
    2. Build NetworkX directed graphs
    3. Extract node features → PyG Data objects
    4. Encode graphs via GraphSAGE → 768d table embeddings
    5. Index embeddings in FAISS
    6. For each question: retrieve top-k tables → generate answer via LLM
    7. Evaluate with EM, F1, R@1, R@5

Usage:
    >>> from src.pipelines.graph_augmented.pipeline import GraphAugmentedPipeline
    >>> pipeline = GraphAugmentedPipeline.from_config(config)
    >>> pipeline.index(records)
    >>> results = pipeline.run(records)
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from src.data.dataset_loader import TableQARecord
from src.graph.table_to_graph import build_graphs_from_config
from src.graph.feature_extraction import GraphFeatureExtractor
from src.graph.graph_embedding import TableGraphEncoder
from src.pipelines.shared.vector_store import FaissVectorStore
from src.pipelines.shared.query_encoder import QueryEncoder
from src.pipelines.shared.retriever import TableRetriever
from src.pipelines.shared.llm_client import create_llm_client, MockLLMClient
from src.pipelines.shared.answer_generator import AnswerGenerator, GenerationResult
from src.evaluation.metrics import evaluate_predictions, EvaluationResult


class GraphAugmentedPipeline:
    """
    Pipeline 3: Graph-Augmented Table Retrieval.

    Parameters
    ----------
    config : dict
        Merged config (base_config + pipeline3_graph).
    feature_extractor : GraphFeatureExtractor
        Converts NX graphs → PyG Data with node features.
    graph_encoder : TableGraphEncoder
        GraphSAGE model that produces table embeddings.
    vector_store : FaissVectorStore
        FAISS index for table embeddings.
    query_encoder : QueryEncoder
        Encodes query strings for retrieval.
    retriever : TableRetriever
        Retrieves top-k tables for a query.
    answer_generator : AnswerGenerator
        Generates answers from retrieved context.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        feature_extractor: GraphFeatureExtractor,
        graph_encoder: TableGraphEncoder,
        vector_store: FaissVectorStore,
        query_encoder: QueryEncoder,
        retriever: TableRetriever,
        answer_generator: AnswerGenerator,
    ):
        self.config = config
        self.feature_extractor = feature_extractor
        self.graph_encoder = graph_encoder
        self.vector_store = vector_store
        self.query_encoder = query_encoder
        self.retriever = retriever
        self.answer_generator = answer_generator

        # Record lookup built during indexing
        self._record_lookup: Dict[str, TableQARecord] = {}

    # ── Indexing phase ───────────────────────────

    def index(self, records: List[TableQARecord]) -> None:
        """
        Build the FAISS index from a list of table records.

        Steps:
            1. Deduplicate tables by record ID
            2. Build NetworkX graphs
            3. Extract PyG features (text embeddings + structural)
            4. Run GraphSAGE encoder → 768d table vectors
            5. Add to FAISS index

        Parameters
        ----------
        records : list[TableQARecord]
            Dataset records to index.
        """
        start = time.perf_counter()

        # Build record lookup (dedup by ID for indexing)
        seen_ids = set()
        unique_records = []
        for rec in records:
            if rec.id not in seen_ids:
                seen_ids.add(rec.id)
                unique_records.append(rec)
            self._record_lookup[rec.id] = rec

        logger.info(
            f"Indexing {len(unique_records)} unique tables "
            f"(from {len(records)} records)..."
        )

        # Step 1: Table → NetworkX graphs
        t0 = time.perf_counter()
        graphs = build_graphs_from_config(unique_records, self.config)
        logger.info(f"  Graph construction: {time.perf_counter() - t0:.2f}s")

        # Step 2: NetworkX → PyG Data (with text embeddings)
        t0 = time.perf_counter()
        pyg_data_list = self.feature_extractor.convert_batch(graphs)
        logger.info(f"  Feature extraction: {time.perf_counter() - t0:.2f}s")

        # Step 3: GraphSAGE encoding → table embeddings
        t0 = time.perf_counter()
        _node_embs, graph_embs = self.graph_encoder.encode_batch(pyg_data_list)
        logger.info(f"  Graph encoding:     {time.perf_counter() - t0:.2f}s")

        # Step 4: Add to FAISS
        record_ids = [d.record_id for d in pyg_data_list]
        metadata = [
            {"dataset": d.dataset, "num_nodes": d.x.size(0)}
            for d in pyg_data_list
        ]

        self.vector_store.reset()
        self.vector_store.add(
            graph_embs.numpy(),
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
            Records with questions to evaluate. The index must already
            be built via :meth:`index`.
        max_samples : int, optional
            Limit evaluation to first N samples.

        Returns
        -------
        EvaluationResult
            Aggregated metrics.
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
        """Save the index and encoder weights to disk."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        self.vector_store.save(str(dir_path / "vector_store"))
        self.graph_encoder.save(str(dir_path / "graph_encoder.pt"))
        logger.info(f"Pipeline saved to {directory}")

    def load_index(self, directory: str) -> None:
        """Load a previously saved index and encoder."""
        dir_path = Path(directory)

        loaded_store = FaissVectorStore.load(str(dir_path / "vector_store"))
        self.vector_store.index = loaded_store.index
        self.vector_store._idx_to_id = loaded_store._idx_to_id
        self.vector_store._id_to_idx = loaded_store._id_to_idx
        self.vector_store._metadata = loaded_store._metadata

        self.graph_encoder.load(str(dir_path / "graph_encoder.pt"))
        logger.info(f"Pipeline loaded from {directory}")

    # ── Config-driven factory ────────────────────

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        fallback_to_mock: bool = True,
    ) -> "GraphAugmentedPipeline":
        """
        Build the full pipeline from the merged config dict.

        Parameters
        ----------
        config : dict
            Merged config from load_config(pipeline="graph").
        fallback_to_mock : bool
            If True, use MockLLMClient when Ollama is unavailable.
        """
        logger.info("Building Graph-Augmented Pipeline from config...")

        feature_extractor = GraphFeatureExtractor.from_config(config)
        graph_encoder = TableGraphEncoder.from_config(config)

        # Load trained checkpoint if available
        train_cfg = config.get("training", {})
        save_dir = train_cfg.get("save_dir", "models/graph_encoder")
        checkpoint_path = Path(save_dir) / "best_encoder.pt"
        if checkpoint_path.exists():
            graph_encoder.load(str(checkpoint_path))
            logger.info(f"Loaded trained GraphSAGE from {checkpoint_path}")
        else:
            logger.warning(
                "No trained GraphSAGE checkpoint found — using random weights. "
                "Run: python scripts/train_graph_encoder.py"
            )

        vector_store = FaissVectorStore.from_config(config)
        query_encoder = QueryEncoder.from_config(config)
        retriever = TableRetriever.from_config(config, vector_store, query_encoder)
        llm_client = create_llm_client(config, fallback_to_mock=fallback_to_mock)
        answer_generator = AnswerGenerator.from_config(config, llm_client)

        logger.info(
            f"Pipeline components ready "
            f"(LLM: {llm_client.__class__.__name__})"
        )

        return cls(
            config=config,
            feature_extractor=feature_extractor,
            graph_encoder=graph_encoder,
            vector_store=vector_store,
            query_encoder=query_encoder,
            retriever=retriever,
            answer_generator=answer_generator,
        )

    def __repr__(self) -> str:
        return (
            f"GraphAugmentedPipeline("
            f"index_size={len(self.vector_store)}, "
            f"encoder={self.graph_encoder.__class__.__name__})"
        )
