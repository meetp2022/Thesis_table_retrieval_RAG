"""
Shared pipeline components — vector store, query encoding, retrieval, and LLM.

These modules are pipeline-agnostic and reused across all three pipelines
(text baseline, image baseline, graph-augmented).
"""

from src.pipelines.shared.vector_store import FaissVectorStore
from src.pipelines.shared.query_encoder import QueryEncoder
from src.pipelines.shared.retriever import TableRetriever, RetrievalResult
from src.pipelines.shared.llm_client import OllamaClient, MockLLMClient, create_llm_client
from src.pipelines.shared.answer_generator import AnswerGenerator, GenerationResult

__all__ = [
    # Phase 4 — retrieval
    "FaissVectorStore",
    "QueryEncoder",
    "TableRetriever",
    "RetrievalResult",
    # Phase 5 — LLM answer generation
    "OllamaClient",
    "MockLLMClient",
    "create_llm_client",
    "AnswerGenerator",
    "GenerationResult",
]
