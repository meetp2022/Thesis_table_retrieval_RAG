"""
Answer generator — constructs prompts from retrieved tables and generates
answers using the LLM client.

Ties together:
    - RetrievalResult (from retriever.py) — which tables were retrieved
    - TableQARecord  (from dataset_loader.py) — the actual table data
    - OllamaClient / MockLLMClient (from llm_client.py) — LLM backend

Usage:
    >>> from src.pipelines.shared.answer_generator import AnswerGenerator
    >>> generator = AnswerGenerator(llm_client)
    >>> result = generator.generate(question, retrieval_results, record_lookup)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from src.data.dataset_loader import TableQARecord
from src.pipelines.shared.retriever import RetrievalResult


# ────────────────────────────────────────────────
#  Default prompt template
# ────────────────────────────────────────────────

DEFAULT_PROMPT_TEMPLATE = """You are a precise table question-answering assistant. Answer the question using ONLY the provided table data. Follow these rules:
- Be concise and direct.
- If the answer requires calculation, show the final result.
- If the table does not contain enough information, say "insufficient data".
- Do not make assumptions beyond what is in the table.

{table_context}

QUESTION: {question}

ANSWER:"""


# ────────────────────────────────────────────────
#  Result dataclass
# ────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """
    Structured result from a single answer generation.

    Attributes
    ----------
    question : str
        The original question.
    answer : str
        The generated answer from the LLM.
    retrieved_record_ids : list[str]
        Record IDs of tables used as context.
    retrieval_scores : list[float]
        Similarity scores of retrieved tables.
    prompt : str
        The full prompt sent to the LLM (for debugging / evaluation).
    model : str
        The LLM model name.
    latency_seconds : float
        Wall-clock time for the LLM generation call.
    """
    question: str
    answer: str
    retrieved_record_ids: List[str]
    retrieval_scores: List[float]
    prompt: str
    model: str
    latency_seconds: float


# ────────────────────────────────────────────────
#  Context formatting helpers
# ────────────────────────────────────────────────

def _format_table_context(
    records: List[TableQARecord],
    include_context_text: bool = True,
    max_rows_per_table: int = 50,
) -> str:
    """
    Format one or more tables into a text context block for the prompt.

    Parameters
    ----------
    records : list[TableQARecord]
        Tables to include, in retrieval-rank order.
    include_context_text : bool
        If True, append paragraph context (for hybrid datasets like TAT-QA).
    max_rows_per_table : int
        Truncate tables with more rows to avoid exceeding context window.

    Returns
    -------
    str
        Formatted context string.
    """
    blocks = []

    for i, record in enumerate(records):
        # Truncate large tables
        truncated_record = record
        if len(record.table_rows) > max_rows_per_table:
            truncated_record = TableQARecord(
                id=record.id,
                question=record.question,
                answers=record.answers,
                table_header=record.table_header,
                table_rows=record.table_rows[:max_rows_per_table],
                table_title=record.table_title,
                context_text=record.context_text,
                dataset=record.dataset,
                domain=record.domain,
            )
            truncation_note = (
                f"\n[... {len(record.table_rows) - max_rows_per_table} "
                f"more rows truncated]"
            )
        else:
            truncation_note = ""

        # Build block
        if len(records) > 1:
            block = f"TABLE {i + 1}:\n"
        else:
            block = "TABLE:\n"

        block += truncated_record.table_to_markdown()
        block += truncation_note

        # Append paragraph context for hybrid datasets
        if include_context_text and record.context_text:
            block += f"\n\nADDITIONAL CONTEXT:\n{record.context_text}"

        blocks.append(block)

    return "\n\n".join(blocks)


# ────────────────────────────────────────────────
#  Answer generator
# ────────────────────────────────────────────────

class AnswerGenerator:
    """
    Generates answers by combining retrieved tables with an LLM.

    Parameters
    ----------
    llm_client : OllamaClient or MockLLMClient
        The LLM backend for generation.
    prompt_template : str
        Prompt template with {table_context} and {question} placeholders.
    max_tables : int
        Maximum number of retrieved tables to include in the prompt.
    include_context_text : bool
        Whether to include paragraph context from hybrid datasets.
    max_rows_per_table : int
        Truncate individual tables to this many rows.
    """

    def __init__(
        self,
        llm_client,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        max_tables: int = 3,
        include_context_text: bool = True,
        max_rows_per_table: int = 50,
    ):
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        self.max_tables = max_tables
        self.include_context_text = include_context_text
        self.max_rows_per_table = max_rows_per_table

    # ── Single generation ────────────────────────

    def generate(
        self,
        question: str,
        retrieval_results: List[RetrievalResult],
        record_lookup: Dict[str, TableQARecord],
    ) -> GenerationResult:
        """
        Generate an answer for a single question.

        Parameters
        ----------
        question : str
            The natural-language question.
        retrieval_results : list[RetrievalResult]
            Ranked retrieval results from TableRetriever.
        record_lookup : dict[str, TableQARecord]
            Mapping from record_id to the full TableQARecord.

        Returns
        -------
        GenerationResult
            Structured result with answer, prompt, latency, etc.
        """
        # Select top tables (up to max_tables)
        top_results = retrieval_results[:self.max_tables]

        # Look up records
        records = []
        used_ids = []
        used_scores = []
        for rr in top_results:
            rec = record_lookup.get(rr.record_id)
            if rec is not None:
                records.append(rec)
                used_ids.append(rr.record_id)
                used_scores.append(rr.score)
            else:
                logger.warning(
                    f"Record '{rr.record_id}' not found in lookup — skipping"
                )

        # Format context
        if records:
            table_context = _format_table_context(
                records,
                include_context_text=self.include_context_text,
                max_rows_per_table=self.max_rows_per_table,
            )
        else:
            table_context = "[No tables retrieved]"

        # Build prompt
        prompt = self.prompt_template.format(
            table_context=table_context,
            question=question,
        )

        # Call LLM
        start = time.perf_counter()
        answer = self.llm_client.generate(prompt)
        latency = time.perf_counter() - start

        logger.debug(
            f"Generated answer ({latency:.2f}s): "
            f"{answer[:80]!r}..."
        )

        return GenerationResult(
            question=question,
            answer=answer,
            retrieved_record_ids=used_ids,
            retrieval_scores=used_scores,
            prompt=prompt,
            model=self.llm_client.model,
            latency_seconds=latency,
        )

    # ── Batch generation ─────────────────────────

    def generate_batch(
        self,
        questions: List[str],
        batch_retrieval_results: List[List[RetrievalResult]],
        record_lookup: Dict[str, TableQARecord],
    ) -> List[GenerationResult]:
        """
        Generate answers for multiple questions.

        Parameters
        ----------
        questions : list[str]
            List of questions.
        batch_retrieval_results : list[list[RetrievalResult]]
            Retrieval results per question (from retrieve_batch).
        record_lookup : dict[str, TableQARecord]
            Record ID → TableQARecord mapping.

        Returns
        -------
        list[GenerationResult]
        """
        if len(questions) != len(batch_retrieval_results):
            raise ValueError(
                f"questions ({len(questions)}) and retrieval_results "
                f"({len(batch_retrieval_results)}) must have the same length"
            )

        results = []
        for i, (q, rr) in enumerate(zip(questions, batch_retrieval_results)):
            result = self.generate(q, rr, record_lookup)
            results.append(result)

            if (i + 1) % 10 == 0:
                logger.info(
                    f"Generated {i + 1}/{len(questions)} answers "
                    f"(avg latency: {sum(r.latency_seconds for r in results) / len(results):.2f}s)"
                )

        total_latency = sum(r.latency_seconds for r in results)
        logger.info(
            f"Batch generation complete: {len(results)} answers, "
            f"total latency: {total_latency:.2f}s"
        )
        return results

    # ── Config-driven factory ────────────────────

    @classmethod
    def from_config(cls, config: dict, llm_client) -> "AnswerGenerator":
        """
        Create an answer generator from the merged pipeline config.

        Reads from config["generation"].
        """
        gen = config.get("generation", {})

        return cls(
            llm_client=llm_client,
            max_tables=gen.get("max_tables", 3),
            include_context_text=gen.get("include_context_text", True),
            max_rows_per_table=gen.get("max_rows_per_table", 50),
        )

    def __repr__(self) -> str:
        return (
            f"AnswerGenerator(model={self.llm_client.model!r}, "
            f"max_tables={self.max_tables})"
        )
