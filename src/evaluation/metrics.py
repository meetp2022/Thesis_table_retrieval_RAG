"""
Evaluation metrics for table QA pipelines.

Computes retrieval and answer-quality metrics:
    - Recall@k     : Was the gold table in the top-k retrieved results?
    - Exact Match  : Does the prediction exactly match the gold answer?
    - F1 Score     : Token-level overlap between prediction and gold answer.
    - Latency      : Average end-to-end query time.

RAGAS-based metrics (faithfulness, context precision) are deferred to a
later phase when the full RAGAS integration is built.

Usage:
    >>> from src.evaluation.metrics import evaluate_predictions
    >>> results = evaluate_predictions(generation_results, records)
"""

import re
import string
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from src.pipelines.shared.answer_generator import GenerationResult


# ────────────────────────────────────────────────
#  Text normalisation (for EM / F1)
# ────────────────────────────────────────────────

def _normalise_answer(text: str) -> str:
    """
    Normalise answer text for fair comparison.

    Lowercases, strips punctuation/articles/whitespace — following the
    standard SQuAD evaluation script convention.
    """
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def _tokenise(text: str) -> List[str]:
    """Split normalised text into tokens."""
    return _normalise_answer(text).split()


# ────────────────────────────────────────────────
#  Per-sample metrics
# ────────────────────────────────────────────────

def exact_match(prediction: str, gold_answers: List[str]) -> float:
    """
    Exact match: 1.0 if the normalised prediction matches any gold answer.

    Parameters
    ----------
    prediction : str
        Model-generated answer.
    gold_answers : list[str]
        List of acceptable gold answers.

    Returns
    -------
    float
        1.0 or 0.0.
    """
    pred_norm = _normalise_answer(prediction)
    for gold in gold_answers:
        if _normalise_answer(gold) == pred_norm:
            return 1.0
    return 0.0


def f1_score(prediction: str, gold_answers: List[str]) -> float:
    """
    Token-level F1: best F1 across all gold answers.

    Parameters
    ----------
    prediction : str
        Model-generated answer.
    gold_answers : list[str]
        List of acceptable gold answers.

    Returns
    -------
    float
        F1 score between 0.0 and 1.0.
    """
    pred_tokens = _tokenise(prediction)

    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = _tokenise(gold)

        if not pred_tokens and not gold_tokens:
            best_f1 = max(best_f1, 1.0)
            continue
        if not pred_tokens or not gold_tokens:
            continue

        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            continue

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


def recall_at_k(
    retrieved_ids: List[str],
    gold_id: str,
    k: int = 5,
) -> float:
    """
    Recall@k: 1.0 if the gold record ID is in the top-k retrieved IDs.

    Parameters
    ----------
    retrieved_ids : list[str]
        Record IDs returned by the retriever, in rank order.
    gold_id : str
        The correct record ID.
    k : int
        Cutoff.

    Returns
    -------
    float
        1.0 or 0.0.
    """
    return 1.0 if gold_id in retrieved_ids[:k] else 0.0


# ────────────────────────────────────────────────
#  Evaluation result dataclass
# ────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """
    Aggregated evaluation metrics across a dataset.

    Attributes
    ----------
    num_samples : int
        Number of evaluated samples.
    exact_match : float
        Average exact match score.
    f1_score : float
        Average token-level F1 score.
    recall_at_1 : float
        Recall@1 (requires gold record IDs).
    recall_at_5 : float
        Recall@5 (requires gold record IDs).
    avg_latency : float
        Average LLM generation latency in seconds.
    total_latency : float
        Total evaluation time in seconds.
    per_sample : List[Dict[str, Any]]
        Per-sample breakdown (for analysis / export).
    """
    num_samples: int
    exact_match: float
    f1_score: float
    recall_at_1: float
    recall_at_5: float
    avg_latency: float
    total_latency: float
    per_sample: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return metrics as a flat dict (for JSON serialisation)."""
        return {
            "num_samples": self.num_samples,
            "exact_match": round(self.exact_match, 4),
            "f1_score": round(self.f1_score, 4),
            "recall_at_1": round(self.recall_at_1, 4),
            "recall_at_5": round(self.recall_at_5, 4),
            "avg_latency_seconds": round(self.avg_latency, 4),
            "total_latency_seconds": round(self.total_latency, 2),
        }

    def summary(self) -> str:
        """Human-readable summary string."""
        return (
            f"Evaluation ({self.num_samples} samples):\n"
            f"  Exact Match : {self.exact_match:.4f}\n"
            f"  F1 Score    : {self.f1_score:.4f}\n"
            f"  Recall@1    : {self.recall_at_1:.4f}\n"
            f"  Recall@5    : {self.recall_at_5:.4f}\n"
            f"  Avg Latency : {self.avg_latency:.3f}s"
        )


# ────────────────────────────────────────────────
#  Batch evaluation
# ────────────────────────────────────────────────

def evaluate_predictions(
    generation_results: List[GenerationResult],
    gold_answers: List[List[str]],
    gold_record_ids: Optional[List[str]] = None,
) -> EvaluationResult:
    """
    Evaluate a batch of generated answers against gold labels.

    Parameters
    ----------
    generation_results : list[GenerationResult]
        Outputs from AnswerGenerator.generate_batch().
    gold_answers : list[list[str]]
        Gold answers for each question (multiple acceptable per sample).
    gold_record_ids : list[str], optional
        Gold table record IDs for retrieval metrics (R@1, R@5).
        If None, recall metrics are reported as 0.0.

    Returns
    -------
    EvaluationResult
        Aggregated and per-sample metrics.
    """
    if len(generation_results) != len(gold_answers):
        raise ValueError(
            f"generation_results ({len(generation_results)}) and "
            f"gold_answers ({len(gold_answers)}) must have the same length"
        )

    em_scores = []
    f1_scores = []
    r1_scores = []
    r5_scores = []
    per_sample = []

    for i, (gen, golds) in enumerate(zip(generation_results, gold_answers)):
        em = exact_match(gen.answer, golds)
        f1 = f1_score(gen.answer, golds)

        r1 = 0.0
        r5 = 0.0
        if gold_record_ids and i < len(gold_record_ids):
            r1 = recall_at_k(gen.retrieved_record_ids, gold_record_ids[i], k=1)
            r5 = recall_at_k(gen.retrieved_record_ids, gold_record_ids[i], k=5)

        em_scores.append(em)
        f1_scores.append(f1)
        r1_scores.append(r1)
        r5_scores.append(r5)

        per_sample.append({
            "question": gen.question,
            "prediction": gen.answer,
            "gold_answers": golds,
            "exact_match": em,
            "f1_score": f1,
            "recall_at_1": r1,
            "recall_at_5": r5,
            "latency_seconds": gen.latency_seconds,
            "retrieved_ids": gen.retrieved_record_ids,
        })

    n = len(generation_results)
    total_latency = sum(g.latency_seconds for g in generation_results)

    result = EvaluationResult(
        num_samples=n,
        exact_match=sum(em_scores) / n if n else 0.0,
        f1_score=sum(f1_scores) / n if n else 0.0,
        recall_at_1=sum(r1_scores) / n if n else 0.0,
        recall_at_5=sum(r5_scores) / n if n else 0.0,
        avg_latency=total_latency / n if n else 0.0,
        total_latency=total_latency,
        per_sample=per_sample,
    )

    logger.info(f"\n{result.summary()}")
    return result
