"""Evaluation module — metrics for retrieval and answer quality."""

from src.evaluation.metrics import (
    evaluate_predictions,
    exact_match,
    f1_score,
    recall_at_k,
    EvaluationResult,
)

__all__ = [
    "evaluate_predictions",
    "exact_match",
    "f1_score",
    "recall_at_k",
    "EvaluationResult",
]
