"""
Unit tests for the evaluation metrics module (src/evaluation/metrics.py).
"""

import pytest

from src.evaluation.metrics import (
    _normalise_answer,
    _tokenise,
    exact_match,
    f1_score,
    recall_at_k,
    evaluate_predictions,
    EvaluationResult,
)
from src.pipelines.shared.answer_generator import GenerationResult


# ────────────────────────────────────────────────
#  Text normalisation
# ────────────────────────────────────────────────

class TestNormaliseAnswer:
    def test_lowercase(self):
        assert _normalise_answer("Hello World") == "hello world"

    def test_strip_punctuation(self):
        assert _normalise_answer("hello, world!") == "hello world"

    def test_remove_articles(self):
        assert _normalise_answer("the quick brown fox") == "quick brown fox"

    def test_collapse_whitespace(self):
        assert _normalise_answer("  hello   world  ") == "hello world"

    def test_combined(self):
        assert _normalise_answer("The Answer is: $42.") == "answer is 42"

    def test_empty(self):
        assert _normalise_answer("") == ""


class TestTokenise:
    def test_basic(self):
        assert _tokenise("hello world") == ["hello", "world"]

    def test_normalises_first(self):
        assert _tokenise("The Cat!") == ["cat"]


# ────────────────────────────────────────────────
#  Exact Match
# ────────────────────────────────────────────────

class TestExactMatch:
    def test_exact_match(self):
        assert exact_match("42", ["42"]) == 1.0

    def test_case_insensitive(self):
        assert exact_match("France", ["france"]) == 1.0

    def test_no_match(self):
        assert exact_match("Germany", ["France"]) == 0.0

    def test_multiple_gold_answers(self):
        assert exact_match("42", ["forty-two", "42", "forty two"]) == 1.0

    def test_with_punctuation(self):
        assert exact_match("$42", ["42"]) == 1.0

    def test_with_articles(self):
        assert exact_match("the answer", ["answer"]) == 1.0

    def test_empty_prediction(self):
        assert exact_match("", ["42"]) == 0.0

    def test_both_empty(self):
        assert exact_match("", [""]) == 1.0


# ────────────────────────────────────────────────
#  F1 Score
# ────────────────────────────────────────────────

class TestF1Score:
    def test_perfect_match(self):
        assert f1_score("hello world", ["hello world"]) == 1.0

    def test_partial_overlap(self):
        # pred: ["hello", "world"], gold: ["hello", "earth"]
        # common: 1 ("hello"), precision=1/2, recall=1/2, F1=0.5
        assert f1_score("hello world", ["hello earth"]) == pytest.approx(0.5)

    def test_no_overlap(self):
        assert f1_score("foo bar", ["baz qux"]) == 0.0

    def test_multiple_gold_takes_best(self):
        score = f1_score("hello world", ["xyz", "hello world"])
        assert score == 1.0

    def test_empty_prediction(self):
        assert f1_score("", ["hello"]) == 0.0

    def test_both_empty(self):
        assert f1_score("", [""]) == 1.0

    def test_superset_prediction(self):
        # pred: ["quick", "brown", "fox"], gold: ["brown", "fox"]
        # common: 2, precision=2/3, recall=2/2=1, F1=2*(2/3*1)/(2/3+1)=4/5=0.8
        assert f1_score("quick brown fox", ["brown fox"]) == pytest.approx(0.8)


# ────────────────────────────────────────────────
#  Recall@k
# ────────────────────────────────────────────────

class TestRecallAtK:
    def test_gold_at_rank_1(self):
        assert recall_at_k(["a", "b", "c"], "a", k=1) == 1.0

    def test_gold_at_rank_3(self):
        assert recall_at_k(["a", "b", "c"], "c", k=3) == 1.0

    def test_gold_not_in_top_k(self):
        assert recall_at_k(["a", "b", "c"], "c", k=2) == 0.0

    def test_gold_not_in_list(self):
        assert recall_at_k(["a", "b", "c"], "x", k=5) == 0.0

    def test_empty_retrieved(self):
        assert recall_at_k([], "a", k=5) == 0.0


# ────────────────────────────────────────────────
#  Batch evaluation
# ────────────────────────────────────────────────

def _make_gen_result(answer, record_ids=None, latency=0.01):
    return GenerationResult(
        question="test?",
        answer=answer,
        retrieved_record_ids=record_ids or [],
        retrieval_scores=[0.9] * len(record_ids or []),
        prompt="test prompt",
        model="test",
        latency_seconds=latency,
    )


class TestEvaluatePredictions:
    def test_perfect_scores(self):
        gen_results = [
            _make_gen_result("42", ["gold-1"]),
            _make_gen_result("France", ["gold-2"]),
        ]
        gold_answers = [["42"], ["France"]]
        gold_ids = ["gold-1", "gold-2"]

        result = evaluate_predictions(gen_results, gold_answers, gold_ids)

        assert result.num_samples == 2
        assert result.exact_match == 1.0
        assert result.f1_score == 1.0
        assert result.recall_at_1 == 1.0
        assert result.recall_at_5 == 1.0

    def test_zero_scores(self):
        gen_results = [
            _make_gen_result("wrong", ["other-1"]),
        ]
        gold_answers = [["correct"]]
        gold_ids = ["gold-1"]

        result = evaluate_predictions(gen_results, gold_answers, gold_ids)

        assert result.exact_match == 0.0
        assert result.f1_score == 0.0
        assert result.recall_at_1 == 0.0

    def test_mixed_scores(self):
        gen_results = [
            _make_gen_result("42", ["gold-1"], latency=0.1),
            _make_gen_result("wrong", ["other"], latency=0.2),
        ]
        gold_answers = [["42"], ["correct"]]
        gold_ids = ["gold-1", "gold-2"]

        result = evaluate_predictions(gen_results, gold_answers, gold_ids)

        assert result.exact_match == 0.5
        assert result.recall_at_1 == 0.5
        assert result.avg_latency == pytest.approx(0.15)

    def test_without_gold_ids(self):
        gen_results = [_make_gen_result("42")]
        gold_answers = [["42"]]

        result = evaluate_predictions(gen_results, gold_answers)

        assert result.exact_match == 1.0
        assert result.recall_at_1 == 0.0  # no gold IDs → 0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            evaluate_predictions(
                [_make_gen_result("a")],
                [["a"], ["b"]],
            )

    def test_per_sample_details(self):
        gen_results = [_make_gen_result("42", ["rec-1"])]
        gold_answers = [["42"]]

        result = evaluate_predictions(gen_results, gold_answers, ["rec-1"])

        assert len(result.per_sample) == 1
        assert result.per_sample[0]["exact_match"] == 1.0
        assert result.per_sample[0]["prediction"] == "42"


# ────────────────────────────────────────────────
#  EvaluationResult
# ────────────────────────────────────────────────

class TestEvaluationResult:
    def test_to_dict(self):
        r = EvaluationResult(
            num_samples=10,
            exact_match=0.8,
            f1_score=0.85,
            recall_at_1=0.7,
            recall_at_5=0.9,
            avg_latency=0.5,
            total_latency=5.0,
        )
        d = r.to_dict()
        assert d["num_samples"] == 10
        assert d["exact_match"] == 0.8
        assert "avg_latency_seconds" in d

    def test_summary_string(self):
        r = EvaluationResult(
            num_samples=10,
            exact_match=0.8,
            f1_score=0.85,
            recall_at_1=0.7,
            recall_at_5=0.9,
            avg_latency=0.5,
            total_latency=5.0,
        )
        s = r.summary()
        assert "10 samples" in s
        assert "0.8000" in s
