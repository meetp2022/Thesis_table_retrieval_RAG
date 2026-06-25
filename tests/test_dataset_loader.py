"""
Tests for dataset loading and standardisation.

Run with: pytest tests/test_dataset_loader.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_loader import (
    TableQARecord,
    load_wikitablequestions,
    get_dataset_stats,
    load_dataset_by_name,
)


# ── TableQARecord unit tests ──


class TestTableQARecord:
    def setup_method(self):
        self.record = TableQARecord(
            id="test-001",
            question="What was the revenue in 2023?",
            answers=["$1.2B"],
            table_header=["Year", "Revenue", "Profit"],
            table_rows=[
                ["2021", "$800M", "$100M"],
                ["2022", "$1.0B", "$150M"],
                ["2023", "$1.2B", "$200M"],
            ],
            table_title="Annual Financials",
            context_text="The company saw strong growth.",
            dataset="test",
            domain="finance",
        )

    def test_basic_properties(self):
        assert self.record.id == "test-001"
        assert self.record.num_rows == 3
        assert self.record.num_cols == 3
        assert self.record.domain == "finance"

    def test_to_dict(self):
        d = self.record.to_dict()
        assert d["id"] == "test-001"
        assert d["table"]["header"] == ["Year", "Revenue", "Profit"]
        assert len(d["table"]["rows"]) == 3
        assert d["context_text"] == "The company saw strong growth."

    def test_table_to_markdown(self):
        md = self.record.table_to_markdown()
        assert "Annual Financials" in md
        assert "| Year | Revenue | Profit |" in md
        assert "| 2023 | $1.2B | $200M |" in md

    def test_repr(self):
        r = repr(self.record)
        assert "test-001" in r
        assert "3×3" in r


# ── Dataset stats ──


class TestDatasetStats:
    def test_stats_computation(self):
        records = [
            TableQARecord(
                id=f"r-{i}",
                question=f"Question number {i}?",
                answers=[f"answer-{i}"],
                table_header=["A", "B"],
                table_rows=[["1", "2"], ["3", "4"]],
                dataset="test",
                domain="general",
            )
            for i in range(5)
        ]
        stats = get_dataset_stats(records)
        assert stats["count"] == 5
        assert stats["avg_table_rows"] == 2.0
        assert stats["avg_table_cols"] == 2.0
        assert stats["domain"] == "general"

    def test_empty_stats(self):
        stats = get_dataset_stats([])
        assert stats["count"] == 0


# ── Integration test (requires network) ──


@pytest.mark.slow
class TestWikiTableQuestionsLoading:
    """These tests download data — mark with @pytest.mark.slow."""

    def test_load_small_sample(self):
        records = load_wikitablequestions(split="test", max_samples=3)
        assert len(records) == 3
        assert all(isinstance(r, TableQARecord) for r in records)
        assert all(r.dataset == "wikitablequestions" for r in records)
        assert all(r.domain == "general" for r in records)
        assert all(len(r.table_header) > 0 for r in records)

    def test_unified_loader(self):
        records = load_dataset_by_name("wikitq", split="test", max_samples=2)
        assert len(records) == 2
        assert records[0].dataset == "wikitablequestions"
