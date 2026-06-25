"""
Unit and integration tests for the Text Baseline Pipeline (Pipeline 1).

Tests cover:
    - Table linearisation (linearise_table)
    - Pipeline indexing
    - Pipeline retrieval
    - End-to-end run (mock LLM)
    - save / load index
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data.dataset_loader import TableQARecord
from src.pipelines.text_baseline.pipeline import TextBaselinePipeline, linearise_table


# ────────────────────────────────────────────────
#  Fixtures
# ────────────────────────────────────────────────

def _make_record(
    id="t1",
    question="Which country has the highest population?",
    answers=["China"],
    header=None,
    rows=None,
    title="World Population",
    context_text=None,
    dataset="test",
):
    return TableQARecord(
        id=id,
        question=question,
        answers=answers,
        table_header=header or ["Country", "Population"],
        table_rows=rows or [["China", "1425"], ["India", "1441"]],
        table_title=title,
        context_text=context_text,
        dataset=dataset,
        domain="general",
    )


BASE_CONFIG = {
    "embedding": {
        "model": "BAAI/bge-base-en-v1.5",
        "dimension": 768,
        "batch_size": 32,
        "normalize": True,
    },
    "vector_store": {
        "type": "faiss",
        "index_type": "IndexFlatIP",
    },
    "retrieval": {"top_k": 3},
    "generation": {
        "max_tables": 3,
        "include_context_text": True,
        "max_rows_per_table": 50,
        "timeout_seconds": 60,
        "fallback_to_mock": False,
    },
    "linearisation": {
        "include_table_title": True,
    },
    "table_parsing": {
        "max_rows": 100,
    },
    "llm": {
        "model": "mock",
        "base_url": "http://localhost:11434",
        "temperature": 0.0,
        "max_tokens": 64,
    },
    "experiment": {"seed": 42},
}


# ────────────────────────────────────────────────
#  linearise_table
# ────────────────────────────────────────────────

class TestLineariseTable:
    def test_includes_title(self):
        rec = _make_record(title="My Table")
        text = linearise_table(rec, BASE_CONFIG)
        assert "Table: My Table" in text

    def test_includes_header(self):
        rec = _make_record(header=["Name", "Value"])
        text = linearise_table(rec, BASE_CONFIG)
        assert "Name" in text
        assert "Value" in text

    def test_includes_rows(self):
        rec = _make_record(rows=[["Alice", "100"], ["Bob", "200"]])
        text = linearise_table(rec, BASE_CONFIG)
        assert "Alice" in text
        assert "Bob" in text

    def test_includes_context(self):
        rec = _make_record(context_text="Some important paragraph.")
        text = linearise_table(rec, BASE_CONFIG)
        assert "Some important paragraph" in text

    def test_no_context_when_disabled(self):
        cfg = {**BASE_CONFIG, "generation": {**BASE_CONFIG["generation"], "include_context_text": False}}
        rec = _make_record(context_text="Should not appear.")
        text = linearise_table(rec, cfg)
        assert "Should not appear" not in text

    def test_no_title_when_disabled(self):
        cfg = {**BASE_CONFIG, "linearisation": {"include_table_title": False}}
        rec = _make_record(title="Hidden Title")
        text = linearise_table(rec, cfg)
        assert "Hidden Title" not in text

    def test_no_title_field(self):
        rec = _make_record(title=None)
        text = linearise_table(rec, BASE_CONFIG)
        assert "Table:" not in text

    def test_max_rows_truncation(self):
        cfg = {**BASE_CONFIG, "table_parsing": {"max_rows": 2}}
        big_rows = [[str(i), str(i * 10)] for i in range(10)]
        rec = _make_record(rows=big_rows)
        text = linearise_table(rec, cfg)
        assert "truncated" in text

    def test_markdown_pipe_separators(self):
        rec = _make_record(header=["A", "B"], rows=[["1", "2"]])
        text = linearise_table(rec, BASE_CONFIG)
        assert "|" in text

    def test_returns_string(self):
        rec = _make_record()
        assert isinstance(linearise_table(rec, BASE_CONFIG), str)


# ────────────────────────────────────────────────
#  Pipeline — build and index
# ────────────────────────────────────────────────

class TestTextBaselinePipeline:
    def test_from_config_builds(self):
        pipeline = TextBaselinePipeline.from_config(BASE_CONFIG, fallback_to_mock=True)
        assert isinstance(pipeline, TextBaselinePipeline)

    def test_index_single_record(self):
        pipeline = TextBaselinePipeline.from_config(BASE_CONFIG, fallback_to_mock=True)
        records = [_make_record()]
        pipeline.index(records)
        assert len(pipeline.vector_store) == 1

    def test_index_multiple_records(self):
        pipeline = TextBaselinePipeline.from_config(BASE_CONFIG, fallback_to_mock=True)
        records = [
            _make_record(id="t1", question="Q1?"),
            _make_record(id="t2", question="Q2?"),
            _make_record(id="t3", question="Q3?"),
        ]
        pipeline.index(records)
        assert len(pipeline.vector_store) == 3

    def test_index_deduplication(self):
        pipeline = TextBaselinePipeline.from_config(BASE_CONFIG, fallback_to_mock=True)
        rec = _make_record(id="same")
        pipeline.index([rec, rec, rec])  # same ID 3 times
        assert len(pipeline.vector_store) == 1

    def test_record_lookup_populated(self):
        pipeline = TextBaselinePipeline.from_config(BASE_CONFIG, fallback_to_mock=True)
        records = [_make_record(id="abc")]
        pipeline.index(records)
        assert "abc" in pipeline._record_lookup

    def test_retrieve_returns_results(self):
        pipeline = TextBaselinePipeline.from_config(BASE_CONFIG, fallback_to_mock=True)
        records = [
            _make_record(id="t1", title="World Population", question="Pop?"),
            _make_record(id="t2", title="Tech Revenue", question="Rev?"),
        ]
        pipeline.index(records)
        results = pipeline.retriever.retrieve("What is the population?", top_k=1)
        assert len(results) >= 1
        assert results[0].record_id in {"t1", "t2"}

    def test_run_returns_evaluation_result(self):
        from src.evaluation.metrics import EvaluationResult
        pipeline = TextBaselinePipeline.from_config(BASE_CONFIG, fallback_to_mock=True)
        records = [_make_record(id="t1"), _make_record(id="t2")]
        pipeline.index(records)
        result = pipeline.run(records)
        assert isinstance(result, EvaluationResult)
        assert result.num_samples == 2

    def test_run_max_samples(self):
        pipeline = TextBaselinePipeline.from_config(BASE_CONFIG, fallback_to_mock=True)
        records = [_make_record(id=f"t{i}") for i in range(5)]
        pipeline.index(records)
        result = pipeline.run(records, max_samples=2)
        assert result.num_samples == 2

    def test_run_raises_if_not_indexed(self):
        pipeline = TextBaselinePipeline.from_config(BASE_CONFIG, fallback_to_mock=True)
        with pytest.raises(RuntimeError, match="empty"):
            pipeline.run([_make_record()])

    def test_repr(self):
        pipeline = TextBaselinePipeline.from_config(BASE_CONFIG, fallback_to_mock=True)
        assert "TextBaselinePipeline" in repr(pipeline)

    def test_len(self):
        pipeline = TextBaselinePipeline.from_config(BASE_CONFIG, fallback_to_mock=True)
        pipeline.index([_make_record(id="x")])
        assert len(pipeline) == 1


# ────────────────────────────────────────────────
#  Save / Load
# ────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_and_load(self):
        pipeline = TextBaselinePipeline.from_config(BASE_CONFIG, fallback_to_mock=True)
        records = [_make_record(id="save1"), _make_record(id="save2")]
        pipeline.index(records)

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline.save(tmpdir)

            # Verify files exist
            vs_dir = Path(tmpdir) / "vector_store"
            assert vs_dir.exists()

            # Load into a fresh pipeline
            fresh = TextBaselinePipeline.from_config(BASE_CONFIG, fallback_to_mock=True)
            fresh.load_index(tmpdir)
            assert len(fresh.vector_store) == 2

    def test_save_creates_directory(self):
        pipeline = TextBaselinePipeline.from_config(BASE_CONFIG, fallback_to_mock=True)
        pipeline.index([_make_record()])

        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = str(Path(tmpdir) / "nested" / "save")
            pipeline.save(new_dir)
            assert Path(new_dir).exists()
