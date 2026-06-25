"""
Unit tests for the LLM answer generation pipeline:
    - OllamaClient     (llm_client.py)
    - MockLLMClient     (llm_client.py)
    - AnswerGenerator   (answer_generator.py)

Uses MockLLMClient throughout — no Ollama server required.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.data.dataset_loader import TableQARecord
from src.pipelines.shared.llm_client import (
    OllamaClient,
    MockLLMClient,
    create_llm_client,
)
from src.pipelines.shared.answer_generator import (
    AnswerGenerator,
    GenerationResult,
    _format_table_context,
    DEFAULT_PROMPT_TEMPLATE,
)
from src.pipelines.shared.retriever import RetrievalResult


# ────────────────────────────────────────────────
#  Fixtures
# ────────────────────────────────────────────────

@pytest.fixture
def mock_client():
    return MockLLMClient(default_response="42", model="test-model")


@pytest.fixture
def sample_records():
    """Dict of record_id → TableQARecord for lookup."""
    rec1 = TableQARecord(
        id="rec-1",
        question="What is the population of France?",
        answers=["67 million"],
        table_header=["Country", "Population"],
        table_rows=[
            ["France", "67,000,000"],
            ["Germany", "83,000,000"],
            ["Spain", "47,000,000"],
        ],
        table_title="European Countries",
        dataset="wikitq",
        domain="general",
    )
    rec2 = TableQARecord(
        id="rec-2",
        question="What was the revenue?",
        answers=["$500M"],
        table_header=["Year", "Revenue", "Profit"],
        table_rows=[
            ["2022", "$400M", "$50M"],
            ["2023", "$500M", "$75M"],
        ],
        table_title=None,
        context_text="The company reported strong growth in fiscal year 2023.",
        dataset="finqa",
        domain="finance",
    )
    rec3 = TableQARecord(
        id="rec-3",
        question="Ratio?",
        answers=["2.5"],
        table_header=["Metric", "Value"],
        table_rows=[["Ratio", "2.5"]],
        dataset="tatqa",
        domain="finance",
    )
    return {"rec-1": rec1, "rec-2": rec2, "rec-3": rec3}


@pytest.fixture
def sample_retrieval_results():
    return [
        RetrievalResult(record_id="rec-1", score=0.95, rank=1, metadata={"dataset": "wikitq"}),
        RetrievalResult(record_id="rec-2", score=0.82, rank=2, metadata={"dataset": "finqa"}),
        RetrievalResult(record_id="rec-3", score=0.71, rank=3, metadata={"dataset": "tatqa"}),
    ]


@pytest.fixture
def generator(mock_client):
    return AnswerGenerator(
        llm_client=mock_client,
        max_tables=3,
        include_context_text=True,
        max_rows_per_table=50,
    )


# ════════════════════════════════════════════════
#  MockLLMClient Tests
# ════════════════════════════════════════════════

class TestMockLLMClient:
    def test_generate_returns_default(self, mock_client):
        assert mock_client.generate("anything") == "42"

    def test_is_available(self, mock_client):
        assert mock_client.is_available() is True

    def test_call_count(self, mock_client):
        mock_client.generate("a")
        mock_client.generate("b")
        assert mock_client.call_count == 2

    def test_last_prompt(self, mock_client):
        mock_client.generate("my prompt")
        assert mock_client.last_prompt == "my prompt"

    def test_generate_batch(self, mock_client):
        results = mock_client.generate_batch(["a", "b", "c"])
        assert results == ["42", "42", "42"]

    def test_repr(self, mock_client):
        assert "test-model" in repr(mock_client)


# ════════════════════════════════════════════════
#  OllamaClient Tests (mocked HTTP)
# ════════════════════════════════════════════════

class TestOllamaClient:
    def test_from_config(self):
        config = {
            "llm": {
                "model": "llama3",
                "base_url": "http://example.com:11434",
                "temperature": 0.5,
                "max_tokens": 256,
            },
            "generation": {"timeout_seconds": 30},
            "experiment": {"seed": 123},
        }
        client = OllamaClient.from_config(config)
        assert client.model == "llama3"
        assert client.base_url == "http://example.com:11434"
        assert client.temperature == 0.5
        assert client.max_tokens == 256
        assert client.timeout == 30
        assert client.seed == 123

    def test_from_config_defaults(self):
        client = OllamaClient.from_config({})
        assert client.model == "mistral"
        assert client.temperature == 0.0

    def test_repr(self):
        client = OllamaClient(model="mistral")
        assert "mistral" in repr(client)

    @patch("src.pipelines.shared.llm_client.requests.get")
    def test_is_available_false_on_connection_error(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        client = OllamaClient()
        # Force re-check by resetting cache
        client._available = None
        assert client.is_available() is False

    @patch("src.pipelines.shared.llm_client.requests.post")
    def test_generate_with_mocked_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "The answer is 42."}
        mock_post.return_value = mock_resp

        client = OllamaClient(model="mistral")
        result = client.generate("What is 6*7?")
        assert result == "The answer is 42."

    @patch("src.pipelines.shared.llm_client.requests.post")
    def test_generate_connection_error(self, mock_post):
        from requests import ConnectionError as ReqConnectionError
        mock_post.side_effect = ReqConnectionError("refused")

        client = OllamaClient()
        with pytest.raises(ConnectionError, match="Cannot connect"):
            client.generate("test")

    @patch("src.pipelines.shared.llm_client.requests.post")
    def test_generate_non_200_raises(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal server error"
        mock_post.return_value = mock_resp

        client = OllamaClient()
        with pytest.raises(RuntimeError, match="Ollama API error"):
            client.generate("test")

    @patch("src.pipelines.shared.llm_client.requests.post")
    def test_generate_batch_handles_errors(self, mock_post):
        """Batch generation should not crash on individual failures."""
        from requests import ConnectionError as ReqConnectionError

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ReqConnectionError("fail")
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"response": f"answer-{call_count}"}
            return resp

        mock_post.side_effect = side_effect
        client = OllamaClient()
        results = client.generate_batch(["q1", "q2", "q3"])
        assert len(results) == 3
        assert results[1] == ""  # failed one


# ════════════════════════════════════════════════
#  create_llm_client factory
# ════════════════════════════════════════════════

class TestCreateLLMClient:
    @patch("src.pipelines.shared.llm_client.requests.get")
    def test_fallback_to_mock(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        client = create_llm_client({}, fallback_to_mock=True)
        assert isinstance(client, MockLLMClient)

    @patch("src.pipelines.shared.llm_client.requests.get")
    def test_no_fallback_raises(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        with pytest.raises(ConnectionError):
            create_llm_client({}, fallback_to_mock=False)


# ════════════════════════════════════════════════
#  Context formatting
# ════════════════════════════════════════════════

class TestFormatTableContext:
    def test_single_table(self, sample_records):
        ctx = _format_table_context([sample_records["rec-1"]])
        assert "TABLE:" in ctx
        assert "Country" in ctx
        assert "France" in ctx
        # Single table should NOT have "TABLE 1:"
        assert "TABLE 1:" not in ctx

    def test_multiple_tables_numbered(self, sample_records):
        recs = [sample_records["rec-1"], sample_records["rec-2"]]
        ctx = _format_table_context(recs)
        assert "TABLE 1:" in ctx
        assert "TABLE 2:" in ctx

    def test_context_text_included(self, sample_records):
        ctx = _format_table_context(
            [sample_records["rec-2"]],
            include_context_text=True,
        )
        assert "ADDITIONAL CONTEXT:" in ctx
        assert "strong growth" in ctx

    def test_context_text_excluded(self, sample_records):
        ctx = _format_table_context(
            [sample_records["rec-2"]],
            include_context_text=False,
        )
        assert "ADDITIONAL CONTEXT:" not in ctx

    def test_truncation(self):
        """Large table should be truncated with a note."""
        rec = TableQARecord(
            id="big",
            question="?",
            answers=["x"],
            table_header=["A", "B"],
            table_rows=[[str(i), str(i * 2)] for i in range(100)],
            dataset="test",
        )
        ctx = _format_table_context([rec], max_rows_per_table=10)
        assert "90 more rows truncated" in ctx

    def test_no_truncation_note_for_small_tables(self, sample_records):
        ctx = _format_table_context([sample_records["rec-1"]], max_rows_per_table=50)
        assert "truncated" not in ctx


# ════════════════════════════════════════════════
#  AnswerGenerator Tests
# ════════════════════════════════════════════════

class TestAnswerGenerator:
    def test_generate_returns_result(
        self, generator, sample_retrieval_results, sample_records
    ):
        result = generator.generate(
            "What is the population of France?",
            sample_retrieval_results,
            sample_records,
        )
        assert isinstance(result, GenerationResult)

    def test_answer_from_mock(
        self, generator, sample_retrieval_results, sample_records
    ):
        result = generator.generate(
            "test?", sample_retrieval_results, sample_records
        )
        assert result.answer == "42"

    def test_result_has_question(
        self, generator, sample_retrieval_results, sample_records
    ):
        result = generator.generate(
            "What is X?", sample_retrieval_results, sample_records
        )
        assert result.question == "What is X?"

    def test_result_has_record_ids(
        self, generator, sample_retrieval_results, sample_records
    ):
        result = generator.generate(
            "test?", sample_retrieval_results, sample_records
        )
        assert result.retrieved_record_ids == ["rec-1", "rec-2", "rec-3"]

    def test_result_has_scores(
        self, generator, sample_retrieval_results, sample_records
    ):
        result = generator.generate(
            "test?", sample_retrieval_results, sample_records
        )
        assert result.retrieval_scores == [0.95, 0.82, 0.71]

    def test_result_has_model(
        self, generator, sample_retrieval_results, sample_records
    ):
        result = generator.generate(
            "test?", sample_retrieval_results, sample_records
        )
        assert result.model == "test-model"

    def test_result_has_latency(
        self, generator, sample_retrieval_results, sample_records
    ):
        result = generator.generate(
            "test?", sample_retrieval_results, sample_records
        )
        assert result.latency_seconds >= 0

    def test_prompt_contains_question(
        self, generator, sample_retrieval_results, sample_records
    ):
        result = generator.generate(
            "What country has 67M people?",
            sample_retrieval_results,
            sample_records,
        )
        assert "What country has 67M people?" in result.prompt

    def test_prompt_contains_table_data(
        self, generator, sample_retrieval_results, sample_records
    ):
        result = generator.generate(
            "test?", sample_retrieval_results, sample_records
        )
        assert "France" in result.prompt
        assert "Revenue" in result.prompt

    def test_max_tables_limits_context(self, mock_client, sample_retrieval_results, sample_records):
        gen = AnswerGenerator(mock_client, max_tables=1)
        result = gen.generate(
            "test?", sample_retrieval_results, sample_records
        )
        # Only rec-1 should be in context
        assert "France" in result.prompt
        assert "Revenue" not in result.prompt
        assert result.retrieved_record_ids == ["rec-1"]

    def test_missing_record_skipped(
        self, generator, sample_records
    ):
        """If a retrieval result points to a missing record, skip it."""
        results = [
            RetrievalResult(record_id="nonexistent", score=0.9, rank=1),
            RetrievalResult(record_id="rec-1", score=0.8, rank=2),
        ]
        gen_result = generator.generate("test?", results, sample_records)
        assert "rec-1" in gen_result.retrieved_record_ids
        assert "nonexistent" not in gen_result.retrieved_record_ids

    def test_empty_retrieval_results(self, generator, sample_records):
        result = generator.generate("test?", [], sample_records)
        assert "No tables retrieved" in result.prompt
        assert result.answer == "42"  # mock still responds

    def test_llm_receives_the_prompt(
        self, mock_client, sample_retrieval_results, sample_records
    ):
        gen = AnswerGenerator(mock_client)
        gen.generate("My question?", sample_retrieval_results, sample_records)
        assert "My question?" in mock_client.last_prompt


class TestAnswerGeneratorBatch:
    def test_batch_returns_list(
        self, generator, sample_retrieval_results, sample_records
    ):
        results = generator.generate_batch(
            ["q1", "q2"],
            [sample_retrieval_results, sample_retrieval_results],
            sample_records,
        )
        assert len(results) == 2
        assert all(isinstance(r, GenerationResult) for r in results)

    def test_batch_mismatched_lengths_raises(
        self, generator, sample_retrieval_results, sample_records
    ):
        with pytest.raises(ValueError, match="same length"):
            generator.generate_batch(
                ["q1", "q2"],
                [sample_retrieval_results],
                sample_records,
            )


class TestAnswerGeneratorFromConfig:
    def test_from_config(self, mock_client):
        config = {
            "generation": {
                "max_tables": 2,
                "include_context_text": False,
                "max_rows_per_table": 20,
            }
        }
        gen = AnswerGenerator.from_config(config, mock_client)
        assert gen.max_tables == 2
        assert gen.include_context_text is False
        assert gen.max_rows_per_table == 20

    def test_from_config_defaults(self, mock_client):
        gen = AnswerGenerator.from_config({}, mock_client)
        assert gen.max_tables == 3
        assert gen.include_context_text is True

    def test_repr(self, generator):
        r = repr(generator)
        assert "test-model" in r
        assert "max_tables=3" in r


# ════════════════════════════════════════════════
#  GenerationResult dataclass
# ════════════════════════════════════════════════

class TestGenerationResult:
    def test_fields(self):
        r = GenerationResult(
            question="What?",
            answer="42",
            retrieved_record_ids=["rec-1"],
            retrieval_scores=[0.95],
            prompt="full prompt",
            model="mistral",
            latency_seconds=1.5,
        )
        assert r.question == "What?"
        assert r.answer == "42"
        assert r.model == "mistral"
        assert r.latency_seconds == 1.5
