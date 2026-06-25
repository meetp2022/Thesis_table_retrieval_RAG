"""
Unified dataset loader for WikiTableQuestions, TAT-QA, and FinQA.

Standardises all datasets into a common format:
{
    "id": str,
    "question": str,
    "answers": List[str],
    "table": {
        "header": List[str],
        "rows": List[List[str]],
        "title": Optional[str]
    },
    "context_text": Optional[str],   # For hybrid datasets (TAT-QA, FinQA)
    "dataset": str,                   # Source dataset name
    "domain": str                     # "general" or "finance"
}
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import load_dataset
from loguru import logger


# ────────────────────────────────────────────────
#  Standardised data record
# ────────────────────────────────────────────────

class TableQARecord:
    """A single table question-answering record in standardised format."""

    def __init__(
        self,
        id: str,
        question: str,
        answers: List[str],
        table_header: List[str],
        table_rows: List[List[str]],
        table_title: Optional[str] = None,
        context_text: Optional[str] = None,
        dataset: str = "",
        domain: str = "general",
    ):
        self.id = id
        self.question = question
        self.answers = answers
        self.table_header = table_header
        self.table_rows = table_rows
        self.table_title = table_title
        self.context_text = context_text
        self.dataset = dataset
        self.domain = domain

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "answers": self.answers,
            "table": {
                "header": self.table_header,
                "rows": self.table_rows,
                "title": self.table_title,
            },
            "context_text": self.context_text,
            "dataset": self.dataset,
            "domain": self.domain,
        }

    @property
    def num_rows(self) -> int:
        return len(self.table_rows)

    @property
    def num_cols(self) -> int:
        return len(self.table_header)

    def table_to_markdown(self) -> str:
        """Convert table to Markdown string (used by Pipeline 1)."""
        lines = []
        if self.table_title:
            lines.append(f"**{self.table_title}**\n")
        lines.append("| " + " | ".join(self.table_header) + " |")
        lines.append("| " + " | ".join(["---"] * len(self.table_header)) + " |")
        for row in self.table_rows:
            # Pad or truncate row to match header length
            padded = row + [""] * (len(self.table_header) - len(row))
            lines.append("| " + " | ".join(padded[:len(self.table_header)]) + " |")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"TableQARecord(id={self.id!r}, dataset={self.dataset!r}, "
            f"q={self.question[:50]!r}..., table={self.num_rows}×{self.num_cols})"
        )


# ────────────────────────────────────────────────
#  WikiTableQuestions Loader
# ────────────────────────────────────────────────

def load_wikitablequestions(
    split: str = "test",
    max_samples: Optional[int] = None,
) -> List[TableQARecord]:
    """
    Load WikiTableQuestions from HuggingFace.

    Args:
        split: 'train', 'test', or 'validation'
        max_samples: Limit number of samples (for quick testing)

    Returns:
        List of standardised TableQARecord objects
    """
    logger.info(f"Loading WikiTableQuestions ({split} split)...")

    ds = load_dataset(
        "stanfordnlp/wikitablequestions",
        name="random-split-1",
        split=split,
        trust_remote_code=True,
    )

    records = []
    for i, item in enumerate(ds):
        if max_samples and i >= max_samples:
            break

        record = TableQARecord(
            id=item["id"],
            question=item["question"],
            answers=item["answers"],
            table_header=item["table"]["header"],
            table_rows=item["table"]["rows"],
            table_title=item["table"].get("name", None),
            context_text=None,
            dataset="wikitablequestions",
            domain="general",
        )
        records.append(record)

    logger.info(f"Loaded {len(records)} WikiTableQuestions records ({split})")
    return records


# ────────────────────────────────────────────────
#  TAT-QA Loader
# ────────────────────────────────────────────────

def load_tatqa(
    data_path: str = "data/raw/tatqa",
    split: str = "train",
    max_samples: Optional[int] = None,
) -> List[TableQARecord]:
    """
    Load TAT-QA dataset.

    TAT-QA has a unique structure with hybrid table + paragraph contexts.
    Each context contains one table and multiple associated paragraphs.

    Args:
        data_path: Path to raw TAT-QA data directory
        split: 'train' or 'dev'
        max_samples: Limit number of samples

    Returns:
        List of standardised TableQARecord objects
    """
    logger.info(f"Loading TAT-QA ({split} split)...")

    # Try HuggingFace first
    try:
        ds = load_dataset("next-tat/TAT-QA", split=split, trust_remote_code=True)
        return _parse_tatqa_hf(ds, max_samples)
    except Exception as e:
        logger.warning(f"HuggingFace load failed: {e}. Trying local files...")

    # Fall back to local JSON files
    split_file = "train" if split == "train" else "dev"
    json_path = Path(data_path) / f"tatqa_dataset_{split_file}.json"

    if not json_path.exists():
        raise FileNotFoundError(
            f"TAT-QA data not found at {json_path}. "
            f"Download from: https://github.com/NExTplusplus/TAT-QA"
        )

    # Explicit UTF-8 encoding (the default on Windows is cp1252, which fails on
    # TAT-QA's non-ASCII bytes such as 0x9d).
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return _parse_tatqa_json(data, max_samples)


def _parse_tatqa_hf(ds, max_samples: Optional[int]) -> List[TableQARecord]:
    """Parse TAT-QA from HuggingFace dataset format."""
    records = []
    count = 0
    for item in ds:
        table_data = item.get("table", {})
        table_content = table_data.get("table", [])
        paragraphs = item.get("paragraphs", [])

        # Extract header and rows
        if table_content and len(table_content) > 0:
            header = [str(cell) for cell in table_content[0]]
            rows = [[str(cell) for cell in row] for row in table_content[1:]]
        else:
            header, rows = [], []

        # Combine paragraph texts
        context = " ".join([p.get("text", "") for p in paragraphs]) if paragraphs else None

        # Process questions
        for q in item.get("questions", []):
            if max_samples and count >= max_samples:
                return records

            answer = q.get("answer", "")
            if isinstance(answer, list):
                answers = [str(a) for a in answer]
            else:
                answers = [str(answer)]

            record = TableQARecord(
                id=q.get("uid", f"tatqa-{count}"),
                question=q.get("question", ""),
                answers=answers,
                table_header=header,
                table_rows=rows,
                table_title=None,
                context_text=context,
                dataset="tatqa",
                domain="finance",
            )
            records.append(record)
            count += 1

    logger.info(f"Loaded {len(records)} TAT-QA records")
    return records


def _parse_tatqa_json(data: List[Dict], max_samples: Optional[int]) -> List[TableQARecord]:
    """Parse TAT-QA from raw JSON format."""
    records = []
    count = 0

    for context in data:
        table_data = context.get("table", {})
        table_content = table_data.get("table", [])
        paragraphs = context.get("paragraphs", [])

        if table_content and len(table_content) > 0:
            header = [str(cell) for cell in table_content[0]]
            rows = [[str(cell) for cell in row] for row in table_content[1:]]
        else:
            header, rows = [], []

        context_text = " ".join([p.get("text", "") for p in paragraphs]) if paragraphs else None

        for q in context.get("questions", []):
            if max_samples and count >= max_samples:
                return records

            answer = q.get("answer", "")
            if isinstance(answer, list):
                answers = [str(a) for a in answer]
            else:
                answers = [str(answer)]

            record = TableQARecord(
                id=q.get("uid", f"tatqa-{count}"),
                question=q.get("question", ""),
                answers=answers,
                table_header=header,
                table_rows=rows,
                table_title=None,
                context_text=context_text,
                dataset="tatqa",
                domain="finance",
            )
            records.append(record)
            count += 1

    logger.info(f"Loaded {len(records)} TAT-QA records")
    return records


# ────────────────────────────────────────────────
#  FinQA Loader
# ────────────────────────────────────────────────

def load_finqa(
    data_path: str = "data/raw/finqa",
    split: str = "test",
    max_samples: Optional[int] = None,
) -> List[TableQARecord]:
    """
    Load FinQA dataset.

    FinQA contains financial report tables with multi-step
    numerical reasoning questions.

    Args:
        data_path: Path to raw FinQA data directory
        split: 'train', 'dev', or 'test'
        max_samples: Limit number of samples

    Returns:
        List of standardised TableQARecord objects
    """
    logger.info(f"Loading FinQA ({split} split)...")

    # Try HuggingFace
    try:
        ds = load_dataset("dreamerdeo/finqa", split=split, trust_remote_code=True)
        return _parse_finqa_hf(ds, max_samples)
    except Exception as e:
        logger.warning(f"HuggingFace load failed: {e}. Trying local files...")

    # Fall back to local JSON
    json_path = Path(data_path) / f"{split}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"FinQA data not found at {json_path}. "
            f"Download from: https://github.com/czyssrs/FinQA"
        )

    # Explicit UTF-8 encoding (same Windows-codec gotcha as the TAT-QA loader).
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return _parse_finqa_json(data, max_samples)


def _parse_finqa_hf(ds, max_samples: Optional[int]) -> List[TableQARecord]:
    """Parse FinQA from HuggingFace dataset format."""
    records = []
    for i, item in enumerate(ds):
        if max_samples and i >= max_samples:
            break

        table = item.get("table", [])
        if table and len(table) > 0:
            header = [str(cell) for cell in table[0]]
            rows = [[str(cell) for cell in row] for row in table[1:]]
        else:
            header, rows = [], []

        # FinQA has pre/post text contexts
        pre_text = " ".join(item.get("pre_text", []))
        post_text = " ".join(item.get("post_text", []))
        context = f"{pre_text} {post_text}".strip() or None

        record = TableQARecord(
            id=item.get("id", f"finqa-{i}"),
            question=item.get("question", ""),
            answers=[str(item.get("answer", ""))],
            table_header=header,
            table_rows=rows,
            table_title=None,
            context_text=context,
            dataset="finqa",
            domain="finance",
        )
        records.append(record)

    logger.info(f"Loaded {len(records)} FinQA records")
    return records


def _parse_finqa_json(data: List[Dict], max_samples: Optional[int]) -> List[TableQARecord]:
    """Parse FinQA from raw JSON format."""
    records = []
    for i, item in enumerate(data):
        if max_samples and i >= max_samples:
            break

        table = item.get("table", [])
        if table and len(table) > 0:
            header = [str(cell) for cell in table[0]]
            rows = [[str(cell) for cell in row] for row in table[1:]]
        else:
            header, rows = [], []

        pre_text = " ".join(item.get("pre_text", []))
        post_text = " ".join(item.get("post_text", []))
        context = f"{pre_text} {post_text}".strip() or None

        record = TableQARecord(
            id=item.get("id", f"finqa-{i}"),
            question=item.get("qa", {}).get("question", ""),
            answers=[str(item.get("qa", {}).get("answer", ""))],
            table_header=header,
            table_rows=rows,
            table_title=None,
            context_text=context,
            dataset="finqa",
            domain="finance",
        )
        records.append(record)

    logger.info(f"Loaded {len(records)} FinQA records")
    return records


# ────────────────────────────────────────────────
#  Unified Loader
# ────────────────────────────────────────────────

def load_dataset_by_name(
    name: str,
    split: str = "test",
    max_samples: Optional[int] = None,
    data_path: Optional[str] = None,
) -> List[TableQARecord]:
    """
    Load any of the three datasets by name.

    Args:
        name: 'wikitablequestions', 'tatqa', or 'finqa'
        split: Dataset split to load
        max_samples: Limit for quick testing
        data_path: Override default data path

    Returns:
        List of standardised TableQARecord objects
    """
    loaders = {
        "wikitablequestions": lambda: load_wikitablequestions(split, max_samples),
        "wikitq": lambda: load_wikitablequestions(split, max_samples),
        "tatqa": lambda: load_tatqa(data_path or "data/raw/tatqa", split, max_samples),
        "finqa": lambda: load_finqa(data_path or "data/raw/finqa", split, max_samples),
    }

    name_lower = name.lower().replace("-", "").replace("_", "")
    # Fuzzy match
    for key in loaders:
        if key.replace("_", "") == name_lower:
            return loaders[key]()

    raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")


def get_dataset_stats(records: List[TableQARecord]) -> Dict[str, Any]:
    """Compute summary statistics for a loaded dataset."""
    if not records:
        return {"count": 0}

    num_rows = [r.num_rows for r in records]
    num_cols = [r.num_cols for r in records]
    q_lengths = [len(r.question.split()) for r in records]
    has_context = sum(1 for r in records if r.context_text)

    return {
        "count": len(records),
        "dataset": records[0].dataset,
        "domain": records[0].domain,
        "avg_table_rows": sum(num_rows) / len(num_rows),
        "avg_table_cols": sum(num_cols) / len(num_cols),
        "max_table_rows": max(num_rows),
        "max_table_cols": max(num_cols),
        "avg_question_words": sum(q_lengths) / len(q_lengths),
        "records_with_context": has_context,
        "pct_with_context": round(has_context / len(records) * 100, 1),
    }
