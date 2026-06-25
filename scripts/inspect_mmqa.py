#!/usr/bin/env python3
"""
Inspect the MMQA multi-table dataset (table-benchmark/mmqa on HuggingFace).

Confirms the schema, the FK/PK annotation format, the per-instance table
counts, and — critically — the size of the retrieval corpus we can build by
pooling unique tables across all instances. Saves three full sample rows to
data/raw/mmqa/_samples.json so the loader can be written against real data.

Usage:
    python scripts/inspect_mmqa.py
"""

from __future__ import annotations

import io
import json
import sys
from collections import Counter
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from datasets import load_dataset


def _parse_table(raw) -> dict:
    """The `table` field is a JSON string (or already a dict)."""
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _table_signature(name: str, columns: list) -> str:
    """Stable identity for a table = name + its column list."""
    cols = "|".join(str(c) for c in (columns or []))
    return f"{name}::{cols}"


def main():
    print("Loading table-benchmark/mmqa [test] (downloads ~1.3 GB first time)...")
    ds = load_dataset("table-benchmark/mmqa", split="test")
    n = len(ds)
    print(f"\nRows: {n}")
    print(f"Features: {list(ds.features.keys())}")

    # ── Parse every instance, pool tables, gather stats ──────────
    tables_per_instance = Counter()
    corpus = {}                 # signature -> {name, columns, n_rows}
    fk_present = pk_present = sql_present = 0
    gold_set_sizes = Counter()

    for ex in ds:
        tbl = _parse_table(ex.get("table"))
        names = tbl.get("table_names") or []
        tables = tbl.get("tables") or []
        tables_per_instance[len(tables)] += 1

        if tbl.get("foreign_keys"):
            fk_present += 1
        if tbl.get("primary_keys"):
            pk_present += 1
        if tbl.get("SQL"):
            sql_present += 1

        gold_sigs = set()
        for i, t in enumerate(tables):
            name = names[i] if i < len(names) else f"tbl{i}"
            cols = t.get("table_columns") or []
            sig = _table_signature(name, cols)
            gold_sigs.add(sig)
            if sig not in corpus:
                corpus[sig] = {
                    "name": name,
                    "columns": cols,
                    "n_rows": len(t.get("table_content") or []),
                }
        gold_set_sizes[len(gold_sigs)] += 1

    print("\n── Per-instance table counts (from `tables` array) ──")
    for k in sorted(tables_per_instance):
        print(f"  {k} tables: {tables_per_instance[k]} instances")

    print("\n── Gold-set sizes (unique tables per question) ──")
    for k in sorted(gold_set_sizes):
        print(f"  {k} gold tables: {gold_set_sizes[k]} questions")

    print("\n── Annotation coverage ──")
    print(f"  foreign_keys present: {fk_present}/{n}")
    print(f"  primary_keys present: {pk_present}/{n}")
    print(f"  SQL present:          {sql_present}/{n}")

    print("\n── Retrieval corpus (pooled unique tables) ──")
    print(f"  Unique tables (by name+columns): {len(corpus)}")
    avg_rows = sum(c["n_rows"] for c in corpus.values()) / max(len(corpus), 1)
    print(f"  Avg rows per table: {avg_rows:.1f}")
    avg_cols = sum(len(c["columns"]) for c in corpus.values()) / max(len(corpus), 1)
    print(f"  Avg columns per table: {avg_cols:.1f}")

    # ── Dump 3 full samples + raw FK/PK format ───────────────────
    out_dir = Path("data/raw/mmqa")
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for ex in list(ds)[:3]:
        tbl = _parse_table(ex.get("table"))
        samples.append({
            "original_dataset_id": ex.get("original_dataset_id"),
            "question": ex.get("question"),
            "answer": ex.get("answer"),
            "table_names": tbl.get("table_names"),
            "foreign_keys": tbl.get("foreign_keys"),
            "primary_keys": tbl.get("primary_keys"),
            "SQL": tbl.get("SQL"),
            "tables_preview": [
                {"columns": t.get("table_columns"),
                 "first_row": (t.get("table_content") or [None])[0]}
                for t in (tbl.get("tables") or [])
            ],
        })
    (out_dir / "_samples.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved 3 full samples to {out_dir / '_samples.json'}")

    # Show the raw FK/PK format of the first instance (the format matters
    # for building inter-table edges)
    print("\n── Raw FK/PK format (first instance) ──")
    print("  table_names :", samples[0]["table_names"])
    print("  primary_keys:", samples[0]["primary_keys"])
    print("  foreign_keys:", samples[0]["foreign_keys"])
    print("  SQL         :", str(samples[0]["SQL"])[:200])


if __name__ == "__main__":
    main()
