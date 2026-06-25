#!/usr/bin/env python3
"""
Phase 0 — build and validate the MMQA retrieval corpus.

Loads MMQA via src/data/mmqa_loader.py, saves processed artifacts to
data/processed/mmqa/, and prints diagnostics + two fully-resolved sample
instances (gold tables + extracted inter-table join edges) so we can confirm
the edges are correct before building the graph.

Usage:
    python scripts/build_mmqa_corpus.py
"""

from __future__ import annotations

import io
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.data.mmqa_loader import load_mmqa


def _short(tid: str) -> str:
    """Readable table id (name + first 2 cols)."""
    name, _, cols = tid.partition("::")
    first = "|".join(cols.split("|")[:2])
    return f"{name}({first}...)"


def main():
    data = load_mmqa()
    corpus = data["corpus"]
    diag = data["diagnostics"]

    out_dir = Path("data/processed/mmqa")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Save corpus ──────────────────────────────
    corpus_json = {
        tid: {"name": t.name, "columns": t.columns, "rows": t.rows}
        for tid, t in corpus.items()
    }
    (out_dir / "corpus.json").write_text(
        json.dumps(corpus_json, ensure_ascii=False), encoding="utf-8"
    )

    # ── Save instances per split ─────────────────
    for split in ("train", "dev", "test"):
        recs = [asdict(inst) for inst in data[split]]
        (out_dir / f"instances_{split}.json").write_text(
            json.dumps(recs, ensure_ascii=False), encoding="utf-8"
        )

    # ── Diagnostics ──────────────────────────────
    print("\n" + "=" * 70)
    print("  MMQA CORPUS BUILD — DIAGNOSTICS")
    print("=" * 70)
    for k, v in diag.items():
        print(f"  {k:32}: {v}")

    # Coverage: fraction of instances with at least one inter-table edge
    n = diag["n_instances"]
    print("\n  Inter-table edge coverage:")
    print(f"    sql-join (alias-aware JOIN/WHERE): "
          f"{diag['instances_with_sql_joins']}/{n} "
          f"({100*diag['instances_with_sql_joins']/n:.1f}%)")
    print(f"    schema-join (FK/PK shared col)   : "
          f"{diag['instances_with_schema_joins']}/{n} "
          f"({100*diag['instances_with_schema_joins']/n:.1f}%)")
    print(f"    value-overlap (fallback only)    : "
          f"{diag['instances_with_value_joins']}/{n} "
          f"({100*diag['instances_with_value_joins']/n:.1f}%)")
    print(f"    GOLD edges (sql or schema)       : "
          f"{diag['instances_with_gold_edges']}/{n} "
          f"({100*diag['instances_with_gold_edges']/n:.1f}%)")
    print(f"    ANY edges (gold or value)        : "
          f"{diag['instances_with_any_edges']}/{n} "
          f"({100*diag['instances_with_any_edges']/n:.1f}%)   <-- usable for inter-table graph")

    # ── Two resolved samples ─────────────────────
    print("\n" + "=" * 70)
    print("  SAMPLE RESOLVED INSTANCES (verify edges by eye)")
    print("=" * 70)
    samples = [data["test"][0], data["test"][1]]
    for s in samples:
        print(f"\n  Q[{s.qid}]: {s.question}")
        print(f"  Answer: {s.answer}")
        print(f"  Gold tables ({len(s.gold_table_ids)}):")
        for tid in s.gold_table_ids:
            print(f"      - {_short(tid)}")
        print(f"  Declared FK cols: {s.fk_cols}")
        print(f"  Declared PK cols: {s.pk_cols}")
        print(f"  Schema join edges:")
        for a, ca, b, cb in s.join_pairs_schema:
            print(f"      {_short(a)}.{ca}  <->  {_short(b)}.{cb}")
        print(f"  SQL join edges:")
        for a, ca, b, cb in s.join_pairs_sql:
            print(f"      {_short(a)}.{ca}  <->  {_short(b)}.{cb}")

    print("\n" + "=" * 70)
    print(f"  Saved corpus + splits to {out_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
