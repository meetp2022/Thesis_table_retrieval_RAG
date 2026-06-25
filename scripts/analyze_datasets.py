"""
Dataset characterization for the cross-table ablation paper.

Computes, from the actual data used in the experiments:
  - single-table (WikiTQ, FinQA, TAT-QA): table-size distribution (rows/cols),
    question length, table-vs-context text balance, numeric content, skewness.
  - multi-table (MMQA): corpus table-size distribution, gold-set-size
    distribution, columns per table, inter-table edge coverage, cell text
    length, plus the same skewness diagnostics.

Pure stdlib + the project loaders. Run from repo root:
    python scripts/analyze_datasets.py
"""

from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path
from typing import List, Sequence

from src.data.dataset_loader import load_dataset_by_name


# ── small stats helpers (no scipy dependency) ────────────────

def skewness(xs: Sequence[float]) -> float:
    """Fisher-Pearson sample skewness. >0 right-tailed, <0 left-tailed."""
    n = len(xs)
    if n < 3:
        return float("nan")
    m = sum(xs) / n
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / n)
    if s == 0:
        return 0.0
    g1 = sum((x - m) ** 3 for x in xs) / (n * s ** 3)
    return g1 * math.sqrt(n * (n - 1)) / (n - 2)  # bias-corrected


def describe(xs: Sequence[float]) -> dict:
    xs = list(xs)
    xs_sorted = sorted(xs)
    n = len(xs)

    def pct(p):
        if n == 0:
            return float("nan")
        k = (n - 1) * p
        lo = int(math.floor(k))
        hi = int(math.ceil(k))
        if lo == hi:
            return xs_sorted[lo]
        return xs_sorted[lo] * (hi - k) + xs_sorted[hi] * (k - lo)

    return {
        "n": n,
        "mean": round(st.mean(xs), 2) if xs else float("nan"),
        "median": round(st.median(xs), 2) if xs else float("nan"),
        "std": round(st.pstdev(xs), 2) if n > 1 else 0.0,
        "min": min(xs) if xs else float("nan"),
        "p25": round(pct(0.25), 2),
        "p75": round(pct(0.75), 2),
        "p95": round(pct(0.95), 2),
        "max": max(xs) if xs else float("nan"),
        "skew": round(skewness(xs), 3),
    }


def row(label: str, d: dict) -> str:
    return (f"  {label:<22} mean={d['mean']:>8}  median={d['median']:>7}  "
            f"std={d['std']:>8}  min={d['min']:>5}  p95={d['p95']:>7}  "
            f"max={d['max']:>6}  skew={d['skew']:>7}")


def is_number(s) -> bool:
    try:
        float(str(s).replace(",", "").replace("$", "").replace("%", "").strip())
        return True
    except (ValueError, AttributeError):
        return False


# ── single-table analysis ────────────────────────────────────

SINGLE = [
    ("wikitablequestions", "validation", 1000, "WikiTQ"),
    ("finqa",              "test",        883, "FinQA"),
    ("tatqa",              "dev",        1000, "TAT-QA"),
]


def analyze_single():
    print("=" * 78)
    print("SINGLE-TABLE DATASETS")
    print("=" * 78)
    summary = {}
    for name, split, cap, label in SINGLE:
        try:
            recs = load_dataset_by_name(name, split=split, max_samples=cap)
        except Exception as e:
            print(f"\n[{label}] FAILED to load ({split}): {e}")
            continue

        rows = [r.num_rows for r in recs]
        cols = [r.num_cols for r in recs]
        cells = [r.num_rows * r.num_cols for r in recs]
        qlen = [len(r.question.split()) for r in recs]
        ctx = [len((r.context_text or "").split()) for r in recs]
        has_ctx = sum(1 for r in recs if r.context_text)

        # text balance: question + context words vs table cell-token count
        tbl_tokens = []
        numeric_frac = []
        for r in recs:
            toks = 0
            num = tot = 0
            for rr in r.table_rows:
                for c in rr:
                    toks += len(str(c).split())
                    tot += 1
                    if is_number(c):
                        num += 1
            tbl_tokens.append(toks)
            numeric_frac.append(num / tot if tot else 0.0)

        print(f"\n[{label}]  n={len(recs)}  split={split}  domain={recs[0].domain}")
        print(row("table rows", describe(rows)))
        print(row("table cols", describe(cols)))
        print(row("table cells (r*c)", describe(cells)))
        print(row("question words", describe(qlen)))
        print(row("table cell-tokens", describe(tbl_tokens)))
        if has_ctx:
            print(row("context words", describe([c for c in ctx if c])))
        print(f"  records w/ context text: {has_ctx}/{len(recs)} "
              f"({100*has_ctx/len(recs):.1f}%)")
        print(f"  mean numeric-cell fraction: {st.mean(numeric_frac):.3f}")
        summary[label] = {
            "n": len(recs), "rows": describe(rows), "cols": describe(cols),
            "qwords": describe(qlen), "pct_context": round(100*has_ctx/len(recs), 1),
            "numeric_frac": round(st.mean(numeric_frac), 3),
        }
    return summary


# ── MMQA analysis ────────────────────────────────────────────

def analyze_mmqa():
    print("\n" + "=" * 78)
    print("MULTI-TABLE DATASET (MMQA)")
    print("=" * 78)
    base = Path("data/processed/mmqa")
    corpus = json.loads((base / "corpus.json").read_text(encoding="utf-8"))

    # corpus-level table size distribution
    c_rows = [len(t["rows"]) for t in corpus.values()]
    c_cols = [len(t["columns"]) for t in corpus.values()]
    cell_tokens = []
    numeric_frac = []
    for t in corpus.values():
        toks = num = tot = 0
        for rr in t["rows"]:
            for c in rr:
                toks += len(str(c).split())
                tot += 1
                if is_number(c):
                    num += 1
        cell_tokens.append(toks)
        if tot:
            numeric_frac.append(num / tot)

    print(f"\n[Corpus]  {len(corpus)} pooled tables")
    print(row("table rows", describe(c_rows)))
    print(row("table cols", describe(c_cols)))
    print(row("cell-tokens/table", describe(cell_tokens)))
    print(f"  mean numeric-cell fraction: {st.mean(numeric_frac):.3f}")

    # instance-level: gold-set size, edges, question length
    all_inst = []
    for sp in ("train", "dev", "test"):
        all_inst += json.loads((base / f"instances_{sp}.json").read_text(encoding="utf-8"))

    gold_sizes = [len(i["gold_table_ids"]) for i in all_inst]
    qlen = [len(i["question"].split()) for i in all_inst]
    n_sql = sum(1 for i in all_inst if i["join_pairs_sql"])
    n_schema = sum(1 for i in all_inst if i["join_pairs_schema"])
    n_value = sum(1 for i in all_inst if i["join_pairs_value"])
    n_gold_edge = sum(1 for i in all_inst if i["join_pairs_sql"] or i["join_pairs_schema"])
    n_any_edge = sum(1 for i in all_inst
                     if i["join_pairs_sql"] or i["join_pairs_schema"] or i["join_pairs_value"])
    N = len(all_inst)

    print(f"\n[Instances]  {N} questions (train+dev+test pooled)")
    print(row("gold tables / question", describe(gold_sizes)))
    print(row("question words", describe(qlen)))
    # gold-set-size histogram
    from collections import Counter
    hist = Counter(gold_sizes)
    print("  gold-set-size distribution:")
    for k in sorted(hist):
        print(f"      {k} tables: {hist[k]:>4}  ({100*hist[k]/N:5.1f}%)")

    print("\n[Inter-table edge coverage]")
    print(f"  SQL-join edges:        {n_sql:>4}/{N}  ({100*n_sql/N:5.1f}%)")
    print(f"  schema FK/PK edges:    {n_schema:>4}/{N}  ({100*n_schema/N:5.1f}%)")
    print(f"  GOLD (sql or schema):  {n_gold_edge:>4}/{N}  ({100*n_gold_edge/N:5.1f}%)")
    print(f"  value-overlap fallback:{n_value:>4}/{N}  ({100*n_value/N:5.1f}%)")
    print(f"  ANY usable edge:       {n_any_edge:>4}/{N}  ({100*n_any_edge/N:5.1f}%)")


if __name__ == "__main__":
    s = analyze_single()
    analyze_mmqa()
    print("\n" + "=" * 78)
    print("Done. (skew > 0 = right-tailed / long upper tail; |skew| > 1 = high)")
    print("=" * 78)
