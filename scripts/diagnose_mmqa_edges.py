#!/usr/bin/env python3
"""
Phase 0b — diagnose WHY inter-table edge extraction misses ~70% of instances.

Categorises the gold SQL by join style, checks whether relaxing the FK/PK
filter or using value-overlap would recover the missing edges, and prints
several no-edge instances in full so we can see the real failure mode.

Usage:
    python scripts/diagnose_mmqa_edges.py
"""

from __future__ import annotations

import io
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.data.mmqa_loader import load_mmqa, _norm


def _value_overlap_join_exists(corpus, gold_ids) -> bool:
    """Does any column pair across two gold tables share >=50% of values?"""
    tabs = [corpus[t] for t in gold_ids if t in corpus]
    for i in range(len(tabs)):
        for j in range(i + 1, len(tabs)):
            A, B = tabs[i], tabs[j]
            for ci, ca in enumerate(A.columns):
                vals_a = {_norm(r[ci]) for r in A.rows if ci < len(r) and r[ci] not in (None, "")}
                if len(vals_a) < 2:
                    continue
                for cj, cb in enumerate(B.columns):
                    vals_b = {_norm(r[cj]) for r in B.rows if cj < len(r) and r[cj] not in (None, "")}
                    if len(vals_b) < 2:
                        continue
                    inter = len(vals_a & vals_b)
                    if inter == 0:
                        continue
                    containment = inter / min(len(vals_a), len(vals_b))
                    if containment >= 0.5:
                        return True
    return False


def main():
    data = load_mmqa()
    corpus = data["corpus"]
    allinst = data["train"] + data["dev"] + data["test"]
    n = len(allinst)

    # ── SQL style histogram ──────────────────────
    sql_styles = Counter()
    for inst in allinst:
        s = (inst.sql or "").upper()
        if " JOIN " in s:
            sql_styles["has_JOIN"] += 1
        if re.search(r"\bAS\s+\w+", s) or re.search(r"\b\w+\s+\w+\s+ON\b", s):
            sql_styles["has_alias_maybe"] += 1
        if "IN (SELECT" in s or "IN(SELECT" in s:
            sql_styles["has_IN_subquery"] += 1
        if " JOIN " not in s and "IN (SELECT" not in s and "IN(SELECT" not in s:
            sql_styles["no_join_no_subquery"] += 1

    print("\n" + "=" * 70)
    print("  SQL JOIN-STYLE HISTOGRAM")
    print("=" * 70)
    for k, v in sql_styles.most_common():
        print(f"  {k:24}: {v}/{n} ({100*v/n:.1f}%)")

    # ── What would each relaxation recover? ──────
    no_edge = [i for i in allinst if not i.join_pairs_schema and not i.join_pairs_sql]
    print(f"\n  Instances with NO edge (current methods): {len(no_edge)}/{n} "
          f"({100*len(no_edge)/n:.1f}%)")

    # Relaxation 1: shared column name WITHOUT the FK/PK filter
    recov_sharedname = 0
    for inst in no_edge:
        tabs = [corpus[t] for t in inst.gold_table_ids if t in corpus]
        found = False
        for a in range(len(tabs)):
            for b in range(a + 1, len(tabs)):
                ca = {_norm(c) for c in tabs[a].columns}
                cb = {_norm(c) for c in tabs[b].columns}
                if ca & cb:
                    found = True
        recov_sharedname += int(found)
    print(f"\n  Relaxation A — ANY shared column name (drop FK/PK filter):")
    print(f"    would recover {recov_sharedname}/{len(no_edge)} no-edge instances")

    # Relaxation 2: value overlap (sample first 200 no-edge for speed)
    sample = no_edge[:200]
    recov_value = sum(_value_overlap_join_exists(corpus, i.gold_table_ids) for i in sample)
    print(f"\n  Relaxation B — value overlap >=50% (sampled {len(sample)} no-edge):")
    print(f"    would recover {recov_value}/{len(sample)} "
          f"({100*recov_value/max(len(sample),1):.1f}%)")

    # ── Print 4 no-edge instances in full ────────
    print("\n" + "=" * 70)
    print("  NO-EDGE INSTANCES (full detail — see the real failure mode)")
    print("=" * 70)
    for inst in no_edge[:4]:
        print(f"\n  Q[{inst.qid}]: {inst.question[:90]}")
        print(f"  FK cols: {inst.fk_cols}   PK cols: {inst.pk_cols}")
        for tid in inst.gold_table_ids:
            t = corpus.get(tid)
            if t:
                print(f"    TABLE {t.name}: cols={t.columns}")
        print(f"  SQL: {inst.sql[:240]}")


if __name__ == "__main__":
    main()
