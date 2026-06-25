#!/usr/bin/env python3
"""
Per-query diagnostic comparing two retrieval pipelines on the same dataset.

Reads two retrieval_details.json files, aligns by record_id, and produces:

  1. A 2x2 contingency table on rank-1 success:
        A succeeds / B succeeds         (both correct)
        A succeeds / B fails            (B regressed)
        A fails    / B succeeds         (B improved -- graph helped here)
        A fails    / B fails            (both wrong)

     plus a McNemar-style symmetry test on whether the gains-vs-losses
     difference is statistically meaningful.

  2. Per-question average rank movement.

  3. A breakdown of the queries B newly gets right (and a count of how many
     B newly gets wrong) so we can read what kinds of questions benefited.

Use case: answer "does the graph leg ever pick the correct table when text
alone misses it?" -- i.e., is the graph leg adding complementary signal
that's being washed out by the aggregate average, or is it genuinely
redundant?

Usage:
    python scripts/per_query_diagnostic.py \\
        data/results/text_finetuned_wikitq_1000 \\
        data/results/hybrid_finetuned_rerank_a0.7_wikitq_1000 \\
        --label-a "FT-BGE alone" \\
        --label-b "Hybrid+Rerank"
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def load_details(folder: str | Path) -> list[dict]:
    p = Path(folder) / "retrieval_details.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    return json.loads(p.read_text(encoding="utf-8"))


def get_r1(d: dict) -> int:
    """Return 1 if the gold table was at rank 1, else 0. Handles both naming
    conventions."""
    if "recall_at_1" in d:
        return int(d["recall_at_1"])
    if "hit@1" in d:
        return int(d["hit@1"])
    return 0


def get_rank(d: dict) -> int | None:
    """Return the gold table's rank in the top-k (None if not present)."""
    if "rank" in d and d["rank"]:
        return int(d["rank"])
    # Derive from reciprocal_rank if present
    if "reciprocal_rank" in d and d["reciprocal_rank"] > 0:
        return int(round(1.0 / d["reciprocal_rank"]))
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder_a", help="Baseline result folder")
    ap.add_argument("folder_b", help="System result folder")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--show-examples", type=int, default=5,
                    help="Print this many example queries from each cell")
    args = ap.parse_args()

    label_a = args.label_a or Path(args.folder_a).name
    label_b = args.label_b or Path(args.folder_b).name

    details_a = {d.get("record_id") or d.get("id"): d
                 for d in load_details(args.folder_a)
                 if d.get("record_id") or d.get("id")}
    details_b = {d.get("record_id") or d.get("id"): d
                 for d in load_details(args.folder_b)
                 if d.get("record_id") or d.get("id")}

    common = sorted(set(details_a) & set(details_b))
    if not common:
        print("ERROR: no matching record_ids", file=sys.stderr)
        sys.exit(1)

    # 2x2 contingency on rank-1 success
    both_right = both_wrong = a_only = b_only = 0
    a_only_examples = []  # questions A got but B missed
    b_only_examples = []  # questions B got but A missed -- the interesting ones
    for rid in common:
        a_hit = get_r1(details_a[rid])
        b_hit = get_r1(details_b[rid])
        if a_hit and b_hit:
            both_right += 1
        elif a_hit and not b_hit:
            a_only += 1
            q = details_a[rid].get("question", "")
            a_only_examples.append((rid, q))
        elif not a_hit and b_hit:
            b_only += 1
            q = details_b[rid].get("question", "")
            b_only_examples.append((rid, q))
        else:
            both_wrong += 1

    n = len(common)
    # McNemar's exact symmetry test on the discordant pairs.
    # Null: pr(A right & B wrong) == pr(A wrong & B right).
    # Statistic: number of B-only wins out of (a_only + b_only) trials,
    # under H0 ~ Binomial(a_only + b_only, 0.5).
    discordant = a_only + b_only
    if discordant > 0:
        from math import comb
        # Two-sided exact p-value
        k = min(a_only, b_only)
        p_one_side = sum(comb(discordant, i) for i in range(k + 1)) \
                     / (2 ** discordant)
        mcnemar_p = min(2 * p_one_side, 1.0)
    else:
        mcnemar_p = 1.0

    print()
    print("=" * 78)
    print(f"  Per-query diagnostic: {label_b}  vs  {label_a}")
    print(f"  Aligned N = {n}")
    print("=" * 78)
    print()
    print(f"  Rank-1 contingency table:")
    print()
    print(f"                                 {label_b:^28}")
    print(f"                          {'correct':>14} {'wrong':>14}")
    print(f"  {label_a + ' correct':<22}  {both_right:>14} {a_only:>14}  "
          f"(A overall: {both_right + a_only})")
    print(f"  {label_a + ' wrong':<22}    {b_only:>14} {both_wrong:>14}  "
          f"(A overall: {b_only + both_wrong})")
    print(f"                          {both_right + b_only:>14} "
          f"{a_only + both_wrong:>14}")
    print(f"                          (B: {both_right + b_only})  "
          f"(B: {a_only + both_wrong})")
    print()
    print(f"  Discordant pairs: {discordant}")
    print(f"    {label_b} won (A wrong, B right)  : {b_only}")
    print(f"    {label_b} lost (A right, B wrong) : {a_only}")
    print(f"    Net (B − A)                       : {b_only - a_only:+d}")
    print()
    print(f"  McNemar exact two-sided p-value: {mcnemar_p:.4f}")
    if mcnemar_p < 0.05:
        print(f"  → Difference is statistically significant at α=0.05.")
    else:
        print(f"  → Difference is NOT statistically significant; gains and")
        print(f"    losses are balanced enough to be consistent with chance.")
    print()

    if b_only > 0 and args.show_examples > 0:
        print(f"  Example queries where {label_b} succeeded and {label_a} failed:")
        print(f"  (these are the queries where the graph leg added unique value)")
        print()
        for rid, q in b_only_examples[:args.show_examples]:
            print(f"    [{rid}] {q[:90]}")
        print()

    if a_only > 0 and args.show_examples > 0:
        print(f"  Example queries where {label_a} succeeded and {label_b} failed:")
        print(f"  (these are the queries where the hybrid lost to text alone)")
        print()
        for rid, q in a_only_examples[:args.show_examples]:
            print(f"    [{rid}] {q[:90]}")
        print()

    # Aggregate rank movement
    rank_changes_b_better = 0  # B has a lower (better) rank than A
    rank_changes_a_better = 0
    rank_changes_same = 0
    for rid in common:
        ra = get_rank(details_a[rid])
        rb = get_rank(details_b[rid])
        if ra is None and rb is None:
            continue
        if ra is None and rb is not None:
            rank_changes_b_better += 1
        elif rb is None and ra is not None:
            rank_changes_a_better += 1
        elif ra is not None and rb is not None:
            if rb < ra:
                rank_changes_b_better += 1
            elif rb > ra:
                rank_changes_a_better += 1
            else:
                rank_changes_same += 1
    print(f"  Rank movement (across all {n} queries):")
    print(f"    {label_b} ranked the gold table higher than {label_a}: "
          f"{rank_changes_b_better:>4}")
    print(f"    {label_a} ranked the gold table higher than {label_b}: "
          f"{rank_changes_a_better:>4}")
    print(f"    Same rank in both pipelines                           : "
          f"{rank_changes_same:>4}")
    print()
    print("=" * 78)


if __name__ == "__main__":
    main()
