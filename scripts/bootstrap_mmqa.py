#!/usr/bin/env python3
"""
Paired bootstrap 95% CI + McNemar on Full-set Recall@10 for two MMQA runs.

Reads per_query.json from two result folders (written by eval_mmqa_hybrid.py),
aligns by qid, and reports the paired-bootstrap CI on the difference in mean
Full-set R@10 plus an exact McNemar test on per-query full-set success.

Usage:
    python scripts/bootstrap_mmqa.py \\
        data/results/mmqa_hybrid_intra_a0.3_test \\
        data/results/mmqa_hybrid_inter_a0.3_test \\
        --label-a "V3 intra" --label-b "V4 inter"
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def load(folder):
    d = json.loads((Path(folder) / "per_query.json").read_text(encoding="utf-8"))
    return {q: r for q, r in zip(d["qid"], d["full_set_recall_at_10"])}


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n) * 2
    return min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder_a")
    ap.add_argument("folder_b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    A, B = load(args.folder_a), load(args.folder_b)
    keys = [q for q in A if q in B]
    a = np.array([A[q] for q in keys], dtype=float)
    b = np.array([B[q] for q in keys], dtype=float)
    n = len(keys)

    rng = np.random.default_rng(args.seed)
    diffs = []
    for _ in range(args.resamples):
        idx = rng.integers(0, n, n)
        diffs.append(b[idx].mean() - a[idx].mean())
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    delta = b.mean() - a.mean()
    p_boot = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())

    # McNemar on discordant pairs
    b_only = int(((b == 1) & (a == 0)).sum())   # B right, A wrong
    a_only = int(((a == 1) & (b == 0)).sum())   # A right, B wrong
    p_mcn = mcnemar_exact(b_only, a_only)

    print("\n" + "=" * 74)
    print(f"  Full-set R@10:  {args.label_b}  vs  {args.label_a}   (N={n})")
    print("=" * 74)
    print(f"  {args.label_a} mean : {a.mean():.4f}")
    print(f"  {args.label_b} mean : {b.mean():.4f}")
    print(f"  Delta (B-A)  : {delta:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
          f"p_boot={p_boot:.4f}")
    sig = "*" if (lo > 0 or hi < 0) else "(NS)"
    print(f"  -> {sig}")
    print(f"\n  McNemar discordant:  {args.label_b}-only={b_only}  "
          f"{args.label_a}-only={a_only}   p={p_mcn:.4f}")
    print("=" * 74)


if __name__ == "__main__":
    main()
