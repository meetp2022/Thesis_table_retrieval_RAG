#!/usr/bin/env python3
"""
Bootstrap 95% confidence intervals on the difference between two retrieval
pipelines.

Reads retrieval_details.json from two result folders, aligns the per-query
results by record_id, and reports paired-bootstrap confidence intervals for
the differences in R@1, R@5, and MRR.

A 95% CI that excludes zero is conventionally taken as statistically
significant at the alpha = 0.05 level (marked with * in the output).

Typical headline comparisons for the EMNLP 2026 Industry Track paper:

    # WikiTQ: hybrid lift over fine-tuned BGE alone
    python scripts/bootstrap_ci.py \\
        data/results/text_finetuned_wikitq_500 \\
        data/results/hybrid_finetuned_rerank_a0.7_wikitq_500 \\
        --label-a "FT-BGE" --label-b "Hybrid+Rerank"

    # FinQA: hybrid lift over fine-tuned BGE alone (matched in-domain GNN)
    python scripts/bootstrap_ci.py \\
        data/results/text_finetuned_finqa \\
        data/results/hybrid_finetuned_rerank_a0.7_finqa_500_with_finqa_gnn \\
        --label-a "FT-BGE" --label-b "Hybrid+Rerank (FinQA)"

    # WikiTQ: hybrid vs Gemini (frontier)
    python scripts/bootstrap_ci.py \\
        data/results/gemini_gemini-embedding-001_wikitq_500 \\
        data/results/hybrid_finetuned_rerank_a0.7_wikitq_500 \\
        --label-a "Gemini" --label-b "Hybrid+Rerank"

    # WikiTQ: no-rerank ablation (graph leg alone effect)
    python scripts/bootstrap_ci.py \\
        data/results/text_finetuned_wikitq_500 \\
        data/results/hybrid_finetuned_norerank_a0.7_wikitq_500 \\
        --label-a "FT-BGE" --label-b "Hybrid no-rerank"
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


# ────────────────────────────────────────────────────────────────────────────
#  Detail-file loading and normalisation
# ────────────────────────────────────────────────────────────────────────────

def load_details(folder: str | Path) -> list[dict]:
    p = Path(folder) / "retrieval_details.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    return json.loads(p.read_text(encoding="utf-8"))


def normalize(details: list[dict]) -> list[tuple[str, float, float, float]]:
    """Convert per-record details to (record_id, r@1, r@5, mrr) tuples.

    Different eval scripts have used different field names for the same
    quantities:
      - eval_retrieval.py            -> "hit@1", "hit@5", and a "rank" field
      - eval_finetuned_bge.py        -> "recall_at_1", "recall_at_5",
                                        "reciprocal_rank"
      - eval_hybrid_finetuned.py     -> "recall_at_1", "recall_at_5",
                                        "reciprocal_rank"
      - eval_gemini_baseline.py      -> "recall_at_1", "recall_at_5",
                                        "reciprocal_rank"

    This function reads whichever set of fields the file uses and computes
    MRR from the rank field if reciprocal_rank is absent.
    """
    out: list[tuple[str, float, float, float]] = []
    for d in details:
        rid = d.get("record_id") or d.get("id")
        if rid is None:
            continue

        # R@1: try recall_at_1, then hit@1
        if "recall_at_1" in d:
            r1 = float(d["recall_at_1"])
        elif "hit@1" in d:
            r1 = float(d["hit@1"])
        else:
            r1 = 0.0

        # R@5 (or whichever recall_at_K is present alongside recall_at_1)
        r5 = 0.0
        r5_key = next(
            (k for k in d if k.startswith("recall_at_") and k != "recall_at_1"),
            None,
        )
        if r5_key is not None:
            r5 = float(d[r5_key])
        elif "hit@5" in d:
            r5 = float(d["hit@5"])

        # MRR / reciprocal rank: prefer the saved value, else derive from
        # a "rank" field (1-based; 0 means the gold table was not in top-k).
        if "reciprocal_rank" in d:
            rr = float(d["reciprocal_rank"])
        elif "rank" in d:
            rank = d["rank"]
            rr = 1.0 / rank if rank else 0.0
        else:
            rr = 0.0

        out.append((rid, r1, r5, rr))
    return out


def align(
    records_a: list[tuple], records_b: list[tuple]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align two normalised records-lists by record_id."""
    by_a = {r[0]: r[1:] for r in records_a}
    by_b = {r[0]: r[1:] for r in records_b}
    common = sorted(set(by_a) & set(by_b))
    if not common:
        raise ValueError(
            "No common record_ids between the two result sets — were they "
            "run on the same dataset / split / n?"
        )
    a = np.array([by_a[rid] for rid in common], dtype=np.float64)
    b = np.array([by_b[rid] for rid in common], dtype=np.float64)
    return a, b, common


# ────────────────────────────────────────────────────────────────────────────
#  Paired bootstrap
# ────────────────────────────────────────────────────────────────────────────

def bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int, seed: int
) -> np.ndarray:
    """Paired bootstrap: resample queries (with replacement), recompute the
    per-pipeline metric means, and return the (n_boot, 3) array of (B − A)
    differences."""
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty((n_boot, 3), dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = b[idx].mean(axis=0) - a[idx].mean(axis=0)
    return diffs


# ────────────────────────────────────────────────────────────────────────────
#  Reporting
# ────────────────────────────────────────────────────────────────────────────

def format_report(
    label_a: str,
    label_b: str,
    n: int,
    n_boot: int,
    a_mean: np.ndarray,
    b_mean: np.ndarray,
    diffs: np.ndarray,
) -> str:
    ci_lo = np.percentile(diffs, 2.5, axis=0)
    ci_hi = np.percentile(diffs, 97.5, axis=0)
    abs_diff = b_mean - a_mean
    # Relative difference, computed on the original (non-bootstrap) means
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.where(a_mean > 0, abs_diff / a_mean * 100, 0.0)

    # Two-sided bootstrap p-values via the percentile method.
    p_values = []
    for k in range(3):
        # Fraction of bootstrap samples on the opposite side of zero from
        # the observed effect. Multiply by 2 for two-sided.
        if abs_diff[k] >= 0:
            p_one_sided = (diffs[:, k] <= 0).mean()
        else:
            p_one_sided = (diffs[:, k] >= 0).mean()
        p_values.append(min(2 * p_one_sided, 1.0))

    sig_flags = [
        "*" if (lo > 0 or hi < 0) else " "
        for lo, hi in zip(ci_lo, ci_hi)
    ]

    lines: list[str] = []
    bar = "=" * 86
    lines.append(bar)
    lines.append(f"  Paired-bootstrap 95% CI:  {label_b}  vs  {label_a}")
    lines.append(f"  N = {n} matched records   ·   {n_boot} bootstrap resamples")
    lines.append(bar)
    lines.append(
        f"  {'Metric':<6}  {label_a + ' mean':<14}  {label_b + ' mean':<16}  "
        f"{'Δ absolute (95% CI)':<32}  {'Δ rel.':<8}  p"
    )
    lines.append("  " + "-" * 84)
    labels = ["R@1", "R@5", "MRR"]
    for k, lbl in enumerate(labels):
        ci_str = f"{abs_diff[k]:+.4f}  [{ci_lo[k]:+.4f}, {ci_hi[k]:+.4f}]"
        rel_str = f"{rel_diff[k]:+.1f}%"
        p_str = f"{p_values[k]:.4f}"
        lines.append(
            f"  {lbl:<6}  {a_mean[k]:.4f}          {b_mean[k]:.4f}            "
            f"{ci_str:<32}  {rel_str:<8}  {p_str}  {sig_flags[k]}"
        )
    lines.append(bar)
    lines.append(
        "  *  =  95% CI excludes 0 (statistically significant at α = 0.05)"
    )
    lines.append("")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("folder_a", help="Baseline result folder")
    ap.add_argument("folder_b", help="System result folder")
    ap.add_argument("--label-a", default=None, help="Display name for A")
    ap.add_argument("--label-b", default=None, help="Display name for B")
    ap.add_argument(
        "--n-bootstrap", type=int, default=10_000,
        help="Number of bootstrap resamples (default: 10000)",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = ap.parse_args()

    label_a = args.label_a or Path(args.folder_a).name
    label_b = args.label_b or Path(args.folder_b).name

    details_a = normalize(load_details(args.folder_a))
    details_b = normalize(load_details(args.folder_b))
    a, b, common = align(details_a, details_b)

    diffs = bootstrap_diff(a, b, n_boot=args.n_bootstrap, seed=args.seed)
    a_mean = a.mean(axis=0)
    b_mean = b.mean(axis=0)

    print(format_report(label_a, label_b, len(common), args.n_bootstrap,
                        a_mean, b_mean, diffs))


if __name__ == "__main__":
    main()
