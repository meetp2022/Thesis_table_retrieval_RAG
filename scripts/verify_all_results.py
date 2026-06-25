#!/usr/bin/env python3
"""
Verify all results — prints every saved metric in one consolidated table.

Scans data/results/*/retrieval_metrics.json and shows R@1, R@5, MRR,
sample size, and latency for every evaluation run on disk. Use this to
cross-check any number quoted in the thesis, slides, or demo materials
against the actual saved result files.

Usage:
    python scripts/verify_all_results.py
    python scripts/verify_all_results.py --sort r1     # sort by Recall@1
    python scripts/verify_all_results.py --filter 500  # only n=500 runs
"""

import argparse
import io
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"


def load_all():
    rows = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        mfile = d / "retrieval_metrics.json"
        if not mfile.exists():
            continue
        try:
            m = json.loads(mfile.read_text())
        except Exception as e:
            print(f"  [skip] {d.name}: {e}")
            continue
        rows.append({
            "folder": d.name,
            "pipeline": m.get("pipeline", "?"),
            "n": m.get("num_samples", 0),
            "r1": m.get("recall_at_1"),
            "r5": m.get("recall_at_5"),
            "mrr": m.get("mrr"),
            "lat": m.get("avg_latency_ms"),
        })
    return rows


def fmt(v, nd=3):
    if v is None:
        return "  -  "
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sort", choices=["folder", "r1", "r5", "mrr", "n"],
                    default="folder", help="sort key")
    ap.add_argument("--filter", default=None,
                    help="only show folders whose name contains this string")
    args = ap.parse_args()

    rows = load_all()
    if args.filter:
        rows = [r for r in rows if args.filter in r["folder"]]

    keymap = {"folder": lambda r: r["folder"], "n": lambda r: -r["n"],
              "r1": lambda r: -(r["r1"] or 0), "r5": lambda r: -(r["r5"] or 0),
              "mrr": lambda r: -(r["mrr"] or 0)}
    rows.sort(key=keymap[args.sort])

    print()
    print("=" * 92)
    print(f"{'ALL SAVED RETRIEVAL RESULTS':^92}")
    print(f"{'(source: data/results/*/retrieval_metrics.json)':^92}")
    print("=" * 92)
    print(f"{'Result folder':<40} {'pipeline':<16} {'n':>5} "
          f"{'R@1':>7} {'R@5':>7} {'MRR':>7} {'ms/q':>8}")
    print("-" * 92)
    for r in rows:
        print(f"{r['folder']:<40} {r['pipeline']:<16} {r['n']:>5} "
              f"{fmt(r['r1']):>7} {fmt(r['r5']):>7} {fmt(r['mrr']):>7} "
              f"{fmt(r['lat'], 1):>8}")
    print("=" * 92)
    print(f"{len(rows)} result file(s) found.")
    print()
    print("To re-measure any number, see scripts/verify_all_results.py header")
    print("or run, e.g.:")
    print("  python scripts/eval_retrieval.py --pipeline text "
          "--max-samples 500 --split validation")
    print()


if __name__ == "__main__":
    main()
