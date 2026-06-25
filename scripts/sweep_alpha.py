#!/usr/bin/env python3
"""
Alpha sweep for hybrid pipeline — finds the optimal text/graph fusion weight.

Indexes tables once, then evaluates R@1, R@5, MRR at multiple alpha values
without re-indexing.  This makes the sweep very fast (minutes not hours).

Usage:
    python scripts/sweep_alpha.py --max-samples 100
    python scripts/sweep_alpha.py --max-samples 500 --alphas 0.3 0.5 0.7
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from rich.console import Console
from rich.table import Table as RichTable
from rich import box

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.data.dataset_loader import load_dataset_by_name
from src.pipelines.hybrid.pipeline import HybridPipeline
from loguru import logger

console = Console(legacy_windows=False, highlight=False)


def eval_alpha(pipeline: HybridPipeline, records, alpha: float, top_k: int = 5):
    """Evaluate retrieval metrics at a given alpha (no re-indexing needed)."""
    pipeline.retriever.alpha = alpha

    hits_at_1, hits_at_5 = 0, 0
    rr_sum = 0.0

    for rec in records:
        results = pipeline.retriever.retrieve(rec.question, top_k=top_k)
        rank = None
        for j, r in enumerate(results):
            if r.record_id == rec.id:
                rank = j + 1
                break

        hits_at_1 += int(rank == 1)
        hits_at_5 += int(rank is not None and rank <= 5)
        rr_sum += (1.0 / rank) if rank else 0.0

    n = len(records)
    return {
        "alpha": alpha,
        "recall_at_1": hits_at_1 / n,
        "recall_at_5": hits_at_5 / n,
        "mrr": rr_sum / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Alpha sweep for hybrid pipeline")
    parser.add_argument("--dataset", default="wikitq")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Alpha values to sweep (space-separated)",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    setup_logging()

    console.print("\n[bold]Hybrid Alpha Sweep[/bold]")
    console.print(f"  Dataset:  {args.dataset} ({args.split})")
    console.print(f"  Samples:  {args.max_samples}")
    console.print(f"  Alphas:   {args.alphas}")
    console.print("=" * 50)

    # Load data
    records = load_dataset_by_name(args.dataset, split=args.split, max_samples=args.max_samples)
    console.print(f"Loaded {len(records)} records")

    # Build pipeline (index ONCE)
    text_config = load_config(pipeline="text")
    graph_config = load_config(pipeline="graph")

    console.print("\n[bold cyan]Building and indexing hybrid pipeline...[/bold cyan]")
    t0 = time.time()
    pipe = HybridPipeline.from_config(
        text_config, graph_config, alpha=0.5, fallback_to_mock=True
    )
    pipe.index(records)
    console.print(f"  Indexed in {time.time() - t0:.1f}s")

    # Sweep alphas
    console.print("\n[bold cyan]Sweeping alpha values...[/bold cyan]")
    results = []
    for alpha in args.alphas:
        r = eval_alpha(pipe, records, alpha, args.top_k)
        results.append(r)
        console.print(
            f"  alpha={alpha:.1f}  R@1={r['recall_at_1']:.4f}  "
            f"R@5={r['recall_at_5']:.4f}  MRR={r['mrr']:.4f}"
        )

    # Best alpha by R@1
    best = max(results, key=lambda x: x["recall_at_1"])
    console.print(f"\n[bold green]Best alpha by R@1: {best['alpha']}[/bold green]")

    # Display table
    table = RichTable(title="Alpha Sweep Results", box=box.ROUNDED)
    table.add_column("Alpha", style="cyan")
    table.add_column("R@1", style="bold green")
    table.add_column("R@5", style="green")
    table.add_column("MRR", style="green")
    for r in results:
        marker = " ←" if r["alpha"] == best["alpha"] else ""
        table.add_row(
            str(r["alpha"]),
            f"{r['recall_at_1']:.4f}{marker}",
            f"{r['recall_at_5']:.4f}",
            f"{r['mrr']:.4f}",
        )
    console.print(table)

    # Save
    output_dir = args.output or f"data/results/alpha_sweep_{args.dataset}_{args.max_samples}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/sweep_results.json", "w") as f:
        json.dump({"best_alpha": best["alpha"], "results": results}, f, indent=2)
    console.print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
