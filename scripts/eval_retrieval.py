#!/usr/bin/env python3
"""
Retrieval-only evaluation — compares R@1, R@5, MRR across pipelines.

Does NOT require Ollama/LLM — only measures how well each pipeline
retrieves the correct table for a given question.

Usage:
    python scripts/eval_retrieval.py --pipeline text --dataset wikitq --max-samples 100
    python scripts/eval_retrieval.py --pipeline graph --dataset wikitq --max-samples 100
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
from loguru import logger

console = Console(legacy_windows=False, highlight=False)


def evaluate_retrieval(
    pipeline_type: str,
    records,
    config,
    top_k: int = 5,
    alpha: float = 0.5,
    use_reranker: bool = False,
):
    """
    Index tables, then evaluate retrieval accuracy.
    Returns dict with R@1, R@5, MRR, and per-sample details.
    """
    # Build pipeline
    if pipeline_type in ("text", "text_finetuned"):
        from src.pipelines.text_baseline.pipeline import TextBaselinePipeline
        pipe = TextBaselinePipeline.from_config(config, fallback_to_mock=True)
    elif pipeline_type == "graph":
        from src.pipelines.graph_augmented.pipeline import GraphAugmentedPipeline
        pipe = GraphAugmentedPipeline.from_config(config, fallback_to_mock=True)
    elif pipeline_type == "hybrid":
        from src.pipelines.hybrid.pipeline import HybridPipeline
        from src.utils.config import load_config as _load_config
        text_config = _load_config(pipeline="text")
        graph_config = _load_config(pipeline="graph")
        pipe = HybridPipeline.from_config(
            text_config, graph_config, alpha=alpha,
            fallback_to_mock=True, use_reranker=use_reranker,
        )
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_type}")

    # Index
    console.print(f"\n[bold cyan]Indexing {len(records)} tables...[/bold cyan]")
    t0 = time.time()
    pipe.index(records)
    index_time = time.time() - t0
    console.print(f"  Indexed in {index_time:.1f}s")

    # Evaluate retrieval
    console.print(f"\n[bold cyan]Evaluating retrieval (top_k={top_k})...[/bold cyan]")

    hits_at_1 = 0
    hits_at_5 = 0
    reciprocal_ranks = []
    per_sample = []

    t0 = time.time()
    for i, rec in enumerate(records):
        if pipeline_type == "hybrid":
            results = pipe.retriever.retrieve(rec.question, top_k=top_k, rerank=use_reranker)
        else:
            results = pipe.retriever.retrieve(rec.question, top_k=top_k)

        # Find rank of correct table
        rank = None
        for j, res in enumerate(results):
            if res.record_id == rec.id:
                rank = j + 1
                break

        hit1 = rank == 1
        hit5 = rank is not None and rank <= 5
        rr = 1.0 / rank if rank else 0.0

        hits_at_1 += int(hit1)
        hits_at_5 += int(hit5)
        reciprocal_ranks.append(rr)

        per_sample.append({
            "record_id": rec.id,
            "question": rec.question[:100],
            "rank": rank,
            "hit@1": hit1,
            "hit@5": hit5,
            "top_score": results[0].score if results else 0.0,
        })

        if (i + 1) % 25 == 0:
            console.print(f"  Evaluated {i+1}/{len(records)}...")

    eval_time = time.time() - t0
    n = len(records)

    metrics = {
        "pipeline": pipeline_type,
        "num_samples": n,
        "recall_at_1": hits_at_1 / n,
        "recall_at_5": hits_at_5 / n,
        "mrr": sum(reciprocal_ranks) / n,
        "hits_at_1": hits_at_1,
        "hits_at_5": hits_at_5,
        "index_time_seconds": index_time,
        "eval_time_seconds": eval_time,
        "avg_latency_ms": (eval_time / n) * 1000,
    }

    return metrics, per_sample


def main():
    parser = argparse.ArgumentParser(description="Retrieval-only evaluation")
    parser.add_argument("--pipeline", choices=["text", "text_finetuned", "graph", "hybrid"], required=True)
    parser.add_argument("--dataset", choices=["wikitq", "tatqa", "finqa"], default="wikitq")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Hybrid fusion weight: alpha*text + (1-alpha)*graph (default: 0.5)")
    parser.add_argument("--rerank", action="store_true",
                        help="Apply cross-encoder re-ranking after fusion (hybrid only)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    setup_logging()
    # Hybrid pipeline builds its own configs internally
    config = load_config(pipeline="text") if args.pipeline == "hybrid" else load_config(pipeline=args.pipeline.replace("text_finetuned", "text_finetuned"))

    console.print(f"\n[bold]Retrieval Evaluation[/bold]")
    console.print(f"  Pipeline: {args.pipeline.upper()}")
    if args.pipeline == "hybrid":
        console.print(f"  Alpha:    {args.alpha}")
        if args.rerank:
            console.print(f"  Rerank:   cross-encoder/ms-marco-MiniLM-L-6-v2")
    console.print(f"  Dataset:  {args.dataset} ({args.split})")
    console.print(f"  Samples:  {args.max_samples}")
    console.print("=" * 50)

    # Load data
    records = load_dataset_by_name(args.dataset, split=args.split, max_samples=args.max_samples)
    console.print(f"Loaded {len(records)} records")

    # Evaluate
    metrics, per_sample = evaluate_retrieval(
        args.pipeline, records, config, args.top_k,
        alpha=args.alpha, use_reranker=args.rerank,
    )

    # Display
    table = RichTable(title=f"Retrieval Results — {args.pipeline.upper()}", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold green")
    table.add_row("Samples", str(metrics["num_samples"]))
    table.add_row("Recall@1", f"{metrics['recall_at_1']:.4f}  ({metrics['hits_at_1']}/{metrics['num_samples']})")
    table.add_row("Recall@5", f"{metrics['recall_at_5']:.4f}  ({metrics['hits_at_5']}/{metrics['num_samples']})")
    table.add_row("MRR", f"{metrics['mrr']:.4f}")
    table.add_row("Index Time", f"{metrics['index_time_seconds']:.1f}s")
    table.add_row("Avg Query Latency", f"{metrics['avg_latency_ms']:.1f}ms")
    console.print(table)

    # Save
    output_dir = args.output or f"data/results/{args.pipeline}_{args.dataset}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{output_dir}/retrieval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{output_dir}/retrieval_details.json", "w") as f:
        json.dump(per_sample, f, indent=2)
    console.print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
