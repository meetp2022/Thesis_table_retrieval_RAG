#!/usr/bin/env python3
"""
Main entry point for running pipelines.

Usage:
    python scripts/run_pipeline.py --pipeline graph --dataset wikitq
    python scripts/run_pipeline.py --pipeline graph --dataset wikitq --max-samples 50
    python scripts/run_pipeline.py --pipeline graph --dataset tatqa --max-samples 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.data.dataset_loader import load_dataset_by_name, get_dataset_stats
from loguru import logger
from rich.console import Console
from rich.table import Table as RichTable


console = Console()


def run_text_pipeline(records, config, args):
    """Run Pipeline 1: Text Baseline (Linearisation + BGE + FAISS)."""
    from src.pipelines.text_baseline.pipeline import TextBaselinePipeline

    # Build pipeline
    pipeline = TextBaselinePipeline.from_config(
        config,
        fallback_to_mock=args.mock_llm,
    )

    # Index all tables
    console.print("\n[bold cyan]Step 1: Indexing tables...[/bold cyan]")
    pipeline.index(records)
    console.print(f"  Indexed {len(pipeline.vector_store)} tables")

    # Run evaluation
    console.print("\n[bold cyan]Step 2: Running evaluation...[/bold cyan]")
    eval_result = pipeline.run(records, max_samples=args.max_samples)

    # Display results
    display_results(eval_result, args)

    # Save results
    if args.output:
        save_results(eval_result, args)

    # Save index for reuse
    if args.save_index:
        pipeline.save(args.save_index)
        console.print(f"\n  Index saved to {args.save_index}")

    return eval_result


def run_graph_pipeline(records, config, args):
    """Run Pipeline 3: Graph-Augmented Table Retrieval."""
    from src.pipelines.graph_augmented.pipeline import GraphAugmentedPipeline

    # Build pipeline
    pipeline = GraphAugmentedPipeline.from_config(
        config,
        fallback_to_mock=args.mock_llm,
    )

    # Index all tables
    console.print("\n[bold cyan]Step 1: Indexing tables...[/bold cyan]")
    pipeline.index(records)
    console.print(f"  Indexed {len(pipeline.vector_store)} tables")

    # Run evaluation
    console.print("\n[bold cyan]Step 2: Running evaluation...[/bold cyan]")
    eval_result = pipeline.run(records, max_samples=args.max_samples)

    # Display results
    display_results(eval_result, args)

    # Save results
    if args.output:
        save_results(eval_result, args)

    # Save index for reuse
    if args.save_index:
        pipeline.save(args.save_index)
        console.print(f"\n  Index saved to {args.save_index}")

    return eval_result


def display_results(eval_result, args):
    """Display evaluation results in a rich table."""
    console.print("\n[bold green]Results:[/bold green]")

    table = RichTable(title="Evaluation Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")

    metrics = eval_result.to_dict()
    table.add_row("Samples", str(metrics["num_samples"]))
    table.add_row("Exact Match", f"{metrics['exact_match']:.4f}")
    table.add_row("F1 Score", f"{metrics['f1_score']:.4f}")
    table.add_row("Recall@1", f"{metrics['recall_at_1']:.4f}")
    table.add_row("Recall@5", f"{metrics['recall_at_5']:.4f}")
    table.add_row("Avg Latency", f"{metrics['avg_latency_seconds']:.3f}s")
    table.add_row("Total Latency", f"{metrics['total_latency_seconds']:.1f}s")

    console.print(table)


def save_results(eval_result, args):
    """Save evaluation results to JSON."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary metrics
    summary_path = output_dir / "metrics.json"
    with open(summary_path, "w") as f:
        json.dump(eval_result.to_dict(), f, indent=2)
    console.print(f"\n  Metrics saved to {summary_path}")

    # Per-sample details
    if args.save_predictions:
        details_path = output_dir / "predictions.json"
        with open(details_path, "w") as f:
            json.dump(eval_result.per_sample, f, indent=2, default=str)
        console.print(f"  Predictions saved to {details_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a table retrieval pipeline"
    )
    parser.add_argument(
        "--pipeline",
        choices=["text", "image", "graph"],
        required=True,
        help="Which pipeline to run",
    )
    parser.add_argument(
        "--dataset",
        choices=["wikitq", "wikitablequestions", "tatqa", "finqa"],
        required=True,
        help="Which dataset to evaluate on",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit samples for quick testing",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Path to config directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save results (e.g. data/results/graph_wikitq)",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        default=False,
        help="Save per-sample predictions alongside metrics",
    )
    parser.add_argument(
        "--save-index",
        type=str,
        default=None,
        help="Directory to save the built index for reuse",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        default=False,
        help="Use mock LLM client (skip Ollama requirement)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override LLM model name (e.g. qwen2:0.5b for low-memory systems)",
    )
    args = parser.parse_args()

    # Setup
    setup_logging()
    config = load_config(pipeline=args.pipeline, config_dir=args.config_dir)

    # Override LLM model if specified
    if args.llm_model:
        config.setdefault("llm", {})["model"] = args.llm_model

    console.print(f"\n[bold]Pipeline: {args.pipeline.upper()}[/bold]")
    console.print(f"[bold]Dataset:  {args.dataset}[/bold]")
    if args.max_samples:
        console.print(f"[bold]Samples:  {args.max_samples}[/bold]")
    console.print("=" * 50)

    # Load data
    start = time.time()
    records = load_dataset_by_name(
        name=args.dataset,
        split="test",
        max_samples=args.max_samples,
    )
    load_time = time.time() - start

    stats = get_dataset_stats(records)
    console.print(f"\nLoaded {stats['count']} records in {load_time:.1f}s")
    console.print(f"  Tables: avg {stats['avg_table_rows']:.0f} x {stats['avg_table_cols']:.0f}")

    # Route to pipeline
    if args.pipeline == "graph":
        run_graph_pipeline(records, config, args)
    elif args.pipeline == "text":
        run_text_pipeline(records, config, args)
    elif args.pipeline == "image":
        console.print("\n[yellow]Image baseline pipeline not yet implemented.[/yellow]")


if __name__ == "__main__":
    main()
