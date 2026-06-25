"""
Phase 1 — Dataset Exploration Notebook
=======================================
Run this to explore and understand all three datasets
before building pipelines.

Usage: python notebooks/01_explore_datasets.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_loader import (
    load_wikitablequestions,
    load_tatqa,
    load_finqa,
    get_dataset_stats,
)
from src.utils.logging import setup_logging
from rich.console import Console
from rich.table import Table as RichTable
from rich.panel import Panel


console = Console()
setup_logging()


def explore_dataset(name, loader_fn, split="test", max_samples=5):
    """Load a dataset and display sample records."""

    console.print(f"\n{'='*60}")
    console.print(Panel(f"[bold]{name}[/bold]", expand=False))

    try:
        # Load a small sample for inspection
        records = loader_fn(split=split, max_samples=max_samples)
    except Exception as e:
        console.print(f"[red]Failed to load: {e}[/red]")
        return

    # Show stats
    stats = get_dataset_stats(records)
    stat_table = RichTable(title="Dataset Statistics")
    stat_table.add_column("Metric", style="cyan")
    stat_table.add_column("Value", style="green", justify="right")
    for k, v in stats.items():
        stat_table.add_row(k, str(v))
    console.print(stat_table)

    # Show sample records
    for i, rec in enumerate(records[:3]):
        console.print(f"\n[bold]─── Sample {i+1} ───[/bold]")
        console.print(f"  [cyan]ID:[/cyan]       {rec.id}")
        console.print(f"  [cyan]Question:[/cyan] {rec.question}")
        console.print(f"  [cyan]Answers:[/cyan]  {rec.answers}")
        console.print(f"  [cyan]Table:[/cyan]    {rec.num_rows} rows × {rec.num_cols} cols")

        # Show table preview (first 3 rows)
        if rec.table_header:
            t = RichTable(show_lines=True, title="Table Preview (first 3 rows)")
            for col in rec.table_header[:8]:  # Limit columns for display
                t.add_column(col, max_width=20)
            for row in rec.table_rows[:3]:
                t.add_row(*[str(c)[:20] for c in row[:8]])
            console.print(t)

        if rec.context_text:
            preview = rec.context_text[:200] + "..." if len(rec.context_text) > 200 else rec.context_text
            console.print(f"  [cyan]Context:[/cyan] {preview}")

        # Show Markdown representation (what Pipeline 1 will use)
        console.print(f"\n  [dim]Markdown representation (Pipeline 1 input):[/dim]")
        md = rec.table_to_markdown()
        for line in md.split("\n")[:6]:
            console.print(f"    {line}")
        if len(md.split("\n")) > 6:
            console.print(f"    ... ({len(md.split(chr(10)))} total lines)")


def main():
    console.print("[bold]Phase 1 — Dataset Exploration[/bold]")
    console.print("Inspecting all three datasets for the thesis.\n")

    # WikiTableQuestions
    explore_dataset(
        "WikiTableQuestions (General Domain)",
        load_wikitablequestions,
        split="test",
        max_samples=5,
    )

    # TAT-QA
    explore_dataset(
        "TAT-QA (Finance Domain)",
        lambda split, max_samples: load_tatqa(split=split, max_samples=max_samples),
        split="train",
        max_samples=5,
    )

    # FinQA
    explore_dataset(
        "FinQA (Finance Domain)",
        lambda split, max_samples: load_finqa(split=split, max_samples=max_samples),
        split="test",
        max_samples=5,
    )

    console.print("\n[bold green]Exploration complete![/bold green]")
    console.print(
        "\nNext step: Phase 2 — Build the graph-augmented pipeline (Pipeline 3)"
    )


if __name__ == "__main__":
    main()
