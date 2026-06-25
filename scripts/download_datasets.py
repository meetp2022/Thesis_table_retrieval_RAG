#!/usr/bin/env python3
"""
Download and verify all three datasets.

Usage:
    python scripts/download_datasets.py                    # Download all
    python scripts/download_datasets.py --dataset wikitq   # Download one
    python scripts/download_datasets.py --verify-only       # Just verify
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_loader import (
    load_wikitablequestions,
    load_tatqa,
    load_finqa,
    get_dataset_stats,
)
from src.utils.logging import setup_logging
from loguru import logger
from rich.console import Console
from rich.table import Table as RichTable


console = Console()


def download_wikitablequestions():
    """Download WikiTableQuestions via HuggingFace."""
    console.print("\n[bold cyan]━━━ WikiTableQuestions ━━━[/bold cyan]")

    for split in ["train", "test", "validation"]:
        try:
            records = load_wikitablequestions(split=split, max_samples=5)
            stats = get_dataset_stats(records)
            console.print(f"  ✓ {split}: loaded {stats['count']} sample records")
        except Exception as e:
            console.print(f"  ✗ {split}: {e}", style="red")
            return False

    # Full load of test split to verify
    records = load_wikitablequestions(split="test")
    stats = get_dataset_stats(records)
    _print_stats("WikiTableQuestions (test)", stats)
    return True


def download_tatqa():
    """Download TAT-QA via HuggingFace or local files."""
    console.print("\n[bold cyan]━━━ TAT-QA ━━━[/bold cyan]")

    try:
        records = load_tatqa(split="train", max_samples=5)
        stats = get_dataset_stats(records)
        console.print(f"  ✓ Loaded {stats['count']} sample records")

        records = load_tatqa(split="train")
        stats = get_dataset_stats(records)
        _print_stats("TAT-QA (train)", stats)
        return True
    except Exception as e:
        console.print(f"  ✗ Failed: {e}", style="red")
        console.print(
            "  ℹ  Download manually from: https://github.com/NExTplusplus/TAT-QA",
            style="yellow",
        )
        console.print(
            "     Place JSON files in: data/raw/tatqa/",
            style="yellow",
        )
        return False


def download_finqa():
    """Download FinQA via HuggingFace or local files."""
    console.print("\n[bold cyan]━━━ FinQA ━━━[/bold cyan]")

    try:
        records = load_finqa(split="test", max_samples=5)
        stats = get_dataset_stats(records)
        console.print(f"  ✓ Loaded {stats['count']} sample records")

        records = load_finqa(split="test")
        stats = get_dataset_stats(records)
        _print_stats("FinQA (test)", stats)
        return True
    except Exception as e:
        console.print(f"  ✗ Failed: {e}", style="red")
        console.print(
            "  ℹ  Download manually from: https://github.com/czyssrs/FinQA",
            style="yellow",
        )
        console.print(
            "     Place JSON files in: data/raw/finqa/",
            style="yellow",
        )
        return False


def _print_stats(name: str, stats: dict):
    """Pretty-print dataset statistics."""
    table = RichTable(title=f"{name} Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Total records", str(stats["count"]))
    table.add_row("Domain", stats["domain"])
    table.add_row("Avg table rows", f"{stats['avg_table_rows']:.1f}")
    table.add_row("Avg table cols", f"{stats['avg_table_cols']:.1f}")
    table.add_row("Max table rows", str(stats["max_table_rows"]))
    table.add_row("Max table cols", str(stats["max_table_cols"]))
    table.add_row("Avg question words", f"{stats['avg_question_words']:.1f}")
    table.add_row("Records with context", f"{stats['records_with_context']} ({stats['pct_with_context']}%)")

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Download thesis datasets")
    parser.add_argument(
        "--dataset",
        choices=["wikitq", "tatqa", "finqa", "all"],
        default="all",
        help="Which dataset to download",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing downloads",
    )
    args = parser.parse_args()

    setup_logging()

    console.print("[bold]Graph-Augmented Table RAG — Dataset Setup[/bold]")
    console.print("=" * 50)

    results = {}

    if args.dataset in ("wikitq", "all"):
        results["WikiTableQuestions"] = download_wikitablequestions()

    if args.dataset in ("tatqa", "all"):
        results["TAT-QA"] = download_tatqa()

    if args.dataset in ("finqa", "all"):
        results["FinQA"] = download_finqa()

    # Summary
    console.print("\n[bold]━━━ Summary ━━━[/bold]")
    for name, status in results.items():
        icon = "✓" if status else "✗"
        colour = "green" if status else "red"
        console.print(f"  [{colour}]{icon} {name}[/{colour}]")

    if all(results.values()):
        console.print("\n[bold green]All datasets ready![/bold green]")
    else:
        console.print("\n[bold yellow]Some datasets need manual download. See instructions above.[/bold yellow]")
        sys.exit(1)


if __name__ == "__main__":
    main()
