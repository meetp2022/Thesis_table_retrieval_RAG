#!/usr/bin/env python3
"""
Interactive demo for the Graph-Augmented Table Retrieval pipeline.

Modes:
  1. Mock mode  (no Ollama needed) — instant answers using retrieved table context
  2. Ollama mode (Mistral running)  — real LLM answers

Usage:
    # Instant demo with built-in sample tables (no downloads, no Ollama):
    python scripts/demo.py

    # With real Mistral LLM:
    python scripts/demo.py --use-ollama

    # Load from a real dataset (downloads ~100 samples from WikiTQ):
    python scripts/demo.py --dataset wikitq --max-samples 100

    # Single query without interactive mode:
    python scripts/demo.py --query "What is the population of France?"
"""

import argparse
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force UTF-8 output on Windows to handle Rich box-drawing characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from rich.console import Console
from rich.table import Table as RichTable
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console(legacy_windows=False, highlight=False)


# ────────────────────────────────────────────────
#  Built-in sample tables (no dataset download needed)
# ────────────────────────────────────────────────

SAMPLE_TABLES = [
    {
        "id": "demo-001",
        "question": "Which country has the highest population?",
        "answers": ["China"],
        "table_title": "World Population 2024",
        "table_header": ["Country", "Population (M)", "Continent", "GDP (USD T)"],
        "table_rows": [
            ["China", "1,425", "Asia", "17.7"],
            ["India", "1,441", "Asia", "3.5"],
            ["United States", "340", "North America", "25.4"],
            ["Indonesia", "278", "Asia", "1.3"],
            ["Pakistan", "240", "Asia", "0.35"],
            ["Brazil", "216", "South America", "2.1"],
            ["Nigeria", "224", "Africa", "0.47"],
            ["Germany", "84", "Europe", "4.1"],
            ["France", "68", "Europe", "2.9"],
        ],
    },
    {
        "id": "demo-002",
        "question": "What was Apple's revenue in 2023?",
        "answers": ["$383.3B"],
        "table_title": "Big Tech Revenue 2022-2024 (USD Billions)",
        "table_header": ["Company", "2022", "2023", "2024", "YoY Growth"],
        "table_rows": [
            ["Apple", "394.3", "383.3", "391.0", "-0.8%"],
            ["Microsoft", "198.3", "211.9", "245.1", "6.9%"],
            ["Alphabet (Google)", "282.8", "307.4", "350.0", "8.7%"],
            ["Amazon", "514.0", "574.8", "620.1", "11.9%"],
            ["Meta", "116.6", "134.9", "156.2", "15.7%"],
            ["NVIDIA", "26.9", "60.9", "130.5", "126.3%"],
        ],
    },
    {
        "id": "demo-003",
        "question": "Who won the most Olympic gold medals?",
        "answers": ["Michael Phelps"],
        "table_title": "All-Time Olympic Gold Medal Leaders",
        "table_header": ["Athlete", "Country", "Sport", "Gold", "Silver", "Bronze", "Total"],
        "table_rows": [
            ["Michael Phelps", "USA", "Swimming", "23", "3", "2", "28"],
            ["Larisa Latynina", "USSR", "Gymnastics", "9", "5", "4", "18"],
            ["Paavo Nurmi", "Finland", "Athletics", "9", "3", "0", "12"],
            ["Mark Spitz", "USA", "Swimming", "9", "1", "1", "11"],
            ["Carl Lewis", "USA", "Athletics", "9", "1", "0", "10"],
            ["Usain Bolt", "Jamaica", "Athletics", "8", "0", "0", "8"],
            ["Simone Biles", "USA", "Gymnastics", "7", "2", "2", "11"],
        ],
    },
    {
        "id": "demo-004",
        "question": "What is the interest coverage ratio for Tesla?",
        "answers": ["8.2x"],
        "table_title": "EV Manufacturer Financial Metrics Q3 2024",
        "table_header": ["Company", "Revenue (B)", "Gross Margin", "D/E Ratio", "Interest Coverage"],
        "table_rows": [
            ["Tesla", "25.2", "17.1%", "0.08", "8.2x"],
            ["BYD", "28.0", "19.4%", "0.51", "6.1x"],
            ["Rivian", "1.34", "-44.8%", "0.95", "-3.2x"],
            ["Lucid", "0.20", "-149.1%", "1.23", "-8.7x"],
            ["NIO", "2.49", "10.7%", "0.78", "-1.4x"],
        ],
        "context_text": "Q3 2024 results reflect ongoing price competition in the EV sector.",
    },
    {
        "id": "demo-005",
        "question": "Which programming language has the highest salary?",
        "answers": ["Zig"],
        "table_title": "Developer Salaries by Language 2024 (Stack Overflow Survey)",
        "table_header": ["Language", "Median Salary (USD)", "Users (M)", "Job Postings"],
        "table_rows": [
            ["Zig", "103,611", "0.1", "120"],
            ["Erlang", "99,492", "0.2", "450"],
            ["Scala", "96,381", "0.8", "3,200"],
            ["Rust", "87,013", "1.5", "4,800"],
            ["Go", "85,120", "3.2", "12,400"],
            ["Python", "75,669", "15.7", "85,000"],
            ["JavaScript", "71,292", "17.4", "102,000"],
            ["PHP", "50,449", "5.1", "22,000"],
        ],
    },
]


def build_sample_records():
    """Convert SAMPLE_TABLES into TableQARecord objects."""
    from src.data.dataset_loader import TableQARecord
    records = []
    for t in SAMPLE_TABLES:
        records.append(TableQARecord(
            id=t["id"],
            question=t["question"],
            answers=t["answers"],
            table_header=t["table_header"],
            table_rows=t["table_rows"],
            table_title=t.get("table_title"),
            context_text=t.get("context_text"),
            dataset="demo",
            domain="general",
        ))
    return records


# ────────────────────────────────────────────────
#  Pipeline builder
# ────────────────────────────────────────────────

def build_pipeline(use_ollama: bool, config: dict, pipeline_type: str = "text"):
    """
    Build the selected pipeline.

    Parameters
    ----------
    use_ollama : bool
        Whether to use real Ollama LLM.
    config : dict
        Merged config.
    pipeline_type : str
        'text' for text baseline (BGE-only, good retrieval),
        'graph' for graph-augmented (GraphSAGE, experimental).
    """
    from src.pipelines.shared.llm_client import MockLLMClient

    if pipeline_type == "graph":
        from src.pipelines.graph_augmented.pipeline import GraphAugmentedPipeline
        pipeline = GraphAugmentedPipeline.from_config(
            config,
            fallback_to_mock=not use_ollama,
        )
    else:
        from src.pipelines.text_baseline.pipeline import TextBaselinePipeline
        pipeline = TextBaselinePipeline.from_config(
            config,
            fallback_to_mock=not use_ollama,
        )

    if not use_ollama:
        # Keep demo mode deterministic even if a local Ollama server is running.
        pipeline.answer_generator.llm_client = MockLLMClient(
            model="mock-demo-llm",
        )

    return pipeline


# ────────────────────────────────────────────────
#  Display helpers
# ────────────────────────────────────────────────

def show_retrieved_tables(results, record_lookup, top_n=2):
    """Display top retrieved tables in a rich panel."""
    for i, r in enumerate(results[:top_n]):
        rec = record_lookup.get(r.record_id)
        if not rec:
            continue

        t = RichTable(
            title=f"[cyan]#{r.rank} {rec.table_title or rec.id}[/cyan]  "
                  f"[dim](score: {r.score:.3f})[/dim]",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold magenta",
        )
        for col in rec.table_header[:6]:
            t.add_column(str(col), overflow="fold", max_width=18)
        for row in rec.table_rows[:5]:
            t.add_row(*[str(c) for c in row[:6]])
        if len(rec.table_rows) > 5:
            t.add_row(*["..." for _ in rec.table_header[:6]])

        console.print(t)


def show_answer(gen_result, use_mock: bool):
    """Display the final answer in a styled panel."""
    label = "[yellow]Mock Answer[/yellow]" if use_mock else "[green]Mistral Answer[/green]"
    console.print(Panel(
        Text(gen_result.answer, style="bold white"),
        title=f"{label}  [dim]({gen_result.latency_seconds:.2f}s)[/dim]",
        border_style="green" if not use_mock else "yellow",
        padding=(0, 2),
    ))


# ────────────────────────────────────────────────
#  Single query handler
# ────────────────────────────────────────────────

def answer_query(query: str, pipeline, record_lookup: dict, use_mock: bool):
    console.print(f"\n[bold]Query:[/bold] {query}")
    console.print("-" * 60, style="dim")

    # Retrieve
    results = pipeline.retriever.retrieve(query, top_k=3)

    # Show retrieved tables
    console.print(f"[dim]Retrieved {len(results)} tables:[/dim]")
    show_retrieved_tables(results, record_lookup, top_n=2)

    # Generate answer
    gen_result = pipeline.answer_generator.generate(
        query, results, record_lookup
    )

    # If mock, show a helpful note explaining what *would* happen with Mistral
    if use_mock:
        # Construct a meaningful mock answer from the retrieved table
        rec = record_lookup.get(results[0].record_id if results else "")
        if rec:
            gen_result_display = gen_result
            # Override display text with a preview of the context
            gen_result_display.answer = (
                "[Mock mode — no LLM] Retrieved context:\n"
                + rec.table_to_markdown()[:400]
                + ("\n..." if len(rec.table_to_markdown()) > 400 else "")
                + "\n\n>> Start Ollama + run with --use-ollama for real answers."
            )
            show_answer(gen_result_display, use_mock=True)
        else:
            show_answer(gen_result, use_mock=True)
    else:
        show_answer(gen_result, use_mock=False)

    return gen_result


# ────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Table RAG interactive demo")
    parser.add_argument("--use-ollama", action="store_true",
                        help="Use real Mistral via Ollama (must be running)")
    parser.add_argument("--pipeline", choices=["text", "graph"], default="text",
                        help="Pipeline to use: text (accurate retrieval) or graph (experimental)")
    parser.add_argument("--dataset", choices=["wikitq", "tatqa", "finqa"],
                        default=None,
                        help="Load from a real dataset (downloads data)")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Max samples from dataset (default: 50)")
    parser.add_argument("--query", type=str, default=None,
                        help="Single query (non-interactive)")
    args = parser.parse_args()

    # ── Header ───────────────────────────────────
    console.print(Panel(
        "[bold cyan]Graph-Augmented Table Retrieval — Demo[/bold cyan]\n"
        "[dim]MSc Thesis · Gisma University of Applied Sciences[/dim]",
        border_style="cyan",
    ))

    # ── Load config ───────────────────────────────
    from src.utils.config import load_config
    from src.utils.logging import setup_logging
    setup_logging(level="WARNING")   # quiet during demo
    config = load_config(pipeline=args.pipeline)

    # ── Load records ──────────────────────────────
    if args.dataset:
        from src.data.dataset_loader import load_dataset_by_name
        console.print(f"\nLoading {args.max_samples} samples from [bold]{args.dataset}[/bold]...")
        records = load_dataset_by_name(args.dataset, split="test", max_samples=args.max_samples)
        console.print(f"Loaded {len(records)} records.")
    else:
        console.print("\nUsing [bold]built-in sample tables[/bold] (5 tables, no download needed)")
        records = build_sample_records()

    record_lookup = {r.id: r for r in records}

    # ── Build & index ─────────────────────────────
    use_mock = not args.use_ollama
    llm_label = "[green]Mistral (Ollama)[/green]" if not use_mock else "[yellow]Mock LLM[/yellow]"
    pipe_label = "[bold cyan]Text Baseline[/bold cyan]" if args.pipeline == "text" else "[bold magenta]Graph-Augmented[/bold magenta]"
    console.print(f"Pipeline:    {pipe_label}")
    console.print(f"LLM backend: {llm_label}")
    if args.pipeline == "graph":
        console.print("[dim](Note: graph pipeline has untrained GraphSAGE -- retrieval may be inaccurate)[/dim]")
    console.print("\nBuilding pipeline and indexing tables...")

    t0 = time.perf_counter()
    pipeline = build_pipeline(use_ollama=args.use_ollama, config=config, pipeline_type=args.pipeline)
    pipeline.index(records)
    elapsed = time.perf_counter() - t0

    console.print(
        f"[green]Ready![/green] Indexed [bold]{len(pipeline.vector_store)}[/bold] tables "
        f"in {elapsed:.1f}s"
    )

    # Show available tables summary
    console.print("\n[bold]Indexed tables:[/bold]")
    summary_table = RichTable(box=box.SIMPLE, show_header=True, header_style="bold")
    summary_table.add_column("#", style="dim", width=3)
    summary_table.add_column("ID")
    summary_table.add_column("Title / Topic")
    summary_table.add_column("Size", justify="right")
    for i, rec in enumerate(records[:10], 1):
        summary_table.add_row(
            str(i),
            rec.id,
            rec.table_title or rec.question[:50],
            f"{rec.num_rows}×{rec.num_cols}",
        )
    if len(records) > 10:
        summary_table.add_row("...", f"...and {len(records)-10} more", "", "")
    console.print(summary_table)

    # ── Single query mode ─────────────────────────
    if args.query:
        answer_query(args.query, pipeline, record_lookup, use_mock)
        return

    # ── Interactive mode ──────────────────────────
    console.print(
        "\n[bold]Interactive mode[/bold] — type a question, [dim]'quit'[/dim] to exit.\n"
        "[dim]Example queries:[/dim]\n"
        "  • Which country has the highest population?\n"
        "  • What was NVIDIA's revenue growth in 2024?\n"
        "  • Who has the most Olympic gold medals?\n"
        "  • Which programming language pays the most?\n"
        "  • What is Tesla's debt-to-equity ratio?\n"
    )

    while True:
        try:
            console.print("[bold cyan]>[/bold cyan] ", end="")
            query = input().strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        try:
            answer_query(query, pipeline, record_lookup, use_mock)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")

        console.print()


if __name__ == "__main__":
    main()
