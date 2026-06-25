#!/usr/bin/env python3
"""
Train the GraphSAGE encoder with contrastive learning.

Aligns graph-encoded table embeddings with BGE-encoded query embeddings
so that FAISS retrieval works for the graph pipeline.

Usage:
    # Quick test with 50 samples:
    python scripts/train_graph_encoder.py --max-samples 50

    # Full training on WikiTQ train split:
    python scripts/train_graph_encoder.py --dataset wikitq --split train --epochs 20

    # Custom hyperparameters:
    python scripts/train_graph_encoder.py --lr 0.0003 --batch-size 32 --temperature 0.05

References / attribution
------------------------
- GraphSAGE architecture: Hamilton, Ying & Leskovec, "Inductive
  Representation Learning on Large Graphs", NeurIPS 2017. The SAGEConv
  layer used here is from the PyTorch Geometric library.
- InfoNCE / contrastive loss with in-batch negatives: van den Oord et al.,
  "Representation Learning with Contrastive Predictive Coding", 2018.
- The typed table-to-graph schema, node features, the GraphSAGE wrapper
  (src/graph/graph_embedding.py) and this training pipeline are this
  project's own code — applying GraphSAGE to table retrieval is novel here.
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from rich.console import Console
from rich.table import Table as RichTable
from rich import box

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.graph.train import ContrastiveTrainer
from loguru import logger

console = Console(legacy_windows=False, highlight=False)


def main():
    parser = argparse.ArgumentParser(
        description="Train GraphSAGE encoder with contrastive learning"
    )
    parser.add_argument(
        "--dataset", choices=["wikitq", "wikitablequestions", "tatqa", "finqa"],
        default="wikitq",
        help="Training dataset (default: wikitq)",
    )
    parser.add_argument(
        "--split", default="train",
        help="Dataset split to train on (default: train)",
    )
    parser.add_argument(
        "--val-split", default="validation",
        help="Dataset split for validation (default: validation)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit samples for quick testing",
    )
    parser.add_argument(
        "--max-val-samples", type=int, default=None,
        help="Limit validation samples",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Directory to save model (default: models/graph_encoder)",
    )
    parser.add_argument(
        "--hard-negatives", type=int, default=None,
        help="Number of hard negatives per sample (0=disabled, default: 8)",
    )
    parser.add_argument(
        "--config-dir", type=str, default="configs",
    )
    args = parser.parse_args()

    # Setup
    setup_logging()
    config = load_config(pipeline="graph", config_dir=args.config_dir)

    # Override training config from CLI args
    train_cfg = config.setdefault("training", {})
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.temperature is not None:
        train_cfg["temperature"] = args.temperature
    if args.patience is not None:
        train_cfg["patience"] = args.patience
    if args.save_dir is not None:
        train_cfg["save_dir"] = args.save_dir
    if args.hard_negatives is not None:
        train_cfg["num_hard_negatives"] = args.hard_negatives

    # Header
    console.print("\n[bold cyan]GraphSAGE Contrastive Training[/bold cyan]")
    console.print(f"Dataset:     {args.dataset} ({args.split})")
    console.print(f"Epochs:      {train_cfg.get('epochs', 20)}")
    console.print(f"Batch size:  {train_cfg.get('batch_size', 16)}")
    console.print(f"LR:          {train_cfg.get('lr', 1e-4)}")
    console.print(f"Temperature: {train_cfg.get('temperature', 0.07)}")
    console.print("=" * 50)

    # Load data
    from src.data.dataset_loader import load_dataset_by_name

    console.print("\n[dim]Loading training data...[/dim]")
    train_records = load_dataset_by_name(
        args.dataset, split=args.split, max_samples=args.max_samples
    )
    console.print(f"  Train: {len(train_records)} records")

    val_records = None
    try:
        console.print("[dim]Loading validation data...[/dim]")
        val_records = load_dataset_by_name(
            args.dataset, split=args.val_split,
            max_samples=args.max_val_samples or (args.max_samples // 5 if args.max_samples else None),
        )
        console.print(f"  Val:   {len(val_records)} records")
    except Exception as e:
        console.print(f"[yellow]  No validation split available: {e}[/yellow]")
        console.print("  Will auto-split 90/10 from training data")

    # Build trainer
    console.print("\n[dim]Building trainer...[/dim]")
    trainer = ContrastiveTrainer.from_config(config)

    # Train
    console.print("\n[bold green]Starting training...[/bold green]\n")
    result = trainer.train(train_records, val_records)

    # Display results
    console.print(f"\n[bold green]{result.summary()}[/bold green]")

    table = RichTable(title="Training History", box=box.SIMPLE)
    table.add_column("Epoch", style="dim", justify="right")
    table.add_column("Train Loss", justify="right")
    table.add_column("Val Loss", justify="right")
    table.add_column("LR", justify="right")
    table.add_column("Time", justify="right")

    for h in result.history:
        table.add_row(
            str(int(h["epoch"])),
            f"{h['train_loss']:.4f}",
            f"{h['val_loss']:.4f}",
            f"{h['lr']:.6f}",
            f"{h['time']:.1f}s",
        )

    console.print(table)

    # Save training log
    save_dir = train_cfg.get("save_dir", "models/graph_encoder")
    log_path = Path(save_dir) / "training_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "split": args.split,
            "num_train": len(train_records),
            "num_val": len(val_records) if val_records else "auto-split",
            "config": {k: v for k, v in train_cfg.items()},
            "result": {
                "epochs_completed": result.epochs_completed,
                "best_val_loss": result.best_val_loss,
                "final_train_loss": result.final_train_loss,
                "time_seconds": result.training_time_seconds,
            },
            "history": result.history,
        }, f, indent=2)
    console.print(f"\nTraining log saved to {log_path}")

    console.print(
        f"\n[bold]Next step:[/bold] Run the demo with the trained encoder:\n"
        f"  python scripts/demo.py --pipeline graph"
    )


if __name__ == "__main__":
    main()
