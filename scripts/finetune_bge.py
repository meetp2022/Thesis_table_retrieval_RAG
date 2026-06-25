#!/usr/bin/env python3
"""
BGE Domain Fine-Tuning on WikiTableQuestions.

Uses sentence-transformers v5 API (SentenceTransformerTrainer) with
MultipleNegativesRankingLoss and in-batch negatives.

Training pairs: (question, linearised_table_markdown) from WikiTQ train split.
Val pairs:      first 500 from validation split.

Outputs
-------
    models/bge_finetuned/best/        — best checkpoint by val InformationRetrievalEvaluator
    models/bge_finetuned/final/       — final epoch weights
    logs/bge_finetune.log             — tee'd training log

Usage (local CPU, small-scale smoke test):
    python scripts/finetune_bge.py --max-train 500 --max-val 50 --epochs 1 --batch-size 4

Usage (GCP T4 GPU, full run):
    python scripts/finetune_bge.py --max-train 15000 --max-val 500 --epochs 5 --batch-size 32

Requirements:
    pip install "sentence-transformers>=3.0" "datasets==2.20.0" transformers accelerate

References / attribution
------------------------
- Fine-tuning recipe (SentenceTransformerTrainer + MultipleNegativesRankingLoss)
  adapted from the official sentence-transformers training documentation:
  https://www.sbert.net/docs/sentence_transformer/training_overview.html
- MultipleNegativesRankingLoss (in-batch negatives) is based on:
  Henderson et al., "Efficient Natural Language Response Suggestion", 2017.
- Base model BAAI/bge-base-en-v1.5: Xiao et al., "C-Pack / BGE", 2023.
- The dataset preparation, table linearisation, evaluation wiring and CLI
  in this file are this project's own code.
"""

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch
from loguru import logger

# ─── Sentence-transformers v5 imports ─────────────────────────────────────────
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from datasets import Dataset as HFDataset

# ─── Project imports ──────────────────────────────────────────────────────────
from src.data.dataset_loader import load_wikitablequestions

# ─── Constants ────────────────────────────────────────────────────────────────
BASE_MODEL  = "BAAI/bge-base-en-v1.5"
OUTPUT_DIR  = "models/bge_finetuned"
BEST_DIR    = "models/bge_finetuned/best"
FINAL_DIR   = "models/bge_finetuned/final"


def linearise_table(record) -> str:
    """Convert a TableQARecord to a markdown string (mirrors Pipeline 1)."""
    rows = []
    if record.table_title:
        rows.append(f"# {record.table_title}")
    # headers
    headers = record.table_data[0] if record.table_data else []
    rows.append(" | ".join(str(h) for h in headers))
    rows.append(" | ".join(["---"] * len(headers)))
    for row in record.table_data[1:]:
        rows.append(" | ".join(str(c) for c in row))
    return "\n".join(rows)


def build_pairs(records, max_pairs: int):
    """Build (question, table_text) pairs from records."""
    pairs = []
    for rec in records:
        if len(pairs) >= max_pairs:
            break
        try:
            tbl = linearise_table(rec)
            pairs.append((rec.question, tbl))
        except Exception as e:
            logger.warning(f"Skipping {rec.id}: {e}")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=15_000)
    parser.add_argument("--max-val",   type=int, default=500)
    parser.add_argument("--epochs",    type=int, default=5)
    parser.add_argument("--batch-size",type=int, default=32)
    parser.add_argument("--lr",        type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--fp16",      action="store_true", default=True)
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}  |  fp16={args.fp16 and device=='cuda'}")

    # ── Load data ────────────────────────────────────────────────────────────
    logger.info("Loading WikiTableQuestions train split...")
    train_records = load_wikitablequestions(split="train")
    logger.info("Loading WikiTableQuestions validation split...")
    val_records   = load_wikitablequestions(split="validation")

    logger.info(f"Building up to {args.max_train} training pairs...")
    train_pairs = build_pairs(train_records, args.max_train)
    logger.info(f"Building up to {args.max_val} validation pairs...")
    val_pairs   = build_pairs(val_records,   args.max_val)

    logger.info(f"Train pairs: {len(train_pairs)}  |  Val pairs: {len(val_pairs)}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_dataset = HFDataset.from_dict({
        "anchor":   [q for q, _ in train_pairs],
        "positive": [t for _, t in train_pairs],
    })

    # ── IR Evaluator ─────────────────────────────────────────────────────────
    queries     = {str(i): q for i, (q, _) in enumerate(val_pairs)}
    corpus      = {str(i): t for i, (_, t) in enumerate(val_pairs)}
    relevant    = {str(i): {str(i)} for i in range(len(val_pairs))}

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant,
        name="wikitq-val",
        show_progress_bar=False,
    )

    # ── Model & loss ─────────────────────────────────────────────────────────
    logger.info(f"Loading base model: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL, device=device)
    loss_fn = MultipleNegativesRankingLoss(model)

    # ── Training arguments ────────────────────────────────────────────────────
    use_fp16 = args.fp16 and device == "cuda"
    training_args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        fp16=use_fp16,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="wikitq-val_cosine_mrr@10",
        greater_is_better=True,
        report_to="none",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss_fn,
        evaluator=evaluator,
    )

    logger.info("Starting fine-tuning...")
    trainer.train()

    # ── Save final ───────────────────────────────────────────────────────────
    model.save(FINAL_DIR)
    logger.info(f"Final model saved to {FINAL_DIR}")

    # Best model was loaded by load_best_model_at_end
    model.save(BEST_DIR)
    logger.info(f"Best model saved to {BEST_DIR}")

    # ── Final eval summary ────────────────────────────────────────────────────
    results = evaluator(model)
    summary = {
        "base_model": BASE_MODEL,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "fp16": use_fp16,
        "final_eval": results,
    }
    with open(f"{OUTPUT_DIR}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training summary saved to {OUTPUT_DIR}/training_summary.json")
    logger.info(f"Final IR eval: {results}")


if __name__ == "__main__":
    main()
