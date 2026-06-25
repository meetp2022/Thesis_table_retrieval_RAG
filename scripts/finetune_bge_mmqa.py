#!/usr/bin/env python3
"""
Script #7 — in-domain BGE fine-tuning on MMQA (de-confound the text leg).

Mirrors scripts/finetune_bge.py (sentence-transformers v5 trainer +
MultipleNegativesRankingLoss, in-batch negatives) but trains on MMQA
query <-> gold-table pairs. Each multi-table question contributes one pair per
gold table. Model selection uses an InformationRetrievalEvaluator over the real
702-table corpus with the dev gold sets.

This produces a text leg trained in-domain on MMQA, so the comparison
"structure-aware retriever vs text+rerank" becomes fair (both legs in-domain).

Outputs:
    models/mmqa_bge_finetuned/best/   — best checkpoint by dev recall@10

Usage (GPU, full):
    python scripts/finetune_bge_mmqa.py --epochs 5 --batch-size 32
Usage (CPU, lighter):
    python scripts/finetune_bge_mmqa.py --epochs 4 --batch-size 16
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

from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from datasets import Dataset as HFDataset

from src.data.mmqa_loader import load_mmqa

BASE_MODEL = "BAAI/bge-base-en-v1.5"
OUTPUT_DIR = "models/mmqa_bge_finetuned"
BEST_DIR = "models/mmqa_bge_finetuned/best"
MAX_ROWS_LINEARISE = 100


def linearise(name, columns, rows):
    parts = [f"Table: {name}",
             "| " + " | ".join(str(c) for c in columns) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows[:MAX_ROWS_LINEARISE]:
        padded = [str(v) if v is not None else "" for v in row]
        padded += [""] * (len(columns) - len(padded))
        parts.append("| " + " | ".join(padded[:len(columns)]) + " |")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"
    logger.info(f"Device: {device}  fp16={use_fp16}")

    data = load_mmqa()
    corpus = data["corpus"]
    table_ids = list(corpus.keys())
    table_text = {t: linearise(corpus[t].name, corpus[t].columns, corpus[t].rows)
                  for t in table_ids}

    # ── Train pairs: (question, gold_table_text), one per gold table ──
    anchors, positives = [], []
    for inst in data["train"]:
        for tid in inst.gold_table_ids:
            if tid in table_text:
                anchors.append(inst.question)
                positives.append(table_text[tid])
    logger.info(f"Train pairs: {len(anchors)} (from {len(data['train'])} queries)")
    train_dataset = HFDataset.from_dict({"anchor": anchors, "positive": positives})

    # ── Dev IR evaluator over the FULL 702-table corpus ──
    dev = data["dev"]
    queries = {str(i): inst.question for i, inst in enumerate(dev)}
    corpus_eval = {t: table_text[t] for t in table_ids}
    relevant = {str(i): set(inst.gold_table_ids) & set(table_ids)
                for i, inst in enumerate(dev)}
    evaluator = InformationRetrievalEvaluator(
        queries=queries, corpus=corpus_eval, relevant_docs=relevant,
        name="mmqa-dev", show_progress_bar=False,
        accuracy_at_k=[1, 5, 10], precision_recall_at_k=[1, 5, 10],
        map_at_k=[10], mrr_at_k=[10], ndcg_at_k=[10])

    model = SentenceTransformer(BASE_MODEL, device=device)
    loss_fn = MultipleNegativesRankingLoss(model)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    targs = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio, fp16=use_fp16,
        logging_steps=50, save_strategy="epoch", eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mmqa-dev_cosine_recall@10",
        greater_is_better=True, report_to="none")

    trainer = SentenceTransformerTrainer(
        model=model, args=targs, train_dataset=train_dataset,
        loss=loss_fn, evaluator=evaluator)

    logger.info("Fine-tuning BGE on MMQA...")
    trainer.train()
    model.save(BEST_DIR)
    logger.info(f"Saved best model to {BEST_DIR}")

    results = evaluator(model)
    (Path(OUTPUT_DIR) / "training_summary.json").write_text(json.dumps({
        "base_model": BASE_MODEL, "train_pairs": len(anchors),
        "dev_queries": len(dev), "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr,
        "final_dev_eval": results}, indent=2, default=str))
    print("\n" + "=" * 60)
    print("  MMQA in-domain BGE fine-tuned")
    print(f"  dev recall@10: {results.get('mmqa-dev_cosine_recall@10')}")
    print(f"  saved: {BEST_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
