#!/usr/bin/env python3
"""
Phase 1 — MMQA multi-table retrieval baselines (V1 text, V2 text+rerank).

Retrieves over the pooled 702-table MMQA corpus and scores with the set-aware
multi-table metrics. This is the harness that V3/V4 (graph variants) will plug
into; here it establishes the text-only and text+rerank baselines, plus the
first-stage recall ceiling that tells us a sane rerank depth.

Variants:
    V1  (default)        : embed corpus + queries, cosine rank.       [text only]
    V2  (--rerank)       : V1 candidates -> cross-encoder rerank.     [text+rerank]

Encoders:
    --encoder generic    : BAAI/bge-base-en-v1.5            (no fine-tuning)
    --encoder finetuned  : models/bge_finetuned/best        (WikiTQ fine-tuned)

Usage:
    python scripts/eval_mmqa_retrieval.py --encoder generic --split dev
    python scripts/eval_mmqa_retrieval.py --encoder generic --split test --rerank
"""

from __future__ import annotations

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

import numpy as np
from loguru import logger

from src.evaluation.multitable_metrics import evaluate_multitable, full_set_recall_at_k

PROC = Path("data/processed/mmqa")
ENCODERS = {
    "generic":   "BAAI/bge-base-en-v1.5",
    "finetuned": "models/bge_finetuned/best",       # WikiTQ-finetuned (cross-domain)
    "mmqa":      "models/mmqa_bge_finetuned/best",  # in-domain MMQA-finetuned
}
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_ROWS_LINEARISE = 100


def linearise_mmqa_table(name: str, columns, rows) -> str:
    """Markdown linearisation mirroring the single-table pipeline format."""
    parts = [f"Table: {name}"]
    parts.append("| " + " | ".join(str(c) for c in columns) + " |")
    parts.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows[:MAX_ROWS_LINEARISE]:
        padded = [str(v) if v is not None else "" for v in row]
        padded += [""] * (len(columns) - len(padded))
        parts.append("| " + " | ".join(padded[:len(columns)]) + " |")
    if len(rows) > MAX_ROWS_LINEARISE:
        parts.append(f"[... {len(rows) - MAX_ROWS_LINEARISE} more rows truncated ...]")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description="MMQA multi-table retrieval baselines")
    ap.add_argument("--encoder", choices=list(ENCODERS), default="generic")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--rerank", action="store_true", help="V2: cross-encoder rerank")
    ap.add_argument("--rerank-depth", type=int, default=50,
                    help="How many top candidates to rerank (V2)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    # ── Load corpus + instances ──────────────────
    logger.info(f"Loading MMQA corpus + {args.split} instances...")
    corpus = json.loads((PROC / "corpus.json").read_text(encoding="utf-8"))
    instances = json.loads((PROC / f"instances_{args.split}.json").read_text(encoding="utf-8"))
    if args.max_samples:
        instances = instances[:args.max_samples]

    table_ids = list(corpus.keys())
    logger.info(f"  Corpus: {len(table_ids)} tables   Queries: {len(instances)}")

    # ── Linearise corpus tables ──────────────────
    table_texts = [
        linearise_mmqa_table(corpus[tid]["name"], corpus[tid]["columns"], corpus[tid]["rows"])
        for tid in table_ids
    ]

    # ── Encode ───────────────────────────────────
    from sentence_transformers import SentenceTransformer
    model_name = ENCODERS[args.encoder]
    logger.info(f"Encoding with {model_name} ...")
    model = SentenceTransformer(model_name)

    t0 = time.perf_counter()
    table_emb = model.encode(table_texts, batch_size=64, normalize_embeddings=True,
                             show_progress_bar=False, convert_to_numpy=True)
    index_time = time.perf_counter() - t0

    questions = [inst["question"] for inst in instances]
    t0 = time.perf_counter()
    query_emb = model.encode(questions, batch_size=64, normalize_embeddings=True,
                             show_progress_bar=False, convert_to_numpy=True)
    query_time = time.perf_counter() - t0

    # ── Rank (cosine = dot, vectors normalised) ──
    sims = query_emb @ table_emb.T                      # (Q, T)
    ranked_idx = np.argsort(-sims, axis=1)              # descending

    # ── Optional cross-encoder rerank (V2) ───────
    reranker = None
    if args.rerank:
        from sentence_transformers import CrossEncoder
        logger.info(f"Reranking top-{args.rerank_depth} with {CROSS_ENCODER} ...")
        reranker = CrossEncoder(CROSS_ENCODER)

    per_query_rankings = []
    rerank_time = 0.0
    for qi, inst in enumerate(instances):
        ranked_ids = [table_ids[j] for j in ranked_idx[qi]]
        if reranker is not None:
            depth = min(args.rerank_depth, len(ranked_ids))
            cand = ranked_ids[:depth]
            pairs = [(questions[qi], table_texts[table_ids.index(c)]) for c in cand]
            t0 = time.perf_counter()
            scores = reranker.predict(pairs, show_progress_bar=False)
            rerank_time += time.perf_counter() - t0
            order = np.argsort(-np.asarray(scores))
            reranked = [cand[o] for o in order]
            ranked_ids = reranked + ranked_ids[depth:]   # keep tail order
        per_query_rankings.append(ranked_ids)
        if (qi + 1) % 100 == 0:
            logger.info(f"  ranked {qi+1}/{len(instances)}")

    per_query_gold = [inst["gold_table_ids"] for inst in instances]

    # ── Metrics ──────────────────────────────────
    metrics = evaluate_multitable(per_query_rankings, per_query_gold,
                                  ks=(1, 5, 10, 20, 50), primary_k=10)

    # First-stage full-set recall ceiling (before rerank) — for rerank-depth choice
    fs_ceiling = {}
    for k in (10, 20, 50, 100):
        vals = [full_set_recall_at_k([table_ids[j] for j in ranked_idx[qi]],
                                     per_query_gold[qi], k)
                for qi in range(len(instances))]
        fs_ceiling[k] = sum(vals) / len(vals) if vals else 0.0

    variant = f"V2_text_rerank" if args.rerank else "V1_text"
    out = {
        "variant": variant,
        "encoder": args.encoder,
        "encoder_model": model_name,
        "split": args.split,
        "rerank": args.rerank,
        "rerank_depth": args.rerank_depth if args.rerank else None,
        "num_corpus_tables": len(table_ids),
        "num_queries": metrics["num_queries"],
        "full_set_recall": metrics["full_set_recall"],
        "per_table_recall": metrics["per_table_recall"],
        "mrr": metrics["mrr"],
        "coverage_k_median": metrics["coverage_k_median"],
        "first_stage_full_set_recall_ceiling": fs_ceiling,
        "index_time_seconds": index_time,
        "query_time_seconds": query_time,
        "rerank_time_seconds": rerank_time,
    }

    out_dir = Path(args.output or
                   f"data/results/mmqa_{variant}_{args.encoder}_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval_metrics.json").write_text(json.dumps(out, indent=2))
    (out_dir / "per_query.json").write_text(json.dumps({
        "gold": per_query_gold,
        "per_query": metrics["per_query"],
    }, indent=2))
    logger.info(f"Saved to {out_dir}")

    # ── Print ────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"  MMQA {variant}  ·  encoder={args.encoder}  ·  split={args.split}")
    print("=" * 78)
    print(f"  Corpus tables : {len(table_ids)}    Queries: {metrics['num_queries']}")
    print(f"  {'k':>4} | {'Full-set R@k':>14} | {'Per-table R@k':>14}")
    print("  " + "-" * 40)
    for k in (1, 5, 10, 20, 50):
        print(f"  {k:>4} | {metrics['full_set_recall'][k]:>14.3f} | "
              f"{metrics['per_table_recall'][k]:>14.3f}")
    print(f"\n  MRR (first gold)     : {metrics['mrr']:.3f}")
    print(f"  Coverage-k (median)  : {metrics['coverage_k_median']}")
    print(f"\n  First-stage Full-set Recall ceiling (pre-rerank):")
    for k, v in fs_ceiling.items():
        print(f"      @{k:<4}: {v:.3f}")
    print("=" * 78)
    print(f"  PRIMARY METRIC — Full-set Recall@10 = "
          f"{metrics['full_set_recall'][10]:.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
