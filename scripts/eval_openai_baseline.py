#!/usr/bin/env python3
"""
Design B — Frontier Embedding Baseline (OpenAI text-embedding-3-large)
======================================================================

Compares your hybrid pipeline against OpenAI's commercial embedding model
on the *same retrieval task*, using identical records and metrics.

Why this experiment:
    Frontier proprietary embedders are the de-facto industry baseline.
    Showing your open-source hybrid + rerank reaches comparable accuracy
    at <1% of the cost (and fully local) is a defensible thesis claim.

Cost (n=500, text-embedding-3-large @ $0.13 / 1M tokens):
    ~ 500 tables × 500 tokens + 500 questions × 20 tokens
    ~ 260k tokens  →  ≈ $0.034   (yes, three cents)

Usage:
    # One-time setup:
    pip install openai
    $env:OPENAI_API_KEY = "sk-..."

    # Run:
    python scripts/eval_openai_baseline.py --max-samples 100
    python scripts/eval_openai_baseline.py --max-samples 500
    python scripts/eval_openai_baseline.py --max-samples 500 --model text-embedding-3-small  # cheaper

Output:
    data/results/openai_<model>_wikitq_<n>/retrieval_metrics.json
    data/results/openai_<model>_wikitq_<n>/retrieval_details.json

Compatible with generate_figures_v2.py — same JSON schema as other pipelines.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
from loguru import logger

from src.data.dataset_loader import load_wikitablequestions
from src.pipelines.text_baseline.pipeline import linearise_table
from src.utils.config import load_config


# Pricing (USD / 1M tokens, as of 2026)
PRICING = {
    "text-embedding-3-small":  0.02,
    "text-embedding-3-large":  0.13,
    "text-embedding-ada-002":  0.10,   # legacy
}

DIMENSIONS = {
    "text-embedding-3-small":  1536,
    "text-embedding-3-large":  3072,
    "text-embedding-ada-002":  1536,
}


def embed_batch(client, texts: list[str], model: str, batch_size: int = 100) -> np.ndarray:
    """Embed a list of texts via OpenAI, batching to respect API limits."""
    out = []
    total_tokens = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # OpenAI accepts up to 8191 tokens per text; we truncate roughly at chars
        batch = [t[:30000] for t in batch]
        resp = client.embeddings.create(model=model, input=batch)
        for d in resp.data:
            out.append(d.embedding)
        total_tokens += resp.usage.total_tokens
        logger.info(f"  Embedded {i + len(batch)}/{len(texts)}  (cumulative tokens: {total_tokens:,})")
    return np.asarray(out, dtype=np.float32), total_tokens


def cosine_topk(query_vec: np.ndarray, table_mat: np.ndarray, k: int = 5):
    """Cosine similarity (vectors assumed L2-normalised) returning (indices, scores)."""
    scores = table_mat @ query_vec  # (N,)
    top_idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return top_idx, scores[top_idx]


def main():
    parser = argparse.ArgumentParser(description="OpenAI embedding retrieval baseline")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--model", default="text-embedding-3-large",
                        choices=list(PRICING.keys()))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("Set OPENAI_API_KEY first:  $env:OPENAI_API_KEY = 'sk-...'")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        logger.error("Install openai:  pip install openai")
        sys.exit(1)

    client = OpenAI()
    config = load_config(pipeline="text")  # reuse linearisation settings

    # ── Load data ───────────────────────────────────────────────
    logger.info(f"Loading WikiTableQuestions [{args.split}]...")
    records = load_wikitablequestions(split=args.split)[: args.max_samples]
    n = len(records)
    logger.info(f"Using {n} records")

    # ── Linearise tables ────────────────────────────────────────
    logger.info("Linearising tables...")
    table_texts = [linearise_table(rec, config) for rec in records]
    questions = [rec.question for rec in records]

    # ── Embed via OpenAI ────────────────────────────────────────
    logger.info(f"Embedding {n} tables with {args.model}...")
    t0 = time.perf_counter()
    table_emb, table_tokens = embed_batch(client, table_texts, args.model)
    index_time = time.perf_counter() - t0

    logger.info(f"Embedding {n} questions...")
    t0 = time.perf_counter()
    query_emb, query_tokens = embed_batch(client, questions, args.model)
    query_time = time.perf_counter() - t0

    # L2-normalise (OpenAI vectors are not unit-norm by default)
    table_emb /= np.linalg.norm(table_emb, axis=1, keepdims=True) + 1e-12
    query_emb /= np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-12

    total_tokens = table_tokens + query_tokens
    cost_usd = total_tokens / 1_000_000 * PRICING[args.model]

    # ── Evaluate retrieval ──────────────────────────────────────
    logger.info("Computing retrieval metrics...")
    r1_hits = r5_hits = 0
    rr_sum = 0.0
    details = []

    t0 = time.perf_counter()
    for i, rec in enumerate(records):
        top_idx, top_scores = cosine_topk(query_emb[i], table_emb, k=args.top_k)
        result_ids = [records[j].id for j in top_idx]

        rank = next((p + 1 for p, rid in enumerate(result_ids) if rid == rec.id), None)
        r1 = int(rank == 1)
        r5 = int(rank is not None and rank <= args.top_k)
        rr = 1.0 / rank if rank else 0.0

        r1_hits += r1
        r5_hits += r5
        rr_sum += rr

        details.append({
            "record_id": rec.id,
            "question": rec.question[:120],
            "rank": rank,
            "recall_at_1": r1,
            f"recall_at_{args.top_k}": r5,
            "reciprocal_rank": rr,
            "top_score": float(top_scores[0]),
        })
    eval_time = time.perf_counter() - t0

    metrics = {
        "pipeline":           f"openai_{args.model}",
        "model":              args.model,
        "embedding_dim":      DIMENSIONS[args.model],
        "num_samples":        n,
        "recall_at_1":        r1_hits / n,
        "recall_at_5":        r5_hits / n,
        "mrr":                rr_sum / n,
        "hits_at_1":          r1_hits,
        "hits_at_5":          r5_hits,
        "index_time_seconds": index_time,
        "query_time_seconds": query_time,
        "eval_time_seconds":  eval_time,
        "avg_latency_ms":     (query_time / n) * 1000,
        "total_tokens":       total_tokens,
        "table_tokens":       table_tokens,
        "query_tokens":       query_tokens,
        "cost_usd":           round(cost_usd, 4),
        "cost_per_query_usd": round(cost_usd / n, 6),
    }

    # ── Save ────────────────────────────────────────────────────
    out_dir = Path(args.output or f"data/results/openai_{args.model}_wikitq_{n}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "retrieval_details.json").write_text(json.dumps(details, indent=2))
    logger.info(f"Saved results to {out_dir}")

    # ── Comparison print-out ────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'OPENAI EMBEDDING BASELINE — ' + args.model:^78}")
    print("=" * 78)
    print(f"  n            : {n}")
    print(f"  R@1          : {metrics['recall_at_1']:.3f}  ({r1_hits}/{n})")
    print(f"  R@5          : {metrics['recall_at_5']:.3f}  ({r5_hits}/{n})")
    print(f"  MRR          : {metrics['mrr']:.3f}")
    print(f"  Tokens       : {total_tokens:,}")
    print(f"  Cost         : ${cost_usd:.4f}   (${cost_usd / n * 1000:.2f} per 1k queries)")
    print(f"  Index time   : {index_time:.1f}s")
    print(f"  Query time   : {query_time:.1f}s   ({metrics['avg_latency_ms']:.1f} ms / query)")
    print("-" * 78)
    print("  Compare against your local pipelines:")
    print("    Generic BGE   (n=500): R@1=0.264  R@5=0.564  MRR=0.377  $0.00")
    print("    Fine-tuned BGE(n=500): R@1=0.330  R@5=0.768  MRR=0.499  $0.00")
    print("    Hybrid+Rerank (target): R@1≈0.40  R@5≈0.83   MRR≈0.55   $0.00")
    print("=" * 78)


if __name__ == "__main__":
    main()
