#!/usr/bin/env python3
"""
Design B — Frontier Embedding Baseline (Google Gemini)
======================================================

Compares your hybrid pipeline against Google's commercial embedding model
on the *same retrieval task*, using identical records and metrics.

Why Gemini rather than OpenAI:
    Google AI Studio offers a free tier — text-embedding-004 is free up to
    1,500 requests/day. The full n=500 run costs $0 instead of $0.04.

Models supported:
    text-embedding-004      768-d   FREE TIER (1500 req/day)
    gemini-embedding-001    3072-d  PAID — current Gemini SOTA embedder

Usage:
    # One-time setup:
    pip install google-genai
    $env:GEMINI_API_KEY = "AIza..."   # from https://aistudio.google.com/apikey

    # Run:
    python scripts/eval_gemini_baseline.py --max-samples 100
    python scripts/eval_gemini_baseline.py --max-samples 500
    python scripts/eval_gemini_baseline.py --max-samples 500 --model gemini-embedding-001

Output:
    data/results/gemini_<model>_wikitq_<n>/retrieval_metrics.json
    data/results/gemini_<model>_wikitq_<n>/retrieval_details.json

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

import hashlib

import numpy as np
from loguru import logger

from src.data.dataset_loader import load_wikitablequestions
from src.pipelines.text_baseline.pipeline import linearise_table
from src.utils.config import load_config


# ── Embedding cache ──────────────────────────────────────────
# Embeddings are a deterministic function of (model, task_type, text):
# the same input always yields the same vector. We therefore cache each
# vector on disk, keyed by a SHA-256 hash of those three components.
#
# Why this is safe and standard:
#   * A cached vector is byte-identical to a freshly computed one — caching
#     changes nothing about the results, only avoids redundant API calls.
#   * The key includes the model and task type, so changing either (or the
#     table linearisation, which changes the text) correctly misses the
#     cache and recomputes.
#   * A crashed run loses nothing: every successful call is already saved,
#     so a re-run resumes instead of starting over.
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "gemini_embeddings"


def _cache_path(text: str, model: str, task_type: str) -> Path:
    """Content-addressed cache path for one (model, task_type, text) triple."""
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(task_type.encode("utf-8"))
    h.update(b"\x00")
    h.update((text or " ").encode("utf-8"))
    return CACHE_DIR / f"{h.hexdigest()}.npy"


# Pricing (USD / 1M tokens, as of 2026)
# AI Studio free tier covers all of these within rate limits → $0
PRICING = {
    "gemini-embedding-001":       0.00,   # Gemini SOTA embedder (3072-d)
    "gemini-embedding-2":         0.00,   # Newer Gemini embedder
    "gemini-embedding-2-preview": 0.00,   # Preview tier
}

DIMENSIONS = {
    "gemini-embedding-001":       3072,
    "gemini-embedding-2":         3072,
    "gemini-embedding-2-preview": 3072,
}

# Free-tier rate limits (requests per minute) — conservative
RATE_LIMITS = {
    "gemini-embedding-001":       100,
    "gemini-embedding-2":         100,
    "gemini-embedding-2-preview":  60,
}


def embed_one(client, text: str, model: str, task_type: str, max_retries: int = 5):
    """
    Embed a single text via Gemini, with exponential back-off on 429.

    Returns (vector, from_cache): from_cache is True when the vector was
    loaded from the on-disk cache and no API call was made.
    """
    text = text[:30000] if text else " "

    # 1. Cache hit — no API call needed
    cpath = _cache_path(text, model, task_type)
    if cpath.exists():
        try:
            return np.load(cpath), True
        except Exception:
            pass  # corrupt cache entry → fall through and recompute

    # 2. Cache miss — call the API, then save the result
    backoff = 5.0
    for attempt in range(max_retries):
        try:
            resp = client.models.embed_content(
                model=model,
                contents=text,
                config={"task_type": task_type},
            )
            vec = np.asarray(resp.embeddings[0].values, dtype=np.float32)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            # Use a tmp name that ends in .npy so numpy does not silently
            # append another .npy extension to it.
            tmp = cpath.with_suffix(".tmp.npy")
            np.save(tmp, vec)
            tmp.replace(cpath)  # atomic write — never leaves a partial file
            return vec, False
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"  Rate-limited; sleeping {backoff:.0f}s then retrying...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
            else:
                raise


def embed_all(client, texts: list[str], model: str, task_type: str, rpm: int) -> np.ndarray:
    """Embed a list of texts, rate-limited to `rpm` requests per minute.

    Cache hits skip the rate-limit delay (no API call was made)."""
    out = []
    delay = 60.0 / rpm  # seconds between calls
    n_cached = n_api = 0
    for i, text in enumerate(texts):
        t0 = time.perf_counter()
        vec, from_cache = embed_one(client, text, model, task_type)
        out.append(vec)
        if from_cache:
            n_cached += 1
        else:
            n_api += 1

        if (i + 1) % 25 == 0 or (i + 1) == len(texts):
            logger.info(f"  Embedded {i + 1}/{len(texts)}  "
                        f"(cache hits: {n_cached}, API calls: {n_api})")

        # rate-limit: only sleep after a real API call, never after a cache hit
        if not from_cache:
            elapsed = time.perf_counter() - t0
            if elapsed < delay and i + 1 < len(texts):
                time.sleep(delay - elapsed)
    return np.asarray(out, dtype=np.float32)


def cosine_topk(query_vec: np.ndarray, table_mat: np.ndarray, k: int = 5):
    """Cosine similarity (vectors L2-normalised) returning (indices, scores)."""
    scores = table_mat @ query_vec
    top_idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return top_idx, scores[top_idx]


def main():
    parser = argparse.ArgumentParser(description="Gemini embedding retrieval baseline")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--model", default="gemini-embedding-001",
                        choices=list(PRICING.keys()))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--rpm", type=int, default=None,
                        help="Override requests-per-minute rate limit")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error(
            "Set GEMINI_API_KEY first:\n"
            "  $env:GEMINI_API_KEY = 'AIza...'\n"
            "Get one free at https://aistudio.google.com/apikey"
        )
        sys.exit(1)

    try:
        from google import genai
    except ImportError:
        logger.error("Install google-genai:  pip install google-genai")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    config = load_config(pipeline="text")  # reuse linearisation settings

    rpm = args.rpm or RATE_LIMITS[args.model]
    logger.info(f"Using model {args.model}  (rate-limited to {rpm} RPM)")

    # ── Load data ───────────────────────────────────────────────
    logger.info(f"Loading WikiTableQuestions [{args.split}]...")
    records = load_wikitablequestions(split=args.split)[: args.max_samples]
    n = len(records)
    logger.info(f"Using {n} records")

    # ── Linearise tables ────────────────────────────────────────
    logger.info("Linearising tables...")
    table_texts = [linearise_table(rec, config) for rec in records]
    questions = [rec.question for rec in records]

    # Gemini supports task-specific embeddings — improves retrieval quality
    # by embedding documents and queries with different "intents"
    # https://ai.google.dev/gemini-api/docs/embeddings#task-types
    TASK_DOC = "RETRIEVAL_DOCUMENT"
    TASK_QUERY = "RETRIEVAL_QUERY"

    # Estimate runtime so user knows what to expect
    total_calls = 2 * n
    est_minutes = total_calls / rpm
    logger.info(
        f"Estimated runtime: ~{est_minutes:.1f} minutes  "
        f"({total_calls} API calls @ {rpm} RPM)"
    )

    # ── Embed via Gemini ────────────────────────────────────────
    logger.info(f"Embedding {n} tables...")
    t0 = time.perf_counter()
    table_emb = embed_all(client, table_texts, args.model, TASK_DOC, rpm)
    index_time = time.perf_counter() - t0

    logger.info(f"Embedding {n} questions...")
    t0 = time.perf_counter()
    query_emb = embed_all(client, questions, args.model, TASK_QUERY, rpm)
    query_time = time.perf_counter() - t0

    # L2-normalise (Gemini doesn't guarantee unit-norm)
    table_emb /= np.linalg.norm(table_emb, axis=1, keepdims=True) + 1e-12
    query_emb /= np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-12

    # Rough token estimate (1 token ≈ 4 chars for English)
    total_chars = sum(len(t) for t in table_texts) + sum(len(q) for q in questions)
    est_tokens = total_chars // 4
    cost_usd = est_tokens / 1_000_000 * PRICING[args.model]

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
        "pipeline":           f"gemini_{args.model}",
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
        "estimated_tokens":   est_tokens,
        "cost_usd":           round(cost_usd, 4),
        "cost_per_query_usd": round(cost_usd / n, 6) if n else 0,
    }

    # ── Save ────────────────────────────────────────────────────
    out_dir = Path(args.output or f"data/results/gemini_{args.model}_wikitq_{n}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "retrieval_details.json").write_text(json.dumps(details, indent=2))
    logger.info(f"Saved results to {out_dir}")

    # ── Comparison print-out ────────────────────────────────────
    free_tag = " (FREE TIER)" if PRICING[args.model] == 0 else ""
    print("\n" + "=" * 78)
    print(f"{'GEMINI EMBEDDING BASELINE — ' + args.model + free_tag:^78}")
    print("=" * 78)
    print(f"  n            : {n}")
    print(f"  R@1          : {metrics['recall_at_1']:.3f}  ({r1_hits}/{n})")
    print(f"  R@5          : {metrics['recall_at_5']:.3f}  ({r5_hits}/{n})")
    print(f"  MRR          : {metrics['mrr']:.3f}")
    print(f"  Est. tokens  : {est_tokens:,}")
    print(f"  Cost         : ${cost_usd:.4f}{free_tag}")
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
