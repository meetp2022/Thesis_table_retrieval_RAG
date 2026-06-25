#!/usr/bin/env python3
"""
The Headline Experiment — Hybrid (Fine-tuned BGE + GraphSAGE) + Cross-Encoder Rerank
====================================================================================

This is the experiment that should produce the thesis headline number:

    "Our open-source hybrid pipeline matches or exceeds Google's frontier
     embedder (gemini-embedding-001) at zero per-query cost."

Pipeline configuration:
    Text leg:    Fine-tuned BGE  (models/bge_finetuned/best)
    Graph leg:   Trained GraphSAGE  (models/graph_encoder/best_encoder.pt)
    Fusion:      0.7 · norm(text) + 0.3 · norm(graph)
    Reranker:    cross-encoder/ms-marco-MiniLM-L-6-v2  (top-15 → top-5)

Targets to beat:
    Generic BGE        (n=500): R@1=0.264  R@5=0.564  MRR=0.377
    Fine-tuned BGE     (n=500): R@1=0.330  R@5=0.768  MRR=0.499
    Gemini-embed-001   (n=500): R@1=0.364  R@5=0.770  MRR=0.528  ← frontier
    Hybrid+Rerank target:       R@1≈0.40  R@5≈0.83   MRR≈0.55  ← us

Usage:
    # Smoke test (small scale, fast)
    python scripts/eval_hybrid_finetuned.py --max-samples 50

    # Full headline run
    python scripts/eval_hybrid_finetuned.py --max-samples 500

    # Disable reranker (ablation)
    python scripts/eval_hybrid_finetuned.py --max-samples 500 --no-rerank

    # Sweep fusion weight
    python scripts/eval_hybrid_finetuned.py --max-samples 100 --alpha 0.5
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from loguru import logger

from src.data.dataset_loader import load_dataset_by_name
from src.utils.config import load_config


# Reference numbers from prior runs at n=500
BASELINES = {
    "generic_bge":        {"r1": 0.264, "r5": 0.564, "mrr": 0.377},
    "finetuned_bge":      {"r1": 0.330, "r5": 0.768, "mrr": 0.499},
    "gemini_emb_001":     {"r1": 0.364, "r5": 0.770, "mrr": 0.528},
    "hybrid_rerank_tgt":  {"r1": 0.400, "r5": 0.830, "mrr": 0.550},
}


def pct_change(new: float, base: float) -> str:
    if base == 0:
        return "n/a"
    return f"{(new - base) / base * 100:+.1f}%"


def main():
    parser = argparse.ArgumentParser(description="Hybrid (fine-tuned BGE + GraphSAGE) + Rerank")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--dataset", default="wikitq",
                        choices=["wikitq", "tatqa", "finqa"],
                        help="Dataset to evaluate on (default: wikitq)")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Fusion weight: alpha*text + (1-alpha)*graph (default: 0.7)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true",
                        help="Skip the cross-encoder rerank stage (ablation)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    use_rerank = not args.no_rerank
    n_req = args.max_samples

    # ── Build configs ───────────────────────────────────────────
    # Text leg: fine-tuned BGE  (key change vs eval_retrieval.py's hybrid)
    text_config = load_config(pipeline="text_finetuned")
    graph_config = load_config(pipeline="graph")

    # ── Build hybrid pipeline ───────────────────────────────────
    from src.pipelines.hybrid.pipeline import HybridPipeline

    logger.info("Building hybrid pipeline:")
    logger.info(f"  Text leg:  fine-tuned BGE  ({text_config['embedding']['model']})")
    logger.info(f"  Graph leg: GraphSAGE       (models/graph_encoder/best_encoder.pt)")
    logger.info(f"  Alpha:     {args.alpha}    Reranker: {use_rerank}")

    pipe = HybridPipeline.from_config(
        text_config, graph_config,
        alpha=args.alpha,
        fallback_to_mock=False,   # FAIL LOUDLY if a model is missing — we want the real result
        use_reranker=use_rerank,
    )

    # ── Load data ───────────────────────────────────────────────
    logger.info(f"Loading {args.dataset} [{args.split}]...")
    records = load_dataset_by_name(args.dataset, split=args.split,
                                    max_samples=n_req)
    n = len(records)
    logger.info(f"Using {n} records from {args.dataset}")

    # ── Index ───────────────────────────────────────────────────
    logger.info(f"Indexing {n} tables (this loads BGE + GraphSAGE; ~1-2 min)...")
    t0 = time.perf_counter()
    pipe.index(records)
    index_time = time.perf_counter() - t0
    logger.info(f"  Indexed in {index_time:.1f}s")

    # ── Evaluate retrieval ──────────────────────────────────────
    logger.info(f"Evaluating retrieval (rerank={use_rerank})...")
    r1_hits = r5_hits = 0
    rr_sum = 0.0
    details = []

    t0 = time.perf_counter()
    for i, rec in enumerate(records):
        results = pipe.retriever.retrieve(rec.question, top_k=args.top_k, rerank=use_rerank)
        result_ids = [r.record_id for r in results]

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
            "top_score": results[0].score if results else 0.0,
        })

        if (i + 1) % 25 == 0:
            running_r1 = r1_hits / (i + 1)
            logger.info(f"  Evaluated {i+1}/{n}   (running R@1 = {running_r1:.3f})")

    eval_time = time.perf_counter() - t0

    metrics = {
        "pipeline":           "hybrid_finetuned" + ("_rerank" if use_rerank else ""),
        "text_leg":           "fine-tuned BGE",
        "graph_leg":          "GraphSAGE",
        "alpha":              args.alpha,
        "rerank":             use_rerank,
        "num_samples":        n,
        "recall_at_1":        r1_hits / n,
        "recall_at_5":        r5_hits / n,
        "mrr":                rr_sum / n,
        "hits_at_1":          r1_hits,
        "hits_at_5":          r5_hits,
        "index_time_seconds": index_time,
        "eval_time_seconds":  eval_time,
        "avg_latency_ms":     (eval_time / n) * 1000,
    }

    # ── Save ────────────────────────────────────────────────────
    rerank_tag = "rerank" if use_rerank else "norerank"
    out_dir = Path(args.output or
                   f"data/results/hybrid_finetuned_{rerank_tag}_a{args.alpha}_{args.dataset}_{n}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "retrieval_details.json").write_text(json.dumps(details, indent=2))
    logger.info(f"Saved results to {out_dir}")

    # ── Comparison print-out ────────────────────────────────────
    print("\n" + "=" * 88)
    title = f"HYBRID (FT-BGE + GRAPHSAGE) " + ("+ RERANK" if use_rerank else "(no rerank)")
    print(f"{title:^88}")
    print(f"{'alpha=' + str(args.alpha):^88}")
    print("=" * 88)
    print(f"{'Pipeline':<32} {'R@1':>10} {'R@5':>10} {'MRR':>10} {'Δ R@1 vs Gemini':>18}")
    print("-" * 88)

    rows = [
        ("Generic BGE",                BASELINES["generic_bge"]),
        ("Fine-tuned BGE",             BASELINES["finetuned_bge"]),
        ("Gemini gemini-embedding-001", BASELINES["gemini_emb_001"]),
        ("Target",                     BASELINES["hybrid_rerank_tgt"]),
    ]
    gemini_r1 = BASELINES["gemini_emb_001"]["r1"]
    for name, b in rows:
        delta = pct_change(b["r1"], gemini_r1)
        print(f"  {name:<30} {b['r1']:>10.3f} {b['r5']:>10.3f} {b['mrr']:>10.3f} {delta:>18}")

    print("-" * 88)
    delta = pct_change(metrics["recall_at_1"], gemini_r1)
    print(f"  {'★ THIS RUN ★':<30} {metrics['recall_at_1']:>10.3f} "
          f"{metrics['recall_at_5']:>10.3f} {metrics['mrr']:>10.3f} {delta:>18}")
    print("=" * 88)
    print(f"  n        : {n}")
    print(f"  Index    : {index_time:.1f}s")
    print(f"  Eval     : {eval_time:.1f}s   ({metrics['avg_latency_ms']:.1f} ms / query)")
    print("=" * 88)

    if metrics["recall_at_1"] >= gemini_r1:
        print("\n  🎯 HEADLINE ACHIEVED: open-source hybrid >= Gemini frontier embedder.")
    elif metrics["recall_at_1"] >= BASELINES["finetuned_bge"]["r1"]:
        print(f"\n  ✓ Hybrid beats fine-tuned BGE alone (+{pct_change(metrics['recall_at_1'], BASELINES['finetuned_bge']['r1'])}).")
    else:
        print("\n  ⚠ Hybrid did not beat fine-tuned BGE alone — check graph encoder quality / alpha.")

    gc.collect()


if __name__ == "__main__":
    main()
