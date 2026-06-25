"""
Retrieval evaluation for the Image Baseline pipeline (Pipeline 2).

Mirrors the evaluation protocol used by :mod:`scripts.eval_retrieval` for
text / graph / hybrid pipelines, so the resulting ``retrieval_metrics.json``
is directly comparable and can be plotted alongside the others.

Metrics: R@1, R@5, MRR, index time, eval time, avg latency.

Example
-------
    python scripts/eval_image_baseline.py \
        --dataset wikitablequestions \
        --max-samples 100 \
        --top-k 5 \
        --out data/results/image_wikitq_100
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.data.dataset_loader import load_wikitablequestions
from src.pipelines.image_baseline.pipeline import ImageBaselinePipeline
from src.utils.config import load_config


# ────────────────────────────────────────────────
#  Retrieval metric helpers
# ────────────────────────────────────────────────

def recall_at_k(pred_ids: List[str], gold_id: str, k: int) -> int:
    return 1 if gold_id in pred_ids[:k] else 0


def reciprocal_rank(pred_ids: List[str], gold_id: str) -> float:
    for rank, pid in enumerate(pred_ids, start=1):
        if pid == gold_id:
            return 1.0 / rank
    return 0.0


# ────────────────────────────────────────────────
#  Dataset loader (shared with other eval scripts)
# ────────────────────────────────────────────────

def load_records(dataset: str, split: str):
    # Accept both the short alias ("wikitq") and the full name
    # ("wikitablequestions") so this script is consistent with
    # eval_retrieval.py and the other eval scripts.
    if dataset in ("wikitablequestions", "wikitq"):
        return load_wikitablequestions(split=split)
    raise ValueError(f"Unsupported dataset: {dataset}")


# ────────────────────────────────────────────────
#  Main eval
# ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wikitq",
                        choices=["wikitq", "wikitablequestions"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--backend",
        choices=["clip", "colpali"],
        default="clip",
        help="Image encoder backend (clip runs on CPU, colpali needs GPU).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Image-encoder batch size (keep small on CPU to avoid OOM).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for retrieval_metrics.json + retrieval_details.json",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Config ─────────────────────────────────
    config = load_config("image")
    # Apply backend override
    config.setdefault("image_encoding", {})["backend"] = args.backend
    # Force conservative batch size for CPU safety (CLIP vision encoder
    # segfaults on Windows with batch_size >= 8 on CPU for some tables).
    config.setdefault("embedding", {})["batch_size"] = args.batch_size

    # Reduce per-image memory footprint (CPU-constrained laptop)
    config.setdefault("table_rendering", {}).update({
        "dpi": 72,
        "font_size": 8,
        "max_rows": 20,
        "max_cols": 10,
    })
    # If ColPali: swap the default model + dim
    if args.backend == "colpali":
        config["image_encoding"].setdefault("model", "vidore/colpali-v1.2")
        config["image_encoding"].setdefault("embedding_dimension", 128)

    logger.info(
        f"Image baseline eval — backend={args.backend} "
        f"n={args.max_samples} top_k={args.top_k}"
    )

    # ── Load data ──────────────────────────────
    all_records = load_records(args.dataset, args.split)
    records = all_records[: args.max_samples]
    logger.info(f"Loaded {len(records)} records from {args.dataset}/{args.split}")

    # ── Build pipeline ─────────────────────────
    pipe = ImageBaselinePipeline.from_config(config)

    # ── Index ──────────────────────────────────
    t_index_start = time.perf_counter()
    pipe.index(records)
    index_time = time.perf_counter() - t_index_start

    # ── Evaluate ───────────────────────────────
    t_eval_start = time.perf_counter()
    questions = [r.question for r in records]
    gold_ids = [r.id for r in records]

    batch_results = pipe.retriever.retrieve_batch(questions, top_k=args.top_k)
    eval_time = time.perf_counter() - t_eval_start

    # ── Metrics ────────────────────────────────
    r1_hits = 0
    r5_hits = 0
    rr_sum = 0.0
    details = []

    for q, gold, retrieved in zip(questions, gold_ids, batch_results):
        pred_ids = [rid for rid, _ in retrieved]
        r1 = recall_at_k(pred_ids, gold, 1)
        r5 = recall_at_k(pred_ids, gold, args.top_k)
        rr = reciprocal_rank(pred_ids, gold)

        r1_hits += r1
        r5_hits += r5
        rr_sum += rr

        details.append(
            {
                "question": q,
                "gold_id": gold,
                "retrieved": [
                    {"id": rid, "score": float(score)}
                    for rid, score in retrieved
                ],
                "recall_at_1": r1,
                "recall_at_5": r5,
                "reciprocal_rank": rr,
            }
        )

    n = len(records)
    metrics = {
        "pipeline": f"image_{args.backend}",
        "num_samples": n,
        "recall_at_1": r1_hits / n if n else 0.0,
        "recall_at_5": r5_hits / n if n else 0.0,
        "mrr": rr_sum / n if n else 0.0,
        "hits_at_1": r1_hits,
        "hits_at_5": r5_hits,
        "index_time_seconds": index_time,
        "eval_time_seconds": eval_time,
        "avg_latency_ms": (eval_time / n * 1000) if n else 0.0,
    }

    # ── Save ───────────────────────────────────
    with open(out_dir / "retrieval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "retrieval_details.json", "w") as f:
        json.dump(details, f, indent=2)

    logger.info("────────────────────────────────────────")
    logger.info(f"R@1      = {metrics['recall_at_1']:.4f}")
    logger.info(f"R@5      = {metrics['recall_at_5']:.4f}")
    logger.info(f"MRR      = {metrics['mrr']:.4f}")
    logger.info(f"Index    = {metrics['index_time_seconds']:.1f}s")
    logger.info(f"Eval     = {metrics['eval_time_seconds']:.1f}s  "
                f"({metrics['avg_latency_ms']:.1f} ms/query)")
    logger.info(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
