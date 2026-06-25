#!/usr/bin/env python3
"""
Evaluate the fine-tuned BGE model against the generic BGE baseline.

Runs retrieval at both n=100 and n=500, saves results compatible with
generate_figures_v2.py, and prints a side-by-side comparison table.

Usage:
    python scripts/eval_finetuned_bge.py

Prerequisites:
    models/bge_finetuned/best/   must exist (download from GCP with
    scripts/gcp_download_model.sh first).
"""

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
from src.data.dataset_loader import load_wikitablequestions
from src.utils.config import load_config


BEST_MODEL_DIR = Path("models/bge_finetuned/best")
# Allow CLI override: python eval_finetuned_bge.py 100   or   ... 500
SCALES = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [100, 500]
OUT_DIRS = {
    100: Path("data/results/text_finetuned_wikitq_100"),
    500: Path("data/results/text_finetuned_wikitq_500"),
}

# Generic BGE baseline results for comparison
BASELINE_RESULTS = {
    100: {"recall_at_1": 0.550, "recall_at_5": 0.820, "mrr": 0.657},
    500: {"recall_at_1": 0.252, "recall_at_5": 0.632, "mrr": 0.397},
}


def run_eval(records, config, top_k: int = 5):
    from src.pipelines.text_baseline.pipeline import TextBaselinePipeline
    pipe = TextBaselinePipeline.from_config(config, fallback_to_mock=False)

    logger.info(f"Indexing {len(records)} tables...")
    t0 = time.perf_counter()
    pipe.index(records)
    index_time = time.perf_counter() - t0

    logger.info("Evaluating retrieval...")
    r1_hits = r5_hits = 0
    rr_sum = 0.0
    details = []

    t_eval = time.perf_counter()
    for rec in records:
        results = pipe.retriever.retrieve(rec.question, top_k=top_k)
        result_ids = [r.record_id for r in results]

        rank = next((i + 1 for i, rid in enumerate(result_ids) if rid == rec.id), None)
        r1 = int(rank == 1)
        r5 = int(rank is not None and rank <= top_k)
        rr = 1.0 / rank if rank else 0.0

        r1_hits += r1
        r5_hits += r5
        rr_sum += rr

        details.append({
            "record_id": rec.id,
            "question": rec.question[:120],
            "recall_at_1": r1,
            f"recall_at_{top_k}": r5,
            "reciprocal_rank": rr,
        })

    eval_time = time.perf_counter() - t_eval
    n = len(records)
    metrics = {
        "pipeline": "text_finetuned",
        "num_samples": n,
        "recall_at_1":  r1_hits / n,
        "recall_at_5":  r5_hits / n,
        "mrr":          rr_sum / n,
        "hits_at_1":    r1_hits,
        "hits_at_5":    r5_hits,
        "index_time_seconds": index_time,
        "eval_time_seconds":  eval_time,
        "avg_latency_ms": eval_time / n * 1000,
    }
    return metrics, details


def pct_change(new, base):
    if base == 0:
        return "n/a"
    return f"{(new - base) / base * 100:+.1f}%"


def main():
    if not BEST_MODEL_DIR.exists():
        logger.error(
            f"Fine-tuned model not found at {BEST_MODEL_DIR}\n"
            "Download it first with: bash scripts/gcp_download_model.sh"
        )
        sys.exit(1)

    logger.info(f"Loading fine-tuned model from {BEST_MODEL_DIR}")

    config = load_config(pipeline="text_finetuned")

    logger.info("Loading WikiTableQuestions validation split...")
    all_records = load_wikitablequestions(split="validation")

    print("\n" + "=" * 72)
    print(f"{'FINE-TUNED BGE EVALUATION':^72}")
    print("=" * 72)
    print(f"{'n':>5} {'Metric':<12} {'Generic BGE':>14} {'Fine-tuned BGE':>16} {'Δ':>10}")
    print("-" * 72)

    for n in SCALES:
        OUT_DIRS[n].mkdir(parents=True, exist_ok=True)
        records = all_records[:n]

        metrics, details = run_eval(records, config)

        # Save results
        with open(OUT_DIRS[n] / "retrieval_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(OUT_DIRS[n] / "retrieval_details.json", "w") as f:
            json.dump(details, f, indent=2)
        logger.info(f"Saved to {OUT_DIRS[n]}")

        base = BASELINE_RESULTS[n]
        for metric_key, label in [("recall_at_1", "R@1"), ("recall_at_5", "R@5"), ("mrr", "MRR")]:
            b = base[metric_key]
            m = metrics[metric_key]
            print(f"{n:>5} {label:<12} {b:>14.3f} {m:>16.3f} {pct_change(m, b):>10}")

        # Release memory before loading next pipeline
        del metrics, details
        gc.collect()
        print()

    print("=" * 72)
    print("Results saved. To regenerate figures with 5 pipelines, update")
    print("generate_figures_v2.py to include text_finetuned rows.")
    print("=" * 72)


if __name__ == "__main__":
    main()
