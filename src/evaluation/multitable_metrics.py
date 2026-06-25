"""
Multi-table retrieval metrics for MMQA.

Unlike single-table R@k (one gold table), MMQA's gold target is a SET of 2-5
tables. We therefore report set-aware metrics, following the pre-registered
protocol:

    Full-set Recall@k  (PRIMARY, k=10):
        1 if ALL gold tables are in the top-k, else 0. This is the
        answerability-relevant quantity for multi-hop retrieval.

    Per-table Recall@k:
        |gold ∩ top-k| / |gold|. Continuity with single-table R@k.

    MRR:
        reciprocal rank of the FIRST gold table. Continuity with the
        existing harness.

    Coverage-k:
        smallest k for which the full gold set is retrieved (None if the
        full set never appears within the corpus ranking). Reported as a
        median over queries whose full set was retrieved.

All functions take `ranked_ids` (corpus table_ids in rank order) and
`gold_ids` (the instance's required tables).
"""

from __future__ import annotations

from statistics import median
from typing import Dict, List, Optional, Sequence


def full_set_recall_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    topk = set(ranked_ids[:k])
    return 1.0 if set(gold_ids).issubset(topk) else 0.0


def per_table_recall_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    topk = set(ranked_ids[:k])
    return len(gold & topk) / len(gold)


def reciprocal_rank_first_gold(ranked_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    gold = set(gold_ids)
    for pos, rid in enumerate(ranked_ids, start=1):
        if rid in gold:
            return 1.0 / pos
    return 0.0


def coverage_k(ranked_ids: Sequence[str], gold_ids: Sequence[str]) -> Optional[int]:
    """Smallest k at which the full gold set is covered; None if never."""
    gold = set(gold_ids)
    if not gold:
        return None
    found = set()
    for pos, rid in enumerate(ranked_ids, start=1):
        if rid in gold:
            found.add(rid)
            if found == gold:
                return pos
    return None


def evaluate_multitable(
    per_query_rankings: List[Sequence[str]],
    per_query_gold: List[Sequence[str]],
    ks: Sequence[int] = (1, 5, 10, 20, 50),
    primary_k: int = 10,
) -> Dict:
    """
    Aggregate metrics over all queries.

    Returns a dict with mean Full-set Recall@k and Per-table Recall@k for each
    k, plus MRR, median Coverage-k, and per-query records (for bootstrap/McNemar
    on Full-set success at primary_k).
    """
    assert len(per_query_rankings) == len(per_query_gold)
    n = len(per_query_rankings)

    fsr = {k: 0.0 for k in ks}
    ptr = {k: 0.0 for k in ks}
    mrr = 0.0
    coverages: List[int] = []
    per_query = []

    for ranked, gold in zip(per_query_rankings, per_query_gold):
        rr = reciprocal_rank_first_gold(ranked, gold)
        mrr += rr
        cov = coverage_k(ranked, gold)
        if cov is not None:
            coverages.append(cov)
        row = {"reciprocal_rank": rr, "coverage_k": cov, "n_gold": len(gold)}
        for k in ks:
            f = full_set_recall_at_k(ranked, gold, k)
            p = per_table_recall_at_k(ranked, gold, k)
            fsr[k] += f
            ptr[k] += p
            row[f"full_set_recall_at_{k}"] = f
            row[f"per_table_recall_at_{k}"] = p
        per_query.append(row)

    return {
        "num_queries": n,
        "primary_k": primary_k,
        "full_set_recall": {k: fsr[k] / n if n else 0.0 for k in ks},
        "per_table_recall": {k: ptr[k] / n if n else 0.0 for k in ks},
        "mrr": mrr / n if n else 0.0,
        "coverage_k_median": median(coverages) if coverages else None,
        "coverage_k_n_full": len(coverages),
        "per_query": per_query,
    }
