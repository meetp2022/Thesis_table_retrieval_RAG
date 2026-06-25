#!/usr/bin/env python3
"""
Phase 3 — MMQA hybrid retrieval eval: V3 (intra) / V4 (inter), fusion + rerank.

Fuses the text leg (generic BGE over linearised tables) with the trained graph
leg (table-node readout) by per-query z-normalised score fusion, optionally
reranks the top candidates with the cross-encoder, and reports the set-aware
multi-table metrics. Saves per-query Full-set-R@10 success for the V3-vs-V4
bootstrap (Script #6).

Modes:
    --mode hybrid   text + graph fusion           (default)
    --mode graph    graph leg only
    --mode text     text leg only (== V1/V2 sanity)

Variants (graph leg):
    --graph-variant intra   V3 graph encoder
    --graph-variant inter   V4 graph encoder

Usage:
    # pick alpha on dev:
    python scripts/eval_mmqa_hybrid.py --graph-variant inter --sweep-alpha
    # eval on test:
    python scripts/eval_mmqa_hybrid.py --graph-variant intra --alpha 0.5 --split test --rerank
    python scripts/eval_mmqa_hybrid.py --graph-variant inter --alpha 0.5 --split test --rerank
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
import torch
from loguru import logger

from src.data.mmqa_loader import load_mmqa
from src.graph.mmqa_graph import (build_corpus_graph, compute_node_features,
                                  edge_index_from_graph, fk_neighbor_map)
from src.graph.graph_embedding import TableGraphEncoder
from src.evaluation.multitable_metrics import evaluate_multitable

GENERIC_BGE = "BAAI/bge-base-en-v1.5"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
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


def znorm_rows(s):
    return (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-6)


def main():
    ap = argparse.ArgumentParser(description="MMQA hybrid retrieval eval")
    ap.add_argument("--mode", choices=["hybrid", "graph", "text"], default="hybrid")
    ap.add_argument("--graph-variant", choices=["intra", "inter"], default="inter")
    ap.add_argument("--edge-kind", choices=["gold", "any"], default="gold")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="fusion: alpha*text + (1-alpha)*graph")
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--joint-rerank", action="store_true",
                    help="§9.2 control: reranker also sees each candidate's "
                         "top FK-neighbour (structure-aware reranking)")
    ap.add_argument("--rerank-depth", type=int, default=50)
    ap.add_argument("--joint-edge-kind", choices=["gold", "any"], default="gold",
                    help="which FK graph the joint reranker uses for neighbours")
    ap.add_argument("--graph-max-rows", type=int, default=30)
    ap.add_argument("--text-encoder", default=GENERIC_BGE,
                    help="model for the TEXT leg (default generic BGE; pass "
                         "models/mmqa_bge_finetuned/best for in-domain)")
    ap.add_argument("--sweep-alpha", action="store_true",
                    help="sweep alpha on the chosen split (no rerank) and exit")
    ap.add_argument("--model-dir", default="models/mmqa_graph_encoder")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = load_mmqa()
    corpus = data["corpus"]
    insts = data[args.split]
    table_ids = list(corpus.keys())
    tid_to_col = {tid: i for i, tid in enumerate(table_ids)}
    gold_lists = [[t for t in i.gold_table_ids if t in tid_to_col] for i in insts]
    questions = [i.question for i in insts]

    table_texts = [linearise(corpus[t].name, corpus[t].columns, corpus[t].rows)
                   for t in table_ids]

    # ── Text leg (configurable encoder) ──
    from sentence_transformers import SentenceTransformer
    logger.info(f"Text leg: {args.text_encoder}")
    text_model = SentenceTransformer(args.text_encoder, device=device)
    query_emb_text = text_model.encode(questions, batch_size=128,
                                       normalize_embeddings=True, convert_to_numpy=True)
    text_table_emb = text_model.encode(table_texts, batch_size=64,
                                       normalize_embeddings=True, convert_to_numpy=True)
    text_sims = query_emb_text @ text_table_emb.T

    # Graph leg was trained against the GENERIC-BGE query space, so it needs
    # generic query embeddings regardless of the text leg's encoder.
    if args.text_encoder == GENERIC_BGE:
        query_emb_graph = query_emb_text
    else:
        query_emb_graph = SentenceTransformer(GENERIC_BGE, device=device).encode(
            questions, batch_size=128, normalize_embeddings=True, convert_to_numpy=True)

    # ── Graph-table embeddings (trained leg) ──
    graph_sims = None
    if args.mode in ("hybrid", "graph"):
        include_inter = (args.graph_variant == "inter")
        G, tnode_of = build_corpus_graph(corpus, data["train"] + data["dev"] + data["test"],
                                         include_inter=include_inter,
                                         edge_kind=args.edge_kind,
                                         graph_max_rows=args.graph_max_rows)
        node_list = list(G.nodes())
        node_to_idx = {nid: i for i, nid in enumerate(node_list)}
        x = compute_node_features(G, node_list, args.graph_max_rows, device).to(device)
        ei = edge_index_from_graph(G, node_to_idx).to(device)
        tidx = torch.tensor([node_to_idx[tnode_of[t]] for t in table_ids],
                            dtype=torch.long, device=device)

        tag = args.graph_variant + ("" if args.edge_kind == "gold" else f"_{args.edge_kind}")
        enc = TableGraphEncoder(input_dim=x.size(1), hidden_dim=256, output_dim=768,
                                num_layers=2, normalize_output=True).to(device)
        state = torch.load(Path(args.model_dir) / f"best_{tag}.pt", map_location=device)
        enc.load_state_dict(state); enc.eval()
        pyg = type("D", (), {"x": x, "edge_index": ei, "batch": None})()
        with torch.no_grad():
            ne, _ = enc(pyg)
            graph_table_emb = ne[tidx].cpu().numpy()
        graph_sims = query_emb_graph @ graph_table_emb.T

    # ── Alpha sweep (dev selection) ──
    if args.sweep_alpha:
        print("\n  alpha sweep (Full-set R@10, no rerank):")
        for a in (0.0, 0.3, 0.5, 0.7, 1.0):
            fused = a * znorm_rows(text_sims) + (1 - a) * znorm_rows(graph_sims)
            rk = np.argsort(-fused, axis=1)
            rankings = [[table_ids[j] for j in rk[qi]] for qi in range(len(insts))]
            m = evaluate_multitable(rankings, gold_lists, ks=(10,), primary_k=10)
            print(f"    alpha={a:.1f}  FullSet R@10 = {m['full_set_recall'][10]:.4f}")
        return

    # ── Build fused ranking ──
    if args.mode == "text":
        fused = text_sims
    elif args.mode == "graph":
        fused = graph_sims
    else:
        fused = args.alpha * znorm_rows(text_sims) + (1 - args.alpha) * znorm_rows(graph_sims)
    ranked_idx = np.argsort(-fused, axis=1)

    # ── Optional rerank (standard or §9.2 joint) ──
    do_rerank = args.rerank or args.joint_rerank
    reranker = None
    fk_nbr = None
    if do_rerank:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(CROSS_ENCODER)
    if args.joint_rerank:
        fk_nbr = fk_neighbor_map(data["train"] + data["dev"] + data["test"],
                                 edge_kind=args.joint_edge_kind)
        logger.info(f"Joint reranker: FK-neighbour map for "
                    f"{len(fk_nbr)} tables")

    def _short(txt, n):     # keep both tables visible within cross-encoder limit
        return txt[:n]

    rankings = []
    rr_time = 0.0
    for qi in range(len(insts)):
        ranked_ids = [table_ids[j] for j in ranked_idx[qi]]
        if reranker is not None:
            depth = min(args.rerank_depth, len(ranked_ids))
            cand = ranked_ids[:depth]
            if args.joint_rerank:
                # augment each candidate with its highest-fused FK-neighbour
                docs = []
                for c in cand:
                    nbrs = fk_nbr.get(c, set())
                    best_nb, best_s = None, -1e9
                    for nb in nbrs:
                        if nb in tid_to_col and fused[qi, tid_to_col[nb]] > best_s:
                            best_s, best_nb = fused[qi, tid_to_col[nb]], nb
                    base = _short(table_texts[tid_to_col[c]], 480)
                    if best_nb is not None:
                        docs.append(base + "\n[Linked table]\n"
                                    + _short(table_texts[tid_to_col[best_nb]], 320))
                    else:
                        docs.append(base)
                pairs = [(questions[qi], d) for d in docs]
            else:
                pairs = [(questions[qi], table_texts[tid_to_col[c]]) for c in cand]
            t0 = time.perf_counter()
            scores = reranker.predict(pairs, show_progress_bar=False)
            rr_time += time.perf_counter() - t0
            order = np.argsort(-np.asarray(scores))
            ranked_ids = [cand[o] for o in order] + ranked_ids[depth:]
        rankings.append(ranked_ids)
        if (qi + 1) % 100 == 0:
            logger.info(f"  ranked {qi+1}/{len(insts)}")

    metrics = evaluate_multitable(rankings, gold_lists, ks=(1, 5, 10, 20, 50), primary_k=10)

    variant = f"{args.mode}"
    if args.mode != "text":
        variant += f"_{args.graph_variant}"
    if args.text_encoder != GENERIC_BGE:
        variant += "_fttext"
    if args.joint_rerank:
        variant += "_jointrr"
    elif args.rerank:
        variant += "_rerank"
    out = {
        "variant": variant, "mode": args.mode, "graph_variant": args.graph_variant,
        "edge_kind": args.edge_kind, "alpha": args.alpha, "split": args.split,
        "rerank": args.rerank, "num_queries": metrics["num_queries"],
        "full_set_recall": metrics["full_set_recall"],
        "per_table_recall": metrics["per_table_recall"],
        "mrr": metrics["mrr"], "coverage_k_median": metrics["coverage_k_median"],
    }
    out_dir = Path(args.output or f"data/results/mmqa_{variant}_a{args.alpha}_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval_metrics.json").write_text(json.dumps(out, indent=2))
    (out_dir / "per_query.json").write_text(json.dumps({
        "qid": [i.qid for i in insts],
        "full_set_recall_at_10": [r["full_set_recall_at_10"] for r in metrics["per_query"]],
        "per_table_recall_at_10": [r["per_table_recall_at_10"] for r in metrics["per_query"]],
        "reciprocal_rank": [r["reciprocal_rank"] for r in metrics["per_query"]],
    }, indent=2))

    print("\n" + "=" * 70)
    print(f"  MMQA {variant}  ·  alpha={args.alpha}  ·  split={args.split}")
    print("=" * 70)
    print(f"  {'k':>4} | {'Full-set R@k':>14} | {'Per-table R@k':>14}")
    for k in (1, 5, 10, 20, 50):
        print(f"  {k:>4} | {metrics['full_set_recall'][k]:>14.3f} | "
              f"{metrics['per_table_recall'][k]:>14.3f}")
    print(f"\n  MRR: {metrics['mrr']:.3f}   Coverage-k median: {metrics['coverage_k_median']}")
    print("=" * 70)
    print(f"  PRIMARY — Full-set Recall@10 = {metrics['full_set_recall'][10]:.3f}")
    print(f"  saved: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
