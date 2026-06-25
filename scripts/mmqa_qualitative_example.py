#!/usr/bin/env python3
"""
Find a real qualitative example for the paper: an MMQA test question where the
standard reranker drops a gold (foreign-key-linked) table out of the top 10 and
the joint reranker rescues it.

Reproduces the paper's reranking setup (generic-text leg + inter-table graph,
alpha=0.3, rerank depth 50), then for each query records the rank each gold
table receives under (i) fusion only, (ii) standard rerank, (iii) joint rerank.
It reports clean rescue cases, preferring small gold sets where the rescued
table is NOT named in the question (the linked-table case the paper describes).

Stops after collecting --max-examples rescue cases to keep runtime down.

Usage:
    python scripts/mmqa_qualitative_example.py --max-examples 8
"""

from __future__ import annotations

import argparse
import io
import re
import sys
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

GENERIC_BGE = "BAAI/bge-base-en-v1.5"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ALPHA = 0.3
DEPTH = 50
MAX_ROWS = 100


def linearise(name, cols, rows):
    p = [f"Table: {name}",
         "| " + " | ".join(str(c) for c in cols) + " |",
         "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows[:MAX_ROWS]:
        pad = [str(v) if v is not None else "" for v in r]
        pad += [""] * (len(cols) - len(pad))
        p.append("| " + " | ".join(pad[:len(cols)]) + " |")
    return "\n".join(p)


def znorm(s):
    return (s - s.mean()) / (s.std() + 1e-6)


def name_in_question(name, question):
    q = re.sub(r"[\s_]+", " ", question.lower())
    toks = [t for t in re.split(r"[\s_]+", name.lower()) if len(t) > 2]
    return any(t in q for t in toks)


def ranks_of(gold, ranked_ids):
    pos = {tid: i + 1 for i, tid in enumerate(ranked_ids)}
    return {g: pos.get(g, 9999) for g in gold}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-examples", type=int, default=8)
    ap.add_argument("--graph-max-rows", type=int, default=30)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    data = load_mmqa()
    corpus = data["corpus"]
    insts = data["test"]
    table_ids = list(corpus.keys())
    col = {t: i for i, t in enumerate(table_ids)}
    table_texts = [linearise(corpus[t].name, corpus[t].columns, corpus[t].rows)
                   for t in table_ids]
    name_of = {t: corpus[t].name for t in table_ids}

    from sentence_transformers import SentenceTransformer, CrossEncoder
    bge = SentenceTransformer(GENERIC_BGE, device=device)
    logger.info("Encoding queries + tables (generic BGE)...")
    questions = [i.question for i in insts]
    q_emb = bge.encode(questions, batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
    t_emb = bge.encode(table_texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    text_sims = q_emb @ t_emb.T

    # graph leg (inter)
    G, tnode = build_corpus_graph(corpus, data["train"] + data["dev"] + data["test"],
                                  include_inter=True, edge_kind="gold",
                                  graph_max_rows=args.graph_max_rows)
    node_list = list(G.nodes()); n2i = {n: i for i, n in enumerate(node_list)}
    x = compute_node_features(G, node_list, args.graph_max_rows, device).to(device)
    ei = edge_index_from_graph(G, n2i).to(device)
    tidx = torch.tensor([n2i[tnode[t]] for t in table_ids], dtype=torch.long, device=device)
    enc = TableGraphEncoder(input_dim=x.size(1), hidden_dim=256, output_dim=768,
                            num_layers=2, normalize_output=True).to(device)
    enc.load_state_dict(torch.load("models/mmqa_graph_encoder/best_inter.pt", map_location=device))
    enc.eval()
    pyg = type("D", (), {"x": x, "edge_index": ei, "batch": None})()
    with torch.no_grad():
        ne, _ = enc(pyg)
        g_emb = ne[tidx].cpu().numpy()
    graph_sims = q_emb @ g_emb.T

    fk = fk_neighbor_map(data["train"] + data["dev"] + data["test"], edge_kind="gold")
    reranker = CrossEncoder(CROSS_ENCODER)

    examples = []
    for qi, inst in enumerate(insts):
        gold = [g for g in inst.gold_table_ids if g in col]
        if not (2 <= len(gold) <= 3):
            continue
        fused = ALPHA * znorm(text_sims[qi]) + (1 - ALPHA) * znorm(graph_sims[qi])
        order = np.argsort(-fused)
        ranked = [table_ids[j] for j in order]
        # only interesting if fusion already has the full set within top-50
        if any(ranked.index(g) >= DEPTH for g in gold):
            continue
        cand = ranked[:DEPTH]

        # standard rerank
        sc = reranker.predict([(questions[qi], table_texts[col[c]]) for c in cand],
                              show_progress_bar=False)
        std = [cand[o] for o in np.argsort(-np.asarray(sc))]

        # joint rerank
        docs = []
        for c in cand:
            nbrs = [nb for nb in fk.get(c, set()) if nb in col]
            best = max(nbrs, key=lambda nb: fused[col[nb]], default=None)
            base = table_texts[col[c]][:480]
            docs.append(base + ("\n[Linked table]\n" + table_texts[col[best]][:320]
                                if best is not None else ""))
        jc = reranker.predict([(questions[qi], d) for d in docs], show_progress_bar=False)
        jnt = [cand[o] for o in np.argsort(-np.asarray(jc))]

        r_std = ranks_of(gold, std)
        r_jnt = ranks_of(gold, jnt)
        std_full = all(v <= 10 for v in r_std.values())
        jnt_full = all(v <= 10 for v in r_jnt.values())
        if jnt_full and not std_full:
            # the rescued table(s): standard >10, joint <=10
            rescued = [g for g in gold if r_std[g] > 10 and r_jnt[g] <= 10]
            examples.append({
                "qid": inst.qid, "question": inst.question, "gold": gold,
                "r_std": r_std, "r_jnt": r_jnt, "rescued": rescued,
            })
            if len(examples) >= args.max_examples:
                break
        if (qi + 1) % 25 == 0:
            logger.info(f"  scanned {qi+1}/{len(insts)}, found {len(examples)}")

    # rank examples: prefer rescued table NOT named in question
    def score(ex):
        not_named = sum(0 if name_in_question(name_of[r], ex["question"]) else 1
                        for r in ex["rescued"])
        return (not_named, -len(ex["gold"]))
    examples.sort(key=score, reverse=True)

    print("\n" + "=" * 74)
    print(f"  RESCUE EXAMPLES (joint reranker recovers a table standard dropped)")
    print("=" * 74)
    for ex in examples:
        print(f"\nQ[{ex['qid']}]: {ex['question']}")
        for g in ex["gold"]:
            nm = name_of[g]
            named = "named in Q" if name_in_question(nm, ex["question"]) else "NOT named in Q"
            tag = "  <-- RESCUED" if g in ex["rescued"] else ""
            print(f"   - {nm:<32} std rank {ex['r_std'][g]:>4}  joint rank {ex['r_jnt'][g]:>3}"
                  f"   ({named}){tag}")


if __name__ == "__main__":
    main()
