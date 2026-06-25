#!/usr/bin/env python3
"""
Phase 2/3 — train the MMQA graph leg (table-node readout) contrastively.

Encodes the WHOLE corpus graph in one GraphSAGE forward pass, reads out the 702
table-node embeddings, and aligns them to BGE query embeddings with a
multi-positive InfoNCE loss (each query's gold table set = the positives).

V3 vs V4 is selected purely by the edge set:
    --variant intra   -> intra-table edges only          (V3)
    --variant inter   -> + inter-table FK edges          (V4)

The contrast is otherwise identical (same nodes, same features, same encoder,
same training), so any V3/V4 difference is attributable to inter-table edges.

Because the corpus encoding is shared across all queries, one optimisation step
covers the whole train set — training is fast even on CPU.

Usage:
    python scripts/train_mmqa_graph.py --variant intra --epochs 60
    python scripts/train_mmqa_graph.py --variant inter --epochs 60
    python scripts/train_mmqa_graph.py --variant inter --edge-kind any --epochs 60
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
import torch.nn.functional as F
from loguru import logger

from src.data.mmqa_loader import load_mmqa
from src.graph.mmqa_graph import build_corpus_graph
from src.graph.graph_embedding import TableGraphEncoder

PROC = Path("data/processed/mmqa")
CACHE = Path("data/cache")
GENERIC_BGE = "BAAI/bge-base-en-v1.5"
NODE_TYPE_MAP = {"header": 0, "data_cell": 1, "table_metadata": 2}


# ── Node features (774-d), cached by graph_max_rows ──────────────
def build_or_load_features(G, node_list, graph_max_rows, device):
    cache_path = CACHE / f"mmqa_nodefeat_r{graph_max_rows}.pt"
    if cache_path.exists():
        blob = torch.load(cache_path)
        if blob["node_list"] == node_list:
            logger.info(f"Loaded cached node features {tuple(blob['x'].shape)}")
            return blob["x"]
        logger.warning("Cache node ordering mismatch; recomputing features.")

    from sentence_transformers import SentenceTransformer
    logger.info(f"Embedding {len(node_list)} node texts with {GENERIC_BGE} ...")
    model = SentenceTransformer(GENERIC_BGE, device=device)
    texts = [G.nodes[nid].get("text", "") for nid in node_list]
    text_emb = model.encode(texts, batch_size=128, normalize_embeddings=True,
                            show_progress_bar=True, convert_to_numpy=True)

    pos, typ, num = [], [], []
    for nid in node_list:
        a = G.nodes[nid]
        pos.append([max(0, a.get("row", 0)) / max(graph_max_rows, 1),
                    max(0, a.get("col", 0)) / 50.0])
        oh = [0.0, 0.0, 0.0]
        oh[NODE_TYPE_MAP.get(a.get("node_type", "data_cell"), 1)] = 1.0
        typ.append(oh)
        num.append([1.0 if a.get("is_numeric", False) else 0.0])
    feats = np.concatenate(
        [text_emb, np.array(pos, np.float32), np.array(typ, np.float32),
         np.array(num, np.float32)], axis=1)
    x = torch.tensor(feats, dtype=torch.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save({"node_list": node_list, "x": x}, cache_path)
    logger.info(f"Saved node features to {cache_path}  {tuple(x.shape)}")
    return x


def edge_index_from_graph(G, node_to_idx):
    src, dst = [], []
    for u, v in G.edges():
        src.append(node_to_idx[u]); dst.append(node_to_idx[v])
    return torch.tensor([src, dst], dtype=torch.long)


def full_set_recall_at_10(table_emb, query_emb, gold_idx_lists):
    sims = query_emb @ table_emb.T
    topk = torch.topk(sims, k=10, dim=1).indices.cpu().numpy()
    hits = 0
    for i, gold in enumerate(gold_idx_lists):
        if set(gold).issubset(set(topk[i].tolist())):
            hits += 1
    return hits / len(gold_idx_lists) if gold_idx_lists else 0.0


def main():
    ap = argparse.ArgumentParser(description="Train MMQA graph leg")
    ap.add_argument("--variant", choices=["intra", "inter"], required=True)
    ap.add_argument("--edge-kind", choices=["gold", "any"], default="gold")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--graph-max-rows", type=int, default=30)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--save-dir", default="models/mmqa_graph_encoder")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Data ─────────────────────────────────────
    data = load_mmqa()
    corpus = data["corpus"]
    train_inst, dev_inst = data["train"], data["dev"]
    table_ids = list(corpus.keys())
    tid_to_col = {tid: i for i, tid in enumerate(table_ids)}

    include_inter = (args.variant == "inter")
    G, table_node_of = build_corpus_graph(
        corpus, train_inst + dev_inst + data["test"],
        include_inter=include_inter, edge_kind=args.edge_kind,
        graph_max_rows=args.graph_max_rows)
    node_list = list(G.nodes())
    node_to_idx = {nid: i for i, nid in enumerate(node_list)}

    x = build_or_load_features(G, node_list, args.graph_max_rows, device).to(device)
    edge_index = edge_index_from_graph(G, node_to_idx).to(device)
    table_node_idx = torch.tensor(
        [node_to_idx[table_node_of[tid]] for tid in table_ids],
        dtype=torch.long, device=device)

    # ── Query embeddings (frozen generic BGE) ────
    from sentence_transformers import SentenceTransformer
    qmodel = SentenceTransformer(GENERIC_BGE, device=device)

    def encode_queries(insts):
        qs = [i.question for i in insts]
        q = qmodel.encode(qs, batch_size=128, normalize_embeddings=True,
                          convert_to_numpy=True)
        gold = [[tid_to_col[t] for t in i.gold_table_ids if t in tid_to_col]
                for i in insts]
        return torch.tensor(q, dtype=torch.float32, device=device), gold

    train_q, train_gold = encode_queries(train_inst)
    dev_q, dev_gold = encode_queries(dev_inst)

    # expanded (query_row, gold_table_col) positive pairs for InfoNCE
    q_rows, t_cols = [], []
    for qi, gold in enumerate(train_gold):
        for tc in gold:
            q_rows.append(qi); t_cols.append(tc)
    q_rows = torch.tensor(q_rows, dtype=torch.long, device=device)
    t_cols = torch.tensor(t_cols, dtype=torch.long, device=device)
    logger.info(f"Train pairs: {len(q_rows)}  ({len(train_inst)} queries)")

    # ── Model ────────────────────────────────────
    encoder = TableGraphEncoder(input_dim=x.size(1), hidden_dim=256,
                                output_dim=768, num_layers=2,
                                normalize_output=True).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    pyg = type("D", (), {"x": x, "edge_index": edge_index, "batch": None})()

    best_dev, best_state, bad = -1.0, None, 0
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.variant}" + ("" if args.edge_kind == "gold" else f"_{args.edge_kind}")

    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        encoder.train(); opt.zero_grad()
        node_emb, _ = encoder(pyg)                       # (N, 768)
        table_emb = node_emb[table_node_idx]             # (702, 768)
        sims = (train_q @ table_emb.T) / args.temperature  # (Q, 702)
        logits = sims[q_rows]                            # (P, 702)
        loss = F.cross_entropy(logits, t_cols)
        loss.backward(); opt.step()

        if epoch % 5 == 0 or epoch == 1:
            encoder.eval()
            with torch.no_grad():
                ne, _ = encoder(pyg)
                te = ne[table_node_idx]
                dev_r10 = full_set_recall_at_10(te, dev_q, dev_gold)
            logger.info(f"epoch {epoch:3d}  loss {loss.item():.4f}  "
                        f"dev FullSet R@10 {dev_r10:.4f}")
            if dev_r10 > best_dev:
                best_dev = dev_r10
                best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    logger.info(f"Early stop at epoch {epoch}")
                    break

    if best_state:
        torch.save(best_state, save_dir / f"best_{tag}.pt")
    train_time = time.perf_counter() - t0
    log = {"variant": args.variant, "edge_kind": args.edge_kind,
           "best_dev_full_set_recall_at_10": best_dev, "epochs": args.epochs,
           "lr": args.lr, "temperature": args.temperature,
           "graph_max_rows": args.graph_max_rows, "train_pairs": int(len(q_rows)),
           "train_seconds": round(train_time, 1)}
    (save_dir / f"train_log_{tag}.json").write_text(json.dumps(log, indent=2))

    print("\n" + "=" * 60)
    print(f"  MMQA graph leg trained — variant={tag}")
    print(f"  best dev Full-set R@10: {best_dev:.4f}")
    print(f"  saved: {save_dir / f'best_{tag}.pt'}   ({train_time:.0f}s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
