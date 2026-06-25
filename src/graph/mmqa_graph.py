"""
Multi-table corpus graph builder for the MMQA cross-table experiment.

Builds ONE NetworkX graph spanning the whole 702-table corpus so that
GraphSAGE message passing can propagate across tables. Each table contributes:

    * one TABLE node (read out as the table's graph-side embedding)
    * column-header nodes
    * data-cell nodes (capped at `graph_max_rows` to keep the corpus graph
      tractable — structural/relational signal lives in headers + table nodes)

Edges are added BIDIRECTIONALLY (undirected message passing) so that, with two
SAGEConv layers, a table node absorbs its headers/cells and — in the inter-table
variant — its FK-neighbours.

Two variants, differing ONLY in the edge set (the controlled contrast):

    include_inter=False  -> V3 hybrid-intra : intra-table edges only
    include_inter=True   -> V4 hybrid-inter : + inter-table FK edges

Inter-table edges are aggregated from the per-instance join pairs produced by
the loader (gold = SQL/schema joins; "any" also uses value-overlap). They link
the two FK/PK column-header nodes across tables, plus the two table nodes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import networkx as nx
from loguru import logger


def _is_numeric(text: str) -> bool:
    cleaned = str(text).strip().replace(",", "").replace("%", "").replace("$", "")
    if not cleaned:
        return False
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def _squash(s: str) -> str:
    import re
    return re.sub(r"[\s_]+", "", str(s).strip().lower())


# node id helpers (globally unique via table index ti)
def _tbl_node(ti: int) -> str: return f"t{ti}_tbl"
def _hdr_node(ti: int, col: int) -> str: return f"t{ti}_h{col}"
def _cell_node(ti: int, row: int, col: int) -> str: return f"t{ti}_c{row}_{col}"


def _add_biedge(G: nx.DiGraph, u: str, v: str, etype: str) -> None:
    G.add_edge(u, v, edge_type=etype)
    G.add_edge(v, u, edge_type=etype)


def _table_fields(entry: Any):
    """Read (name, columns, rows) from either an MMQATable or a dict."""
    if isinstance(entry, dict):
        return entry["name"], entry["columns"], entry["rows"]
    return entry.name, entry.columns, entry.rows


def build_corpus_graph(
    corpus: Dict[str, Dict[str, Any]],
    instances: List[Any],
    *,
    include_inter: bool = False,
    edge_kind: str = "gold",          # "gold" (sql/schema) or "any" (+ value)
    graph_max_rows: int = 30,
) -> Tuple[nx.DiGraph, Dict[str, str]]:
    """
    Returns (graph, table_id -> table_node_id).

    `instances` is the list of MMQAInstance (or asdict dicts) used only to
    aggregate corpus-level inter-table edges; the corpus structure itself is
    fixed regardless of split.
    """
    G = nx.DiGraph()
    tid_to_ti: Dict[str, int] = {tid: ti for ti, tid in enumerate(corpus.keys())}
    table_node_of: Dict[str, str] = {}

    # ── Per-table intra-structure ────────────────
    for tid, ti in tid_to_ti.items():
        name, cols, rows = _table_fields(corpus[tid])
        rows = rows[:graph_max_rows]
        ncols = len(cols)

        # table node (read-out node) — text = name + columns for a rich repr
        tnode = _tbl_node(ti)
        G.add_node(tnode, text=f"{name} | " + " | ".join(str(c) for c in cols),
                   node_type="table_metadata", row=-2, col=-1, is_numeric=False)
        table_node_of[tid] = tnode

        # header nodes
        for c in range(ncols):
            G.add_node(_hdr_node(ti, c), text=str(cols[c]),
                       node_type="header", row=-1, col=c,
                       is_numeric=_is_numeric(cols[c]))
            _add_biedge(G, tnode, _hdr_node(ti, c), "table_to_header")

        # cell nodes + edges
        for r, row in enumerate(rows):
            padded = list(row) + [""] * (ncols - len(row))
            for c in range(ncols):
                cell = _cell_node(ti, r, c)
                txt = str(padded[c]) if padded[c] is not None else ""
                G.add_node(cell, text=txt, node_type="data_cell",
                           row=r, col=c, is_numeric=_is_numeric(txt))
                _add_biedge(G, _hdr_node(ti, c), cell, "header_to_cell")
        # row / column adjacency
        for r in range(len(rows)):
            for c in range(ncols - 1):
                _add_biedge(G, _cell_node(ti, r, c), _cell_node(ti, r, c + 1),
                            "row_adjacency")
        for c in range(ncols):
            for r in range(len(rows) - 1):
                _add_biedge(G, _cell_node(ti, r, c), _cell_node(ti, r + 1, c),
                            "column_adjacency")

    G.graph["record_id"] = "mmqa_corpus"
    G.graph["dataset"] = "mmqa"

    n_inter = 0
    if include_inter:
        n_inter = _add_inter_table_edges(G, corpus, instances, tid_to_ti,
                                         edge_kind=edge_kind)

    logger.info(
        f"Corpus graph ({'inter' if include_inter else 'intra'}): "
        f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
        f"{len(table_node_of)} table nodes, {n_inter} inter-table edge groups"
    )
    return G, table_node_of


NODE_TYPE_MAP = {"header": 0, "data_cell": 1, "table_metadata": 2}


def edge_index_from_graph(G, node_to_idx):
    import torch
    src, dst = [], []
    for u, v in G.edges():
        src.append(node_to_idx[u]); dst.append(node_to_idx[v])
    return torch.tensor([src, dst], dtype=torch.long)


def compute_node_features(G, node_list, graph_max_rows, device,
                          model_name="BAAI/bge-base-en-v1.5",
                          cache_dir="data/cache"):
    """774-d node features (text768 + pos2 + type3 + numeric1), cached by rows."""
    import numpy as np
    import torch
    from pathlib import Path

    cache_path = Path(cache_dir) / f"mmqa_nodefeat_r{graph_max_rows}.pt"
    if cache_path.exists():
        blob = torch.load(cache_path)
        if blob["node_list"] == node_list:
            logger.info(f"Loaded cached node features {tuple(blob['x'].shape)}")
            return blob["x"]
        logger.warning("Cache node ordering mismatch; recomputing features.")

    from sentence_transformers import SentenceTransformer
    logger.info(f"Embedding {len(node_list)} node texts with {model_name} ...")
    model = SentenceTransformer(model_name, device=device)
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
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"node_list": node_list, "x": x}, cache_path)
    logger.info(f"Saved node features to {cache_path}  {tuple(x.shape)}")
    return x


def _col_index(corpus_entry: Any, col_name: str) -> int:
    _, cols, _ = _table_fields(corpus_entry)
    target = _squash(col_name)
    for i, c in enumerate(cols):
        if _squash(c) == target:
            return i
    return -1


def _iter_join_pairs(inst: Any, edge_kind: str):
    """Yield (tid_a, col_a, tid_b, col_b) from a dataclass or dict instance."""
    if isinstance(inst, dict):
        if edge_kind == "any":
            pairs = inst.get("join_pairs_sql") or inst.get("join_pairs_schema") \
                or inst.get("join_pairs_value") or []
        else:
            pairs = inst.get("join_pairs_sql") or inst.get("join_pairs_schema") or []
    else:
        pairs = inst.any_edges() if edge_kind == "any" else inst.gold_edges()
    for p in pairs:
        yield tuple(p)


def fk_neighbor_map(instances, edge_kind: str = "gold") -> Dict[str, set]:
    """table_id -> set of FK-linked table_ids, aggregated across instances.

    Used by the joint reranker (§9.2): when scoring a candidate table, it can
    look up the candidate's relational neighbours independently of the GNN.
    """
    from collections import defaultdict
    nbr: Dict[str, set] = defaultdict(set)
    for inst in instances:
        for tid_a, _, tid_b, _ in _iter_join_pairs(inst, edge_kind):
            if tid_a != tid_b:
                nbr[tid_a].add(tid_b)
                nbr[tid_b].add(tid_a)
    return nbr


def _add_inter_table_edges(G, corpus, instances, tid_to_ti, *, edge_kind: str) -> int:
    """
    Aggregate corpus-level FK edges from per-instance join pairs and add
    bidirectional header<->header and table<->table edges. Returns the number
    of distinct inter-table edge groups added.
    """
    seen: set = set()
    n_groups = 0
    for inst in instances:
        for tid_a, col_a, tid_b, col_b in _iter_join_pairs(inst, edge_kind):
            if tid_a not in tid_to_ti or tid_b not in tid_to_ti or tid_a == tid_b:
                continue
            key = tuple(sorted([f"{tid_a}::{_squash(col_a)}",
                                f"{tid_b}::{_squash(col_b)}"]))
            if key in seen:
                continue
            seen.add(key)
            ti_a, ti_b = tid_to_ti[tid_a], tid_to_ti[tid_b]
            ci_a = _col_index(corpus[tid_a], col_a)
            ci_b = _col_index(corpus[tid_b], col_b)
            if ci_a < 0 or ci_b < 0:
                continue
            _add_biedge(G, _hdr_node(ti_a, ci_a), _hdr_node(ti_b, ci_b), "fk_link")
            _add_biedge(G, _tbl_node(ti_a), _tbl_node(ti_b), "table_link")
            n_groups += 1
    return n_groups
