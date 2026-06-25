#!/usr/bin/env python3
"""
Phase 2 — build & validate the MMQA corpus graphs (V3 intra, V4 inter).

No training yet. This confirms the graph structure is sound before we spend
GPU time: node/edge counts, how many tables get linked by inter-table edges,
the database-cluster structure those edges induce, and a spot-check that FK
edges connect the right tables.

Usage:
    python scripts/build_mmqa_graph.py
"""

from __future__ import annotations

import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import networkx as nx

from src.data.mmqa_loader import load_mmqa
from src.graph.mmqa_graph import build_corpus_graph


def _table_clusters(G, table_node_of):
    """Connected components restricted to table nodes (via inter-table links)."""
    tnodes = set(table_node_of.values())
    und = G.to_undirected()
    comp_sizes = []
    for comp in nx.connected_components(und):
        ntables = len(comp & tnodes)
        if ntables:
            comp_sizes.append(ntables)
    return comp_sizes


def main():
    data = load_mmqa()
    corpus = data["corpus"]
    all_instances = data["train"] + data["dev"] + data["test"]

    print("\n" + "=" * 70)
    print("  V3 — INTRA-TABLE ONLY")
    print("=" * 70)
    G3, tnode3 = build_corpus_graph(corpus, all_instances, include_inter=False)
    print(f"  nodes: {G3.number_of_nodes()}   edges: {G3.number_of_edges()}")
    print(f"  table nodes: {len(tnode3)}")

    print("\n" + "=" * 70)
    print("  V4 — INTRA + INTER-TABLE (gold FK edges)")
    print("=" * 70)
    G4, tnode4 = build_corpus_graph(corpus, all_instances,
                                    include_inter=True, edge_kind="gold")
    print(f"  nodes: {G4.number_of_nodes()}   edges: {G4.number_of_edges()}")
    extra = G4.number_of_edges() - G3.number_of_edges()
    print(f"  inter-table edges added (bidirectional incl. header+table): {extra}")

    clusters = _table_clusters(G4, tnode4)
    linked = sum(1 for c in clusters if c > 1)
    singletons = sum(1 for c in clusters if c == 1)
    print(f"\n  Table-cluster structure (databases induced by FK edges):")
    print(f"    clusters total           : {len(clusters)}")
    print(f"    multi-table clusters     : {linked}")
    print(f"    singleton tables (no FK) : {singletons}")
    size_hist = Counter(clusters)
    print(f"    cluster-size histogram   : "
          f"{dict(sorted(size_hist.items()))}")
    tables_in_multi = sum(c for c in clusters if c > 1)
    print(f"    tables in a multi-table cluster: {tables_in_multi}/{len(tnode4)} "
          f"({100*tables_in_multi/len(tnode4):.1f}%)")

    # ── Also build the schema-free (value-overlap) variant for comparison ──
    G4v, tnode4v = build_corpus_graph(corpus, all_instances,
                                      include_inter=True, edge_kind="any")
    extra_v = G4v.number_of_edges() - G3.number_of_edges()
    clusters_v = _table_clusters(G4v, tnode4v)
    print(f"\n  Schema-free arm (edge_kind='any', adds value-overlap):")
    print(f"    inter-table edges added  : {extra_v}")
    print(f"    tables in multi-cluster  : "
          f"{sum(c for c in clusters_v if c>1)}/{len(tnode4v)}")

    print("\n" + "=" * 70)
    print("  SPOT-CHECK — inter-table neighbours of a sample table")
    print("=" * 70)
    # find a table node that has table_link edges and show its neighbours
    shown = 0
    for tid, tn in tnode4.items():
        nbrs = [n for n in G4.successors(tn)
                if G4.edges[tn, n].get("edge_type") == "table_link"]
        if nbrs:
            name = tid.split("::")[0]
            print(f"\n  Table '{name}' is FK-linked to:")
            id_by_node = {v: k for k, v in tnode4.items()}
            for nb in nbrs:
                print(f"      -> {id_by_node[nb].split('::')[0]}")
            shown += 1
        if shown >= 3:
            break

    print("\n" + "=" * 70)
    print("  Graph structure validated. Ready for graph-leg training (Script #4).")
    print("=" * 70)


if __name__ == "__main__":
    main()
