"""
Graph module — table-to-graph construction, feature extraction, and embedding.

Submodules:
    table_to_graph      : TableQARecord → NetworkX DiGraph
    feature_extraction  : NetworkX DiGraph → PyG Data (with node features)
    graph_embedding     : GraphSAGE encoder (PyG Data → fixed-size embeddings)
"""

from src.graph.table_to_graph import (
    batch_table_to_graph,
    build_graphs_from_config,
    graph_summary,
    table_to_graph,
)
from src.graph.feature_extraction import GraphFeatureExtractor
from src.graph.graph_embedding import TableGraphEncoder

__all__ = [
    # Phase 2 — graph construction
    "table_to_graph",
    "batch_table_to_graph",
    "build_graphs_from_config",
    "graph_summary",
    # Phase 3 — feature extraction & embedding
    "GraphFeatureExtractor",
    "TableGraphEncoder",
]
