"""
Node feature extraction — converts NetworkX table graphs into
PyTorch Geometric Data objects with rich node features.

Feature vector per node (concatenated):
    1. Text embedding   — 768d from sentence-transformers (BAAI/bge-base-en-v1.5)
    2. Position encoding — 2d normalised (row, col)
    3. Cell type one-hot — 3d (header, data_cell, table_metadata)
    4. Numeric flag      — 1d (1.0 if cell is numeric, else 0.0)
                         ─────
    Total per node:       774d  (before GraphSAGE projection)

Usage:
    >>> from src.graph.feature_extraction import GraphFeatureExtractor
    >>> extractor = GraphFeatureExtractor()
    >>> pyg_data = extractor.convert(nx_graph)
    >>> pyg_data_list = extractor.convert_batch(nx_graphs)
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from loguru import logger

try:
    import torch_geometric
    from torch_geometric.data import Data
except ImportError:
    raise ImportError(
        "torch-geometric is required for feature extraction. "
        "Install with: pip install torch-geometric"
    )

import networkx as nx


# ────────────────────────────────────────────────
#  Node type → one-hot mapping
# ────────────────────────────────────────────────

NODE_TYPE_MAP: Dict[str, int] = {
    "header": 0,
    "data_cell": 1,
    "table_metadata": 2,
}
NUM_NODE_TYPES = len(NODE_TYPE_MAP)


# ────────────────────────────────────────────────
#  Edge type → integer mapping (for typed GNNs)
# ────────────────────────────────────────────────

EDGE_TYPE_MAP: Dict[str, int] = {
    "header_to_cell": 0,
    "row_adjacency": 1,
    "column_adjacency": 2,
    "table_to_header": 3,
}


# ────────────────────────────────────────────────
#  Feature extractor class
# ────────────────────────────────────────────────

class GraphFeatureExtractor:
    """
    Converts NetworkX table graphs into PyTorch Geometric Data objects.

    Parameters
    ----------
    embedding_model_name : str
        HuggingFace sentence-transformer model name for text embeddings.
    embedding_dim : int
        Expected dimensionality of text embeddings.
    max_rows : int
        Maximum row index for position normalisation.
    max_cols : int
        Maximum column index for position normalisation.
    device : str
        Device for the sentence-transformer model ('cpu' or 'cuda').
    """

    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-base-en-v1.5",
        embedding_dim: int = 768,
        max_rows: int = 100,
        max_cols: int = 50,
        device: Optional[str] = None,
    ):
        self.embedding_model_name = embedding_model_name
        self.embedding_dim = embedding_dim
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None  # lazy-loaded

    # ── Lazy model loading ───────────────────────

    @property
    def model(self):
        """Lazy-load the sentence-transformer model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(
                f"Loading embedding model: {self.embedding_model_name} "
                f"(device={self.device})"
            )
            self._model = SentenceTransformer(
                self.embedding_model_name,
                device=self.device,
            )
        return self._model

    # ── Text embedding ───────────────────────────

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of strings using the sentence-transformer model.

        Returns
        -------
        np.ndarray of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        # Use smaller batch size to avoid OOM on memory-constrained systems
        embed_batch = min(64, max(8, len(texts) // 10)) if len(texts) > 500 else 32
        embeddings = self.model.encode(
            texts,
            batch_size=embed_batch,
            show_progress_bar=len(texts) > 1000,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    # ── Position encoding ────────────────────────

    def _position_features(self, row: int, col: int) -> np.ndarray:
        """
        Normalised (row, col) position features.

        Headers use row=-1 (mapped to 0), metadata uses row=-2 (mapped to 0).
        """
        norm_row = max(0, row) / max(self.max_rows, 1)
        norm_col = max(0, col) / max(self.max_cols, 1)
        return np.array([norm_row, norm_col], dtype=np.float32)

    # ── Cell type one-hot ────────────────────────

    @staticmethod
    def _cell_type_onehot(node_type: str) -> np.ndarray:
        """One-hot encoding of node type (3 classes)."""
        vec = np.zeros(NUM_NODE_TYPES, dtype=np.float32)
        idx = NODE_TYPE_MAP.get(node_type, 1)  # default to data_cell
        vec[idx] = 1.0
        return vec

    # ── Numeric flag ─────────────────────────────

    @staticmethod
    def _numeric_flag(is_numeric: bool) -> np.ndarray:
        return np.array([1.0 if is_numeric else 0.0], dtype=np.float32)

    # ── Core conversion ──────────────────────────

    def convert(self, G: nx.DiGraph) -> Data:
        """
        Convert a single NetworkX table graph to a PyG Data object.

        Parameters
        ----------
        G : nx.DiGraph
            A table graph produced by ``table_to_graph()``.

        Returns
        -------
        torch_geometric.data.Data
            Attributes:
            - x            : node feature matrix   [num_nodes, 774]
            - edge_index   : COO edge tensor       [2, num_edges]
            - edge_type    : edge type IDs          [num_edges]
            - node_texts   : list of raw text strings (for debugging)
            - node_types   : list of node type strings
            - record_id    : source record ID from graph metadata
        """
        # Establish a fixed node ordering
        node_list = list(G.nodes())
        node_to_idx = {nid: i for i, nid in enumerate(node_list)}
        num_nodes = len(node_list)

        if num_nodes == 0:
            return self._empty_data(G)

        # ── Collect text and per-node features ───
        texts = []
        position_feats = []
        type_feats = []
        numeric_feats = []
        node_type_labels = []

        for nid in node_list:
            attrs = G.nodes[nid]
            texts.append(attrs.get("text", ""))
            position_feats.append(
                self._position_features(attrs.get("row", 0), attrs.get("col", 0))
            )
            type_feats.append(
                self._cell_type_onehot(attrs.get("node_type", "data_cell"))
            )
            numeric_feats.append(
                self._numeric_flag(attrs.get("is_numeric", False))
            )
            node_type_labels.append(attrs.get("node_type", "data_cell"))

        # ── Text embeddings (batch) ──────────────
        text_emb = self._embed_texts(texts)  # (N, 768)

        # ── Concatenate all features ─────────────
        position_arr = np.stack(position_feats)   # (N, 2)
        type_arr = np.stack(type_feats)           # (N, 3)
        numeric_arr = np.stack(numeric_feats)     # (N, 1)

        features = np.concatenate(
            [text_emb, position_arr, type_arr, numeric_arr],
            axis=1,
        )  # (N, 774)

        x = torch.tensor(features, dtype=torch.float32)

        # ── Edge index + edge types ──────────────
        src_indices = []
        dst_indices = []
        edge_types = []

        for u, v, attrs in G.edges(data=True):
            src_indices.append(node_to_idx[u])
            dst_indices.append(node_to_idx[v])
            et = attrs.get("edge_type", "row_adjacency")
            edge_types.append(EDGE_TYPE_MAP.get(et, 1))

        if src_indices:
            edge_index = torch.tensor(
                [src_indices, dst_indices], dtype=torch.long
            )
            edge_type = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)

        # ── Assemble Data object ─────────────────
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
        )

        # Store metadata for downstream use
        data.node_texts = texts
        data.node_types = node_type_labels
        data.record_id = G.graph.get("record_id", "unknown")
        data.dataset = G.graph.get("dataset", "unknown")
        data.num_header_cols = G.graph.get("num_header_cols", 0)
        data.num_data_rows = G.graph.get("num_data_rows", 0)

        return data

    # ── Batch conversion ─────────────────────────

    def convert_batch(
        self,
        graphs: List[nx.DiGraph],
        show_progress: bool = True,
        chunk_size: int = 200,
    ) -> List[Data]:
        """
        Convert a list of NetworkX graphs to PyG Data objects.

        Processes graphs in memory-safe chunks to avoid OOM errors when
        embedding millions of nodes (e.g. full WikiTQ = 1.7M nodes).

        Parameters
        ----------
        graphs : list[nx.DiGraph]
            Table graphs from ``table_to_graph()`` or ``batch_table_to_graph()``.
        show_progress : bool
            Log progress after each chunk.
        chunk_size : int
            Number of graphs to embed per chunk. Reduce if OOM errors occur.
            500 graphs ≈ 75K nodes ≈ ~0.5 GB RAM peak.

        Returns
        -------
        list[Data]
        """
        if not graphs:
            return []

        total_nodes = sum(G.number_of_nodes() for G in graphs)
        logger.info(
            f"Batch embedding {total_nodes} nodes across {len(graphs)} graphs "
            f"(chunk_size={chunk_size})..."
        )

        data_list = []

        for chunk_start in range(0, len(graphs), chunk_size):
            chunk = graphs[chunk_start : chunk_start + chunk_size]

            # Collect all texts in this chunk
            all_texts = []
            for G in chunk:
                for nid in G.nodes():
                    all_texts.append(G.nodes[nid].get("text", ""))

            # Embed all texts in chunk at once
            chunk_embeddings = self._embed_texts(all_texts)  # (chunk_nodes, 768)

            # Build Data objects for each graph in chunk
            emb_offset = 0
            for G in chunk:
                node_list = list(G.nodes())
                num_nodes = len(node_list)
                node_to_idx = {nid: j for j, nid in enumerate(node_list)}

                if num_nodes == 0:
                    data_list.append(self._empty_data(G))
                    continue

                text_emb = chunk_embeddings[emb_offset : emb_offset + num_nodes]
                emb_offset += num_nodes

                texts = []
                position_feats = []
                type_feats = []
                numeric_feats = []
                node_type_labels = []

                for nid in node_list:
                    attrs = G.nodes[nid]
                    texts.append(attrs.get("text", ""))
                    position_feats.append(
                        self._position_features(
                            attrs.get("row", 0), attrs.get("col", 0)
                        )
                    )
                    type_feats.append(
                        self._cell_type_onehot(attrs.get("node_type", "data_cell"))
                    )
                    numeric_feats.append(
                        self._numeric_flag(attrs.get("is_numeric", False))
                    )
                    node_type_labels.append(attrs.get("node_type", "data_cell"))

                features = np.concatenate(
                    [
                        text_emb,
                        np.stack(position_feats),
                        np.stack(type_feats),
                        np.stack(numeric_feats),
                    ],
                    axis=1,
                )
                x = torch.tensor(features, dtype=torch.float32)

                src_indices, dst_indices, edge_types = [], [], []
                for u, v, attrs in G.edges(data=True):
                    src_indices.append(node_to_idx[u])
                    dst_indices.append(node_to_idx[v])
                    et = attrs.get("edge_type", "row_adjacency")
                    edge_types.append(EDGE_TYPE_MAP.get(et, 1))

                if src_indices:
                    edge_index = torch.tensor(
                        [src_indices, dst_indices], dtype=torch.long
                    )
                    edge_type = torch.tensor(edge_types, dtype=torch.long)
                else:
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
                    edge_type = torch.zeros(0, dtype=torch.long)

                data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
                data.node_texts = texts
                data.node_types = node_type_labels
                data.record_id = G.graph.get("record_id", "unknown")
                data.dataset = G.graph.get("dataset", "unknown")
                data.num_header_cols = G.graph.get("num_header_cols", 0)
                data.num_data_rows = G.graph.get("num_data_rows", 0)

                data_list.append(data)

            if show_progress:
                done = min(chunk_start + chunk_size, len(graphs))
                logger.info(
                    f"Embedded {done}/{len(graphs)} graphs "
                    f"({len(data_list)} Data objects built)"
                )

        logger.info(f"Batch conversion complete: {len(data_list)} PyG Data objects")
        return data_list

    # ── Empty graph fallback ─────────────────────

    def _empty_data(self, G: nx.DiGraph) -> Data:
        """Return a valid but empty Data object."""
        feature_dim = self.embedding_dim + 2 + NUM_NODE_TYPES + 1  # 774
        data = Data(
            x=torch.zeros((0, feature_dim), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros(0, dtype=torch.long),
        )
        data.node_texts = []
        data.node_types = []
        data.record_id = G.graph.get("record_id", "unknown")
        data.dataset = G.graph.get("dataset", "unknown")
        data.num_header_cols = 0
        data.num_data_rows = 0
        return data

    # ── Config-driven factory ────────────────────

    @classmethod
    def from_config(cls, config: dict) -> "GraphFeatureExtractor":
        """
        Create an extractor from the merged pipeline config.

        Reads:
        - config["embedding"]["model"]       → embedding_model_name
        - config["embedding"]["dimension"]    → embedding_dim
        - config["table_parsing"]["max_rows"] → max_rows
        - config["table_parsing"]["max_cols"] → max_cols
        """
        emb_cfg = config.get("embedding", {})
        tp_cfg = config.get("table_parsing", {})

        return cls(
            embedding_model_name=emb_cfg.get("model", "BAAI/bge-base-en-v1.5"),
            embedding_dim=emb_cfg.get("dimension", 768),
            max_rows=tp_cfg.get("max_rows", 100),
            max_cols=tp_cfg.get("max_cols", 50),
        )
