"""
GraphSAGE embedding model for table graphs.

Takes PyG Data objects (from feature_extraction.py) and produces:
    - Per-node embeddings    : (num_nodes, output_dim)   — 768d
    - Per-table embedding    : (output_dim,)             — mean-pooled

Architecture (from pipeline3_graph.yaml):
    Input (774d) → SAGEConv(774, 256) → ReLU → Dropout
                 → SAGEConv(256, 768) → L2-normalise
                 → Mean pool → table-level vector

Usage:
    >>> from src.graph.graph_embedding import TableGraphEncoder
    >>> encoder = TableGraphEncoder()
    >>> node_emb, table_emb = encoder(pyg_data)
    >>> table_embs = encoder.encode_batch(pyg_data_list)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import SAGEConv, global_mean_pool
except ImportError:
    raise ImportError(
        "torch-geometric is required. Install with: pip install torch-geometric"
    )


class TableGraphEncoder(nn.Module):
    """
    GraphSAGE-based encoder that maps table graphs to fixed-size embeddings.

    Parameters
    ----------
    input_dim : int
        Dimension of input node features (774 = 768 text + 2 pos + 3 type + 1 numeric).
    hidden_dim : int
        Hidden layer dimension (default 256 from config).
    output_dim : int
        Output embedding dimension (default 768 to match text embedding space).
    num_layers : int
        Number of SAGEConv layers (default 2).
    dropout : float
        Dropout rate between layers (default 0.1).
    normalize_output : bool
        L2-normalise output embeddings (default True, needed for cosine FAISS).
    pool_method : str
        Graph-level pooling method: 'mean' (default) or 'max'.
    """

    def __init__(
        self,
        input_dim: int = 774,
        hidden_dim: int = 256,
        output_dim: int = 768,
        num_layers: int = 2,
        dropout: float = 0.1,
        normalize_output: bool = True,
        pool_method: str = "mean",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.normalize_output = normalize_output
        self.pool_method = pool_method

        # Build SAGEConv layers
        self.convs = nn.ModuleList()

        # First layer: input_dim → hidden_dim
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggr="mean"))

        # Intermediate layers (if num_layers > 2)
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr="mean"))

        # Last layer: hidden_dim → output_dim
        if num_layers >= 2:
            self.convs.append(SAGEConv(hidden_dim, output_dim, aggr="mean"))
        else:
            # Single layer: replace first conv
            self.convs = nn.ModuleList([
                SAGEConv(input_dim, output_dim, aggr="mean")
            ])

        logger.debug(
            f"TableGraphEncoder: {num_layers} SAGEConv layers, "
            f"{input_dim}→{hidden_dim}→{output_dim}, "
            f"dropout={dropout}, pool={pool_method}"
        )

    def forward(
        self,
        data: Data,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single graph (or a batched graph).

        Parameters
        ----------
        data : torch_geometric.data.Data
            Must have ``x`` (node features) and ``edge_index``.
            Optionally ``batch`` for batched graphs.

        Returns
        -------
        node_embeddings : torch.Tensor
            Shape (num_nodes, output_dim).
        graph_embedding : torch.Tensor
            Shape (num_graphs, output_dim) — pooled per-graph vectors.
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else None

        # If no batch attribute, treat as single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Message passing layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            # Apply ReLU + dropout to all layers except the last
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # L2 normalisation (for cosine similarity in FAISS)
        if self.normalize_output:
            x = F.normalize(x, p=2, dim=-1)

        node_embeddings = x

        # Graph-level pooling
        if self.pool_method == "mean":
            graph_embedding = global_mean_pool(x, batch)
        elif self.pool_method == "max":
            from torch_geometric.nn import global_max_pool
            graph_embedding = global_max_pool(x, batch)
        else:
            raise ValueError(f"Unknown pool method: {self.pool_method}")

        # Re-normalise pooled embedding
        if self.normalize_output:
            graph_embedding = F.normalize(graph_embedding, p=2, dim=-1)

        return node_embeddings, graph_embedding

    # ── Batch encoding (inference mode) ──────────

    @torch.no_grad()
    def encode_batch(
        self,
        data_list: List[Data],
        batch_size: int = 32,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Encode a list of PyG Data objects in inference mode.

        Parameters
        ----------
        data_list : list[Data]
            PyG Data objects from ``GraphFeatureExtractor.convert_batch()``.
        batch_size : int
            Number of graphs per forward pass.

        Returns
        -------
        all_node_embeddings : list[torch.Tensor]
            Per-graph node embedding tensors.
        all_graph_embeddings : torch.Tensor
            Shape (len(data_list), output_dim) — one vector per table.
        """
        self.eval()

        if not data_list:
            return [], torch.zeros((0, self.output_dim))

        all_node_embs: List[torch.Tensor] = []
        all_graph_embs: List[torch.Tensor] = []

        for start in range(0, len(data_list), batch_size):
            chunk = data_list[start : start + batch_size]
            batched = Batch.from_data_list(chunk)

            node_emb, graph_emb = self.forward(batched)

            # Split node embeddings back per graph
            node_counts = [d.x.size(0) for d in chunk]
            node_splits = torch.split(node_emb, node_counts)
            all_node_embs.extend([s.cpu() for s in node_splits])
            all_graph_embs.append(graph_emb.cpu())

            if (start + batch_size) % 200 == 0 or start + batch_size >= len(data_list):
                logger.debug(
                    f"Encoded {min(start + batch_size, len(data_list))}"
                    f"/{len(data_list)} graphs"
                )

        all_graph_embeddings = torch.cat(all_graph_embs, dim=0)

        logger.info(
            f"Encoded {len(data_list)} graphs → "
            f"table embeddings shape: {all_graph_embeddings.shape}"
        )
        return all_node_embs, all_graph_embeddings

    # ── Config-driven factory ────────────────────

    @classmethod
    def from_config(cls, config: dict) -> "TableGraphEncoder":
        """
        Create encoder from the merged pipeline config.

        Reads from config["graph_embedding"] and config["graph"]["node_features"].
        """
        ge = config.get("graph_embedding", {})
        nf = config.get("graph", {}).get("node_features", {})

        # Calculate input_dim from enabled features
        input_dim = 0
        if nf.get("text_embedding", True):
            input_dim += config.get("embedding", {}).get("dimension", 768)
        if nf.get("position_encoding", True):
            input_dim += 2
        if nf.get("cell_type", True):
            input_dim += 3  # 3 node types
        if nf.get("numeric_flag", True):
            input_dim += 1

        return cls(
            input_dim=input_dim,
            hidden_dim=ge.get("hidden_dim", 256),
            output_dim=ge.get("output_dim", 768),
            num_layers=ge.get("num_layers", 2),
            dropout=ge.get("dropout", 0.1),
            normalize_output=ge.get("normalize_output", True),
            pool_method=config.get("indexing", {}).get(
                "pool_graph_embedding", "mean"
            ),
        )

    # ── Persistence ──────────────────────────────

    def save(self, path: str) -> None:
        """Save model weights to disk."""
        torch.save(self.state_dict(), path)
        logger.info(f"Saved TableGraphEncoder to {path}")

    def load(self, path: str, device: str = "cpu") -> "TableGraphEncoder":
        """Load model weights from disk."""
        self.load_state_dict(torch.load(path, map_location=device))
        logger.info(f"Loaded TableGraphEncoder from {path}")
        return self
