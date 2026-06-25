"""
Contrastive training for the GraphSAGE table encoder.

Aligns graph embeddings (from GraphSAGE) with text query embeddings
(from BGE) using InfoNCE contrastive loss with in-batch negatives.

Training pairs:
    - Positive: (graph_embedding_of_table_i, query_embedding_of_question_i)
    - Negatives: all other questions in the same batch (in-batch negatives)

After training, FAISS cosine similarity between graph-encoded tables
and BGE-encoded queries becomes meaningful (vs ~0.003 with random weights).

Usage:
    >>> from src.graph.train import ContrastiveTrainer
    >>> trainer = ContrastiveTrainer.from_config(config)
    >>> trainer.train(train_records, val_records)
    >>> trainer.encoder.save("models/graph_encoder.pt")

References / attribution
------------------------
- InfoNCE contrastive loss with in-batch negatives: van den Oord et al.,
  "Representation Learning with Contrastive Predictive Coding", 2018.
- Aligning two encoders into a shared retrieval space follows the
  dense-passage-retrieval pattern of Karpukhin et al., "Dense Passage
  Retrieval for Open-Domain QA", EMNLP 2020.
- The contrastive training loop, hard-negative mining and config wiring
  in this file are this project's own code.
"""

import hashlib
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from torch_geometric.data import Data, Batch

from src.data.dataset_loader import TableQARecord
from src.graph.table_to_graph import build_graphs_from_config
from src.graph.feature_extraction import GraphFeatureExtractor
from src.graph.graph_embedding import TableGraphEncoder


# ────────────────────────────────────────────────
#  InfoNCE loss
# ────────────────────────────────────────────────

def info_nce_loss(
    graph_embs: torch.Tensor,
    query_embs: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric InfoNCE loss with in-batch negatives.

    Parameters
    ----------
    graph_embs : Tensor (B, D)
        L2-normalised graph embeddings from GraphSAGE.
    query_embs : Tensor (B, D)
        L2-normalised query embeddings from BGE.
    temperature : float
        Temperature scaling for logits (lower = sharper).

    Returns
    -------
    Tensor (scalar)
        Mean of graph→query and query→graph contrastive losses.
    """
    # Cosine similarity matrix: (B, B)
    logits = torch.mm(graph_embs, query_embs.t()) / temperature

    # Labels: diagonal entries are positives
    labels = torch.arange(logits.size(0), device=logits.device)

    # Symmetric loss
    loss_g2q = F.cross_entropy(logits, labels)
    loss_q2g = F.cross_entropy(logits.t(), labels)

    return (loss_g2q + loss_q2g) / 2


# ────────────────────────────────────────────────
#  Hard negative mining
# ────────────────────────────────────────────────

def mine_hard_negatives(
    query_embs: np.ndarray,
    num_hard: int = 8,
    exclude_top: int = 1,
) -> np.ndarray:
    """
    For each query, find the most similar OTHER queries by cosine
    similarity. Their corresponding tables are hard negatives —
    tables whose questions look similar but are different tables.

    Parameters
    ----------
    query_embs : ndarray (N, D)
        L2-normalised query embeddings.
    num_hard : int
        Number of hard negatives to mine per sample.
    exclude_top : int
        Skip the top-k most similar (exclude self, k=1).

    Returns
    -------
    hard_neg_indices : ndarray (N, num_hard)
        Indices of hard negatives for each sample.
    """
    # Cosine similarity matrix (N, N)
    sim = query_embs @ query_embs.T

    # Zero out self-similarity
    np.fill_diagonal(sim, -1.0)

    # For each row, get indices of top-k most similar queries
    # These are the hardest negatives (similar question, different table)
    hard_neg_indices = np.zeros((len(sim), num_hard), dtype=np.int64)
    for i in range(len(sim)):
        top_indices = np.argpartition(sim[i], -num_hard)[-num_hard:]
        # Sort by similarity descending
        top_indices = top_indices[np.argsort(sim[i][top_indices])[::-1]]
        hard_neg_indices[i] = top_indices

    logger.info(
        f"Mined {num_hard} hard negatives per sample "
        f"(avg hard-neg similarity: {sim[np.arange(len(sim))[:, None], hard_neg_indices].mean():.3f})"
    )
    return hard_neg_indices


# ────────────────────────────────────────────────
#  Training result
# ────────────────────────────────────────────────

@dataclass
class TrainingResult:
    """Summary of a training run."""
    epochs_completed: int
    best_val_loss: float
    final_train_loss: float
    training_time_seconds: float
    history: List[Dict[str, float]]

    def summary(self) -> str:
        return (
            f"Training complete: {self.epochs_completed} epochs, "
            f"best val loss={self.best_val_loss:.4f}, "
            f"final train loss={self.final_train_loss:.4f}, "
            f"time={self.training_time_seconds:.1f}s"
        )


# ────────────────────────────────────────────────
#  Contrastive trainer
# ────────────────────────────────────────────────

class ContrastiveTrainer:
    """
    Trains GraphSAGE via contrastive learning to align
    graph embeddings with BGE query embeddings.

    Parameters
    ----------
    encoder : TableGraphEncoder
        The GraphSAGE model to train.
    feature_extractor : GraphFeatureExtractor
        Converts NX graphs → PyG Data.
    config : dict
        Merged pipeline config.
    lr : float
        Learning rate.
    temperature : float
        InfoNCE temperature parameter.
    epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size (number of table-question pairs).
    patience : int
        Early stopping patience (epochs without improvement).
    save_dir : str or None
        Directory to save best model checkpoint.
    """

    def __init__(
        self,
        encoder: TableGraphEncoder,
        feature_extractor: GraphFeatureExtractor,
        config: dict,
        lr: float = 1e-4,
        temperature: float = 0.07,
        epochs: int = 20,
        batch_size: int = 16,
        patience: int = 5,
        save_dir: Optional[str] = None,
        num_hard_negatives: int = 8,
    ):
        self.encoder = encoder
        self.feature_extractor = feature_extractor
        self.config = config
        self.lr = lr
        self.temperature = temperature
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.save_dir = save_dir
        self.num_hard_negatives = num_hard_negatives

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder.to(self.device)

        self._query_encoder = None  # lazy

    @property
    def query_encoder(self):
        """Lazy-load query encoder for embedding questions."""
        if self._query_encoder is None:
            from src.pipelines.shared.query_encoder import QueryEncoder
            self._query_encoder = QueryEncoder.from_config(self.config)
        return self._query_encoder

    # ── Cache helpers ────────────────────────────

    def _cache_key(self, records: List[TableQARecord]) -> str:
        """Generate a cache key based on record IDs and embedding model."""
        ids = "".join(r.id for r in records)
        model = self.config.get("embedding", {}).get("model", "bge")
        return hashlib.md5(f"{ids}{model}".encode()).hexdigest()[:12]

    def _cache_path(self, key: str) -> Path:
        save_dir = self.save_dir or "models/graph_encoder"
        return Path(save_dir) / "cache" / f"pyg_{key}.pkl"

    def _load_cache(self, key: str) -> Optional[Tuple[List[Data], np.ndarray]]:
        path = self._cache_path(key)
        if path.exists():
            logger.info(f"Loading cached PyG data from {path}")
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cache(
        self,
        key: str,
        pyg_data: List[Data],
        query_embs: np.ndarray,
    ) -> None:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump((pyg_data, query_embs), f)
        logger.info(f"Cached PyG data to {path} ({path.stat().st_size / 1e6:.1f} MB)")

    # ── Prepare data ────────────────────────────

    def _prepare_data(
        self,
        records: List[TableQARecord],
        use_cache: bool = True,
    ) -> Tuple[List[Data], np.ndarray]:
        """
        Convert records → (PyG Data list, query embeddings).

        Results are cached to disk so repeated training runs skip the
        slow BGE node-embedding step (which takes ~1 hour for 11k tables).

        Returns
        -------
        pyg_data : list[Data]
            PyG Data objects with graph features.
        query_embeddings : ndarray (N, 768)
            BGE embeddings of questions (normalised).
        """
        logger.info(f"Preparing data: {len(records)} records...")

        # Deduplicate by ID (keep first occurrence)
        seen = set()
        unique = []
        for r in records:
            if r.id not in seen:
                seen.add(r.id)
                unique.append(r)

        # Try cache first
        if use_cache:
            key = self._cache_key(unique)
            cached = self._load_cache(key)
            if cached is not None:
                pyg_data, query_embs = cached
                logger.info(
                    f"Cache hit: {len(pyg_data)} graphs + "
                    f"{query_embs.shape} query embeddings"
                )
                return pyg_data, query_embs

        # Build graphs → PyG features (slow: BGE embeds every node)
        graphs = build_graphs_from_config(unique, self.config)
        pyg_data = self.feature_extractor.convert_batch(graphs)

        # Embed questions with BGE
        questions = [r.question for r in unique]
        query_embs = self.query_encoder.encode(questions, batch_size=64)

        logger.info(
            f"Prepared {len(pyg_data)} graphs + "
            f"{query_embs.shape} query embeddings"
        )

        # Save to cache for next run
        if use_cache:
            self._save_cache(key, pyg_data, query_embs)

        return pyg_data, query_embs

    # ── Train one epoch ─────────────────────────

    def _build_hard_negative_batches(
        self,
        n: int,
        hard_neg_indices: Optional[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Build batches that include hard negatives for each anchor.

        Each batch contains `batch_size` anchors. For each anchor,
        its hard negatives are also included in the batch so the
        contrastive loss must discriminate against confusing tables.

        Falls back to random batching if hard_neg_indices is None.
        """
        shuffled = np.random.permutation(n)

        if hard_neg_indices is None:
            # Random batching (no hard negatives)
            batches = []
            for start in range(0, n, self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                if len(batch) >= 2:
                    batches.append(batch)
            return batches

        # Hard-negative-aware batching:
        # For each anchor group, include the anchor + some hard negatives
        # This ensures the batch contains confusing pairs
        anchors_per_batch = max(self.batch_size // 2, 4)
        batches = []

        for start in range(0, n, anchors_per_batch):
            anchor_ids = shuffled[start : start + anchors_per_batch]
            if len(anchor_ids) < 2:
                continue

            # Collect anchor indices + their hard negatives
            batch_set = set(anchor_ids.tolist())
            for aid in anchor_ids:
                # Add a few hard negatives per anchor (not all, to keep batch manageable)
                hn = hard_neg_indices[aid]
                # Add top 2-3 hard negatives per anchor
                for h in hn[:3]:
                    batch_set.add(int(h))

            batch = np.array(list(batch_set), dtype=np.int64)
            if len(batch) >= 2:
                batches.append(batch)

        return batches

    def _train_epoch(
        self,
        pyg_data: List[Data],
        query_embs: np.ndarray,
        optimizer: torch.optim.Optimizer,
        hard_neg_indices: Optional[np.ndarray] = None,
    ) -> float:
        """Run one training epoch. Returns average loss."""
        self.encoder.train()

        n = len(pyg_data)
        batches = self._build_hard_negative_batches(n, hard_neg_indices)

        total_loss = 0.0
        num_batches = 0

        for batch_idx in batches:
            # Batch graphs
            batch_data = [pyg_data[i] for i in batch_idx]
            batched = Batch.from_data_list(batch_data).to(self.device)

            # Forward through GraphSAGE
            _node_embs, graph_embs = self.encoder(batched)

            # Get corresponding query embeddings
            batch_query_embs = torch.tensor(
                query_embs[batch_idx],
                dtype=torch.float32,
                device=self.device,
            )
            batch_query_embs = F.normalize(batch_query_embs, p=2, dim=-1)

            # Contrastive loss
            loss = info_nce_loss(graph_embs, batch_query_embs, self.temperature)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    # ── Validate ────────────────────────────────

    @torch.no_grad()
    def _validate(
        self,
        pyg_data: List[Data],
        query_embs: np.ndarray,
    ) -> float:
        """Compute validation loss. Returns average loss."""
        self.encoder.eval()

        n = len(pyg_data)
        total_loss = 0.0
        num_batches = 0

        for start in range(0, n, self.batch_size):
            batch_idx = list(range(start, min(start + self.batch_size, n)))
            if len(batch_idx) < 2:
                continue

            batch_data = [pyg_data[i] for i in batch_idx]
            batched = Batch.from_data_list(batch_data).to(self.device)

            _node_embs, graph_embs = self.encoder(batched)

            batch_query_embs = torch.tensor(
                query_embs[batch_idx],
                dtype=torch.float32,
                device=self.device,
            )
            batch_query_embs = F.normalize(batch_query_embs, p=2, dim=-1)

            loss = info_nce_loss(graph_embs, batch_query_embs, self.temperature)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    # ── Main training loop ──────────────────────

    def train(
        self,
        train_records: List[TableQARecord],
        val_records: Optional[List[TableQARecord]] = None,
    ) -> TrainingResult:
        """
        Train the GraphSAGE encoder with contrastive learning.

        Parameters
        ----------
        train_records : list[TableQARecord]
            Training set (table + question pairs).
        val_records : list[TableQARecord], optional
            Validation set. If None, uses 10% of train as validation.

        Returns
        -------
        TrainingResult
            Training summary with loss history.
        """
        start_time = time.perf_counter()

        # Auto-split if no validation set
        if val_records is None:
            n = len(train_records)
            split_idx = max(int(n * 0.9), 1)
            np.random.seed(self.config.get("experiment", {}).get("seed", 42))
            shuffled = list(train_records)
            np.random.shuffle(shuffled)
            train_records = shuffled[:split_idx]
            val_records = shuffled[split_idx:]
            logger.info(
                f"Auto-split: {len(train_records)} train, "
                f"{len(val_records)} val"
            )

        # Prepare data (cached after first run — skips slow BGE embedding)
        train_pyg, train_query_embs = self._prepare_data(train_records, use_cache=True)
        val_pyg, val_query_embs = self._prepare_data(val_records, use_cache=True)

        # Mine hard negatives from query embedding similarity
        hard_neg_indices = None
        if self.num_hard_negatives > 0:
            hard_neg_indices = mine_hard_negatives(
                train_query_embs,
                num_hard=self.num_hard_negatives,
            )

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )

        # LR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=self.lr * 0.01,
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        history: List[Dict[str, float]] = []

        logger.info(
            f"Starting contrastive training: {self.epochs} epochs, "
            f"lr={self.lr}, temp={self.temperature}, batch={self.batch_size}, "
            f"hard_negatives={self.num_hard_negatives}"
        )

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.perf_counter()

            train_loss = self._train_epoch(
                train_pyg, train_query_embs, optimizer,
                hard_neg_indices=hard_neg_indices,
            )
            val_loss = self._validate(val_pyg, val_query_embs)

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            epoch_time = time.perf_counter() - epoch_start

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "time": epoch_time,
            })

            logger.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"lr={current_lr:.6f} | "
                f"{epoch_time:.1f}s"
            )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                if self.save_dir:
                    save_path = Path(self.save_dir)
                    save_path.mkdir(parents=True, exist_ok=True)
                    self.encoder.save(str(save_path / "best_encoder.pt"))
                    logger.info(f"  Saved best model (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {self.patience} epochs)"
                    )
                    break

        total_time = time.perf_counter() - start_time

        # Load best model if saved
        if self.save_dir:
            best_path = Path(self.save_dir) / "best_encoder.pt"
            if best_path.exists():
                self.encoder.load(str(best_path))
                logger.info("Loaded best checkpoint")

        result = TrainingResult(
            epochs_completed=len(history),
            best_val_loss=best_val_loss,
            final_train_loss=history[-1]["train_loss"] if history else 0.0,
            training_time_seconds=total_time,
            history=history,
        )

        logger.info(result.summary())
        return result

    # ── Config-driven factory ────────────────────

    @classmethod
    def from_config(
        cls,
        config: dict,
        encoder: Optional[TableGraphEncoder] = None,
        feature_extractor: Optional[GraphFeatureExtractor] = None,
    ) -> "ContrastiveTrainer":
        """
        Build trainer from merged pipeline config.

        Parameters
        ----------
        config : dict
            Merged config (base_config + pipeline3_graph).
        encoder : TableGraphEncoder, optional
            Pre-built encoder. If None, creates from config.
        feature_extractor : GraphFeatureExtractor, optional
            Pre-built feature extractor. If None, creates from config.
        """
        if encoder is None:
            encoder = TableGraphEncoder.from_config(config)
        if feature_extractor is None:
            feature_extractor = GraphFeatureExtractor.from_config(config)

        train_cfg = config.get("training", {})

        return cls(
            encoder=encoder,
            feature_extractor=feature_extractor,
            config=config,
            lr=train_cfg.get("lr", 1e-4),
            temperature=train_cfg.get("temperature", 0.07),
            epochs=train_cfg.get("epochs", 20),
            batch_size=train_cfg.get("batch_size", 16),
            patience=train_cfg.get("patience", 5),
            save_dir=train_cfg.get("save_dir", "models/graph_encoder"),
            num_hard_negatives=train_cfg.get("num_hard_negatives", 8),
        )
