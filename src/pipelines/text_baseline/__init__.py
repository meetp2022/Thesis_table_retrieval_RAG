"""Pipeline 1: Text Baseline (Table Linearisation + BGE + FAISS)."""

from src.pipelines.text_baseline.pipeline import TextBaselinePipeline, linearise_table

__all__ = ["TextBaselinePipeline", "linearise_table"]
