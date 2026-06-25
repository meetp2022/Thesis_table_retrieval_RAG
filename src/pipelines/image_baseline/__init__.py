"""Image Baseline Pipeline (Pipeline 2) — table-as-image + CLIP/ColPali retrieval."""

from src.pipelines.image_baseline.pipeline import (
    ImageBaselinePipeline,
    ImageRetriever,
)

__all__ = ["ImageBaselinePipeline", "ImageRetriever"]
