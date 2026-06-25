"""
Configuration loader — merges base config with pipeline-specific overrides.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger


def load_config(
    pipeline: str = "graph",
    config_dir: str = "configs",
) -> Dict[str, Any]:
    """
    Load and merge configuration files.

    Args:
        pipeline: One of 'text', 'image', 'graph'
        config_dir: Path to the configs directory

    Returns:
        Merged configuration dictionary
    """
    config_path = Path(config_dir)

    # Load base config
    base_path = config_path / "base_config.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    with open(base_path, "r") as f:
        config = yaml.safe_load(f)

    # Map pipeline name to config file
    pipeline_map = {
        "text":           "pipeline1_text.yaml",
        "text_finetuned": "pipeline1_text_finetuned.yaml",
        "image":          "pipeline2_image.yaml",
        "graph":          "pipeline3_graph.yaml",
    }

    if pipeline not in pipeline_map:
        raise ValueError(f"Unknown pipeline: {pipeline}. Choose from {list(pipeline_map.keys())}")

    # Load and merge pipeline-specific config
    pipeline_path = config_path / pipeline_map[pipeline]
    if pipeline_path.exists():
        with open(pipeline_path, "r") as f:
            pipeline_config = yaml.safe_load(f)
        config = _deep_merge(config, pipeline_config)
        logger.info(f"Loaded pipeline config: {pipeline_path}")
    else:
        logger.warning(f"Pipeline config not found: {pipeline_path}, using base only")

    return config


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override dict into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_dataset_config(config: Dict, dataset_name: str) -> Dict:
    """Extract dataset-specific configuration."""
    datasets = config.get("datasets", {})
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(datasets.keys())}")
    return datasets[dataset_name]
