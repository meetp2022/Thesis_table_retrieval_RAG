"""
Logging configuration using loguru.
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    """
    Configure loguru for the project.

    Args:
        log_dir: Directory to store log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console handler — clean, colourful output
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> — <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # File handler — detailed logs with rotation
    logger.add(
        str(log_path / "experiment_{time:YYYY-MM-DD}.log"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
    )

    logger.info("Logging initialised.")
