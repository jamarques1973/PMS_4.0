"""Logging setup utilities for PMS 4.0.0.

Provides a consistent logging configuration across all modules with
rotating file handlers and optional console output.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def ensure_directory(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def setup_logging(
    *,
    level: int | str = logging.INFO,
    log_dir: str | os.PathLike[str] = "/workspace/logs",
    log_file_name: str = "pms4.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
    use_console: bool = True,
) -> None:
    """Configure root logger for PMS.

    Parameters
    ----------
    level: int | str
        Logging level or name.
    log_dir: str | Path
        Directory path where the log file will be created.
    log_file_name: str
        Log file name.
    max_bytes: int
        Max size per log file before rotation.
    backup_count: int
        Number of rotated files to keep.
    use_console: bool
        Whether to log to console in addition to file.
    """

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    log_directory = Path(log_dir)
    ensure_directory(log_directory)
    log_file = log_directory / log_file_name

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    if use_console:
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            fmt="%(levelname)s | %(name)s | %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger.

    If logging has not been configured yet, set up a sensible default.
    """
    if not logging.getLogger().handlers:
        setup_logging()
    return logging.getLogger(name)

