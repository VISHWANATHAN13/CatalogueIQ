"""
Logging configuration for the CatalogueIQ RAG system.

Provides a consistent logging setup across all application modules.

Logs are written to:
    1. Console (stdout)
    2. Rotating log file: logs/app.log

The rotating file handler prevents the log file from growing
indefinitely during ingestion, retrieval, and evaluation runs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


# ============================================================
# LOG DIRECTORY
# ============================================================

# Project root:
#
# CatalogueIQ/
# ├── app/
# │   └── logger.py
# └── logs/
#
# Therefore:
# Path(__file__).resolve().parent.parent
# points to the project root.

LOG_DIR = (
    Path(__file__).resolve().parent.parent / "logs"
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE = LOG_DIR / "app.log"


# ============================================================
# FORMATTER
# ============================================================

_LOG_FORMAT = (
    "[%(asctime)s] "
    "%(levelname)-8s | "
    "%(name)-18s | "
    "%(funcName)-25s | "
    "%(message)s"
)

_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_formatter = logging.Formatter(
    _LOG_FORMAT,
    datefmt=_DATE_FORMAT,
)


# ============================================================
# LOGGER FACTORY
# ============================================================

def get_logger(
    name: str,
) -> logging.Logger:
    """
    Return a configured logger for a CatalogueIQ module.

    The logger writes to:

    - Console (stdout)
    - Rotating file: logs/app.log

    The file handler:
    - Maximum file size: 10 MB
    - Backup files: 5

    Example:

        from .logger import get_logger

        logger = get_logger(__name__)

        logger.info("CatalogueIQ started")
        logger.warning("No documents found")
        logger.error("Retrieval failed")
    """

    logger = logging.getLogger(name)

    # --------------------------------------------------------
    # Prevent duplicate handlers
    # --------------------------------------------------------

    if logger.handlers:
        return logger

    logger.setLevel(
        logging.DEBUG
    )

    # --------------------------------------------------------
    # Console Handler
    # --------------------------------------------------------

    console_handler = logging.StreamHandler(
        sys.stdout
    )

    console_handler.setLevel(
        logging.DEBUG
    )

    console_handler.setFormatter(
        _formatter
    )

    logger.addHandler(
        console_handler
    )

    # --------------------------------------------------------
    # Rotating File Handler
    # --------------------------------------------------------

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )

    file_handler.setLevel(
        logging.DEBUG
    )

    file_handler.setFormatter(
        _formatter
    )

    logger.addHandler(
        file_handler
    )

    # Prevent messages from being duplicated
    # by the root logger.
    logger.propagate = False

    return logger