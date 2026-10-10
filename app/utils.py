from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_lancedb import LanceDB

from .logger import get_logger


logger = get_logger(__name__)


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv()


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LANCEDB_DIR = Path(
    os.getenv(
        "LANCEDB_DIR",
        str(PROJECT_ROOT / "data" / "lancedb"),
    )
)

LANCEDB_TABLE = os.getenv(
    "LANCEDB_TABLE",
    "catalogueiq",
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small",
)


embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL
)


logger.info(
    "OpenAIEmbeddings initialized: model=%s",
    EMBEDDING_MODEL,
)


# ============================================================
# LANCEDB CONFIGURATION
# ============================================================

# Make sure the local LanceDB directory exists.
LANCEDB_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


logger.info(
    "LanceDB directory: %s",
    LANCEDB_DIR,
)

logger.info(
    "LanceDB table: %s",
    LANCEDB_TABLE,
)


# ============================================================
# VECTOR STORE FACTORY
# ============================================================

def get_vector_store() -> LanceDB:
    """
    Create and return the CatalogueIQ LanceDB vector store.
    
    """

    logger.info(
        "Creating CatalogueIQ LanceDB vector store"
    )

    logger.info(
        "LanceDB path: %s",
        LANCEDB_DIR,
    )

    logger.info(
        "LanceDB table: %s",
        LANCEDB_TABLE,
    )

    # store = LanceDB(
    #     embedding=embeddings,
    #     uri=str(LANCEDB_DIR),
    #     table_name=LANCEDB_TABLE,
    # )
    store = LanceDB(
    embedding_function=embeddings,
    uri=str(LANCEDB_DIR),
    table_name=LANCEDB_TABLE,
    )

    logger.info(
        "CatalogueIQ LanceDB vector store ready"
    )

    return store