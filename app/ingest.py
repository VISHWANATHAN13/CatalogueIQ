from __future__ import annotations

import os
import glob
import json
import csv
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    BSHTMLLoader,
)

from .utils import get_vector_store
from .logger import get_logger


logger = get_logger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(
    os.getenv(
        "DATA_DIR",
        str(PROJECT_ROOT / "data"),
    )
)

logger.info(
    "CatalogueIQ DATA_DIR = %s",
    DATA_DIR,
)


# ============================================================
# HELPERS
# ============================================================

def _get_category(path: str, base: str) -> str:
    """
    Determine category from the first directory under DATA_DIR.

    Examples:

        data/catalogue/products.csv
        -> catalogue

        data/policies/returns_refunds_policy.md
        -> policies

        data/faqs/buyer_faq.html
        -> faqs

    For product records, the actual product category comes from
    the CSV/JSON record itself when available.
    """

    relative_path = os.path.relpath(
        path,
        base,
    )

    parts = Path(relative_path).parts

    if len(parts) > 1:
        return parts[0]

    return "general"


def _safe_str(value: Any) -> str:
    """
    Convert values safely into strings.
    """

    if value is None:
        return ""

    return str(value).strip()


# ============================================================
# PRODUCT RECORD FORMATTING
# ============================================================

def _format_product_record(
    record: Dict[str, Any],
    source: str,
    category: str,
    record_index: int,
) -> Document:
    """
    Convert one product catalogue record into one LangChain Document.

    Product records are kept logically together whenever possible
    so comparative recommendation queries can retrieve complete
    product information.
    """

    # --------------------------------------------------------
    # PRODUCT IDENTIFIERS
    # --------------------------------------------------------

    product_id = (
        record.get("product_id")
        or record.get("id")
        or record.get("sku")
        or f"product_{record_index}"
    )

    name = (
        record.get("name")
        or record.get("product_name")
        or record.get("title")
        or "Unknown Product"
    )

    # --------------------------------------------------------
    # BASIC FIELDS
    # --------------------------------------------------------

    brand = record.get("brand", "")
    price = record.get("price", "")
    specs = record.get("specs", "")
    rating = record.get("rating", "")
    seller_id = record.get("seller_id", "")

    # --------------------------------------------------------
    # USE PRODUCT'S OWN CATEGORY IF AVAILABLE
    # --------------------------------------------------------

    record_category = (
        record.get("category")
        or category
    )

    # --------------------------------------------------------
    # BUILD SEARCHABLE PRODUCT TEXT
    # --------------------------------------------------------

    parts = [
        f"Product Name: {_safe_str(name)}",
        f"Product ID: {_safe_str(product_id)}",
        f"Category: {_safe_str(record_category)}",
    ]

    if brand:
        parts.append(
            f"Brand: {_safe_str(brand)}"
        )

    if price != "":
        parts.append(
            f"Price: {_safe_str(price)}"
        )

    if rating != "":
        parts.append(
            f"Rating: {_safe_str(rating)}"
        )

    if seller_id:
        parts.append(
            f"Seller ID: {_safe_str(seller_id)}"
        )

    # --------------------------------------------------------
    # SPECIFICATIONS
    # --------------------------------------------------------

    if isinstance(specs, dict):

        parts.append(
            "Specifications:"
        )

        for key, value in specs.items():

            parts.append(
                f"- {_safe_str(key)}: "
                f"{_safe_str(value)}"
            )

    elif specs:

        parts.append(
            f"Specifications: "
            f"{_safe_str(specs)}"
        )

    # --------------------------------------------------------
    # ADDITIONAL PRODUCT FIELDS
    # --------------------------------------------------------

    ignored_fields = {
        "product_id",
        "id",
        "sku",
        "name",
        "product_name",
        "title",
        "brand",
        "price",
        "specs",
        "rating",
        "seller_id",
        "category",
    }

    additional_fields = []

    for key, value in record.items():

        if key in ignored_fields:
            continue

        if value is None or value == "":
            continue

        if isinstance(value, dict):

            value = json.dumps(
                value,
                ensure_ascii=False,
            )

        additional_fields.append(
            f"{key}: {_safe_str(value)}"
        )

    if additional_fields:

        parts.append(
            "Additional Information:"
        )

        for item in additional_fields:

            parts.append(
                f"- {item}"
            )

    page_content = "\n".join(parts)

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata = {
        "source": source,
        "source_type": "product_catalogue",
        "category": _safe_str(record_category),
        "product_id": _safe_str(product_id),
        "product_name": _safe_str(name),
        "brand": _safe_str(brand),
        "seller_id": _safe_str(seller_id),
        "chunk_type": "product",
    }

    # --------------------------------------------------------
    # NUMERIC METADATA
    # --------------------------------------------------------

    try:

        if price != "":
            metadata["price"] = float(price)

    except (ValueError, TypeError):

        metadata["price"] = _safe_str(price)

    try:

        if rating != "":
            metadata["rating"] = float(rating)

    except (ValueError, TypeError):

        metadata["rating"] = _safe_str(rating)

    return Document(
        page_content=page_content,
        metadata=metadata,
    )


# ============================================================
# CSV LOADER
# ============================================================

def _load_csv(
    path: str,
    category: str,
) -> List[Document]:
    """
    Load product catalogue CSV.

    Each CSV row becomes one Document.
    """

    documents: List[Document] = []

    logger.info(
        "Loading CSV catalogue: %s",
        path,
    )

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for index, row in enumerate(
            reader,
            start=1,
        ):

            # Ignore completely empty rows.
            if not any(
                value not in (None, "")
                for value in row.values()
            ):
                continue

            documents.append(
                _format_product_record(
                    record=dict(row),
                    source=path,
                    category=category,
                    record_index=index,
                )
            )

    logger.info(
        "Loaded %d product records from %s",
        len(documents),
        path,
    )

    return documents


# ============================================================
# JSON LOADER
# ============================================================

def _load_json(
    path: str,
    category: str,
) -> List[Document]:
    """
    Load product catalogue JSON.

    Supports:

        [
            {...},
            {...}
        ]

    and:

        {
            "products": [
                {...},
                {...}
            ]
        }
    """

    documents: List[Document] = []

    logger.info(
        "Loading JSON catalogue: %s",
        path,
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    # --------------------------------------------------------
    # DETERMINE RECORD STRUCTURE
    # --------------------------------------------------------

    if isinstance(data, list):

        records = data

    elif isinstance(data, dict):

        if isinstance(
            data.get("products"),
            list,
        ):

            records = data["products"]

        elif isinstance(
            data.get("items"),
            list,
        ):

            records = data["items"]

        elif isinstance(
            data.get("data"),
            list,
        ):

            records = data["data"]

        else:

            records = [data]

    else:

        raise ValueError(
            f"Unsupported JSON structure in {path}"
        )

    # --------------------------------------------------------
    # CONVERT RECORDS TO DOCUMENTS
    # --------------------------------------------------------

    for index, record in enumerate(
        records,
        start=1,
    ):

        if not isinstance(
            record,
            dict,
        ):

            logger.warning(
                "Skipping non-object JSON "
                "record %d in %s",
                index,
                path,
            )

            continue

        documents.append(
            _format_product_record(
                record=record,
                source=path,
                category=category,
                record_index=index,
            )
        )

    logger.info(
        "Loaded %d product records from %s",
        len(documents),
        path,
    )

    return documents


# ============================================================
# GENERAL DOCUMENT LOADERS
# ============================================================

def _load_general_document(
    path: str,
    category: str,
) -> List[Document]:
    """
    Load policy, FAQ, seller guide, taxonomy,
    review summary, or other knowledge documents.
    """

    ext = Path(path).suffix.lower()

    logger.debug(
        "Loading document: %s (category=%s)",
        path,
        category,
    )

    if ext == ".md":

        return UnstructuredMarkdownLoader(
            path
        ).load()

    if ext == ".pdf":

        return PyMuPDFLoader(
            path
        ).load()

    if ext == ".docx":

        return UnstructuredWordDocumentLoader(
            path
        ).load()

    if ext == ".txt":

        return TextLoader(
            path,
            encoding="utf-8",
        ).load()

    if ext in {".html", ".htm"}:

        return BSHTMLLoader(
            path
        ).load()

    return []


# ============================================================
# LOAD ALL DOCUMENTS
# ============================================================

def _load_docs(
    base: str | Path = DATA_DIR,
) -> List[Document]:
    """
    Load CatalogueIQ knowledge-base files.

    Product data:
        .csv
        .json

    Knowledge documents:
        .md
        .pdf
        .docx
        .txt
        .html
        .htm
    """

    docs: List[Document] = []

    base = str(base)

    logger.info(
        "Loading CatalogueIQ knowledge base from: %s",
        base,
    )

    if not os.path.exists(base):

        raise FileNotFoundError(
            f"DATA_DIR does not exist: {base}"
        )

    for path in glob.glob(
        os.path.join(
            base,
            "**",
            "*",
        ),
        recursive=True,
    ):

        # Skip directories.
        if os.path.isdir(path):
            continue

        # Skip hidden files.
        if Path(path).name.startswith("."):
            continue

        ext = Path(path).suffix.lower()

        category = _get_category(
            path,
            base,
        )

        try:

            # ------------------------------------------------
            # PRODUCT CATALOGUE
            # ------------------------------------------------

            if ext == ".csv":

                loaded_docs = _load_csv(
                    path,
                    category,
                )

            elif ext == ".json":

                loaded_docs = _load_json(
                    path,
                    category,
                )

            # ------------------------------------------------
            # GENERAL KNOWLEDGE
            # ------------------------------------------------

            elif ext in {
                ".md",
                ".pdf",
                ".docx",
                ".txt",
                ".html",
                ".htm",
            }:

                loaded_docs = _load_general_document(
                    path,
                    category,
                )

            else:

                logger.debug(
                    "Skipping unsupported file: %s",
                    path,
                )

                continue

            # ------------------------------------------------
            # ATTACH METADATA TO GENERAL DOCUMENTS
            # ------------------------------------------------

            for document in loaded_docs:

                if document.metadata.get(
                    "source_type"
                ) != "product_catalogue":

                    document.metadata.update(
                        {
                            "source": path,
                            "source_type":
                                "knowledge_document",
                            "category": category,
                        }
                    )

                docs.append(document)

            if loaded_docs:

                logger.info(
                    "Loaded %d document(s) from %s",
                    len(loaded_docs),
                    path,
                )

        except Exception:

            logger.error(
                "Failed to load %s",
                path,
                exc_info=True,
            )

    logger.info(
        "Total documents loaded: %d",
        len(docs),
    )

    return docs


# ============================================================
# CHUNKING
# ============================================================

def _chunk(
    docs: List[Document],
) -> List[Document]:
    """
    Chunk CatalogueIQ knowledge documents.

    Product records remain together whenever possible because
    comparative queries need complete product information.

    General knowledge documents are split using
    RecursiveCharacterTextSplitter.
    """

    logger.info(
        "Starting CatalogueIQ chunking: %d documents",
        len(docs),
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks: List[Document] = []

    product_count = 0
    general_count = 0

    for document in docs:

        source_type = document.metadata.get(
            "source_type"
        )

        # ----------------------------------------------------
        # PRODUCT RECORD
        # ----------------------------------------------------

        if source_type == "product_catalogue":

            if len(
                document.page_content
            ) <= 1200:

                # Keep complete product together.
                document.metadata[
                    "chunk_type"
                ] = "product"

                chunks.append(document)

            else:

                product_chunks = (
                    splitter.split_documents(
                        [document]
                    )
                )

                for chunk in product_chunks:

                    chunk.metadata[
                        "chunk_type"
                    ] = "product"

                    chunks.append(chunk)

            product_count += 1

        # ----------------------------------------------------
        # GENERAL KNOWLEDGE
        # ----------------------------------------------------

        else:

            general_chunks = (
                splitter.split_documents(
                    [document]
                )
            )

            for chunk in general_chunks:

                chunk.metadata[
                    "chunk_type"
                ] = "knowledge"

                chunks.append(chunk)

            general_count += 1

    logger.info(
        "Chunking complete: %d total chunks",
        len(chunks),
    )

    logger.info(
        "CatalogueIQ product documents processed: %d",
        product_count,
    )

    logger.info(
        "CatalogueIQ knowledge documents processed: %d",
        general_count,
    )

    return chunks


# ============================================================
# COMPLETE INGESTION PIPELINE
# ============================================================

async def run_ingest_async() -> dict:
    """
    Run the complete CatalogueIQ ingestion pipeline.

        Load
          ↓
        Chunk
          ↓
        LanceDB
          ↓
        OpenAI Embeddings
          ↓
        Store vectors
    """

    logger.info("=" * 70)

    logger.info(
        "CATALOGUEIQ INGESTION PIPELINE STARTED"
    )

    logger.info("=" * 70)

    # --------------------------------------------------------
    # STEP 1: LOAD
    # --------------------------------------------------------

    docs = _load_docs()

    logger.info(
        "Step 1/4: Loading → %d documents/records",
        len(docs),
    )

    if not docs:

        raise ValueError(
            "No documents were found in DATA_DIR. "
            "Check the CatalogueIQ data directory."
        )

    # --------------------------------------------------------
    # STEP 2: CHUNK
    # --------------------------------------------------------

    chunks = _chunk(docs)

    logger.info(
        "Step 2/4: Chunking → %d chunks",
        len(chunks),
    )

    if not chunks:

        raise ValueError(
            "Chunking produced zero documents."
        )

    # --------------------------------------------------------
    # STEP 3: LANCEDB
    # --------------------------------------------------------

    logger.info(
        "Step 3/4: Connecting to LanceDB..."
    )

    store = get_vector_store()

    logger.info(
        "Step 3/4: LanceDB vector store ready"
    )

    # --------------------------------------------------------
    # STEP 4: EMBEDDING + STORAGE
    # --------------------------------------------------------

    logger.info(
        "Step 4/4: Embedding and storing "
        "%d chunks...",
        len(chunks),
    )

    await store.aadd_documents(
        chunks
    )

    logger.info(
        "Step 4/4: Embedding and storage complete"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    logger.info("=" * 70)

    logger.info(
        "CATALOGUEIQ INGESTION PIPELINE COMPLETED"
    )

    logger.info(
        "Documents/records: %d",
        len(docs),
    )

    logger.info(
        "Chunks: %d",
        len(chunks),
    )

    logger.info("=" * 70)

    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "vector_store": "LanceDB",
    }