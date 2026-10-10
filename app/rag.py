from __future__ import annotations

from typing import List, Tuple, Optional
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.docstore.document import Document

from langchain_core.globals import set_llm_cache

from .utils import get_vector_store
from .logger import get_logger

# from langchain_cohere import CohereRerank
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
)


logger = get_logger(__name__)


# ============================================================
# CATALOGUEIQ SYSTEM PROMPT
# ============================================================

SYSTEM = """
You are CatalogueIQ, an e-commerce knowledge assistant for
ShopSmart India.

Your job is to answer questions about:

1. Products
2. Product specifications
3. Prices and ratings
4. Product comparisons and recommendations
5. Returns and refunds
6. Buyer FAQs
7. Seller policies and onboarding
8. Category-specific guidelines

You MUST answer using only the information present in the
retrieved context.

IMPORTANT GROUNDING RULES:

- Do not invent product specifications.
- Do not invent prices, ratings, brands, sellers, policies,
  return windows, or eligibility conditions.
- If the requested information is not present in the context,
  clearly say that the information is not available in the
  CatalogueIQ knowledge base.
- Do not use outside knowledge.
- When multiple products are relevant, compare them using the
  information actually present in the context.
- For recommendation questions, consider ALL relevant retrieved
  products before recommending one.
- Never recommend a product simply because it appears first.
- Numerical constraints such as price, rating, battery life,
  storage, size, or other specifications must be checked against
  the retrieved product information.
- If a requested constraint cannot be verified from the context,
  say that it cannot be verified.

COMPARATIVE QUESTIONS:

When the user asks for the "best", "which is better",
"compare", "top", "under ₹X", or similar recommendation:

1. Identify all relevant products in the retrieved context.
2. Apply the user's explicit constraints first.
3. Compare the remaining products.
4. Explain the recommendation using evidence from the context.
5. If multiple products satisfy the requirements, mention the
   strongest alternatives when useful.

POLICY QUESTIONS:

For returns, refunds, seller policies, FAQs, or eligibility:

- Follow the policy information in the retrieved context.
- Prefer the most recent policy when an effective date is given.
- Do not assume that a general policy applies when a more
  specific policy is present.

TRACEABILITY:

Use product names and relevant source information when useful.
Keep answers concise but sufficiently detailed to show why the
answer was reached.
"""


# ============================================================
# PROMPT
# ============================================================

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM,
        ),
        (
            "user",
            """
Question:
{input}

Retrieved CatalogueIQ Context:
{context}

Answer the question using only the retrieved context.
""",
        ),
    ]
)


# ============================================================
# LLM CACHE
# ============================================================

# Semantic caching is intentionally disabled.
#
# CatalogueIQ contains many questions that can be semantically
# similar while requiring different product/policy evidence.
# Returning a cached answer for a different question can therefore
# produce a plausible but incorrect response.

set_llm_cache(None)

logger.info(
    "CatalogueIQ semantic LLM caching disabled"
)


# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

def _get_retrieval_k(
    category: Optional[str] = None,
) -> int:
    """
    Determine the number of documents retrieved from PostgreSQL.

    Default is deliberately higher than the old generic RAG
    configuration because CatalogueIQ contains comparative
    recommendation queries.

    Example:

        User:
        "Which wireless earbuds under ₹3000 have the best
        battery life?"

    We need multiple candidate products before reranking.
    """

    default_k = 10

    try:
        return max(
            1,
            int(
                os.getenv(
                    "RETRIEVAL_K",
                    str(default_k),
                )
            ),
        )

    except ValueError:

        logger.warning(
            "Invalid RETRIEVAL_K value. "
            "Using default=%d",
            default_k,
        )

        return default_k


def _get_rerank_top_n() -> int:
    """
    Number of documents retained after Cohere reranking.

    CatalogueIQ benefits from keeping several products because
    recommendation queries require comparison rather than
    single-document lookup.
    """

    default_top_n = 5

    try:

        return max(
            1,
            int(
                os.getenv(
                    "RERANK_TOP_N",
                    str(default_top_n),
                )
            ),
        )

    except ValueError:

        logger.warning(
            "Invalid RERANK_TOP_N value. "
            "Using default=%d",
            default_top_n,
        )

        return default_top_n


# ============================================================
# BUILD RAG CHAIN
# ============================================================

async def _build_chain(
    category: Optional[str] = None,
):
    """
    Build the complete CatalogueIQ RAG pipeline.

    Pipeline:

        User Question
             |
             v
        PostgreSQL Vector Search
             |
             v
        Retrieve top-K candidates
             |
             v
        Cohere Reranker
             |
             v
        Keep top-N relevant documents
             |
             v
        Grounded Prompt
             |
             v
        GPT-4o-mini
             |
             v
        CatalogueIQ Answer
    """

    logger.info(
        "Building CatalogueIQ RAG chain "
        "(category=%s)",
        category,
    )

    # --------------------------------------------------------
    # VECTOR STORE
    # --------------------------------------------------------

    store = get_vector_store()
    logger.info(
        "PostgreSQL vector store obtained"
    )

    # --------------------------------------------------------
    # RETRIEVAL SETTINGS
    # --------------------------------------------------------

    retrieval_k = _get_retrieval_k(
        category
    )

    rerank_top_n = _get_rerank_top_n()

    logger.info(
        "Retrieval configuration: "
        "retrieval_k=%d, rerank_top_n=%d",
        retrieval_k,
        rerank_top_n,
    )

    search_kwargs = {
        "k": retrieval_k,
    }

    # --------------------------------------------------------
    # CATEGORY FILTER
    # --------------------------------------------------------

    if category:

        search_kwargs["filter"] = {
            "category": category,
        }

        logger.info(
            "Metadata filter applied: category=%s",
            category,
        )

    # --------------------------------------------------------
    # BASE VECTOR RETRIEVER
    # --------------------------------------------------------

    base_retriever = store.as_retriever(
        search_kwargs=search_kwargs
    )

    logger.info(
        "LanceDB vector retriever created"
    )

    # --------------------------------------------------------
    # COHERE RERANKER
    # --------------------------------------------------------

    logger.info(
        "Initializing Cohere reranker: "
        "model=rerank-multilingual-v3.0, top_n=%d",
        rerank_top_n,
    )

    # compressor = CohereRerank(
    #     top_n=rerank_top_n,
    #     model="rerank-multilingual-v3.0",
    # )

    # retriever = ContextualCompressionRetriever(
    #     base_retriever=base_retriever,
    #     base_compressor=compressor,
    # )
    retriever = store.as_retriever(
    search_kwargs={
        "k": 10
    }
    )

    logger.info(
        "ContextualCompressionRetriever created"
    )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    llm = ChatOpenAI(
        model=os.getenv(
            "CATALOGUEIQ_LLM_MODEL",
            "gpt-4o-mini",
        ),
    )

    logger.info(
        "CatalogueIQ LLM initialized: %s",
        os.getenv(
            "CATALOGUEIQ_LLM_MODEL",
            "gpt-4o-mini",
        ),
    )

    # --------------------------------------------------------
    # DOCUMENT CHAIN
    # --------------------------------------------------------

    doc_chain = create_stuff_documents_chain(
        llm,
        PROMPT,
    )

    # --------------------------------------------------------
    # RETRIEVAL CHAIN
    # --------------------------------------------------------

    rag_chain = create_retrieval_chain(
        retriever,
        doc_chain,
    )

    logger.info(
        "CatalogueIQ RAG chain built successfully"
    )

    return rag_chain


# ============================================================
# FORMAT RETRIEVED CONTEXT
# ============================================================

def _format_context(
    document: Document,
) -> str:
    """
    Convert a retrieved document into a traceable context string.

    The metadata is included so that the evaluation/debugging
    layer can identify where the retrieved information came from.
    """

    metadata = document.metadata or {}

    source = metadata.get(
        "source",
        "unknown",
    )

    category = metadata.get(
        "category",
        "unknown",
    )

    product_id = metadata.get(
        "product_id",
    )

    product_name = metadata.get(
        "product_name",
    )

    chunk_type = metadata.get(
        "chunk_type",
        "knowledge",
    )

    header = [
        f"Source: {source}",
        f"Category: {category}",
        f"Chunk Type: {chunk_type}",
    ]

    if product_id:
        header.append(
            f"Product ID: {product_id}"
        )

    if product_name:
        header.append(
            f"Product: {product_name}"
        )

    return (
        "\n".join(header)
        + "\n"
        + "Content:\n"
        + document.page_content
    )


# ============================================================
# MAIN RAG EXECUTION
# ============================================================

async def answer_with_docs_async(
    question: str,
    category: Optional[str] = None,
) -> Tuple[str, List[str], List[str]]:
    """
    Execute the complete CatalogueIQ RAG workflow.

    Returns:

        (
            answer,
            sources,
            contexts
        )

    where:

        answer
            -> final LLM response

        sources
            -> unique source files

        contexts
            -> retrieved contexts including traceability metadata
    """

    logger.info("=" * 70)

    logger.info(
        "CATALOGUEIQ QUERY: question='%s', category=%s",
        question,
        category,
    )

    logger.info("=" * 70)

    # --------------------------------------------------------
    # BUILD CHAIN
    # --------------------------------------------------------

    chain = await _build_chain(
        category
    )

    logger.info(
        "Invoking CatalogueIQ RAG chain..."
    )

    # --------------------------------------------------------
    # EXECUTE RAG
    # --------------------------------------------------------

    result = await chain.ainvoke(
        {
            "input": question,
        }
    )

    # --------------------------------------------------------
    # EXTRACT ANSWER
    # --------------------------------------------------------

    answer = result.get(
        "answer",
        "",
    )

    docs: List[Document] = result.get(
        "context",
        [],
    )

    logger.info(
        "RAG invocation completed"
    )

    logger.info(
        "Retrieved %d context document(s)",
        len(docs),
    )

    # --------------------------------------------------------
    # FORMAT CONTEXTS
    # --------------------------------------------------------

    contexts: List[str] = []

    for index, document in enumerate(
        docs,
        start=1,
    ):

        formatted_context = _format_context(
            document
        )

        contexts.append(
            formatted_context
        )

        metadata = document.metadata or {}

        logger.info(
            "Context %d: "
            "product=%s, category=%s, source=%s",
            index,
            metadata.get(
                "product_name",
                "N/A",
            ),
            metadata.get(
                "category",
                "N/A",
            ),
            metadata.get(
                "source",
                "N/A",
            ),
        )

    # --------------------------------------------------------
    # UNIQUE SOURCES
    # --------------------------------------------------------

    unique_sources = {
        document.metadata.get(
            "source"
        )
        for document in docs
        if document.metadata.get(
            "source"
        )
    }

    sources = sorted(
        unique_sources
    )

    logger.info(
        "Unique sources: %s",
        sources,
    )

    # --------------------------------------------------------
    # FINAL LOG
    # --------------------------------------------------------

    logger.info(
        "Answer (first 300 chars): %s...",
        answer[:300]
        if answer
        else "(empty)",
    )

    logger.info("=" * 70)

    return (
        answer,
        sources,
        contexts,
    )