import asyncio
import json
import math
import os

import requests
from dotenv import load_dotenv

from ragas import evaluate
from ragas import SingleTurnSample, EvaluationDataset
from langchain_openai import ChatOpenAI

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)

from ragas.run_config import RunConfig

from .logger import get_logger


# ---------------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------------

# Load environment variables before creating the LLM client.
load_dotenv()


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# RAGAS JUDGE LLM
# ---------------------------------------------------------------------------

# This LLM is used only by RAGAS as the evaluation/judge model.
#
# It does NOT answer the user's CatalogueIQ question.
# Your actual RAG application generates the answer through /ask.
#
# CatalogueIQ evaluation requirement:
# RAGAS judge = gpt-4o-mini
#
oai_llm = ChatOpenAI(
    model="gpt-4o-mini"
)


logger.info(
    "ChatOpenAI initialized for CatalogueIQ RAGAS evaluation: "
    "model=gpt-4o-mini"
)


# ---------------------------------------------------------------------------
# CATALOGUEIQ CONFIGURATION
# ---------------------------------------------------------------------------

# CatalogueIQ RAG API.
#
# Expected API request:
#
#     POST http://localhost:8000/ask
#     {
#         "question": "..."
#     }
#
# Expected response:
#
#     {
#         "answer": "...",
#         "contexts": [...]
#     }
#
RAG_API_URL = os.getenv(
    "CATALOGUEIQ_API_URL",
    "http://localhost:8000/ask",
)


# Resolve test data relative to this file.
#
# Example:
#
# project/
# ├── app/
# │   └── eval_ragas.py
# └── seed/
#     └── qna_test.json
#
_DEFAULT_TEST_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "documentation",
        "seed",
        "qna_test.json",
    )
)


# CatalogueIQ query types.
#
# These are useful when analysing RAGAS results because the project
# specifically expects performance to be examined across different
# query categories.
CATALOGUEIQ_QUERY_TYPES = {
    "product_factual_lookup",
    "policy_eligibility",
    "comparative_recommendation",
    "seller_policy_lookup",
    "multi_hop",
}


# ---------------------------------------------------------------------------
# LOAD TEST DATA
# ---------------------------------------------------------------------------

def load_test_data(path):
    """
    Load CatalogueIQ evaluation questions and reference answers.

    The evaluator supports:

    1. JSON array:
       [
           {
               "question": "...",
               "answer": "...",
               "query_type": "product_factual_lookup"
           }
       ]

    2. JSON object containing a list:
       {
           "questions": [...]
       }

    3. JSONL:
       {"question": "...", "answer": "..."}
       {"question": "...", "answer": "..."}
    """

    logger.info(
        "Loading CatalogueIQ test data from: %s",
        path,
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CatalogueIQ test file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        raise ValueError(
            "CatalogueIQ test data file is empty."
        )

    # ---------------------------------------------------------
    # Try normal JSON first.
    # ---------------------------------------------------------

    try:
        data = json.loads(content)

        if isinstance(data, list):
            test_data = data

        elif isinstance(data, dict):
            # Support:
            # {
            #     "questions": [...]
            # }
            test_data = data.get("questions", [])

            if not isinstance(test_data, list):
                raise ValueError(
                    "Expected 'questions' to contain a list."
                )

        else:
            raise ValueError(
                "Unsupported JSON structure in test data."
            )

    except json.JSONDecodeError:

        # -----------------------------------------------------
        # Fall back to JSONL.
        # -----------------------------------------------------

        test_data = [
            json.loads(line)
            for line in content.splitlines()
            if line.strip()
        ]

    logger.info(
        "Loaded %d CatalogueIQ evaluation questions",
        len(test_data),
    )

    if not test_data:
        raise ValueError(
            "No evaluation questions were found."
        )

    return test_data


# ---------------------------------------------------------------------------
# NORMALIZE QUERY TYPE
# ---------------------------------------------------------------------------

def get_query_type(item):
    """
    Extract the CatalogueIQ query type from a test case.

    Supported field names:

        query_type
        type
        category

    If the test data does not contain a query type,
    classify it as 'unknown'.
    """

    query_type = (
        item.get("query_type")
        or item.get("type")
        or item.get("category")
        or "unknown"
    )

    return str(query_type).strip().lower()


# ---------------------------------------------------------------------------
# NORMALIZE REFERENCE ANSWER
# ---------------------------------------------------------------------------

def get_reference_answer(item):
    """
    Extract the expected/reference answer.

    CatalogueIQ test data may use either:
        answer
    or:
        reference
    or:
        ground_truth
    """

    reference = (
        item.get("answer")
        or item.get("reference")
        or item.get("ground_truth")
    )

    if reference is None:
        raise ValueError(
            "Test case does not contain a reference answer."
        )

    return str(reference)


# ---------------------------------------------------------------------------
# NORMALIZE CONTEXTS
# ---------------------------------------------------------------------------

def normalize_contexts(contexts):
    """
    Convert retrieved CatalogueIQ contexts into strings.

    The RAG API may return contexts as:

        ["text 1", "text 2"]

    or:

        [
            {"page_content": "text 1"},
            {"page_content": "text 2"}
        ]

    or another serializable structure.
    """

    if contexts is None:
        return []

    if not isinstance(contexts, list):
        contexts = [contexts]

    normalized = []

    for context in contexts:

        if isinstance(context, str):
            normalized.append(context)

        elif isinstance(context, dict):

            # LangChain-style document representation.
            if "page_content" in context:
                normalized.append(
                    str(context["page_content"])
                )

            elif "content" in context:
                normalized.append(
                    str(context["content"])
                )

            else:
                normalized.append(
                    json.dumps(
                        context,
                        ensure_ascii=False,
                    )
                )

        else:
            normalized.append(str(context))

    return normalized


# ---------------------------------------------------------------------------
# SAFE API CALL
# ---------------------------------------------------------------------------

def call_catalogueiq(question):
    """
    Send a question to the running CatalogueIQ RAG API.

    Returns:
        answer, contexts
    """

    payload = {
        "question": question
    }

    logger.debug(
        "POST %s with question='%s'",
        RAG_API_URL,
        question,
    )

    try:

        response = requests.post(
            RAG_API_URL,
            json=payload,
            timeout=60,
        )

        response.raise_for_status()

    except requests.RequestException as exc:

        logger.error(
            "CatalogueIQ API request failed: %s",
            exc,
        )

        raise RuntimeError(
            f"CatalogueIQ API request failed: {exc}"
        ) from exc

    try:
        result = response.json()

    except ValueError as exc:

        logger.error(
            "CatalogueIQ API returned invalid JSON."
        )

        raise RuntimeError(
            "CatalogueIQ API returned invalid JSON."
        ) from exc

    # ---------------------------------------------------------
    # Extract generated answer.
    # ---------------------------------------------------------

    answer = (
        result.get("answer")
        or result.get("response")
        or result.get("output")
        or ""
    )

    # ---------------------------------------------------------
    # Extract retrieved contexts.
    # ---------------------------------------------------------

    contexts = (
        result.get("contexts")
        or result.get("retrieved_contexts")
        or result.get("documents")
        or []
    )

    answer = str(answer)

    contexts = normalize_contexts(contexts)

    return answer, contexts


# ---------------------------------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------------------------------

def print_eval_res(eval_result, test_data):
    """
    Display individual CatalogueIQ RAGAS scores,
    overall averages, and query-type averages.
    """

    scores = eval_result.scores

    if not scores:
        logger.warning(
            "RAGAS returned no scores."
        )
        return

    # ---------------------------------------------------------
    # Detailed question-level results.
    # ---------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("CATALOGUEIQ RAGAS EVALUATION RESULTS")
    print("=" * 100)

    print(
        f"{'Q':<5}"
        f"{'Query Type':<30}"
        f"{'Faithfulness':<16}"
        f"{'Answer Relevancy':<18}"
        f"{'Context Precision':<18}"
    )

    print("-" * 100)

    for index, score in enumerate(scores):

        query_type = get_query_type(
            test_data[index]
        )

        faith = score.get(
            "faithfulness",
            0.0,
        )

        relevance = score.get(
            "answer_relevancy",
            0.0,
        )

        precision = score.get(
            "context_precision",
            0.0,
        )

        print(
            f"{index + 1:<5}"
            f"{query_type:<30}"
            f"{faith:<16.3f}"
            f"{relevance:<18.3f}"
            f"{precision:<18.3f}"
        )

    # ---------------------------------------------------------
    # Convert RAGAS results to DataFrame.
    # ---------------------------------------------------------

    result_df = eval_result.to_pandas()

    # ---------------------------------------------------------
    # Overall averages.
    # ---------------------------------------------------------

    means = result_df.mean(
        numeric_only=True
    ).to_dict()

    logger.info(
        "CatalogueIQ overall RAGAS averages: %s",
        means,
    )

    print("\n")
    print("=" * 60)
    print("OVERALL AVERAGES")
    print("=" * 60)

    for metric, value in means.items():

        if isinstance(value, (int, float)):

            if math.isfinite(value):
                print(
                    f"{metric:<25}: {value:.3f}"
                )

    # ---------------------------------------------------------
    # Query-type analysis.
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("CATALOGUEIQ QUERY-TYPE ANALYSIS")
    print("=" * 80)

    query_type_scores = {}

    for index, score in enumerate(scores):

        query_type = get_query_type(
            test_data[index]
        )

        if query_type not in query_type_scores:
            query_type_scores[query_type] = []

        query_type_scores[query_type].append(score)

    for query_type, query_scores in query_type_scores.items():

        print(f"\n[{query_type}]")

        metric_values = {}

        for score in query_scores:

            for metric, value in score.items():

                if isinstance(value, (int, float)):
                    metric_values.setdefault(
                        metric,
                        [],
                    ).append(value)

        for metric, values in metric_values.items():

            if values:

                average = sum(values) / len(values)

                print(
                    f"  {metric:<22}: {average:.3f}"
                )

    # ---------------------------------------------------------
    # Comparative recommendation highlight.
    # ---------------------------------------------------------

    comparative_scores = query_type_scores.get(
        "comparative_recommendation",
        [],
    )

    if comparative_scores:

        print("\n")
        print("=" * 80)
        print("COMPARATIVE RECOMMENDATION PERFORMANCE")
        print("=" * 80)

        for metric in [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
        ]:

            values = [
                score[metric]
                for score in comparative_scores
                if metric in score
            ]

            if values:

                average = sum(values) / len(values)

                print(
                    f"{metric:<25}: {average:.3f}"
                )

        print(
            "\nComparative recommendation queries are "
            "particularly important because they require "
            "retrieving and comparing multiple products."
        )

    print("\n")


# ---------------------------------------------------------------------------
# MAIN RAGAS EVALUATION
# ---------------------------------------------------------------------------

async def evaluate_rag_system(
    test_path=_DEFAULT_TEST_PATH,
):
    """
    Evaluate the CatalogueIQ RAG system using RAGAS.

    Workflow:

    1. Load the 20 CatalogueIQ evaluation questions.
    2. Send every question to the CatalogueIQ RAG API.
    3. Collect generated answers.
    4. Collect retrieved product/policy contexts.
    5. Build RAGAS SingleTurnSamples.
    6. Create an EvaluationDataset.
    7. Evaluate using:
         - faithfulness
         - answer_relevancy
         - context_precision
    8. Display individual scores.
    9. Display overall averages.
    10. Display scores grouped by CatalogueIQ query type.
    """

    logger.info("=" * 70)
    logger.info("CATALOGUEIQ RAGAS EVALUATION STARTED")
    logger.info("=" * 70)

    # ---------------------------------------------------------
    # Load evaluation dataset.
    # ---------------------------------------------------------

    test_data = load_test_data(
        test_path
    )

    results = []

    # ---------------------------------------------------------
    # Validate test set size.
    # ---------------------------------------------------------

    if len(test_data) != 20:

        logger.warning(
            "CatalogueIQ specification expects 20 evaluation "
            "questions, but %d were found.",
            len(test_data),
        )

    # ---------------------------------------------------------
    # Count query types.
    # ---------------------------------------------------------

    query_type_counts = {}

    for item in test_data:

        query_type = get_query_type(item)

        query_type_counts[query_type] = (
            query_type_counts.get(
                query_type,
                0,
            )
            + 1
        )

    logger.info(
        "CatalogueIQ query type distribution: %s",
        query_type_counts,
    )

    # ---------------------------------------------------------
    # Evaluate every question.
    # ---------------------------------------------------------

    for idx, item in enumerate(test_data):

        question = str(
            item["question"]
        ).strip()

        reference_answer = get_reference_answer(
            item
        )

        query_type = get_query_type(
            item
        )

        logger.info(
            "[%d/%d] Evaluating CatalogueIQ query "
            "type='%s': %s",
            idx + 1,
            len(test_data),
            query_type,
            question,
        )

        # -----------------------------------------------------
        # Call CatalogueIQ RAG application.
        # -----------------------------------------------------

        answer, contexts = call_catalogueiq(
            question
        )

        logger.info(
            "Generated answer: %s...",
            answer[:150]
            if answer
            else "(empty)",
        )

        logger.info(
            "Retrieved %d context(s)",
            len(contexts),
        )

        # -----------------------------------------------------
        # Build RAGAS evaluation sample.
        # -----------------------------------------------------

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference_answer,
        )

        results.append(sample)

        logger.debug(
            "CatalogueIQ sample %d appended",
            idx + 1,
        )

    # ---------------------------------------------------------
    # Create RAGAS EvaluationDataset.
    # ---------------------------------------------------------

    logger.info(
        "Creating EvaluationDataset with %d samples",
        len(results),
    )

    dataset = EvaluationDataset(
        results
    )

    # ---------------------------------------------------------
    # CatalogueIQ-required metrics.
    #
    # IMPORTANT:
    #
    # The CatalogueIQ specification requires:
    #
    #   faithfulness
    #   answer_relevancy
    #   context_precision
    #
    # Context recall is intentionally NOT included.
    # ---------------------------------------------------------

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
    ]

    logger.info(
        "CatalogueIQ RAGAS metrics: %s",
        [metric.name for metric in metrics],
    )

    # ---------------------------------------------------------
    # RAGAS execution configuration.
    # ---------------------------------------------------------

    run_config = RunConfig(
        max_workers=16,
        timeout=60,
    )

    logger.info(
        "Running RAGAS evaluation "
        "(max_workers=16, timeout=60s)..."
    )

    # ---------------------------------------------------------
    # Execute evaluation.
    # ---------------------------------------------------------

    eval_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=oai_llm,
        run_config=run_config,
    )

    logger.info(
        "CatalogueIQ RAGAS evaluation complete"
    )

    # ---------------------------------------------------------
    # Print results.
    # ---------------------------------------------------------

    print_eval_res(
        eval_result,
        test_data,
    )

    logger.info("=" * 70)
    logger.info("CATALOGUEIQ RAGAS EVALUATION FINISHED")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    logger.info(
        "Starting CatalogueIQ RAGAS evaluation script"
    )

    asyncio.run(
        evaluate_rag_system()
    )