from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import gradio as gr

try:
    from .ingest import run_ingest_async
    from .logger import get_logger
    from .rag import answer_with_docs_async
except ImportError:  # pragma: no cover
    from app.ingest import run_ingest_async
    from app.logger import get_logger
    from app.rag import answer_with_docs_async

logger = get_logger(__name__)

CATEGORY_OPTIONS = [
    "",
    "Electronics",
    "Fashion",
    "Home & Kitchen",
    "Beauty",
    "Books",
]

DOCUMENT_TYPE_OPTIONS = [
    "",
    "product_catalogue",
    "product_reviews",
    "category_guide",
    "seller_guide",
    "policy",
    "faq",
]

_ingest_lock = asyncio.Lock()
_ingest_task: asyncio.Task | None = None
_ingest_last = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "stats": None,
    "error": None,
}


def _build_metadata_filters(
    category: str | None,
    document_type: str | None,
    max_price: float | int | None,
) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if category:
        filters["category"] = category
    if document_type:
        filters["document_type"] = document_type
    if max_price is not None:
        filters["max_price"] = float(max_price)
    return filters


async def _run_ingest_job() -> None:
    global _ingest_last

    logger.info("Background CatalogueIQ ingestion job started")
    _ingest_last.update({
        "status": "running",
        "started_at": time.time(),
        "finished_at": None,
        "stats": None,
        "error": None,
    })

    try:
        stats = await run_ingest_async()
        _ingest_last.update({
            "status": "succeeded",
            "finished_at": time.time(),
            "stats": stats,
            "error": None,
        })
        logger.info("CatalogueIQ ingestion completed successfully: %s", stats)
    except Exception as exc:  # pragma: no cover
        _ingest_last.update({
            "status": "failed",
            "finished_at": time.time(),
            "stats": None,
            "error": str(exc),
        })
        logger.error("CatalogueIQ ingestion failed: %s", str(exc), exc_info=True)


async def trigger_ingest() -> str:
    global _ingest_task

    async with _ingest_lock:
        if _ingest_task and not _ingest_task.done():
            logger.warning("Ingestion already running")
            return "Ingestion is already running. Please wait for the current sync to finish."

        _ingest_task = asyncio.create_task(_run_ingest_job())
        logger.info("Background ingestion task created")
        return "CatalogueIQ ingestion started in the background."


def get_ingestion_status() -> str:
    state = _ingest_last.copy()
    status = state.get("status", "idle")
    if state.get("stats"):
        return f"Status: {status}\nStats: {state['stats']}"
    if state.get("error"):
        return f"Status: {status}\nError: {state['error']}"
    return f"Status: {status}"


def ask_catalogueiq(
    question: str,
    category: str | None,
    document_type: str | None,
    max_price: float | int | None,
):
    if not question or not question.strip():
        raise gr.Error("Please enter a question before asking CatalogueIQ.")

    q = question.strip()
    filters = _build_metadata_filters(category, document_type, max_price)
    logger.info("CatalogueIQ Gradio request: question='%s', filters=%s", q, filters)

    start = time.perf_counter()
    answer, sources, contexts = asyncio.run(answer_with_docs_async(q, filters))
    elapsed = round(time.perf_counter() - start, 2)

    sources_text = "\n".join(str(item) for item in (sources or [])) or "No sources returned."
    contexts_text = "\n\n---\n\n".join(str(item) for item in (contexts or [])) or "No context was returned."

    summary = f"Answer generated in {elapsed} seconds."
    return (
        f"## Answer\n\n{answer}\n\n> {summary}",
        f"## Sources\n\n{sources_text}",
        f"## Retrieved context\n\n{contexts_text}",
        f"Filters used: {filters or 'none'} | Time: {elapsed}s",
    )


CSS = """
.gradio-container {
    max-width: 1400px !important;
}
#catalogueiq-header {
    padding-bottom: 0.5rem;
}
#result-box {
    min-height: 220px;
}
"""


def build_gradio_app() -> gr.Blocks:
    with gr.Blocks(
        title="CatalogueIQ",
        theme=gr.themes.Soft(),
        css=CSS,
    ) as demo:
        gr.Markdown(
            "# CatalogueIQ\n"
            "### Product Intelligence Assistant for ShopSmart India",
            elem_id="catalogueiq-header",
        )

        with gr.Row():
            with gr.Column(scale=2):
                question = gr.Textbox(
                    label="Question",
                    lines=5,
                    placeholder="Ask about a product, refund policy, comparison, or seller requirement...",
                )

            with gr.Column(scale=1):
                category = gr.Dropdown(
                    choices=CATEGORY_OPTIONS,
                    label="Category",
                    value="",
                    allow_custom_value=True,
                )
                document_type = gr.Dropdown(
                    choices=DOCUMENT_TYPE_OPTIONS,
                    label="Source Type",
                    value="",
                    allow_custom_value=True,
                )
                max_price = gr.Number(
                    label="Max price (₹)",
                    value=None,
                    precision=0,
                    minimum=0,
                    info="Optional filter for product recommendation queries",
                )
                ask_button = gr.Button("Ask CatalogueIQ", variant="primary")

        with gr.Row():
            refresh_button = gr.Button("Refresh Knowledge Base")
            ingestion_status = gr.Textbox(
                label="Ingestion status",
                value=get_ingestion_status(),
                interactive=False,
            )

        answer_output = gr.Markdown(label="Answer", elem_id="result-box")
        sources_output = gr.Markdown(label="Sources")
        contexts_output = gr.Markdown(label="Retrieved Context")
        filter_output = gr.Textbox(label="Request summary", interactive=False)

        ask_button.click(
            fn=ask_catalogueiq,
            inputs=[question, category, document_type, max_price],
            outputs=[answer_output, sources_output, contexts_output, filter_output],
        )

        refresh_button.click(
            fn=trigger_ingest,
            outputs=[ingestion_status],
        )

        demo.load(fn=get_ingestion_status, outputs=[ingestion_status])

    return demo


demo = build_gradio_app()


if __name__ == "__main__":
    demo.launch(
        # server_name=os.getenv("HOST", "0.0.0.0"),
        # server_port=int(os.getenv("PORT", "7860")),
        # share=False,
    )
