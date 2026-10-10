# CatalogueIQ

CatalogueIQ is a RAG-powered product intelligence assistant for ShopSmart India. It helps users ask questions about product specifications, price comparisons, return policies, seller requirements, and knowledge-base guidance using a grounded retrieval workflow.

This project originally used a static HTML frontend served through FastAPI. It has now been replaced with a Gradio interface for a simpler, faster, and more interactive chat experience.

## Features

- Product lookup and factual answers
- Product comparison and recommendation flow
- Return/refund and policy Q&A
- Seller guidance and category-based knowledge retrieval
- Background ingestion pipeline for the vector store
- Gradio-based browser UI for everyday use

## Tech Stack

- Python
- LangChain
- OpenAI / Azure OpenAI-compatible models
- LanceDB
- Gradio
- RAGAS for evaluation workflows

## Project Structure

```text
CatalogueIQ/
├── app/
│   ├── api.py
│   ├── gradio_ui.py
│   ├── ingest.py
│   ├── rag.py
│   ├── eval_ragas.py
│   ├── logger.py
│   ├── utils.py
│   └── static/
├── data/
├── documentation/
├── logs/
├── requirements.txt
├── README.md
└── .env
```

## Environment Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Configure your environment variables in a `.env` file.

```env
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini
```

Optional variables: `DATA_DIR`, `HOST`, `PORT`, `RETRIEVAL_K`, `RERANK_TOP_N`.

## Run the App

Start the Gradio UI:

```bash
python -m app.api
```

Then open the URL shown in the terminal, typically: `http://localhost:7860`

## What the UI Does

The Gradio app lets you:

- Ask questions in natural language
- Apply optional filters such as category, source type, and max price
- Start a knowledge-base refresh/inference cycle
- View answer, source references, and retrieved context

## Data Ingestion

CatalogueIQ loads product and knowledge data into a vector store for retrieval. You can trigger ingestion from the Gradio interface or via the project pipeline in the app code.

## Evaluation

The repository also includes RAGAS evaluation scripts for testing answer quality. You can use the evaluation utilities in `app/eval_ragas.py` after the main app is running.

## Notes

- The Gradio interface replaces the earlier static HTML + FastAPI web UI.
- The retrieval and generation pipeline remains the same; only the interface layer changed.
- This project is meant for grounded retail and product knowledge use cases rather than general-purpose web search.
