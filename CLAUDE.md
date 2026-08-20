# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Lahmajo is an experimental Agentic RAG (Retrieval-Augmented Generation) system with a FastAPI web UI and CLI. It combines BM25 keyword search with vector semantic search ("hybrid search") over ingested documents, and answers questions using a pluggable LLM (Ollama local/cloud or OpenAI).

## Commands

### Run the app

```bash
# Web server (local dev, with reload)
python main.py
# or
python -m lahmajo.api.routes
# or
uvicorn lahmajo.api.routes:app --host 0.0.0.0 --port 8000

# CLI
python cli.py
# or
python -m lahmajo.cli
```

### Docker (includes Elasticsearch)

```bash
./docker-dev.sh start     # docker-compose up -d (web UI :8000, ES :9200)
./docker-dev.sh stop
./docker-dev.sh restart
./docker-dev.sh logs [service]
./docker-dev.sh rebuild
./docker-dev.sh health
./docker-dev.sh reset
```

Note: inside the container uvicorn binds to port 8080 (see `Dockerfile` CMD); docker-compose maps host `8000` → container `8080`.

### Tests

```bash
python run_tests.py
# or
python -m unittest discover tests
# single test file
python -m unittest tests.test_retrieval_service
# single test case/method
python -m unittest tests.test_retrieval_service.TestRetrievalService.test_some_method
```

Tests use the standard library `unittest` (with `unittest.mock`), not pytest. There is no linter/formatter configured in this repo.

### Dependencies

```bash
pip install -r requirements.txt
# optional extra, only needed for the OpenAI provider:
pip install langchain-openai
```

`requirements.txt` is version-pinned (resolved against Python 3.12). `langchain-elasticsearch`/`elasticsearch` are already pinned in `requirements.txt` even though ES is an opt-in provider.

### CI

`.github/workflows/tests.yml` runs `python run_tests.py` on every push/PR to `main` (Python 3.12, deps from `requirements.txt`). Keep it green — this is the only automated check in the repo.

## Architecture

The codebase is a strict layered pipeline, each layer only calling the one below it — API → Services → Ingestion/Index/Search → LLM providers. Preserve this separation when adding code (no business logic in `api/`, no direct index access from `api/`, etc.).

```
lahmajo/
├── api/routes.py           # FastAPI endpoints only (HTTP in/out, no business logic)
├── services/                # Business logic orchestration
│   ├── rag_service.py           # Builds the LangChain agent, answers questions
│   ├── retrieval_service.py     # Document retrieval orchestration
│   └── ingestion_service.py     # Upload handling / ingestion workflow
├── ingestion/processing.py  # Document loading (TXT/PDF/MD), adaptive chunking, embedding
├── indexes/                 # Pluggable search index providers + state
│   ├── state.py                  # Lazy-init singleton for the active vector index; tracks all documents for BM25
│   ├── vector_provider.py        # VectorIndexProvider ABC + in_memory / elasticsearch impls (factory: get_vector_index_provider)
│   ├── bm25_provider.py          # BM25Provider ABC + rank_bm25 / elasticsearch impls (factory: get_bm25_provider)
│   └── elasticsearch_hybrid_provider.py  # ES-native single-query hybrid (BM25 `match` + `knn` in one request)
├── search/hybrid_search.py  # HybridRetriever: combines BM25 + vector results (weighted score or RRF)
├── llm/
│   ├── llm_provider.py           # get_llm() factory: ollama_local / ollama_cloud / openai
│   └── embedding_provider.py     # get_embeddings() factory, same provider set
└── cli.py                   # CLI entry point
```

### Provider pattern (used throughout)

Every pluggable component (LLM, embeddings, vector index, BM25 index) is selected purely via environment variables at call time through a `get_x()` factory function that reads `os.getenv(...)`, e.g. `LLM_PROVIDER`, `EMBEDDING_PROVIDER`, `VECTOR_INDEX_PROVIDER`, `BM25_PROVIDER`. Each provider implements an ABC interface (`VectorIndexProvider`, `BM25Provider`). When adding a new provider, implement the interface and add a branch to the corresponding factory function — see README.md "Configuration" section for the full env-var reference and defaults.

### Elasticsearch native hybrid search

When `VECTOR_INDEX_PROVIDER=elasticsearch` AND `BM25_PROVIDER=elasticsearch` (optionally forced via `ELASTICSEARCH_USE_NATIVE_HYBRID=true`), `HybridRetriever` detects this and delegates to `ElasticsearchHybridProvider`, which issues a single ES query (`bool` query combining `match` and `knn`) instead of two separate BM25 + vector queries combined in Python. This is the auto-detected fast path; mixed providers (e.g. ES vector + `rank_bm25`) fall back to Python-side weighted-score combination in `HybridRetriever.search()`.

### Index state (`indexes/state.py`)

The vector index is a lazily-initialized module-level singleton (`get_vector_index()`). For non-ES providers, ingested documents are also kept in an in-memory list (`_all_documents`) because BM25 (`rank_bm25`) needs the full corpus to build its index; for ES-based setups, ES itself is the source of truth and `get_all_documents()` fetches from ES instead. `reset_vector_index()` clears this state — useful in tests.

### Adaptive chunking (`ingestion/processing.py`)

Document type (structured vs. long-form) is auto-detected during ingestion to pick chunk size: ~300 chars/50 overlap for structured docs (resumes, technical docs) vs. ~600 chars/100 overlap for long-form content, using `RecursiveCharacterTextSplitter` by default (a `Semantic` strategy using embeddings-based breakpoints is also selectable). See README.md "Adaptive Chunking" / "Chunking Strategies" for the full rationale.

### Query-time hybrid weighting

`HybridRetriever.search()` combines BM25 and vector scores with configurable weights (default 40% BM25 / 60% vector for semantic queries; the caller can flip this to 60/40 for name/keyword-style queries — see README.md "Hybrid Search"). A separate `reciprocal_rank_fusion()` helper implements RRF as an alternative combination method.

## Notes for editing

- `ARCHITECTURE.md` is the single source of truth for the layering/data-flow diagrams; README.md links to it instead of repeating it. Update `ARCHITECTURE.md` if you change layer boundaries — don't let a second copy of the diagram grow back in README.
- `indexes/state.py` has a duplicated `get_vector_store()` function definition (harmless — the second silently shadows the first) — be aware when editing that file.
- Config is entirely environment-variable driven (no config files); see `.env.docker` for the full set used by the Docker Compose setup and README.md for local-dev defaults.
