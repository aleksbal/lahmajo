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

Note: `Dockerfile`/`docker-compose.yml` live under `docker/` (see `docker-dev.sh`, which passes `-f docker/docker-compose.yml`); `.dockerignore` stays at the repo root since Docker resolves it relative to the build context, not the Dockerfile's location. Inside the container uvicorn binds to port 8080 (see `docker/Dockerfile` CMD); docker-compose maps host `8000` → container `8080`.

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
pip install --no-deps -e .   # registers the local package (src/ layout, see pyproject.toml)
# optional extra, only needed for the OpenAI provider:
pip install langchain-openai
```

`requirements.txt` is version-pinned (resolved against Python 3.12). `langchain-elasticsearch`/`elasticsearch` are already pinned in `requirements.txt` even though ES is an opt-in provider. The package lives under `src/lahmajo/` (see "Architecture" below) and is installed editable via `pyproject.toml`; `import lahmajo...` won't resolve without the `pip install -e .` step.

### CI

`.github/workflows/tests.yml` runs `python run_tests.py` on every push/PR to `main` (Python 3.12, deps from `requirements.txt`, then `pip install --no-deps -e .` to register the `src/` package). Keep it green — this is the only automated check in the repo.

## Architecture

The codebase is a strict layered pipeline, each layer only calling the one below it — API → Services → Ingestion/Index/Search → LLM providers. Preserve this separation when adding code (no business logic in `api/`, no direct index access from `api/`, etc.).

```
src/lahmajo/                 # package root - see pyproject.toml for the src/ layout
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
├── search/
│   ├── hybrid_search.py          # HybridRetriever: combines BM25 + vector results (RRF, or ES native hybrid)
│   └── rerank_provider.py        # RerankProvider ABC + none/llm impls (factory: get_rerank_provider)
├── llm/
│   ├── llm_provider.py           # get_llm() factory: ollama_local / ollama_cloud / openai
│   └── embedding_provider.py     # get_embeddings() factory, same provider set
└── cli.py                   # CLI entry point
```

Repo root also has `docker/` (Dockerfile, docker-compose.yml), `docs/` (supplementary docs, e.g. `docs/EMBEDDINGS_EXPLANATION.md`), and `pyproject.toml` (packaging).

### Provider pattern (used throughout)

Every pluggable component (LLM, embeddings, vector index, BM25 index, reranking) is selected purely via environment variables at call time through a `get_x()` factory function that reads `os.getenv(...)`, e.g. `LLM_PROVIDER`, `EMBEDDING_PROVIDER`, `VECTOR_INDEX_PROVIDER`, `BM25_PROVIDER`, `RERANK_PROVIDER`. Each provider implements an ABC interface (`VectorIndexProvider`, `BM25Provider`, `RerankProvider`). When adding a new provider, implement the interface and add a branch to the corresponding factory function — see README.md "Configuration" section for the full env-var reference and defaults.

### Elasticsearch native hybrid search

When `VECTOR_INDEX_PROVIDER=elasticsearch` AND `BM25_PROVIDER=elasticsearch` (optionally forced via `ELASTICSEARCH_USE_NATIVE_HYBRID=true`), `HybridRetriever` detects this and delegates to `ElasticsearchHybridProvider`, which issues a single ES query (`bool` query combining `match` and `knn`) instead of two separate BM25 + vector queries combined in Python. This is the auto-detected fast path; mixed providers (e.g. ES vector + `rank_bm25`) fall back to Python-side Reciprocal Rank Fusion (RRF) in `HybridRetriever.search()`.

### Index state (`indexes/state.py`)

The vector index is a lazily-initialized module-level singleton (`get_vector_index()`). For non-ES providers, ingested documents are also kept in an in-memory list (`_all_documents`) because BM25 (`rank_bm25`) needs the full corpus to build its index; for ES-based setups, ES itself is the source of truth and `get_all_documents()` fetches from ES instead. `reset_vector_index()` clears this state — useful in tests.

### Adaptive chunking (`ingestion/processing.py`)

Document type (structured vs. long-form) is auto-detected during ingestion to pick chunk size: ~300 chars/50 overlap for structured docs (resumes, technical docs) vs. ~600 chars/100 overlap for long-form content, using `RecursiveCharacterTextSplitter` by default (a `Semantic` strategy using embeddings-based breakpoints is also selectable). Detection is `_detect_document_type()` — scores on total size, the fraction of non-blank lines that look like list/bullet items (anchored at line start), and average line length; deliberately *not* "contains a hyphen anywhere in the first 500 chars" (an earlier version of this heuristic did that and misfired on ordinary hyphenated prose — don't reintroduce it). The detection result is logged unconditionally via `logging` (not gated on `show_progress`), since the real `/ingest` API path always calls this with `show_progress=False`. See README.md "Adaptive Chunking" / "Chunking Strategies" for the full rationale.

### Query-time hybrid combination, dedup, and reranking

`HybridRetriever.search()`'s Python-side path combines BM25 and vector candidates via `reciprocal_rank_fusion()` — fusion by each list's rank order, not raw score magnitude, since BM25 scores and vector scores aren't on a comparable scale and different vector providers don't even agree on whether a higher or lower score means "more similar" (a naive weighted-average of normalized scores got this wrong for the default in-memory provider — don't reintroduce that). The ES native hybrid path (previous section) still combines by query-level boost weights (`bm25_weight`/`vector_weight` params on `search()`), since that's a different, ES-internal combination mechanism.

`retrieve_context()` (`services/retrieval_service.py`) then dedupes: `_dedupe_chunks()` drops chunks that are near-duplicates (`difflib.SequenceMatcher` ratio ≥ 0.8) of an already-kept chunk *from the same source* — catches adjacent, overlapping adaptive-chunking windows surfacing as two near-identical candidates; different sources with similar text are left alone.

`retrieve_context()` optionally reranks the deduped candidates afterward via `get_rerank_provider()` — opt-in, `RERANK_PROVIDER=none` by default. When enabled (`RERANK_PROVIDER=llm`), a larger candidate pool (20 instead of 10) is fetched so the reranker has more to choose from, then `LLMRerankProvider` asks the already-configured LLM to reorder them; a malformed/failed rerank response falls back to hybrid search's own order rather than erroring.

Both `use_hybrid` (bool) and `use_rerank` (`Optional[bool]`, `None` = defer to `RERANK_PROVIDER`) are per-call overrides threaded from `retrieve_context()` up through `create_rag_agent()`/`ask_question()` (`rag_service.py`) to the `/ask` and `/debug/search` API endpoints and the GUI's "Hybrid search"/"Rerank results" checkboxes (`static/index.html`) — so retrieval behavior can be compared per-request without restarting the server or touching env vars.

## Notes for editing

- `ARCHITECTURE.md` is the single source of truth for the layering/data-flow diagrams; README.md links to it instead of repeating it. Update `ARCHITECTURE.md` if you change layer boundaries — don't let a second copy of the diagram grow back in README.
- `indexes/state.py` has a duplicated `get_vector_store()` function definition (harmless — the second silently shadows the first) — be aware when editing that file.
- Config is entirely environment-variable driven (no config files); see `.env.docker` for the full set used by the Docker Compose setup and README.md for local-dev defaults.
- `docs/EMBEDDINGS_EXPLANATION.md` explains the two separate embedding-model instances used during ingestion (`semantic_chunker_embeddings` vs. the document embedding model) — read it before touching `ingestion/processing.py`'s embedding calls.
- `GET /` (`api/routes.py`) locates `static/index.html` via a `parents[3]` traversal from `__file__`, which only holds for an editable install (`pip install -e .` - how this project is actually installed, locally and in `docker/Dockerfile`). `LAHMAJO_STATIC_DIR` overrides it for a non-editable `pip install .`, where `static/` isn't bundled into the installed package. See README.md "Static UI Directory".
