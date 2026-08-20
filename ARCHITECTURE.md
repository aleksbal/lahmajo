# Architecture

## Overview

The project follows a clean, domain-driven, layered structure — each layer only calls the one below it:

```
lahmajo/                   # repo root
├── pyproject.toml         # packaging (src/ layout, `pip install -e .`)
├── src/
│   └── lahmajo/            # Main package (project name)
│       ├── api/               # API layer - HTTP endpoints
│       │   └── routes.py
│       ├── services/          # Service layer - business logic
│       │   ├── rag_service.py
│       │   ├── retrieval_service.py
│       │   └── ingestion_service.py
│       ├── ingestion/         # Ingestion layer - document processing
│       │   └── processing.py
│       ├── indexes/           # Index layer - search indexes
│       │   ├── state.py
│       │   ├── vector_provider.py
│       │   ├── bm25_provider.py
│       │   └── elasticsearch_hybrid_provider.py
│       ├── search/            # Search layer - retrieval algorithms
│       │   └── hybrid_search.py
│       ├── llm/               # Provider layer - model abstraction
│       │   ├── llm_provider.py
│       │   └── embedding_provider.py
│       └── cli.py             # CLI entry point
├── tests/                 # Unit tests
├── static/                # Static files (HTML UI)
├── docker/                # Dockerfile, docker-compose.yml
├── docs/                  # Supplementary docs (not the layer/data-flow reference - that's this file)
└── requirements.txt       # Dependencies
```

```
┌─────────────────┐
│   API Layer     │  FastAPI routes (lahmajo/api/routes.py)
│  (Endpoints)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Service Layer  │  Business logic (lahmajo/services/)
│ (Orchestration) │  - rag_service.py
└────────┬────────┘  - retrieval_service.py
         │           - ingestion_service.py
         │
         ├──────────────────┬──────────────────┐
         ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Ingestion Layer │ │   Index Layer   │ │  Provider Layer │
│  (Processing)   │ │   (Indexes)     │ │  (llm/)         │
│  ingestion/      │ │   indexes/      │ │  llm_provider.py│
│  processing.py  │ │   state.py      │ │  embedding_     │
│                 │ │   vector_       │ │  provider.py    │
│                 │ │   provider.py   │ │                 │
│                 │ │   bm25_         │ │                 │
│                 │ │   provider.py   │ │                 │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                  │                  │
         └──────────────────┴──────────────────┘
                            │
                            ▼
                 ┌─────────────────┐
                 │  Search Layer   │  Combines both indexes
                 │  (Retrieval)    │  search/hybrid_search.py
                 └─────────────────┘
```

## Layer Details

### API Layer (`src/lahmajo/api/`)
**Responsibility**: HTTP interface only
- Route handlers, request/response validation, error handling
- **No business logic**

**Files**: `routes.py` — all FastAPI endpoints

### Service Layer (`src/lahmajo/services/`)
**Responsibility**: Business logic orchestration
- Coordinates between API, ingestion, index, and search layers
- **No direct data access**

**Files**:
- `rag_service.py` — RAG agent creation and question answering
- `retrieval_service.py` — document retrieval operations
- `ingestion_service.py` — document ingestion workflow

### Ingestion Layer (`src/lahmajo/ingestion/`)
**Responsibility**: Document ingestion pipeline (loading, chunking, embedding)
- **No business logic**

**Files**: `processing.py`

### Index Layer (`src/lahmajo/indexes/`)
**Responsibility**: Search index providers and state management
- **Vector Index**: stores embedded documents for semantic search
- **BM25 Index**: keyword search index
- **Elasticsearch Hybrid Provider**: native ES hybrid search (combines BM25 + vector in a single query)
- **No business logic**

**Files**:
- `state.py` — index state management (lazy singleton, document tracking)
- `vector_provider.py` — vector index provider (`in_memory`, `elasticsearch`, ...)
- `bm25_provider.py` — BM25 index provider (`rank_bm25`, `elasticsearch`, ...)
- `elasticsearch_hybrid_provider.py` — ES native hybrid search provider

### Search Layer (`src/lahmajo/search/`)
**Responsibility**: Search algorithms that use the indexes
- Combines vector (semantic) + BM25 (keyword) search
- **No business logic**

**Files**: `hybrid_search.py`

### Provider Layer (`src/lahmajo/llm/`)
**Responsibility**: Model provider abstraction
- **Embedding Models**: convert text to vectors (used by the vector index)
- **LLM Models**: generate text responses
- **No business logic**
- Configuration is entirely via environment variables (see README.md)

**Files**:
- `embedding_provider.py` — embedding model factory
- `llm_provider.py` — LLM model factory

## Data Flow

### Question Answering
```
User → API (routes.py)
     → Service (rag_service.py)
     → Service (retrieval_service.py)
     → Search (hybrid_search.py)
         ├→ Indexes (vector_provider.py) - semantic search
         └→ Indexes (bm25_provider.py) - keyword search
     → Service (rag_service.py) - combines results
     → LLM (llm_provider.py) - generates answer
     → API (routes.py)
     → User
```

### Document Ingestion
```
User → API (routes.py)
     → Service (ingestion_service.py)
     → Ingestion (processing.py)
         ├→ Embedding Model (embedding_provider.py) - converts text to vectors
         ├→ Indexes (vector_provider.py) - stores embedded documents
         └→ Indexes (bm25_provider.py) - indexes for keyword search
     → Service (ingestion_service.py)
     → API (routes.py)
     → User
```

## Terminology

### Embedding Models vs Vector Index

- **Embedding Models** (`llm/embedding_provider.py`): AI models that convert text into numerical vectors. Transform text → vectors (e.g., "cat" → `[0.1, 0.5, -0.3, ...]`). Used by the vector index during ingestion and search.
- **Vector Index** (`indexes/vector_provider.py`): storage system that holds embedded documents and performs semantic similarity search, using an embedding model to convert queries and documents.

**Analogy**: Embedding model = translator (text→numbers), Vector index = library (stores translated books).

### Vector Index vs BM25 Index

- **Vector Index** (Semantic Search): finds documents by meaning/context/concepts. Example: query "feline" finds documents about "cats". Location: `indexes/vector_provider.py`.
- **BM25 Index** (Keyword Search): exact matches, names, technical terms, identifiers. Example: query "John Smith" finds documents containing "John Smith". Location: `indexes/bm25_provider.py`.

**Both are used together** in hybrid search for optimal retrieval.

### Elasticsearch Native Hybrid Search

When both the vector and BM25 providers are Elasticsearch-based, the system uses **Elasticsearch native hybrid search**:
- **Single Query**: ES combines BM25 (`match` query) and vector (`knn` query) in one request
- **Native Scoring**: ES's optimized score combination (query-level boost weights, vs. the Python-side path's Reciprocal Rank Fusion)
- **Better Performance**: single network round trip, ES query optimization
- **Scalability**: ES handles large datasets efficiently without keeping documents in memory

**Implementation**: `ElasticsearchHybridProvider` uses an ES `bool` query with `should` clauses combining `match` (BM25) and `knn` (vector) queries, with configurable boost weights.

**Auto-detection**: the system automatically detects when both providers are ES-based and uses native hybrid search. It falls back to Python-side combination for mixed providers (e.g., ES vector + `rank-bm25`).

## Import Examples

```python
# From API layer
from lahmajo.services.rag_service import ask_question

# From Service layer
from lahmajo.indexes.state import get_vector_index
from lahmajo.search.hybrid_search import HybridRetriever

# From Ingestion layer
from lahmajo.ingestion.processing import ingest_documents

# From Index layer
from lahmajo.indexes import get_vector_index_provider, get_bm25_provider
```

## Benefits of This Structure

1. **Separation of concerns**: each layer has a single responsibility
2. **Testability**: layers can be tested independently
3. **Maintainability**: easy to locate and modify code
4. **Scalability**: easy to add new features without mixing concerns
5. **Reusability**: services can be used by the CLI, API, or future interfaces
