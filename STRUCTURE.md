# Project Structure

## Overview

The project follows a clean, domain-driven structure:

```
lahmajo/
├── lahmajo/              # Main package (project name)
│   ├── api/              # API layer - HTTP endpoints
│   │   └── routes.py
│   ├── services/         # Service layer - business logic
│   │   ├── rag_service.py
│   │   ├── retrieval_service.py
│   │   └── ingestion_service.py
│   ├── ingestion/        # Ingestion layer - document processing
│   │   └── processing.py
│   ├── indexes/          # Index layer - search indexes
│   │   ├── state.py
│   │   ├── vector_provider.py
│   │   └── bm25_provider.py
│   ├── search/           # Search layer - retrieval algorithms
│   │   └── hybrid_search.py
│   ├── llm/              # LLM provider abstraction
│   │   ├── llm_provider.py
│   │   └── embedding_provider.py
│   └── cli.py            # CLI entry point
├── tests/                # Unit tests
├── static/               # Static files (HTML UI)
└── requirements.txt      # Dependencies
```

## Layer Responsibilities

### API Layer (`lahmajo/api/`)
- **Purpose**: HTTP interface
- **Contains**: FastAPI route handlers
- **No business logic** - delegates to services

### Service Layer (`lahmajo/services/`)
- **Purpose**: Business logic orchestration
- **Contains**: 
  - `rag_service.py` - RAG agent creation and Q&A
  - `retrieval_service.py` - Document retrieval operations
  - `ingestion_service.py` - Document ingestion workflow

### Ingestion Layer (`lahmajo/ingestion/`)
- **Purpose**: Document ingestion pipeline
- **Contains**:
  - `processing.py` - Document loading, chunking, and embedding

### Index Layer (`lahmajo/indexes/`)
- **Purpose**: Search index providers and state management
- **Contains**:
  - `state.py` - Index state management (singleton, document tracking)
  - `vector_provider.py` - Vector index provider abstraction (in_memory, elasticsearch, etc.)
  - `bm25_provider.py` - BM25 index provider abstraction (rank_bm25, elasticsearch, etc.)

### Search Layer (`lahmajo/search/`)
- **Purpose**: Search algorithms that use indexes
- **Contains**:
  - `hybrid_search.py` - Combines vector (semantic) + BM25 (keyword) search

### Provider Layer (`lahmajo/llm/`)
- **Purpose**: Model provider abstraction
- **Contains**:
  - `embedding_provider.py` - **Embedding models** (convert text → vectors for vector index)
  - `llm_provider.py` - **LLM models** (generate text responses)
- **Configuration**: Via environment variables (see README)

## Key Distinctions

**Vector Index vs Embedding Models:**
- **Embedding Models** (`llm/embedding_provider.py`): Convert text to numerical vectors
- **Vector Index** (`indexes/vector_provider.py`): Stores embedded documents and performs semantic similarity search

**BM25 Index vs Vector Index:**
- **BM25 Index** (`indexes/bm25_provider.py`): Keyword-based search (exact matches, names, terms)
- **Vector Index** (`indexes/vector_provider.py`): Semantic search (meaning, similarity)

**Both indexes are used together** in `hybrid_search.py` for optimal retrieval.

## Why This Structure?

1. **Domain-driven**: Each directory represents a clear domain (api, services, storage, search)
2. **Standard Python**: Package name matches project name (`lahmajo`)
3. **Logical grouping**: Related functionality grouped together
4. **No vague names**: "storage" and "search" are clearer than "core"
5. **Scalable**: Easy to add new domains (e.g., `models/`, `utils/`)

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
