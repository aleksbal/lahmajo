# Architecture

## Overview

The project follows a clean domain-driven structure:

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
         │
         ├──────────────────┬──────────────────┐
         ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Ingestion Layer │ │   Index Layer   │ │  Embedding      │
│  (Processing)   │ │   (Indexes)     │ │  Models         │
│  ingestion/     │ │   indexes/      │ │  llm/           │
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
                 │  Search Layer    │  Combines both indexes
                 │  (Retrieval)     │  search/hybrid_search.py
                 └─────────────────┘
```

## Layer Details

### API Layer (`lahmajo/api/`)
**Responsibility**: HTTP interface only
- Route handlers
- Request/response validation
- Error handling
- **No business logic**

**Files**:
- `routes.py` - All FastAPI endpoints

### Service Layer (`lahmajo/services/`)
**Responsibility**: Business logic orchestration
- Coordinates between API, Storage, and Search layers
- Implements use cases
- **No direct data access**

**Files**:
- `rag_service.py` - RAG agent creation and question answering
- `retrieval_service.py` - Document retrieval operations
- `ingestion_service.py` - Document ingestion workflow

### Ingestion Layer (`lahmajo/ingestion/`)
**Responsibility**: Document ingestion pipeline
- Document loading, chunking, and embedding
- **No business logic**

**Files**:
- `processing.py` - Document loading, chunking, and embedding

### Index Layer (`lahmajo/indexes/`)
**Responsibility**: Search index providers and state management
- **Vector Index**: Stores embedded documents for semantic search
- **BM25 Index**: Keyword search index
- **Elasticsearch Hybrid Provider**: Native ES hybrid search (combines BM25 + vector in single query)
- **No business logic**

**Files**:
- `state.py` - Index state management (singleton, document tracking)
- `vector_provider.py` - Vector index provider (in_memory, elasticsearch, etc.)
- `bm25_provider.py` - BM25 index provider (rank_bm25, elasticsearch, etc.)
- `elasticsearch_hybrid_provider.py` - ES native hybrid search provider (BM25 + vector in single query)

### Search Layer (`lahmajo/search/`)
**Responsibility**: Search algorithms that use indexes
- **Hybrid Search**: Combines vector (semantic) + BM25 (keyword) search
- **No business logic**

**Files**:
- `hybrid_search.py` - Combines vector + BM25 search results

### Provider Layer (`lahmajo/llm/`)
**Responsibility**: Model provider abstraction
- **Embedding Models**: Convert text to vectors (used by vector index)
- **LLM Models**: Generate text responses
- **No business logic**

**Files**:
- `embedding_provider.py` - Embedding model factory (for vector index)
- `llm_provider.py` - LLM model factory (for RAG responses)

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

To avoid confusion, here are the key terms and their distinctions:

### Embedding Models vs Vector Index

- **Embedding Models** (`llm/embedding_provider.py`):
  - **What**: AI models that convert text into numerical vectors
  - **Purpose**: Transform text → vectors (e.g., "cat" → [0.1, 0.5, -0.3, ...])
  - **Used by**: Vector index during ingestion and search

- **Vector Index** (`indexes/vector_provider.py`):
  - **What**: Storage system that holds embedded documents
  - **Purpose**: Store vectors and perform semantic similarity search
  - **Uses**: Embedding models to convert queries and documents

**Analogy**: Embedding model = translator (text→numbers), Vector index = library (stores translated books)

### Vector Index vs BM25 Index

- **Vector Index** (Semantic Search):
  - **Type**: Semantic similarity search
  - **Best for**: Finding documents by meaning, context, concepts
  - **Example**: Query "feline" finds documents about "cats"
  - **Location**: `indexes/vector_provider.py`

- **BM25 Index** (Keyword Search):
  - **Type**: Keyword matching search
  - **Best for**: Exact matches, names, technical terms, identifiers
  - **Example**: Query "John Smith" finds documents containing "John Smith"
  - **Location**: `indexes/bm25_provider.py`

**Both are used together** in hybrid search for optimal retrieval.

### Elasticsearch Native Hybrid Search

When both vector and BM25 providers are Elasticsearch-based, the system uses **Elasticsearch native hybrid search**:

- **Single Query**: ES combines BM25 (`match` query) and vector (`knn` query) in one request
- **Native Scoring**: ES's optimized score combination (more sophisticated than Python-side weighted average)
- **Better Performance**: Single network round trip, ES query optimization
- **Scalability**: ES handles large datasets efficiently without keeping documents in memory

**Implementation**: `ElasticsearchHybridProvider` uses ES `bool` query with `should` clauses combining `match` (BM25) and `knn` (vector) queries, with configurable boost weights.

**Auto-detection**: The system automatically detects when both providers are ES-based and uses native hybrid search. Falls back to Python-side combination for mixed providers (e.g., ES vector + rank-bm25).

## Benefits

1. **Separation of Concerns**: Each layer has a single responsibility
2. **Testability**: Layers can be tested independently
3. **Maintainability**: Easy to locate and modify code
4. **Scalability**: Easy to add new features without mixing concerns
5. **Reusability**: Services can be used by CLI, API, or future interfaces
