# Lahmajo - RAG System with Hybrid Document Search

Experimental Retrieval-Augmented Generation (RAG) system with Web UI interfaces for testing, experimenting and prototyping. Supports multiple LLM providers (Ollama local/cloud, OpenAI) and implements standard hybrid search combining BM25 (keyword) and vector (semantic) retrieval.

## Features

- **Agentic RAG**: Search agent implemented with LangChain
- **Hybrid Search**: Agent BM25 keyword matching with vector semantic search for optimal retrieval
- **Adaptive Chunking**: Automatically detects document types (structured vs unstructured content) and uses appropriate chunk sizes
- **Multiple File Formats**: Supports TXT, PDF, and Markdown files
- **Flexible Chunking Strategies**: Choose between Recursive (structured docs) or Semantic (long-form) chunking
- **Web UI**: Single-page interface for document ingestion and question answering
- **CLI Interface**: Command-line interface for interactive Q&A

## Architecture

### Hybrid Search

The system uses **hybrid search**, which is the standard approach for production RAG systems:

- **BM25 (Keyword Matching)**: Excellent for exact matches, names, keywords
- **Vector Search (Semantic)**: Good for semantic similarity and understanding
- **Weighted Combination**: 
  - Name queries: 60% BM25, 40% Vector (prioritizes exact matches)
  - Semantic queries: 40% BM25, 60% Vector (prioritizes meaning)

This approach solves the problem where pure vector search struggles with exact name/keyword matching, as embeddings aren't always good at capturing exact matches.

### Adaptive Chunking

The system automatically detects document types and uses appropriate chunking:

- **Structured Documents** (resumes, technical docs, formatted data):
  - Chunk size: 300 characters
  - Overlap: 50 characters
  - Uses RecursiveCharacterTextSplitter
  - Enables granular retrieval of specific sections

- **Long-form Content (Articles, Blog Posts)**:
  - Chunk size: 600 characters
  - Overlap: 100 characters
  - Uses RecursiveCharacterTextSplitter by default
  - Preserves more context

### Chunking Strategies

You can choose between two chunking strategies when ingesting documents:

1. **Recursive** (Recommended for structured documents):
   - Consistent chunks, preserves structure
   - Predictable and reliable
   - Industry standard for structured documents

2. **Semantic** (Better for articles, blog posts):
   - Chunks by meaning
   - Can be unpredictable but better for long-form content
   - Uses embeddings to detect semantic breakpoints

## Requirements

- Python 3.12+
- **LLM Provider** (choose one):
  - **Ollama (default)**: Running locally on `http://127.0.0.1:11434`
    - Required models: `gpt-oss:120b-cloud` (or `llama3`) for chat/LLM, `embeddinggemma` for embeddings
  - **OpenAI**: API key required (see Configuration below)

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows

# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running
ollama serve
```

## Configuration

The system supports multiple LLM and embedding providers via environment variables. By default, it uses Ollama locally.

### LLM Provider Configuration

Set `LLM_PROVIDER` to choose your LLM backend:

**Ollama Local (default):**
```bash
export LLM_PROVIDER=ollama_local
export LLM_MODEL="gpt-oss:120b-cloud"  # Optional, defaults to gpt-oss:120b-cloud
export LLM_BASE_URL="http://127.0.0.1:11434"  # Optional, defaults to http://127.0.0.1:11434
export LLM_TEMPERATURE=0.1  # Optional, defaults to 0.1
```

**Ollama Cloud:**
```bash
export LLM_PROVIDER=ollama_cloud
export LLM_MODEL="gpt-oss:120b-cloud"
export LLM_BASE_URL="https://your-ollama-cloud-url.com"  # Your Ollama cloud endpoint
export LLM_TEMPERATURE=0.1
```

**OpenAI:**
```bash
export LLM_PROVIDER=openai
export LLM_MODEL="gpt-4"  # Optional, defaults to gpt-4
export OPENAI_API_KEY="your-api-key-here"
export LLM_TEMPERATURE=0.1
```

### Embedding Provider Configuration

Set `EMBEDDING_PROVIDER` to choose your embedding backend:

**Ollama Local (default):**
```bash
export EMBEDDING_PROVIDER=ollama_local
export EMBEDDING_MODEL="embeddinggemma"  # Optional, defaults to embeddinggemma
export EMBEDDING_BASE_URL="http://127.0.0.1:11434"  # Optional, defaults to http://127.0.0.1:11434
```

**Ollama Cloud:**
```bash
export EMBEDDING_PROVIDER=ollama_cloud
export EMBEDDING_MODEL="embeddinggemma"
export EMBEDDING_BASE_URL="https://your-ollama-cloud-url.com"
```

**OpenAI:**
```bash
export EMBEDDING_PROVIDER=openai
export EMBEDDING_MODEL="text-embedding-ada-002"  # Optional, defaults to text-embedding-ada-002
export OPENAI_API_KEY="your-api-key-here"  # Same key as LLM if using OpenAI for both
```

### Quick Examples

**Use OpenAI for both LLM and embeddings:**
```bash
export LLM_PROVIDER=openai
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY="your-key-here"
```

**Use Ollama locally (no configuration needed - this is the default):**
```bash
# No environment variables needed, just ensure Ollama is running
ollama serve
```

**Mix providers (e.g., Ollama for LLM, OpenAI for embeddings):**
```bash
export LLM_PROVIDER=ollama_local
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY="your-key-here"
```

**Note:** If you want to use OpenAI, you'll need to install the optional dependency:
```bash
pip install langchain-openai
```

### Vector Index Provider Configuration

The system supports multiple vector index implementations via environment variables. By default, it uses in-memory storage.

**In-Memory (default):**
```bash
export VECTOR_INDEX_PROVIDER=in_memory
# No additional configuration needed
```

**Elasticsearch:**
```bash
export VECTOR_INDEX_PROVIDER=elasticsearch
export ELASTICSEARCH_URL="http://localhost:9200"  # Optional, defaults to http://localhost:9200
export ELASTICSEARCH_INDEX="lahmajo_vectors"  # Optional, defaults to lahmajo_vectors
```

**Note:** If you want to use Elasticsearch, you'll need to install the optional dependency:
```bash
pip install langchain-elasticsearch
```

### BM25 Index Provider Configuration

The system supports multiple BM25/keyword search implementations via environment variables. By default, it uses `rank-bm25`.

**rank-bm25 (default):**
```bash
export BM25_PROVIDER=rank_bm25
# No additional configuration needed
```

**Note:** Additional BM25 providers (e.g., Elasticsearch, Whoosh) can be added by implementing the `BM25Provider` interface in `lahmajo/indexes/bm25_provider.py`.

Additional vector index providers (e.g., Pinecone, Weaviate) can be added by implementing the `VectorIndexProvider` interface in `lahmajo/indexes/vector_provider.py`.

## Usage

### Web UI

Start the web server:

```bash
python -m lahmajo.api.routes
```

Or using uvicorn directly:

```bash
uvicorn lahmajo.api.routes:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` in your browser.

**Features:**
- **Ingest Documents**: Upload TXT, PDF, or MD files, or provide URLs
- **Choose Chunking Strategy**: Select Recursive (structured docs) or Semantic (long-form) chunking
- **Ask Questions**: Query the knowledge base and get answers based on retrieved context

### API Endpoints

- `GET /` - Web UI
- `POST /ask` - Ask a question (JSON: `{"question": "your question"}`)
- `POST /ingest` - Ingest documents (Form data: `url`, `files`, `chunking_strategy`)
- `GET /debug/search?query=...` - Debug endpoint to test retrieval directly

## How It Works

### Document Ingestion Flow

1. **Load Documents**: Files are loaded using appropriate loaders (TextLoader, PyPDFLoader)
2. **Detect Document Type**: System automatically detects if it's a structured document or unstructured long-form content
3. **Chunk Documents**: 
   - Structured documents → 300 char chunks (enables granular retrieval)
   - Unstructured long-form → 600 char chunks (preserves context)
4. **Process Chunks**: Long chunks are intelligently split to fit embedding limits (1200 chars)
5. **Index Documents**: 
   - Added to vector index (for semantic search)
   - Added to BM25 index (for keyword matching)

### Query Flow

1. **Query Analysis**: System detects if query is a name/keyword query or semantic query
2. **Hybrid Retrieval**:
   - BM25 finds exact keyword matches
   - Vector index finds semantically similar content
   - Results are combined with weighted scores
3. **Filtering**: Very small chunks (< 100 chars) are filtered out
4. **Top-K Selection**: Top 8 most relevant chunks are selected
5. **Context Assembly**: Selected chunks are formatted and sent to LLM
6. **Answer Generation**: LLM generates answer based on retrieved context only

### Why Hybrid Search?

Pure vector search has limitations:
- Embeddings aren't good at exact keyword matching
- Similarity scores can be too close (0.56 vs 0.54) making ranking unreliable
- Specific terms, names, and technical keywords may not embed well

Hybrid search solves this by:
- BM25 handles exact matches (keywords, technical terms, specific identifiers)
- Vector index handles semantic similarity and meaning
- Combined scores provide better ranking and relevance

## Project Structure

```
lahmajo/
├── lahmajo/
│   ├── api/
│   │   └── routes.py     # FastAPI web server and endpoints
│   ├── services/
│   │   ├── rag_service.py        # RAG agent orchestration
│   │   ├── retrieval_service.py  # Document retrieval
│   │   └── ingestion_service.py  # Document ingestion
│   ├── ingestion/
│   │   └── processing.py     # Document loading, chunking, embedding
│   ├── indexes/
│   │   ├── state.py          # Index state management
│   │   ├── vector_provider.py # Vector index implementations
│   │   └── bm25_provider.py   # BM25 index implementations
│   ├── search/
│   │   └── hybrid_search.py  # BM25 + Vector hybrid search
│   └── cli.py            # Command-line interface
├── static/
│   └── index.html        # Web UI (single-page application)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Key Design Decisions

### Empty Vector Store Initialization

The `build_vector_store()` function creates an **empty** vector index (pluggable implementation):
- Users ingest documents

### Adaptive Chunking

The system uses an adaptive chunking strategy:
- Automatically detects structured vs unstructured documents
- Uses appropriate chunk sizes for different document types
- Uses RecursiveCharacterTextSplitter to split large blocks while preserving context
- Ensures meaningful chunks are created from the start (no tiny fragments)

### Details

The implementation follows hybrid search approach:
- **Hybrid Search**: BM25 + Vector similarity, a common pattern in production RAG systems
- **RecursiveCharacterTextSplitter**: Most reliable for structured documents
- **Chunk sizes**: 200-400 chars for structured docs, 500-800 for long-form
- **Overlap**: 10-20% for context preservation
- **Filtering**: Only filters chunks < 100 chars (safety check, not workaround)

### Architecture

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
         ▼
┌─────────────────┐
│ Ingestion Layer │  Document ingestion (lahmajo/ingestion/)
│  (Processing)  │  - processing.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Index Layer    │  Search indexes (lahmajo/indexes/)
│  (Indexes)      │  - state.py
└────────┬────────┘  - vector_provider.py
         │           - bm25_provider.py
         │
         ▼
┌─────────────────┐
│  Search Layer   │  Search algorithms (lahmajo/search/)
│  (Retrieval)    │  - hybrid_search.py
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
- Index state management
- Pluggable index implementations
- **No business logic**

**Files**:
- `state.py` - Index state management (singleton, document tracking)
- `vector_provider.py` - Vector index provider (in_memory, elasticsearch, etc.)
- `bm25_provider.py` - BM25 index provider (rank_bm25, elasticsearch, etc.)

### Search Layer (`lahmajo/search/`)
**Responsibility**: Search algorithms
- Hybrid search implementation
- **No business logic**

**Files**:
- `hybrid_search.py` - BM25 + Vector search

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

## License

See LICENSE file for details.

## Notes

- The system requires Ollama to be running locally
- All embeddings and LLM inference happen locally (no external API calls)
- The vector store is in-memory (resets on restart)
- For production, consider using a persistent vector database (e.g., Chroma, Pinecone)
