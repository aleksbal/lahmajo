# Lahmajo - Agentic RAG with Hybrid Document Search

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
- **Combination**:
  - **In-memory / rank-bm25 (default)**: results are combined via **Reciprocal Rank Fusion (RRF)** — each candidate's BM25 rank and vector-search rank are fused into a single score, so the combination only depends on relative ranking, not on raw score magnitude (which isn't comparable across BM25 and vector scores, or even consistent in "higher/lower is better" direction across vector store implementations).
  - **Elasticsearch native hybrid** (`VECTOR_INDEX_PROVIDER=elasticsearch` + `BM25_PROVIDER=elasticsearch`): ES combines BM25 and kNN scores at the query level using configurable boost weights (default 40% BM25 / 60% vector).

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

### Option 1: Docker (Recommended for Elasticsearch)

```bash
# Start with Docker (includes Elasticsearch)
./docker-dev.sh start

# Access the application
# Web UI: http://localhost:8000
# Elasticsearch: http://localhost:9200
```

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows

# Install dependencies
pip install -r requirements.txt

# Register the local package (src/ layout - see pyproject.toml). Needed so
# `import lahmajo...` resolves regardless of your working directory.
pip install --no-deps -e .

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

**Use Ollama Cloud via local Ollama:**
```bash
# No environment variables needed, register your local Ollama with your Ollama Cloud account and ensure that Ollama is running!
# Use a Olllama Cloud running LLM modell (with suffix -cloud). An example how to test once local Ollama has been successfully registered:
ollama run gpt-oss:120b-cloud
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

**Note:** If you want to use Elasticsearch, you'll need to install the optional dependencies:
```bash
pip install langchain-elasticsearch elasticsearch
```

### BM25 Index Provider Configuration

The system supports multiple BM25/keyword search implementations via environment variables. By default, it uses `rank-bm25`.

**rank-bm25 (default):**
```bash
export BM25_PROVIDER=rank_bm25
# No additional configuration needed
```

**Elasticsearch (Native BM25):**
```bash
export BM25_PROVIDER=elasticsearch
export ELASTICSEARCH_URL="http://localhost:9200"  # Optional, defaults to http://localhost:9200
export ELASTICSEARCH_INDEX="lahmajo_vectors"  # Optional, defaults to lahmajo_vectors
```

### Elasticsearch Native Hybrid Search

When both `VECTOR_INDEX_PROVIDER=elasticsearch` and `BM25_PROVIDER=elasticsearch`, the system automatically uses **Elasticsearch native hybrid search**, which provides:

- **Single Query**: Combines BM25 and vector search in one ES query (instead of two separate queries)
- **Better Performance**: ES native scoring and optimization
- **Improved Scalability**: ES handles large datasets efficiently
- **Native Scoring**: ES's optimized BM25 and vector scoring algorithms

**Configuration:**
```bash
export VECTOR_INDEX_PROVIDER=elasticsearch
export BM25_PROVIDER=elasticsearch
export ELASTICSEARCH_URL="http://localhost:9200"
export ELASTICSEARCH_INDEX="lahmajo_vectors"
export ELASTICSEARCH_USE_NATIVE_HYBRID=true  # Optional, auto-detected when both providers are ES
```

**Benefits over Python-side combination:**
- Single network round trip instead of two
- ES native score combination (query-level boost weights) instead of the Python-side path's Reciprocal Rank Fusion
- Better performance at scale
- No need to keep all documents in memory

**Note:** The system automatically detects when both providers are ES-based and uses native hybrid search. You can also explicitly enable it with `ELASTICSEARCH_USE_NATIVE_HYBRID=true`.

Additional vector index providers (e.g., Pinecone, Weaviate) can be added by implementing the `VectorIndexProvider` interface in `lahmajo/indexes/vector_provider.py`.

### Rerank Provider Configuration

The system supports an optional reranking pass over retrieved candidates, applied after hybrid search and before the results are sent to the LLM. **Disabled by default** - existing behavior/latency is unchanged unless you opt in.

**None (default):**
```bash
export RERANK_PROVIDER=none
# No additional configuration needed - hybrid search's own ordering is used as-is
```

**LLM-based:**
```bash
export RERANK_PROVIDER=llm
# No additional configuration needed - reuses the already-configured LLM_PROVIDER
```

When enabled, a larger candidate pool (20 instead of 10) is fetched from hybrid search so the reranker has more to work with, then the configured LLM is asked to reorder them by relevance to the query before the final top-k is selected. If the LLM's response can't be parsed into a valid ranking, or the call fails, retrieval falls back to hybrid search's own order rather than failing the request.

Additional rerank providers (e.g., a local cross-encoder model) can be added by implementing the `RerankProvider` interface in `lahmajo/search/rerank_provider.py`.

### Static UI Directory

`GET /` serves `static/index.html`, located by default relative to the installed package (`src/lahmajo/api/routes.py` → repo root → `static/`). This works for how the project is actually run - an editable install (`pip install -e .`), both locally and in `docker/Dockerfile`. It does **not** work with a normal, non-editable `pip install .`, since `static/` lives outside the installed package and isn't bundled into it. If you install that way, point `LAHMAJO_STATIC_DIR` at wherever `static/` actually lives:

```bash
export LAHMAJO_STATIC_DIR=/path/to/lahmajo/static
```

## Usage

### Docker (Recommended)

```bash
# Start all services (Elasticsearch + Lahmajo)
./docker-dev.sh start

# View logs
./docker-dev.sh logs

# Check health
./docker-dev.sh health

# Stop services
./docker-dev.sh stop
```

### Local Development

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
- **Retrieval Options**: Toggle hybrid search and reranking per-question (checkboxes next to the question input) without restarting the server or changing env vars

### API Endpoints

- `GET /` - Web UI
- `POST /ask` - Ask a question (JSON: `{"question": "your question", "use_hybrid": true, "use_rerank": null}`). `use_hybrid` defaults to `true`; `use_rerank` defaults to `null`, which defers to the `RERANK_PROVIDER` env var — pass `true`/`false` to force it on/off for this call.

  The response is `{"answer": "...", "sources": [...]}`. Each entry in `sources` is one
  retrieved excerpt the answer could cite:

  ```json
  {
    "answer": "The deadline is 30 June [source 2].",
    "sources": [
      {"index": 1, "source": "notes.md", "score": 0.0312, "length": 412, "preview": "..."},
      {"index": 2, "source": "contract.pdf", "score": 0.0298, "length": 380, "preview": "..."}
    ]
  }
  ```

  `index` matches the `[source N]` marker the model cites inline, so a citation resolves
  to the file and text it came from. Indices are unique within a response even if the
  agent retrieved more than once — numbering continues across retrieval calls rather
  than restarting at 1. `sources` is empty when the agent answered without
  retrieving, or when retrieval found nothing. `score` is provider-specific and only
  comparable within a single response — it is `null` when the ranking path could not
  supply one.
- `POST /ingest` - Ingest documents (Form data: `url`, `files`, `chunking_strategy`)
- `GET /debug/search?query=...&use_hybrid=true&use_rerank=false` - Debug endpoint to test retrieval directly. `use_rerank=true` previews reranked results for this one call (uses `RERANK_PROVIDER` if configured, otherwise falls back to the LLM reranker) regardless of the global `RERANK_PROVIDER` setting.

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

1. **Hybrid Retrieval**:
   - BM25 finds exact keyword matches
   - Vector index finds semantically similar content
   - Results are combined via Reciprocal Rank Fusion (RRF) — fused by rank order, not raw score magnitude (see "Hybrid Search" under Architecture above). `use_hybrid=false` skips BM25 and uses vector-only search instead.
2. **Filtering**: Very small chunks (< 100 chars) are filtered out
3. **Deduplication**: Near-duplicate chunks from the same source (e.g. two adjacent, overlapping adaptive-chunking windows) are collapsed to one
4. **Reranking (optional)**: If `RERANK_PROVIDER` is set (or `use_rerank=true` for this call), the deduped candidates are reordered by the configured reranker; disabled by default
5. **Top-K Selection**: Top 8 most relevant chunks are selected
6. **Context Assembly**: Selected chunks are formatted and sent to LLM
7. **Answer Generation**: LLM generates answer based on retrieved context only

### Why Hybrid Search?

Pure vector search has limitations:
- Embeddings aren't good at exact keyword matching
- Similarity scores can be too close (0.56 vs 0.54) making ranking unreliable
- Specific terms, names, and technical keywords may not embed well

Hybrid search solves this by:
- BM25 handles exact matches (keywords, technical terms, specific identifiers)
- Vector index handles semantic similarity and meaning
- Combined scores provide better ranking and relevance

## Project Structure & Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full layer breakdown, data flow, and terminology reference.

## Key Design Decisions

### Empty Vector Store Initialization

The `build_vector_store()` function creates an **empty** vector index (pluggable implementation):
- Users ingest documents

### Adaptive Chunking

The system uses an adaptive chunking strategy:
- Automatically detects structured vs unstructured documents, based on document size plus how list/bullet-heavy and line-dense the text is (not just "contains a hyphen somewhere") — always logged, so you can see which branch a given upload took
- Uses appropriate chunk sizes for different document types
- Uses RecursiveCharacterTextSplitter to split large blocks while preserving context
- Ensures meaningful chunks are created from the start (no tiny fragments)

### Details

The implementation follows hybrid search approach:
- **Hybrid Search**: BM25 + Vector similarity, combined via RRF, a common pattern in production RAG systems
- **Deduplication**: near-duplicate chunks from the same source (overlapping adaptive-chunking windows) are collapsed before reranking/generation
- **Reranking**: optional, pluggable (`RERANK_PROVIDER`), off by default
- **RecursiveCharacterTextSplitter**: Most reliable for structured documents
- **Chunk sizes**: 200-400 chars for structured docs, 500-800 for long-form
- **Overlap**: 10-20% for context preservation
- **Filtering**: Only filters chunks < 100 chars (safety check, not workaround)

## License

See LICENSE file for details.

## Notes

- The system requires Ollama to be running locally
- All embeddings and LLM inference happen locally (no external API calls)
- The vector store is in-memory (resets on restart)
- For production, consider using a persistent vector database (e.g., Chroma, Pinecone)
