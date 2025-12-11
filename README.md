# Lahmajo - RAG System with Hybrid Search

A production-ready Retrieval-Augmented Generation (RAG) system with both CLI and Web UI interfaces. The system uses Ollama for local LLM inference and implements industry-standard hybrid search combining BM25 (keyword) and vector (semantic) retrieval.

## Features

- **Hybrid Search**: Combines BM25 keyword matching with vector semantic search for optimal retrieval
- **Adaptive Chunking**: Automatically detects document types (structured vs unstructured content) and uses appropriate chunk sizes
- **Multiple File Formats**: Supports TXT, PDF, and Markdown files
- **Flexible Chunking Strategies**: Choose between Recursive (structured docs) or Semantic (long-form) chunking
- **Web UI**: Modern single-page interface for document ingestion and question answering
- **CLI Interface**: Command-line interface for interactive Q&A
- **Empty Vector Store**: Clean initialization - users control all content ingestion

## Architecture

### Hybrid Search (Industry Standard)

The system uses **hybrid search**, which is the industry standard approach for production RAG systems:

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
- Ollama running locally on `http://127.0.0.1:11434`
- Required Ollama models:
  - `mistral` (or `llama3`) for chat/LLM
  - `nomic-embed-text` for embeddings

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

## Usage

### Web UI

Start the web server:

```bash
python -m app.web
# or
uvicorn app.web:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` in your browser.

**Features:**
- **Ingest Documents**: Upload TXT, PDF, or MD files, or provide URLs
- **Choose Chunking Strategy**: Select Recursive (structured docs) or Semantic (long-form) chunking
- **Ask Questions**: Query the knowledge base and get answers based on retrieved context

### CLI Interface

Run the CLI for interactive Q&A:

```bash
python -m app.cli
```

Or ask a single question:

```bash
python -m app.cli "What information do you have about me?"
```

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
   - Added to vector store (for semantic search)
   - Added to BM25 index (for keyword matching)

### Query Flow

1. **Query Analysis**: System detects if query is a name/keyword query or semantic query
2. **Hybrid Retrieval**:
   - BM25 finds exact keyword matches
   - Vector search finds semantically similar content
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
- Vector search handles semantic similarity and meaning
- Combined scores provide better ranking and relevance

## Project Structure

```
lahmajo/
├── app/
│   ├── agent.py          # RAG agent with hybrid search retrieval
│   ├── cli.py            # Command-line interface
│   ├── hybrid_search.py  # BM25 + Vector hybrid search implementation
│   ├── indexing.py       # Document loading, chunking, and ingestion
│   ├── web.py            # FastAPI web server and endpoints
│   └── main.py           # Legacy Ollama proxy (separate functionality)
├── static/
│   └── index.html        # Web UI (single-page application)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Key Design Decisions

### Empty Vector Store Initialization

The `build_vector_store()` function creates an **empty** vector store. This is intentional:
- Users have full control over what content gets ingested
- No hardcoded documents or URLs
- Clean, maintainable architecture

### Adaptive Chunking

The system uses intelligent chunking:
- Automatically detects structured vs unstructured documents
- Uses appropriate chunk sizes for different document types
- Intelligently splits long chunks using RecursiveCharacterTextSplitter
- Ensures meaningful chunks are created from the start (no tiny fragments)

### Industry Standards

The implementation follows industry best practices:
- **Hybrid Search**: BM25 + Vector (standard in production RAG systems)
- **RecursiveCharacterTextSplitter**: Most reliable for structured documents
- **Chunk sizes**: 200-400 chars for structured docs, 500-800 for long-form
- **Overlap**: 10-20% for context preservation
- **Filtering**: Only filters chunks < 100 chars (safety check, not workaround)

## Troubleshooting

### Vector Search Returns Irrelevant Results

- **Solution**: The system uses hybrid search by default. Ensure documents are properly indexed for BM25.
- **Check**: Use `/debug/search?query=...` to see what's being retrieved

### Too Few Chunks Created

- **Problem**: Chunk size might be too large
- **Solution**: System auto-detects structured documents and uses 300-char chunks. For unstructured documents, try semantic chunking.

### Documents Not Found in Search

- **Check**: Verify documents were successfully ingested (check logs)
- **Try**: Use hybrid search (default) which combines keyword and semantic matching
- **Debug**: Use `/debug/search` endpoint to test retrieval directly

## Development

### Adding New File Types

1. Add loader import in `app/indexing.py`
2. Add file extension check in `load_from_file()`
3. Update validation in `app/web.py`
4. Update HTML file input `accept` attribute

### Modifying Chunking

Edit `_process_documents()` in `app/indexing.py`:
- Adjust chunk sizes for different document types
- Modify detection logic for structured vs long-form documents
- Change chunking strategy parameters

## License

See LICENSE file for details.

## Notes

- The system requires Ollama to be running locally
- All embeddings and LLM inference happen locally (no external API calls)
- The vector store is in-memory (resets on restart)
- For production, consider using a persistent vector database (e.g., Chroma, Pinecone)
