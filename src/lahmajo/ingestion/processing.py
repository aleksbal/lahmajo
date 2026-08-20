# lahmajo/ingestion/processing.py
import bs4
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

# Set USER_AGENT environment variable early to suppress WebBaseLoader warning
# This must be set before importing WebBaseLoader
if "USER_AGENT" not in os.environ:
    os.environ["USER_AGENT"] = "Mozilla/5.0"

from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lahmajo.llm import get_embeddings
from lahmajo.indexes.vector_provider import get_vector_index_provider, VectorIndexProvider

logger = logging.getLogger(__name__)

# Matches a line that *starts* with a bullet/list marker: "•", "▪", "●", "‣", "∙",
# "-", "*", or a numbered/lettered list item ("1.", "2)", "a.", ...). Deliberately
# anchored to line start (not "contains a hyphen anywhere in the first 500 chars",
# which matches almost any English text - hyphenated words, em dashes, dates).
_BULLET_LINE_RE = re.compile(r'^\s*(?:[•▪●‣∙*-]|\d+[.)]|[a-zA-Z][.)])\s+')


def build_vector_store(show_progress: bool = True) -> VectorIndexProvider:
    """
    This creates a clean vector store ready for document ingestion.
    """
    if show_progress:
        print("🔧 Initializing vector index and embeddings...", end=" ", flush=True)
    
    # Get vector index provider (configurable via environment variables)
    vector_index = get_vector_index_provider()
    
    # Verify the embedding model is properly configured
    # Test the embedding connection by embedding a small test string
    # Note: The provider initializes its own embeddings, but we test here for early error detection
    try:
        from lahmajo.llm import get_embeddings
        test_embeddings = get_embeddings()
        test_embedding = test_embeddings.embed_query("test")
        if show_progress:
            print(f"✓ (embedding dim: {len(test_embedding)})")
    except Exception as e:
        if show_progress:
            print(f"✗")
        error_msg = str(e)
        raise ConnectionError(
            f"Failed to connect to embeddings provider. "
            f"Please check your embedding provider configuration (EMBEDDING_PROVIDER environment variable). "
            f"For Ollama: ensure Ollama is running: 'ollama serve' and the embedding model is available. "
            f"Error: {error_msg}"
        )
    
    if show_progress:
        print("✓")
        print("✅ Vector index initialized (empty, ready for document ingestion)\n")
    
    return vector_index


def _get_semantic_chunker_embeddings():
    """
    Get embedding model for semantic chunking analysis.
    
    Returns:
        Embeddings instance used by SemanticChunker to find semantic breakpoints
        in text (determines where to split documents based on meaning).
    """
    # Use the same embedding model configuration as the vector store
    # This ensures consistency across the system
    return get_embeddings()


def _detect_document_type(docs: list[Document]) -> tuple[bool, dict]:
    """
    Heuristically classify a batch of ingested documents as "structured"
    (resume/CV-like: short, list-heavy) vs. "long-form" (prose-heavy), to pick
    chunk size/overlap.

    Looks at:
    - total size (structured docs are typically short, e.g. CVs)
    - the fraction of non-blank lines that actually look like list/bullet items
      (anchored at line start - not "any hyphen anywhere in the first 500 chars",
      which fires on virtually any English text with a hyphenated word or an
      em dash)
    - average line length (structured docs tend to have many short lines;
      long-form prose has fewer, longer paragraph-wrapped lines)

    Returns (is_structured, stats) - stats is included so callers can log why a
    document was classified the way it was.
    """
    if not docs:
        return False, {}

    total_chars = sum(len(d.page_content) for d in docs)
    non_blank_lines = [
        line
        for d in docs
        for line in d.page_content.splitlines()
        if line.strip()
    ]

    if not non_blank_lines:
        return False, {"total_chars": total_chars, "non_blank_lines": 0}

    bullet_lines = sum(1 for line in non_blank_lines if _BULLET_LINE_RE.match(line))
    bullet_ratio = bullet_lines / len(non_blank_lines)
    avg_line_length = sum(len(line) for line in non_blank_lines) / len(non_blank_lines)

    is_short = total_chars < 15000
    is_list_heavy = bullet_ratio >= 0.15  # ~1 in 7 non-blank lines looks like a list item
    is_line_dense = avg_line_length < 80  # many short lines, not flowing paragraphs

    is_structured = is_short and (is_list_heavy or is_line_dense)

    stats = {
        "total_chars": total_chars,
        "non_blank_lines": len(non_blank_lines),
        "bullet_ratio": round(bullet_ratio, 3),
        "avg_line_length": round(avg_line_length, 1),
        "is_short": is_short,
        "is_list_heavy": is_list_heavy,
        "is_line_dense": is_line_dense,
    }
    return is_structured, stats


def _process_documents(
    docs: list[Document], 
    semantic_chunker_embeddings=None, 
    use_semantic: bool = False,
    show_progress: bool = False
) -> list[Document]:
    """
    Process documents into chunks for embedding.
    
    Standard approach:
    - RecursiveCharacterTextSplitter: Most reliable, predictable, works well for structured docs (CVs, resumes)
    - SemanticChunker: Better for long-form content, but can be unpredictable
    
    For well structured documents like CVs/resumes, RecursiveCharacterTextSplitter is preferred as it:
    - Preserves section boundaries better
    - Creates consistent chunk sizes
    - Doesn't create tiny fragments
    - Is faster and more reliable
    """
    if show_progress:
        chunk_type = "semantic" if use_semantic else "recursive character"
        print(f"✂️  Creating {chunk_type} chunks...", end=" ", flush=True)
    
    # RecursiveCharacterTextSplitter for structured documents
    # For CVs/resumes: Smaller chunks (200-400 chars) enable granular retrieval
    # For long-form content: Larger chunks (500-800 chars) preserve context
    if not use_semantic:
        # Detect if this looks like a well structured document (CV/resume-like)
        # vs. long-form prose - see _detect_document_type() for the heuristic.
        is_structured_doc, detection_stats = _detect_document_type(docs)

        if is_structured_doc:
            # Smaller chunks for CVs/resumes - enables granular retrieval
            chunk_size = 300
            chunk_overlap = 50  # Smaller overlap for smaller chunks
            doc_type = "CV/resume (structured)"
        else:
            # Larger chunks for long-form content
            chunk_size = 600
            chunk_overlap = 100
            doc_type = "long-form content"

        # Always logged (not just when show_progress=True) - the real /ingest API
        # path calls this with show_progress=False, so without this log line
        # there was no way to see which branch fired outside the CLI.
        logger.info(
            f"Chunking: detected '{doc_type}' -> chunk_size={chunk_size}, "
            f"overlap={chunk_overlap} (stats: {detection_stats})"
        )
        if show_progress:
            print(f"  Detected: {doc_type}, using chunk_size={chunk_size}, overlap={chunk_overlap}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],  # Try to break at paragraphs, sentences, words
            is_separator_regex=False,
        )
    else:
        # Semantic chunking for long-form content (experimental)
        # Uses embeddings to analyze text and find semantic breakpoints
        if semantic_chunker_embeddings is None:
            semantic_chunker_embeddings = _get_semantic_chunker_embeddings()
        text_splitter = SemanticChunker(
            semantic_chunker_embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85,  # Lower threshold = more chunks, fewer tiny fragments
            min_chunk_size=300,  # Minimum chunk size
        )
    
    # Process each document separately to preserve metadata
    all_chunks = []
    for doc in docs:
        # Get base metadata from original document
        base_metadata = doc.metadata.copy()
        
        # Create chunks from this document
        chunks = text_splitter.create_documents([doc.page_content])
        
        # Preserve metadata in each chunk
        for chunk in chunks:
            # Merge base metadata with any chunk metadata
            chunk.metadata.update(base_metadata)
            # Ensure source is always set
            if "source" not in chunk.metadata or not chunk.metadata["source"]:
                chunk.metadata["source"] = base_metadata.get("source", "unknown")
        
        all_chunks.extend(chunks)
    
    if show_progress:
        print(f"✓ ({len(all_chunks)} chunks created)")
    
    # Split chunks that are too long for embedding (Ollama limit ~1500 chars)
    # But do it intelligently to avoid tiny fragments
    MAX_CHARS_FOR_EMBED = 1200  # Safe limit for Ollama embeddings
    # Adjust minimum based on average chunk size
    avg_chunk_size = sum(len(c.page_content) for c in all_chunks) / len(all_chunks) if all_chunks else 500
    MIN_CHARS = max(50, int(avg_chunk_size * 0.15))  # 15% of average chunk size, minimum 50
    OVERLAP = 100  # Overlap for sub-chunks
    
    safe_chunks = []
    for i, chunk in enumerate(all_chunks):
        text = chunk.page_content.strip()
        
        # Skip chunks that are too small to be meaningful
        if len(text) < MIN_CHARS:
            continue
        
        # If chunk is within embedding limit, use as-is
        if len(text) <= MAX_CHARS_FOR_EMBED:
            safe_chunks.append(chunk)
        else:
            # Split long chunks intelligently
            # Use RecursiveCharacterTextSplitter for sub-chunking to avoid tiny fragments
            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=MAX_CHARS_FOR_EMBED,
                chunk_overlap=OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            sub_chunks = sub_splitter.create_documents([text])
            
            # Only add sub-chunks that meet minimum size
            for sub_chunk in sub_chunks:
                if len(sub_chunk.page_content.strip()) >= MIN_CHARS:
                    sub_metadata = chunk.metadata.copy()
                    sub_metadata["original_chunk_index"] = i
                    sub_metadata["is_sub_chunk"] = True
                    
                    safe_chunks.append(Document(
                        page_content=sub_chunk.page_content,
                        metadata=sub_metadata,
                    ))
    
    if show_progress:
        print(f"  (final: {len(safe_chunks)} chunks, avg size: {sum(len(c.page_content) for c in safe_chunks) // len(safe_chunks) if safe_chunks else 0} chars)")
    
    return safe_chunks


def load_from_url(url: str, show_progress: bool = False) -> list[Document]:
    """Load documents from a URL."""
    if show_progress:
        print(f"📚 Loading from URL: {url}...", end=" ", flush=True)
    
    loader = WebBaseLoader(
        web_paths=(url,),
        header_template={"User-Agent": os.environ.get("USER_AGENT", "Mozilla/5.0")},
    )
    docs = loader.load()
    
    # Ensure source metadata is set properly
    for doc in docs:
        if not doc.metadata:
            doc.metadata = {}
        doc.metadata["source"] = url
    
    if show_progress:
        print("✓")
    
    return docs


def load_from_file(file_path: str, show_progress: bool = False) -> list[Document]:
    """Load documents from a file (txt, PDF, or markdown)."""
    path = Path(file_path)
    
    if show_progress:
        print(f"📚 Loading from file: {path.name}...", end=" ", flush=True)
    
    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    elif path.suffix.lower() in [".txt", ".md"]:
        # TextLoader works for both .txt and .md files (markdown is plain text)
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Supported: .txt, .pdf, .md")
    
    docs = loader.load()
    
    # Ensure source metadata is set properly
    file_source = str(path.name)  # Use just filename for cleaner metadata
    for doc in docs:
        # Always set source, don't use setdefault which might not work if metadata is empty
        if not doc.metadata:
            doc.metadata = {}
        doc.metadata["source"] = file_source
        doc.metadata["file_path"] = str(path)  # Keep full path for reference
    
    if show_progress:
        print("✓")
    
    return docs


def ingest_documents(
    vector_index: VectorIndexProvider,
    urls: Optional[list[str]] = None,
    file_paths: Optional[list[str]] = None,
    use_semantic: bool = False,
    show_progress: bool = False,
    track_documents: bool = True
) -> tuple[int, list[Document]]:
    """Ingest documents from URLs and/or files into an existing vector store."""
    all_docs = []
    
    # Load from URLs
    if urls:
        for url in urls:
            docs = load_from_url(url, show_progress=show_progress)
            all_docs.extend(docs)
    
    # Load from files
    if file_paths:
        for file_path in file_paths:
            docs = load_from_file(file_path, show_progress=show_progress)
            all_docs.extend(docs)
    
    if not all_docs:
        return 0, []
    
    # Process documents into chunks
    # use_semantic=False: RecursiveCharacterTextSplitter (better for structured docs like CVs)
    # use_semantic=True: SemanticChunker (better for long-form content like blog posts)
    safe_chunks = _process_documents(
        all_docs, 
        use_semantic=use_semantic,
        show_progress=show_progress
    )
    
    # Add to vector index one chunk at a time with retries
    # Ollama can be overwhelmed even with small batches, so process sequentially
    if show_progress:
        print(f"💾 Adding {len(safe_chunks)} chunks to vector index...", end=" ", flush=True)
    
    # Process chunks one at a time with retry logic
    # This prevents overwhelming Ollama and handles transient connection errors
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds between retries
    CHUNK_DELAY = 0.1  # small delay between chunks to avoid overwhelming Ollama
    
    total_added = 0
    failed_chunks = []
    
    for i, chunk in enumerate(safe_chunks):
        retry_count = 0
        success = False
        
        while retry_count < MAX_RETRIES and not success:
            try:
                # Add single chunk
                vector_index.add_documents([chunk])
                total_added += 1
                success = True
                
                # Small delay between chunks to avoid overwhelming Ollama
                if i < len(safe_chunks) - 1:  # Don't delay after last chunk
                    time.sleep(CHUNK_DELAY)
                
                # Show progress for large files
                if show_progress and len(safe_chunks) > 10:
                    progress = min(100, int((total_added / len(safe_chunks)) * 100))
                    print(f"\r💾 Adding chunks... {total_added}/{len(safe_chunks)} ({progress}%)", end="", flush=True)
                    
            except Exception as chunk_error:
                retry_count += 1
                error_msg = str(chunk_error)
                
                # Check for connection errors
                if "EOF" in error_msg or "500" in error_msg or "embedding" in error_msg.lower():
                    if retry_count < MAX_RETRIES:
                        # Exponential backoff: wait longer on each retry
                        wait_time = RETRY_DELAY * (2 ** (retry_count - 1))
                        if show_progress:
                            print(f"\r💾 Retrying chunk {i+1}/{len(safe_chunks)} (attempt {retry_count + 1}/{MAX_RETRIES})...", end="", flush=True)
                        time.sleep(wait_time)
                        continue
                    else:
                        # Max retries reached, mark as failed
                        failed_chunks.append((i, chunk, error_msg))
                        if show_progress:
                            print(f"\r⚠️  Failed to embed chunk {i+1}/{len(safe_chunks)} after {MAX_RETRIES} retries", flush=True)
                        break
                else:
                    # Non-retryable error, fail immediately
                    raise RuntimeError(
                        f"Error embedding chunk {i+1}/{len(safe_chunks)}: {error_msg}"
                    )
        
        if not success:
            # Continue with next chunk even if this one failed
            continue
    
    if show_progress:
        if len(safe_chunks) > 10:
            print(f"\r💾 Added {total_added}/{len(safe_chunks)} chunks to vector store", end="", flush=True)
        print(f" ✓", flush=True)
    
    # Report any failed chunks
    if failed_chunks:
        failed_count = len(failed_chunks)
        if show_progress:
            print(f"⚠️  Warning: {failed_count} chunks failed to embed after retries", flush=True)
        # For now, we continue - the successfully embedded chunks are still useful
        # In the future, we could log these or retry them separately
    
    if total_added == 0:
        raise RuntimeError(
            f"Failed to embed any chunks. All {len(safe_chunks)} chunks failed. "
            f"Please check your embedding provider configuration and ensure it's accessible."
        )
    
    # Return both count and documents for hybrid search indexing
    return len(safe_chunks), safe_chunks

