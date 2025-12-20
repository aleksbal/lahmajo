# lahmajo/storage/indexing.py
import bs4
import os
import time
from pathlib import Path
from typing import Optional

# Set USER_AGENT environment variable early to suppress WebBaseLoader warning
# This must be set before importing WebBaseLoader
if "USER_AGENT" not in os.environ:
    os.environ["USER_AGENT"] = "Mozilla/5.0"

from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings


def build_vector_store(show_progress: bool = True) -> InMemoryVectorStore:
    """
    This creates a clean vector store ready for document ingestion.
    """
    if show_progress:
        print("🔧 Initializing vector store and embeddings...", end=" ", flush=True)
    
    # Create embedding model for embedding documents into the vector store
    # Explicitly set base_url to ensure correct Ollama connection
    document_embedding_model = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://127.0.0.1:11434",
    )
    
    # Verify the embedding model is properly configured
    # Test the embedding connection by embedding a small test string
    try:
        test_embedding = document_embedding_model.embed_query("test")
        if show_progress:
            print(f"✓ (embedding dim: {len(test_embedding)})")
    except Exception as e:
        if show_progress:
            print(f"✗")
        error_msg = str(e)
        # Check if the error mentions a wrong port
        if "63480" in error_msg or "63480" in str(e):
            raise ConnectionError(
                f"Embedding connection error - wrong port detected. "
                f"This may indicate a configuration issue with OllamaEmbeddings. "
                f"Expected base_url: http://127.0.0.1:11434. "
                f"Error: {error_msg}"
            )
        raise ConnectionError(
            f"Failed to connect to Ollama embeddings at http://127.0.0.1:11434. "
            f"Please ensure Ollama is running: 'ollama serve' and the 'nomic-embed-text' model is available: 'ollama pull nomic-embed-text'. "
            f"Error: {error_msg}"
        )
    
    # Create empty vector store with the embedding model
    vector_store = InMemoryVectorStore(document_embedding_model)
    
    if show_progress:
        print("✓")
        print("✅ Vector store initialized (empty, ready for document ingestion)\n")
    
    return vector_store


def _get_semantic_chunker_embeddings():
    """
    Get embedding model for semantic chunking analysis.
    
    Returns:
        OllamaEmbeddings instance used by SemanticChunker to find semantic breakpoints
        in text (determines where to split documents based on meaning).
    """
    # Use the same embedding model configuration as the vector store
    # This ensures consistency and proper base_url configuration
    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://127.0.0.1:11434",
    )


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
        # Detect if this looks like a well structured document
        # CVs typically have sections, bullet points, and are relatively short
        is_structured_doc = False
        if docs:
            total_chars = sum(len(d.page_content) for d in docs)
            # CVs are usually 2000-10000 chars, have many newlines, bullet points
            has_bullets = any('•' in d.page_content or '-' in d.page_content[:500] for d in docs)
            has_sections = any('\n\n' in d.page_content or d.page_content.count('\n') > 10 for d in docs)
            is_structured_doc = (total_chars < 15000) and (has_bullets or has_sections)
        
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
    vector_store: InMemoryVectorStore,
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
    
    # Add to vector store one chunk at a time with retries
    # Ollama can be overwhelmed even with small batches, so process sequentially
    if show_progress:
        print(f"💾 Adding {len(safe_chunks)} chunks to vector store...", end=" ", flush=True)
    
    # Ensure the vector store's embedding model is properly configured
    if hasattr(vector_store, 'embeddings') and hasattr(vector_store.embeddings, 'base_url'):
        if vector_store.embeddings.base_url != "http://127.0.0.1:11434":
            raise ValueError(
                f"Vector store embedding model has incorrect base_url: {vector_store.embeddings.base_url}. "
                f"Expected: http://127.0.0.1:11434"
            )
    
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
                vector_store.add_documents([chunk])
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
            f"Please check Ollama is running and the 'nomic-embed-text' model is available."
        )
    
    # Return both count and documents for hybrid search indexing
    return len(safe_chunks), safe_chunks

