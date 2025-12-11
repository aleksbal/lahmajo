# app/indexing.py
import bs4
import os
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
    Initialize an empty vector store with embeddings.
    
    This creates a clean vector store ready for document ingestion.
    Users control what content gets added via the ingest endpoint.
    """
    if show_progress:
        print("🔧 Initializing vector store and embeddings...", end=" ", flush=True)
    
    # Create embeddings instance for the vector store
    vector_store_embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://127.0.0.1:11434",
    )
    
    # Create empty vector store
    vector_store = InMemoryVectorStore(vector_store_embeddings)
    
    if show_progress:
        print("✓")
        print("✅ Vector store initialized (empty, ready for document ingestion)\n")

    return vector_store


def _get_embeddings():
    """Get embeddings instance for chunking and vector store."""
    chunking_embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://127.0.0.1:11434",
    )
    vector_store_embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://127.0.0.1:11434",
    )
    return chunking_embeddings, vector_store_embeddings


def _process_documents(
    docs: list[Document], 
    chunking_embeddings=None, 
    use_semantic: bool = False,
    show_progress: bool = False
) -> list[Document]:
    """
    Process documents into chunks for embedding.
    
    Industry standard approach:
    - RecursiveCharacterTextSplitter: Most reliable, predictable, works well for structured docs (CVs, resumes)
    - SemanticChunker: Better for long-form content, but can be unpredictable
    
    For CVs/resumes, RecursiveCharacterTextSplitter is preferred as it:
    - Preserves section boundaries better
    - Creates consistent chunk sizes
    - Doesn't create tiny fragments
    - Is faster and more reliable
    """
    if show_progress:
        chunk_type = "semantic" if use_semantic else "recursive character"
        print(f"✂️  Creating {chunk_type} chunks...", end=" ", flush=True)
    
    # Industry standard: RecursiveCharacterTextSplitter for structured documents
    # For CVs/resumes: Smaller chunks (200-400 chars) enable granular retrieval
    # For long-form content: Larger chunks (500-800 chars) preserve context
    if not use_semantic:
        # Detect if this looks like a CV/resume (structured document)
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
        if chunking_embeddings is None:
            chunking_embeddings, _ = _get_embeddings()
        text_splitter = SemanticChunker(
            chunking_embeddings,
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
        return 0
    
    # Process documents into chunks
    # use_semantic=False: RecursiveCharacterTextSplitter (better for structured docs like CVs)
    # use_semantic=True: SemanticChunker (better for long-form content like blog posts)
    safe_chunks = _process_documents(
        all_docs, 
        use_semantic=use_semantic,
        show_progress=show_progress
    )
    
    # Add to vector store
    if show_progress:
        print("💾 Adding chunks to vector store...", end=" ", flush=True)
    
    vector_store.add_documents(safe_chunks)
    
    if show_progress:
        print(f"✓ ({len(safe_chunks)} chunks added)")
    
    # Return both count and documents for hybrid search indexing
    return len(safe_chunks), safe_chunks

