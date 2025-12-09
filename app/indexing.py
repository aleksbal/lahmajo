# app/indexing.py
import bs4
import os

# Set USER_AGENT environment variable early to suppress WebBaseLoader warning
# This must be set before importing WebBaseLoader
if "USER_AGENT" not in os.environ:
    os.environ["USER_AGENT"] = "Mozilla/5.0"

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings


def build_vector_store(show_progress: bool = True) -> InMemoryVectorStore:
    """Build vector store with optional progress indicators."""
    if show_progress:
        print("📚 Loading document from web...", end=" ", flush=True)
    
    # 1) Load the raw document from the web
    bs4_strainer = bs4.SoupStrainer(
        class_=("post-title", "post-header", "post-content")
    )

    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        header_template={"User-Agent": os.environ.get("USER_AGENT", "Mozilla/5.0")},
        bs_kwargs={"parse_only": bs4_strainer},
    )

    docs = loader.load()  # -> list[Document]
    
    if show_progress:
        print("✓")

    if show_progress:
        print("🔧 Initializing embeddings...", end=" ", flush=True)
    
    # 2) Create separate embeddings instances:
    #    - One for semantic chunking (gets used heavily during chunking)
    #    - One for vector store (stays clean for embedding chunks)
    #    This prevents the chunking process from affecting the vector store embeddings
    chunking_embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://127.0.0.1:11434",
    )
    
    vector_store_embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://127.0.0.1:11434",
    )
    
    if show_progress:
        print("✓")

    if show_progress:
        print("✂️  Creating semantic chunks (this may take a moment)...", end=" ", flush=True)

    # 3) Semantic chunker instead of RecursiveCharacterTextSplitter
    #    It detects semantic breakpoints using embeddings.
    # Tune SemanticChunker for tighter, more focused breaks:
    # - percentile mode with a higher cutoff = fewer, more meaningful breakpoints
    # - min_chunk_size=400 chars to avoid tiny fragments
    text_splitter = SemanticChunker(
        chunking_embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=92,
        min_chunk_size=400,
    )

    # SemanticChunker expects a list of strings or texts;
    # we take the page_content from each loaded Document.
    raw_texts = [d.page_content for d in docs]

    # This returns list[Document] again, but now *semantically* chunked.
    semantic_chunks = text_splitter.create_documents(raw_texts)
    
    if show_progress:
        print(f"✓ ({len(semantic_chunks)} chunks created)")

    # (Optional) Preserve a simple "source" metadata
    for chunk in semantic_chunks:
        chunk.metadata.setdefault("source", "lilianweng_agent_blog")

    if show_progress:
        print("💾 Building vector store and embedding chunks...", end=" ", flush=True)

    # 4) Build vector store and add chunks using the fresh embeddings instance.
    #    Ollama's embedding endpoint fails with very long payloads (>1500 chars).
    #    Instead of truncating (which loses information), we split long semantic chunks
    #    into smaller sub-chunks while preserving their semantic structure.
    #    This way we keep all the information while making it embeddable.
    MAX_CHARS_FOR_EMBED = 1200  # Safe limit for Ollama embeddings
    OVERLAP = 100  # Small overlap to preserve context between sub-chunks
    
    safe_chunks = []
    for i, chunk in enumerate(semantic_chunks):
        text = chunk.page_content
        
        # If chunk is small enough, use it as-is
        if len(text) <= MAX_CHARS_FOR_EMBED:
            safe_chunks.append(chunk)
        else:
            # Split long semantic chunk into smaller sub-chunks with overlap
            # This preserves all information while making chunks embeddable
            start = 0
            sub_chunk_num = 0
            while start < len(text):
                end = start + MAX_CHARS_FOR_EMBED
                sub_text = text[start:end]
                
                # Create sub-chunk with metadata indicating it's part of a larger semantic chunk
                sub_metadata = chunk.metadata.copy()
                sub_metadata["original_chunk_index"] = i
                sub_metadata["sub_chunk"] = sub_chunk_num
                
                safe_chunks.append(Document(
                    page_content=sub_text,
                    metadata=sub_metadata,
                ))
                
                # Move start position with overlap
                start = end - OVERLAP
                sub_chunk_num += 1

    vector_store = InMemoryVectorStore(vector_store_embeddings)
    vector_store.add_documents(safe_chunks)
    
    if show_progress:
        print("✓")
        print("✅ Vector store ready!\n")

    return vector_store

