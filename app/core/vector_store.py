# app/core/vector_store.py
"""Vector store management and state."""
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from app.indexing import build_vector_store


# Lazy initialization - build vector store only when first needed
_vector_store: Optional[InMemoryVectorStore] = None
_all_documents: List[Document] = []  # Store all documents for hybrid search


def get_vector_store() -> InMemoryVectorStore:
    """Get or build the vector store (lazy initialization)."""
    global _vector_store, _all_documents
    if _vector_store is None:
        _vector_store = build_vector_store(show_progress=False)  # Silent initialization
        # Initialize documents list - we'll populate it as documents are added
        _all_documents = []
    return _vector_store


def get_all_documents() -> List[Document]:
    """Get all documents in the vector store for hybrid search."""
    global _all_documents
    return _all_documents


def add_documents(documents: List[Document]):
    """Add documents to the global index for hybrid search."""
    global _all_documents
    _all_documents.extend(documents)


def reset_vector_store():
    """Reset the vector store (useful for testing)."""
    global _vector_store, _all_documents
    _vector_store = None
    _all_documents = []
