# lahmajo/indexes/state.py
"""Vector index state management."""
from typing import List, Optional
from langchain_core.documents import Document

from lahmajo.indexes.vector_provider import get_vector_index_provider, VectorIndexProvider


# Lazy initialization - build vector index only when first needed
_vector_index: Optional[VectorIndexProvider] = None
_all_documents: List[Document] = []  # Store all documents for hybrid search


def get_vector_index() -> VectorIndexProvider:
    """
    Get or build the vector index (lazy initialization).
    
    Returns:
        VectorIndexProvider instance (pluggable implementation)
    """
    global _vector_index, _all_documents
    if _vector_index is None:
        _vector_index = get_vector_index_provider()
        # Initialize documents list - we'll populate it as documents are added
        _all_documents = []
    return _vector_index


def get_all_documents() -> List[Document]:
    """
    Get all documents in the vector store.
    
    Note: This is used to initialize the BM25 index (which needs all documents
    to build the keyword index). During search, only top candidates from each
    method are retrieved and combined - not all documents.
    
    Returns:
        List of all documents in the store
    """
    global _all_documents
    return _all_documents


def add_documents(documents: List[Document]):
    """Add documents to the global index for hybrid search."""
    global _all_documents
    _all_documents.extend(documents)


def get_vector_store() -> VectorIndexProvider:
    """
    Get vector index (backward compatibility alias).
    
    Note: Use get_vector_index() for new code.
    """
    return get_vector_index()


def get_vector_store() -> VectorIndexProvider:
    """
    Get vector index (backward compatibility alias).
    
    Note: Use get_vector_index() for new code.
    """
    return get_vector_index()


def reset_vector_index():
    """Reset the vector index (useful for testing)."""
    global _vector_index, _all_documents
    _vector_index = None
    _all_documents = []


def reset_vector_store():
    """Reset vector index (backward compatibility alias)."""
    reset_vector_index()
