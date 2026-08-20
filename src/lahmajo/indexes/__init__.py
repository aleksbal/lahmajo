# lahmajo/indexes/__init__.py
"""Index providers and state management for vector and BM25 search."""
from lahmajo.indexes.vector_provider import get_vector_index_provider, VectorIndexProvider
from lahmajo.indexes.bm25_provider import get_bm25_provider, BM25Provider
from lahmajo.indexes.state import get_vector_index, get_all_documents, add_documents

__all__ = [
    "get_vector_index_provider",
    "VectorIndexProvider",
    "get_bm25_provider",
    "BM25Provider",
    "get_vector_index",
    "get_all_documents",
    "add_documents",
]

