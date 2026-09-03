# lahmajo/indexes/state.py
"""Vector index state management."""
import os
from typing import List, Optional
from langchain_core.documents import Document

from lahmajo.indexes.vector_provider import get_vector_index_provider, VectorIndexProvider


# Lazy initialization - build vector index only when first needed
_vector_index: Optional[VectorIndexProvider] = None
_all_documents: List[Document] = []  # Store all documents for hybrid search (only for non-ES providers)
_use_elasticsearch: Optional[bool] = None  # Cache ES detection


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


def _is_elasticsearch_configured() -> bool:
    """Check if Elasticsearch is configured as the provider."""
    global _use_elasticsearch
    if _use_elasticsearch is None:
        vector_provider = os.getenv("VECTOR_INDEX_PROVIDER", "in_memory").lower()
        bm25_provider = os.getenv("BM25_PROVIDER", "rank_bm25").lower()
        _use_elasticsearch = vector_provider == "elasticsearch" or bm25_provider == "elasticsearch"
    return _use_elasticsearch


def get_all_documents() -> List[Document]:
    """
    Get all documents in the vector store.
    
    For Elasticsearch: Fetches documents from ES (ES is source of truth).
    For in-memory providers: Returns in-memory document list.
    
    Note: This is used to initialize the BM25 index for non-ES providers.
    For ES providers, documents are already in ES and don't need to be loaded.
    
    Returns:
        List of all documents in the store
    """
    global _all_documents, _vector_index
    
    # If ES is configured, fetch from ES
    if _is_elasticsearch_configured():
        try:
            vector_index = get_vector_index()
            
            # Check if vector_index has method to get all documents from ES
            if hasattr(vector_index, 'get_es_client'):
                es_client = vector_index.get_es_client()
                index_name = vector_index.get_index_name()
                
                # Fetch all documents from ES
                try:
                    response = es_client.search(
                        index=index_name,
                        body={
                            "query": {"match_all": {}},
                            "size": 10000,  # Adjust based on your needs
                            "_source": ["text", "metadata"]
                        }
                    )
                    
                    documents = []
                    for hit in response["hits"]["hits"]:
                        source = hit["_source"]
                        text_content = source.get("text", "") or source.get("page_content", "")
                        metadata = source.get("metadata", {})
                        if not isinstance(metadata, dict):
                            metadata = {}
                        
                        doc = Document(
                            page_content=text_content,
                            metadata=metadata
                        )
                        documents.append(doc)
                    
                    return documents
                except Exception:
                    # If ES fetch fails, return empty list
                    # Documents are in ES, but we can't fetch them all
                    return []
        except Exception:
            # If ES operations fail, fall back to in-memory list
            pass
    
    # For non-ES providers, return in-memory list
    return _all_documents


def add_documents(documents: List[Document]):
    """
    Add documents to the global index for hybrid search.
    
    For Elasticsearch: Documents are already in ES (added via vector provider),
    so we don't need to store them separately.
    For in-memory providers: Add to in-memory list for BM25 indexing.
    """
    global _all_documents
    
    # Only store in memory for non-ES providers
    if not _is_elasticsearch_configured():
        _all_documents.extend(documents)
    # For ES, documents are already in ES via vector provider's add_documents()


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
