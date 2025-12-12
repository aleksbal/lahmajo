# app/services/retrieval_service.py
"""Retrieval service - handles document retrieval using hybrid search."""
import logging
from typing import List, Tuple
from langchain_core.documents import Document

from app.core.vector_store import get_vector_store, get_all_documents
from app.hybrid_search import HybridRetriever

logger = logging.getLogger(__name__)


def retrieve_context(query: str, k: int = 8) -> Tuple[str, List[Document]]:
    """
    Retrieve relevant documents for a query using hybrid search.
    
    Args:
        query: Search query
        k: Number of documents to return
        
    Returns:
        Tuple of (serialized_context, documents)
    """
    vector_store = get_vector_store()
    all_docs = get_all_documents()
    
    # Use hybrid search if we have documents indexed, otherwise fall back to vector only
    if all_docs and len(all_docs) > 0:
        try:
            # Hybrid search: BM25 (keyword) + Vector (semantic)
            hybrid_retriever = HybridRetriever(vector_store, all_docs)
            
            # Determine query type and weights
            query_lower = query.lower()
            is_name_query = any(word.isupper() or len(word) > 8 for word in query.split())
            
            if is_name_query or len(query.split()) <= 3:
                # Name or short query - prioritize keyword matching
                bm25_weight, vector_weight = 0.6, 0.4
            else:
                # Longer semantic query - balance both
                bm25_weight, vector_weight = 0.4, 0.6
            
            results = hybrid_retriever.search(query, k=10, bm25_weight=bm25_weight, vector_weight=vector_weight)
            top_docs = [doc for doc, score in results]
            
            logger.info(f"Hybrid search - Query: {query}, BM25 weight: {bm25_weight}, Vector weight: {vector_weight}")
            for i, (doc, score) in enumerate(results[:5]):
                logger.info(f"Doc {i+1} source: {doc.metadata.get('source', 'unknown')}, hybrid_score: {score:.4f}")
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to vector search: {e}")
            # Fallback to vector search
            try:
                retrieved_with_scores = vector_store.similarity_search_with_score(query, k=10)
                top_docs = [doc for doc, score in retrieved_with_scores]
            except:
                top_docs = vector_store.similarity_search(query, k=10)
    else:
        # Fallback to vector-only search
        try:
            retrieved_with_scores = vector_store.similarity_search_with_score(query, k=10)
            top_docs = [doc for doc, score in retrieved_with_scores]
        except:
            top_docs = vector_store.similarity_search(query, k=10)
    
    # Filter out very small chunks
    MIN_CHARS = 100
    filtered_docs = [doc for doc in top_docs if len(doc.page_content.strip()) >= MIN_CHARS]
    
    # Take top k
    top_docs = filtered_docs[:k]
    
    # Debug logging
    logger.info(f"Retrieval query: {query}")
    logger.info(f"Retrieved {len(top_docs)} documents")
    for i, doc in enumerate(top_docs[:5]):
        logger.info(f"Doc {i+1} source: {doc.metadata.get('source', 'unknown')}")
        logger.info(f"Doc {i+1} preview: {doc.page_content[:100]}...")
    
    # If no results, return a message indicating no context found
    if not top_docs:
        return "No relevant documents found in the knowledge base for this query.", []
    
    # Format results with source information
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
        for doc in top_docs
    )
    
    return serialized, top_docs
