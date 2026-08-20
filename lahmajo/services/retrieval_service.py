# lahmajo/services/retrieval_service.py
"""Retrieval service - handles document retrieval using hybrid search."""
import logging
from typing import List, Optional, Tuple
from langchain_core.documents import Document

from lahmajo.indexes.state import get_vector_index, get_all_documents
from lahmajo.search.hybrid_search import HybridRetriever
from lahmajo.search.rerank_provider import get_rerank_provider, LLMRerankProvider

logger = logging.getLogger(__name__)


def retrieve_context(
    query: str,
    k: int = 8,
    use_hybrid: bool = True,
    use_rerank: Optional[bool] = None,
) -> Tuple[str, List[Document]]:
    """
    Retrieve relevant documents for a query using hybrid search, optionally reranked.

    Args:
        query: Search query
        k: Number of documents to return
        use_hybrid: If False, skip BM25 entirely and use vector-only search - mainly
            useful for comparing retrieval techniques (e.g. from the GUI/API).
        use_rerank: Per-call override for reranking. None (default) uses the
            RERANK_PROVIDER env var. True forces reranking on for this call even if
            RERANK_PROVIDER=none globally (falling back to the LLM reranker). False
            forces it off even if a rerank provider is configured globally.

    Returns:
        Tuple of (serialized_context, documents)
    """
    vector_index = get_vector_index()
    all_docs = get_all_documents()

    # Reranking is opt-in by default (RERANK_PROVIDER env var, "none" unless a per-call
    # override says otherwise). When enabled, fetch a larger candidate pool so the
    # reranker has more to work with than the final k.
    if use_rerank is None:
        rerank_provider = get_rerank_provider()
    elif use_rerank:
        rerank_provider = get_rerank_provider() or LLMRerankProvider()
    else:
        rerank_provider = None
    candidate_k = 20 if rerank_provider else 10

    # Use hybrid search if requested and we have documents indexed, otherwise fall back
    # to vector only.
    if use_hybrid and all_docs and len(all_docs) > 0:
        try:
            # Hybrid search: BM25 (keyword) + Vector (semantic)
            # Both vector_index and BM25 are indexes - vector_index is the semantic index,
            # and BM25 index is created from the provider factory
            hybrid_retriever = HybridRetriever(vector_index, all_docs)

            # Combined via Reciprocal Rank Fusion (RRF) for the Python-side path (see
            # HybridRetriever.search()); bm25_weight/vector_weight are only consulted
            # when ES native hybrid search is active.
            results = hybrid_retriever.search(query, k=candidate_k)
            top_docs = [doc for doc, score in results]

            logger.info(f"Hybrid search - Query: {query}")
            for i, (doc, score) in enumerate(results[:5]):
                logger.info(f"Doc {i+1} source: {doc.metadata.get('source', 'unknown')}, hybrid_score: {score:.4f}")
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to vector search: {e}")
            # Fallback to vector search
            try:
                retrieved_with_scores = vector_index.similarity_search_with_score(query, k=candidate_k)
                top_docs = [doc for doc, score in retrieved_with_scores]
            except:
                top_docs = vector_index.similarity_search(query, k=candidate_k)
    else:
        # Fallback to vector-only search
        try:
            retrieved_with_scores = vector_index.similarity_search_with_score(query, k=candidate_k)
            top_docs = [doc for doc, score in retrieved_with_scores]
        except:
            top_docs = vector_index.similarity_search(query, k=candidate_k)

    # Filter out very small chunks
    MIN_CHARS = 100
    filtered_docs = [doc for doc in top_docs if len(doc.page_content.strip()) >= MIN_CHARS]

    # Rerank the filtered candidates if a rerank provider is configured, otherwise keep
    # them in hybrid-search order.
    if rerank_provider:
        try:
            reranked = rerank_provider.rerank(query, filtered_docs, top_k=k)
            top_docs = [doc for doc, score in reranked]
            logger.info(f"Reranked {len(filtered_docs)} candidates down to {len(top_docs)}")
        except Exception as e:
            logger.warning(f"Reranking failed, falling back to hybrid-search order: {e}")
            top_docs = filtered_docs[:k]
    else:
        top_docs = filtered_docs[:k]

    # Debug logging - clarify chunks vs documents
    logger.info(f"Retrieval query: {query}")
    logger.info(f"Retrieved {len(top_docs)} chunks (from {len(set(doc.metadata.get('source', 'unknown') for doc in top_docs))} unique document(s))")
    for i, doc in enumerate(top_docs):
        source = doc.metadata.get('source', 'unknown')
        chunk_length = len(doc.page_content)
        logger.info(f"Chunk {i+1}/{len(top_docs)}: source='{source}', length={chunk_length} chars, preview: {doc.page_content[:100]}...")
    
    # If no results, return a message indicating no context found
    if not top_docs:
        return "No relevant documents found in the knowledge base for this query.", []
    
    # Format results with source information
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
        for doc in top_docs
    )
    
    return serialized, top_docs
