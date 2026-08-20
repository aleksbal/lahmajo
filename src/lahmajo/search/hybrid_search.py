# lahmajo/search/hybrid_search.py
"""
Hybrid search implementation combining BM25 (keyword) and Vector (semantic) search.
This is the industry standard approach for production RAG systems.

Uses Reciprocal Rank Fusion (RRF) to combine results from both methods.
When both providers are Elasticsearch-based, uses ES native hybrid search for better performance.
"""
from typing import List, Tuple, Optional

from langchain_core.documents import Document
from lahmajo.indexes.bm25_provider import get_bm25_provider, BM25Provider
from lahmajo.indexes.vector_provider import VectorIndexProvider


class HybridRetriever:
    """
    Combines BM25 (keyword matching) and Vector (semantic) search.
    
    Standard approach:
    - BM25: Excellent for exact matches, names, keywords
    - Vector: Good for semantic similarity
    - RRF: Combines both effectively
    
    When both providers are Elasticsearch-based, automatically uses ES native hybrid search
    for better performance (single query instead of two separate queries).
    
    Terminology Note:
    - Both parameters are called "index" for consistency (both are search indexes)
    - vector_index: VectorIndexProvider (pluggable - in_memory, elasticsearch, etc.)
    - bm25_index: BM25Provider (pluggable - rank_bm25, elasticsearch, etc.)
    Both are fully pluggable via environment variables for maximum flexibility.
    """
    
    def __init__(self, vector_index: VectorIndexProvider, documents: List[Document], bm25_index: BM25Provider = None):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_index: The vector index provider for semantic search (pluggable implementation)
            documents: All documents in the index (needed to build BM25 index for non-ES providers)
            bm25_index: Optional BM25 index provider instance (default: from factory)
        
        Note: Both vector_index and bm25_index are search indexes that store documents.
        - vector_index: VectorIndexProvider (pluggable - can be in_memory, elasticsearch, etc.)
        - bm25_index: BM25Provider (pluggable - can be rank_bm25, elasticsearch, etc.)
        For ES native hybrid, documents are not needed (ES is source of truth).
        For non-ES providers, documents are needed to build the BM25 index.
        """
        self.vector_index = vector_index
        self.documents = documents
        
        # Get BM25 index (from factory or use provided one)
        if bm25_index is None:
            self.bm25_index = get_bm25_provider()
        else:
            self.bm25_index = bm25_index
        
        # Check if we can use ES native hybrid search
        self.use_es_native_hybrid = self._can_use_es_native_hybrid()
        
        # Index documents with BM25 index (only needed for non-ES providers)
        if not self.use_es_native_hybrid:
            self.bm25_index.index_documents(documents)
    
    def _can_use_es_native_hybrid(self) -> bool:
        """
        Check if both providers support ES native hybrid search.
        
        Returns:
            True if ES native hybrid can be used, False otherwise
        """
        try:
            from lahmajo.indexes.elasticsearch_hybrid_provider import ElasticsearchHybridProvider
            
            # Check if vector_index is already a hybrid provider
            if isinstance(self.vector_index, ElasticsearchHybridProvider):
                return True
            
            # Check if vector index supports native hybrid
            vector_supports = hasattr(self.vector_index, 'supports_native_hybrid') and \
                             self.vector_index.supports_native_hybrid()
            
            # Check if BM25 index is ES-based by class name
            bm25_class_name = type(self.bm25_index).__name__
            bm25_is_es = 'Elasticsearch' in bm25_class_name
            
            # If both are ES and use the same index, we can use native hybrid
            if vector_supports and bm25_is_es:
                # Check if both use the same ES index
                if hasattr(self.vector_index, 'get_index_name') and \
                   hasattr(self.bm25_index, 'get_index_name'):
                    return self.vector_index.get_index_name() == self.bm25_index.get_index_name()
                # If we can't check index names, assume they're compatible if both are ES
                return True
        except ImportError:
            pass
        
        return False
    
    def _get_es_hybrid_provider(self):
        """Get or create ES hybrid provider for native hybrid search."""
        try:
            from lahmajo.indexes.elasticsearch_hybrid_provider import ElasticsearchHybridProvider
            
            # If vector_index is already a hybrid provider, use it
            if isinstance(self.vector_index, ElasticsearchHybridProvider):
                return self.vector_index
            
            # Otherwise, create a new one
            return ElasticsearchHybridProvider()
        except ImportError:
            return None
    
    def search(
        self,
        query: str,
        k: int = 10,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6
    ) -> List[Tuple[Document, float]]:
        """
        Perform hybrid search combining BM25 and vector search.

        If ES native hybrid is available, uses single ES query for better performance,
        combining scores via the given weights at the query level.
        Otherwise, combines results from separate BM25 and vector queries using
        Reciprocal Rank Fusion (RRF) - see reciprocal_rank_fusion() below.

        Args:
            query: Search query
            k: Number of results to return
            bm25_weight: Weight for BM25 scores (0-1). Only used for ES native hybrid;
                the Python-side RRF fallback combines by rank, not score magnitude,
                so it doesn't need (or use) these weights.
            vector_weight: Weight for vector scores (0-1). See bm25_weight above.

        Returns:
            List of (document, combined_score) tuples, sorted by score (higher is better)
        """
        # Use ES native hybrid if available
        if self.use_es_native_hybrid:
            es_hybrid = self._get_es_hybrid_provider()
            if es_hybrid and hasattr(es_hybrid, 'hybrid_search'):
                try:
                    return es_hybrid.hybrid_search(query, k=k, bm25_weight=bm25_weight, vector_weight=vector_weight)
                except Exception as e:
                    # If native hybrid fails, fall back to Python-side combination
                    import logging
                    logging.warning(f"ES native hybrid search failed, falling back to Python-side combination: {e}")

        # Fallback to Python-side combination via Reciprocal Rank Fusion (RRF).
        # RRF combines by each list's rank order rather than raw score magnitude, which
        # avoids having to know whether a given provider's score is a similarity (higher
        # is better, e.g. LangChain's InMemoryVectorStore) or a distance (lower is better) -
        # a distinction a naive weighted-average of normalized scores gets wrong in practice.
        #
        # Get top candidates from each method (retrieve more than k to allow for better fusion)
        # Using 2k to ensure we have enough candidates to combine, but not all documents
        candidate_count = max(k * 2, 20)  # At least 2k, minimum 20

        # BM25 search (keyword matching) - already sorted best-first
        bm25_results = self.bm25_index.search(query, top_k=candidate_count)

        # Vector search (semantic similarity) - already sorted best-first
        try:
            vector_results = self.vector_index.similarity_search_with_score(query, k=candidate_count)
        except Exception:
            # Fallback if similarity_search_with_score not available; RRF only needs rank
            # order, so a placeholder score is fine here.
            vector_results = [(doc, 0.0) for doc in self.vector_index.similarity_search(query, k=candidate_count)]

        fused_results = reciprocal_rank_fusion(bm25_results, vector_results)

        # Return top k
        return fused_results[:k]


def reciprocal_rank_fusion(
    bm25_results: List[Tuple[Document, float]],
    vector_results: List[Tuple[Document, float]],
    k: int = 60
) -> List[Tuple[Document, float]]:
    """
    Reciprocal Rank Fusion (RRF) to combine results from multiple retrieval methods.
    
    RRF formula: score = sum(1 / (k + rank)) for each method
    """
    # Create rank dictionaries
    bm25_ranks = {doc.page_content: rank + 1 for rank, (doc, _) in enumerate(bm25_results)}
    vector_ranks = {doc.page_content: rank + 1 for rank, (doc, _) in enumerate(vector_results)}
    
    # Get all unique documents
    all_docs = {}
    for doc, _ in bm25_results:
        all_docs[doc.page_content] = doc
    for doc, _ in vector_results:
        all_docs[doc.page_content] = doc
    
    # Calculate RRF scores
    rrf_scores = []
    for content, doc in all_docs.items():
        bm25_rank = bm25_ranks.get(content, k + 1)
        vector_rank = vector_ranks.get(content, k + 1)
        
        rrf_score = (1.0 / (k + bm25_rank)) + (1.0 / (k + vector_rank))
        rrf_scores.append((doc, rrf_score))
    
    # Sort by RRF score (higher is better)
    rrf_scores.sort(key=lambda x: x[1], reverse=True)
    
    return rrf_scores
