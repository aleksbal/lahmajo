# lahmajo/search/hybrid_search.py
"""
Hybrid search implementation combining BM25 (keyword) and Vector (semantic) search.
This is the industry standard approach for production RAG systems.

Uses Reciprocal Rank Fusion (RRF) to combine results from both methods.
"""
from typing import List, Tuple

from langchain_core.documents import Document
from lahmajo.search.bm25_provider import get_bm25_provider, BM25Provider
from lahmajo.storage.vector_index_provider import VectorIndexProvider


class HybridRetriever:
    """
    Combines BM25 (keyword matching) and Vector (semantic) search.
    
    Standard approach:
    - BM25: Excellent for exact matches, names, keywords
    - Vector: Good for semantic similarity
    - RRF: Combines both effectively
    
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
            documents: All documents in the index (needed to build BM25 index)
            bm25_index: Optional BM25 index provider instance (default: from factory)
        
        Note: Both vector_index and bm25_index are search indexes that store documents.
        - vector_index: VectorIndexProvider (pluggable - can be in_memory, elasticsearch, etc.)
        - bm25_index: BM25Provider (pluggable - can be rank_bm25, elasticsearch, etc.)
        All documents are needed here to build the BM25 index. During search,
        only top candidates from each method are retrieved and combined for efficiency.
        """
        self.vector_index = vector_index
        self.documents = documents
        
        # Get BM25 index (from factory or use provided one)
        if bm25_index is None:
            self.bm25_index = get_bm25_provider()
        else:
            self.bm25_index = bm25_index
        
        # Index documents with BM25 index
        self.bm25_index.index_documents(documents)
    
    def search(
        self, 
        query: str, 
        k: int = 10,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6
    ) -> List[Tuple[Document, float]]:
        """
        Perform hybrid search combining BM25 and vector search.
        
        Optimization: Only retrieves top candidates (2k) from each method, not all documents.
        This is much more efficient than processing all documents.
        
        Args:
            query: Search query
            k: Number of results to return
            bm25_weight: Weight for BM25 scores (0-1)
            vector_weight: Weight for vector scores (0-1)
            
        Returns:
            List of (document, combined_score) tuples, sorted by score (higher is better)
        """
        # Get top candidates from each method (retrieve more than k to allow for better combination)
        # Using 2k to ensure we have enough candidates to combine, but not all documents
        candidate_count = max(k * 2, 20)  # At least 2k, minimum 20
        
        # BM25 search (keyword matching) - only get top candidates
        bm25_results = self.bm25_index.search(query, top_k=candidate_count)
        
        # Normalize BM25 scores to 0-1 range
        bm25_scores_dict = {}
        if bm25_results:
            max_score = max(score for _, score in bm25_results) if bm25_results else 0
            if max_score > 0:
                bm25_scores_dict = {
                    doc.page_content: score / max_score
                    for doc, score in bm25_results
                }
            else:
                bm25_scores_dict = {doc.page_content: 0.0 for doc, _ in bm25_results}
        
        # Vector search (semantic similarity) - only get top candidates
        vector_scores_dict = {}
        vector_docs = []
        try:
            vector_results = self.vector_index.similarity_search_with_score(query, k=candidate_count)
            # Create a mapping from document content to normalized similarity score
            # Note: vector scores are distances (lower = better), so we convert to similarity
            if vector_results:
                # Find max distance for normalization
                max_distance = max(abs(score) for _, score in vector_results) if vector_results else 1.0
                if max_distance > 0:
                    for doc, score in vector_results:
                        # Convert distance to similarity (normalized 0-1)
                        similarity = 1.0 / (1.0 + abs(score) / max_distance)
                        vector_scores_dict[doc.page_content] = similarity
                        vector_docs.append(doc)
        except:
            # Fallback if similarity_search_with_score not available
            vector_results = self.vector_index.similarity_search(query, k=candidate_count)
            for doc in vector_results:
                vector_scores_dict[doc.page_content] = 0.5  # Default similarity
                vector_docs.append(doc)
        
        # Get unique documents from both result sets
        candidate_docs = {}
        for doc, _ in bm25_results:
            candidate_docs[doc.page_content] = doc
        for doc in vector_docs:
            candidate_docs[doc.page_content] = doc
        
        # Combine scores using weighted average (only for candidate documents)
        combined_scores = []
        for content, doc in candidate_docs.items():
            # Get BM25 score (normalized 0-1, default 0 if not in BM25 results)
            bm25_score = bm25_scores_dict.get(content, 0.0)
            # Get vector score (normalized 0-1, default 0 if not in vector results)
            vector_score = vector_scores_dict.get(content, 0.0)
            
            # Weighted combination
            combined_score = (bm25_weight * bm25_score) + (vector_weight * vector_score)
            combined_scores.append((doc, combined_score))
        
        # Sort by combined score (higher is better)
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        return combined_scores[:k]


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
