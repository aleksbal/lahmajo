# app/hybrid_search.py
"""
Hybrid search implementation combining BM25 (keyword) and Vector (semantic) search.
This is the industry standard approach for production RAG systems.

Uses Reciprocal Rank Fusion (RRF) to combine results from both methods.
"""
import re
from typing import List, Tuple
from collections import defaultdict

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class HybridRetriever:
    """
    Combines BM25 (keyword matching) and Vector (semantic) search.
    
    Industry standard approach:
    - BM25: Excellent for exact matches, names, keywords
    - Vector: Good for semantic similarity
    - RRF: Combines both effectively
    """
    
    def __init__(self, vector_store, documents: List[Document]):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: The vector store for semantic search
            documents: All documents in the store (for BM25 indexing)
        """
        self.vector_store = vector_store
        self.documents = documents
        
        # Build BM25 index from document texts
        # Tokenize documents for BM25
        tokenized_docs = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Convert to lowercase and split on non-word characters
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def search(
        self, 
        query: str, 
        k: int = 10,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6
    ) -> List[Tuple[Document, float]]:
        """
        Perform hybrid search combining BM25 and vector search.
        
        Args:
            query: Search query
            k: Number of results to return
            bm25_weight: Weight for BM25 scores (0-1)
            vector_weight: Weight for vector scores (0-1)
            
        Returns:
            List of (document, combined_score) tuples, sorted by score (higher is better)
        """
        # BM25 search (keyword matching)
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Normalize BM25 scores to 0-1 range
        if max(bm25_scores) > 0:
            bm25_scores = [s / max(bm25_scores) for s in bm25_scores]
        else:
            bm25_scores = [0.0] * len(bm25_scores)
        
        # Vector search (semantic similarity)
        try:
            vector_results = self.vector_store.similarity_search_with_score(query, k=len(self.documents))
            # Create a dict mapping documents to vector scores
            # Note: vector scores are distances (lower = better), so we need to convert
            vector_scores_dict = {}
            for doc, score in vector_results:
                # Find matching document by content (fuzzy match for efficiency)
                for i, stored_doc in enumerate(self.documents):
                    # Match by content (exact or first 100 chars for efficiency)
                    if (stored_doc.page_content == doc.page_content or 
                        stored_doc.page_content[:100] == doc.page_content[:100]):
                        # Convert distance to similarity (1 / (1 + distance))
                        similarity = 1.0 / (1.0 + abs(score))
                        vector_scores_dict[i] = similarity
                        break
        except:
            # Fallback if similarity_search_with_score not available
            vector_results = self.vector_store.similarity_search(query, k=len(self.documents))
            vector_scores_dict = {}
            for doc in vector_results:
                for i, stored_doc in enumerate(self.documents):
                    if stored_doc.page_content == doc.page_content:
                        vector_scores_dict[i] = 0.5  # Default similarity
                        break
        
        # Combine scores using weighted average
        combined_scores = []
        for i, doc in enumerate(self.documents):
            bm25_score = bm25_scores[i]
            vector_score = vector_scores_dict.get(i, 0.0)
            
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
    
    This is the industry standard way to combine heterogeneous search results.
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
