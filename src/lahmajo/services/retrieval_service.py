# lahmajo/services/retrieval_service.py
"""Retrieval service - handles document retrieval using hybrid search."""
import logging
from difflib import SequenceMatcher
from typing import List, Optional, Tuple
from langchain_core.documents import Document

from lahmajo.indexes.state import get_vector_index, get_all_documents
from lahmajo.search.hybrid_search import HybridRetriever
from lahmajo.search.rerank_provider import get_rerank_provider, LLMRerankProvider

logger = logging.getLogger(__name__)

# Similarity ratio (0-1) above which two chunks from the same source are treated
# as near-duplicates. Adaptive chunking overlaps adjacent chunks by 50-100 chars
# (see ingestion/processing.py), so hybrid/vector search can surface two chunks
# that are mostly the same text - this threshold is deliberately high so it only
# catches that kind of overlap, not merely related content.
DEDUPE_SIMILARITY_THRESHOLD = 0.8


def _is_near_duplicate(a: str, b: str, threshold: float = DEDUPE_SIMILARITY_THRESHOLD) -> bool:
    """Near-duplicate check between two chunks' text.

    Uses SequenceMatcher.ratio() (real sequence alignment), not quick_ratio() -
    quick_ratio() is only an upper bound based on shared character counts, not
    matching sequences, so two unrelated chunks that happen to share vocabulary
    (e.g. two differently-ordered sentences built from similar words) can score
    above the threshold and get wrongly dropped. autojunk=False is required too:
    with autojunk on (the default), SequenceMatcher treats any character that
    recurs often enough in a 200+-char sequence as "popular" and excludes it from
    matching - which silently zeroes out the ratio for genuinely overlapping
    chunks once they're long/repetitive enough to trip that heuristic.
    """
    return SequenceMatcher(None, a, b, autojunk=False).ratio() >= threshold


def _dedupe_chunks(docs: List[Document], threshold: float = DEDUPE_SIMILARITY_THRESHOLD) -> List[Document]:
    """Drop chunks that are near-duplicates of an already-kept chunk from the same
    source, so overlapping adaptive-chunking windows don't waste context budget
    (or reranker candidate slots) on redundant text. Keeps the first occurrence,
    i.e. respects the incoming rank/relevance order.
    """
    kept: List[Document] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        is_duplicate = any(
            kept_doc.metadata.get("source", "unknown") == source
            and _is_near_duplicate(doc.page_content, kept_doc.page_content, threshold)
            for kept_doc in kept
        )
        if not is_duplicate:
            kept.append(doc)
    return kept


def retrieve_context(
    query: str,
    k: int = 8,
    use_hybrid: bool = True,
    use_rerank: Optional[bool] = None,
) -> Tuple[str, List[Document]]:
    """
    Retrieve relevant documents for a query using hybrid search, deduplicated and
    optionally reranked.

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
            except Exception:
                top_docs = vector_index.similarity_search(query, k=candidate_k)
    else:
        # Fallback to vector-only search
        try:
            retrieved_with_scores = vector_index.similarity_search_with_score(query, k=candidate_k)
            top_docs = [doc for doc, score in retrieved_with_scores]
        except Exception:
            top_docs = vector_index.similarity_search(query, k=candidate_k)

    # Filter out very small chunks
    MIN_CHARS = 100
    filtered_docs = [doc for doc in top_docs if len(doc.page_content.strip()) >= MIN_CHARS]

    # Collapse near-duplicate chunks before reranking/truncation, so overlapping
    # adaptive-chunking windows don't eat into the reranker's candidate pool or
    # the final top-k.
    deduped_count = len(filtered_docs)
    filtered_docs = _dedupe_chunks(filtered_docs)
    if len(filtered_docs) < deduped_count:
        logger.info(f"Deduped {deduped_count - len(filtered_docs)} near-duplicate chunk(s)")

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
