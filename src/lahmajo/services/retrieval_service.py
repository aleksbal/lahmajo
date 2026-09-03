# lahmajo/services/retrieval_service.py
"""Retrieval service - handles document retrieval using hybrid search."""
import logging
import re
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple
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

# Characters of chunk text carried in a SourceRef preview, matching what
# GET /debug/search already returns per result.
PREVIEW_CHARS = 300

NO_CONTEXT_MESSAGE = "No relevant documents found in the knowledge base for this query."

# Citation markers as they appear in the serialized context. Ingested text can
# contain the same shape - this project's own README does, in its /ask example - and
# a chunk carrying a literal "[source 2]" would be indistinguishable from the header
# this module generates, letting the model cite a marker that resolves to an
# unrelated chunk or to nothing at all. Matched loosely (case, inner spacing) so a
# near-miss in the source text is neutralized too.
CITATION_MARKER_RE = re.compile(r"\[\s*source\s+\d+\s*\]", re.IGNORECASE)


@dataclass(frozen=True)
class SourceRef:
    """A citable reference to one retrieved chunk.

    `index` is the number the excerpt carries in the serialized context handed to
    the LLM, so a `[source N]` marker in an answer resolves to the SourceRef with
    `index == N`. Numbering is 1-based, and continues across retrieval calls within
    one agent run via the `start_index` argument below - the agent may retrieve more
    than once, and restarting at 1 each time would make `[source 1]` ambiguous.

    `score` is whatever the final ranking stage produced - a fused hybrid score, a
    reranker score, or a raw vector distance. It is provider-specific and only
    comparable within a single response (see RerankProvider.rerank()); it is None
    when the ranking path could not supply one (e.g. the bare similarity_search
    fallback).

    Note there is deliberately no chunk position: chunk index within a document is
    not recorded at ingestion time (see ingestion/processing.py, which copies the
    source document's metadata onto each chunk without a position), so it cannot be
    derived here without changing ingestion.
    """

    index: int
    source: str
    score: Optional[float]
    length: int
    preview: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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


def _source_of(doc: Document) -> str:
    return doc.metadata.get("source", "unknown")


def _neutralize_citation_markers(text: str) -> str:
    """Rewrite citation-shaped text in a chunk to round brackets.

    Only the headers this module writes may look like citations, or the model can
    cite a marker that came out of a document rather than one of its excerpts. The
    text stays readable - "[source 2]" becomes "(source 2)" - because it is still
    content the answer may need to discuss.

    Applied to every untrusted string that reaches the model-facing header - the
    chunk text and the source label, since an uploaded file can be named
    "notes [source 2].txt". SourceRef keeps the originals in `preview` and
    `source`: those are evidence shown to a human beside their own header, where
    there is nothing to confuse, and `source` has to stay a usable filename.
    """
    return CITATION_MARKER_RE.sub(lambda match: f"({match.group(0)[1:-1].strip()})", text)


def _build_source_refs(
    scored_docs: List[Tuple[Document, Optional[float]]], start_index: int = 1
) -> List[SourceRef]:
    """Build citable references for the final ranked chunks.

    Numbering starts at `start_index` (1 by default) so a caller making several
    retrieval calls can keep every reference distinct; it must match the numbering
    _serialize_context() used for the same chunks.
    """
    refs = []
    for i, (doc, score) in enumerate(scored_docs, start=start_index):
        content = doc.page_content
        preview = content[:PREVIEW_CHARS] + "..." if len(content) > PREVIEW_CHARS else content
        refs.append(
            SourceRef(
                index=i,
                source=_source_of(doc),
                score=float(score) if score is not None else None,
                length=len(content),
                preview=preview,
            )
        )
    return refs


def _serialize_context(
    scored_docs: List[Tuple[Document, Optional[float]]], start_index: int = 1
) -> str:
    """Format retrieved chunks as the context string handed to the LLM.

    Each excerpt is labelled `[source N]` with its originating file, so the
    citation instruction in the agent's system prompt ("cite which excerpts you
    used, e.g. [source 1]") refers to something the model can actually resolve.
    Before this, chunks were serialized with a bare `Source: <filename>` header and
    no numbering, leaving `[source N]` an unresolvable placeholder.

    Labels start at `start_index` so consecutive retrieval calls in one agent run
    produce distinct markers rather than each restarting at "[source 1]".

    Citation-shaped text in the chunk and in its source label is neutralized
    first - see _neutralize_citation_markers().
    """
    blocks = []
    for i, (doc, score) in enumerate(scored_docs, start=start_index):
        header = f"[source {i}] {_neutralize_citation_markers(_source_of(doc))}"
        if score is not None:
            header += f" (score: {score:.4f})"
        blocks.append(f"{header}\n{_neutralize_citation_markers(doc.page_content)}")
    return "\n\n".join(blocks)


def retrieve_context_with_sources(
    query: str,
    k: int = 8,
    use_hybrid: bool = True,
    use_rerank: Optional[bool] = None,
    start_index: int = 1,
) -> Tuple[str, List[Document], List[SourceRef]]:
    """
    Retrieve relevant documents for a query, and describe what was retrieved.

    Same retrieval behaviour as retrieve_context(), but additionally returns the
    structured SourceRefs matching the `[source N]` labels in the serialized
    context, so callers can report real references instead of placeholders.

    Args:
        query: Search query
        k: Number of documents to return
        use_hybrid: If False, skip BM25 entirely and use vector-only search - mainly
            useful for comparing retrieval techniques (e.g. from the GUI/API).
        use_rerank: Per-call override for reranking. None (default) uses the
            RERANK_PROVIDER env var. True forces reranking on for this call even if
            RERANK_PROVIDER=none globally (falling back to the LLM reranker). False
            forces it off even if a rerank provider is configured globally.
        start_index: Number the first returned excerpt gets in both the `[source N]`
            labels and the SourceRefs. Callers that retrieve several times for one
            answer pass the running total so markers stay unique across calls.

    Returns:
        Tuple of (serialized_context, documents, source_refs)
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

    # Candidates are carried as (document, score) pairs all the way through, so the
    # score that the final ranking stage produced survives into the SourceRefs
    # instead of being discarded at the first step. score is None where the path
    # cannot supply one (the bare similarity_search fallback).
    scored_docs: List[Tuple[Document, Optional[float]]] = []

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
            scored_docs = [(doc, score) for doc, score in results]

            logger.info(f"Hybrid search - Query: {query}")
            for i, (doc, score) in enumerate(results[:5]):
                logger.info(f"Doc {i+1} source: {_source_of(doc)}, hybrid_score: {score:.4f}")
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to vector search: {e}")
            # Fallback to vector search
            try:
                retrieved_with_scores = vector_index.similarity_search_with_score(query, k=candidate_k)
                scored_docs = [(doc, score) for doc, score in retrieved_with_scores]
            except Exception:
                scored_docs = [(doc, None) for doc in vector_index.similarity_search(query, k=candidate_k)]
    else:
        # Fallback to vector-only search
        try:
            retrieved_with_scores = vector_index.similarity_search_with_score(query, k=candidate_k)
            scored_docs = [(doc, score) for doc, score in retrieved_with_scores]
        except Exception:
            scored_docs = [(doc, None) for doc in vector_index.similarity_search(query, k=candidate_k)]

    # Filter out very small chunks
    MIN_CHARS = 100
    scored_docs = [(doc, score) for doc, score in scored_docs if len(doc.page_content.strip()) >= MIN_CHARS]

    # Collapse near-duplicate chunks before reranking/truncation, so overlapping
    # adaptive-chunking windows don't eat into the reranker's candidate pool or the
    # final top-k. _dedupe_chunks() works on documents, so the surviving pairs are
    # recovered by identity - the same approach GET /debug/search uses.
    pre_dedupe_count = len(scored_docs)
    kept_ids = {id(doc) for doc in _dedupe_chunks([doc for doc, _ in scored_docs])}
    scored_docs = [(doc, score) for doc, score in scored_docs if id(doc) in kept_ids]
    if len(scored_docs) < pre_dedupe_count:
        logger.info(f"Deduped {pre_dedupe_count - len(scored_docs)} near-duplicate chunk(s)")

    # Rerank the filtered candidates if a rerank provider is configured, otherwise keep
    # them in hybrid-search order. Reranking replaces the scores with its own, which is
    # correct: the reported score should be the one that decided the final ordering.
    if rerank_provider:
        try:
            candidates = [doc for doc, _ in scored_docs]
            reranked = rerank_provider.rerank(query, candidates, top_k=k)
            logger.info(f"Reranked {len(candidates)} candidates down to {len(reranked)}")
            scored_docs = [(doc, score) for doc, score in reranked]
        except Exception as e:
            logger.warning(f"Reranking failed, falling back to hybrid-search order: {e}")
            scored_docs = scored_docs[:k]
    else:
        scored_docs = scored_docs[:k]

    top_docs = [doc for doc, _ in scored_docs]

    # Debug logging - clarify chunks vs documents
    logger.info(f"Retrieval query: {query}")
    logger.info(f"Retrieved {len(top_docs)} chunks (from {len(set(_source_of(doc) for doc in top_docs))} unique document(s))")
    for i, doc in enumerate(top_docs):
        chunk_length = len(doc.page_content)
        logger.info(f"Chunk {i+1}/{len(top_docs)}: source='{_source_of(doc)}', length={chunk_length} chars, preview: {doc.page_content[:100]}...")

    # If no results, return a message indicating no context found
    if not top_docs:
        return NO_CONTEXT_MESSAGE, [], []

    return (
        _serialize_context(scored_docs, start_index),
        top_docs,
        _build_source_refs(scored_docs, start_index),
    )


def retrieve_context(
    query: str,
    k: int = 8,
    use_hybrid: bool = True,
    use_rerank: Optional[bool] = None,
) -> Tuple[str, List[Document]]:
    """
    Retrieve relevant documents for a query using hybrid search, deduplicated and
    optionally reranked.

    Thin wrapper over retrieve_context_with_sources() for callers that only need
    the serialized context and the documents.

    Returns:
        Tuple of (serialized_context, documents)
    """
    serialized, docs, _ = retrieve_context_with_sources(
        query, k=k, use_hybrid=use_hybrid, use_rerank=use_rerank
    )
    return serialized, docs
