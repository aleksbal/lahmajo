# tests/test_hybrid_search.py
"""Unit tests for hybrid search (BM25 + vector combination via RRF)."""
import unittest
from typing import List, Tuple
from langchain_core.documents import Document

from lahmajo.indexes.bm25_provider import BM25Provider
from lahmajo.indexes.vector_provider import VectorIndexProvider
from lahmajo.search.hybrid_search import HybridRetriever, reciprocal_rank_fusion


class FakeBM25Provider(BM25Provider):
    """Stub BM25 provider that returns a fixed, pre-ranked result list."""

    def __init__(self, results: List[Tuple[Document, float]]):
        self._results = results

    def index_documents(self, documents: List[Document]) -> None:
        pass

    def search(self, query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        return self._results[:top_k] if top_k is not None else self._results


class FakeVectorIndexProvider(VectorIndexProvider):
    """
    Stub vector provider mimicking LangChain's InMemoryVectorStore contract:
    similarity_search_with_score() returns (doc, score) with a HIGHER score
    meaning a BETTER match, already sorted best-first.
    """

    def __init__(self, results: List[Tuple[Document, float]]):
        self._results = results

    def add_documents(self, documents: List[Document]) -> None:
        pass

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        return [doc for doc, _ in self._results[:k]]

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        return self._results[:k]


class TestHybridSearchRRF(unittest.TestCase):
    """Test that the Python-side combination (RRF) ranks by rank order, not score magnitude/direction."""

    def test_best_vector_match_is_not_inverted(self):
        """
        Regression test: a weighted-average combination that assumes vector scores are
        distances (lower = better) will rank the single best vector match LAST when the
        provider actually returns similarities (higher = better, as InMemoryVectorStore does).
        RRF must not have this failure mode, since it only looks at rank order.
        """
        best_doc = Document(page_content="The best semantic match for this query. " * 5, metadata={"source": "best"})
        worst_doc = Document(page_content="A barely related document about something else. " * 5, metadata={"source": "worst"})

        # Vector provider: best_doc ranked first with the highest (best) similarity score.
        vector_index = FakeVectorIndexProvider([(best_doc, 0.95), (worst_doc, 0.10)])

        # BM25 finds no keyword match for either document (neither appears in results),
        # so it shouldn't be able to favor one over the other - only the vector ranking should.
        bm25_index = FakeBM25Provider([])

        retriever = HybridRetriever(vector_index, [best_doc, worst_doc], bm25_index=bm25_index)
        results = retriever.search("test query", k=2)

        ranked_sources = [doc.metadata["source"] for doc, _ in results]
        self.assertEqual(ranked_sources[0], "best", f"Expected 'best' doc ranked first, got order: {ranked_sources}")

    def test_reciprocal_rank_fusion_combines_both_lists(self):
        """A document ranked #1 in both BM25 and vector results should come out on top."""
        doc_a = Document(page_content="Document A content here.", metadata={"source": "a"})
        doc_b = Document(page_content="Document B content here.", metadata={"source": "b"})
        doc_c = Document(page_content="Document C content here.", metadata={"source": "c"})

        bm25_results = [(doc_a, 5.0), (doc_b, 3.0), (doc_c, 1.0)]
        vector_results = [(doc_a, 0.9), (doc_c, 0.5), (doc_b, 0.2)]

        fused = reciprocal_rank_fusion(bm25_results, vector_results)

        self.assertEqual(fused[0][0].metadata["source"], "a")


if __name__ == '__main__':
    unittest.main()
