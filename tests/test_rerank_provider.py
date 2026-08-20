# tests/test_rerank_provider.py
"""Unit tests for the rerank provider (opt-in reranking of hybrid search candidates)."""
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from lahmajo.search.rerank_provider import (
    LLMRerankProvider,
    get_rerank_provider,
    RERANK_PROVIDER_ENV,
)


def _make_docs(n):
    return [Document(page_content=f"Content {i}", metadata={"source": f"doc{i}"}) for i in range(n)]


class TestLLMRerankProviderParsing(unittest.TestCase):
    """Test the defensive parsing of the LLM's rerank response."""

    def setUp(self):
        # Skip real get_llm() construction (would need a configured provider).
        with patch("lahmajo.llm.get_llm", return_value=MagicMock()):
            self.provider = LLMRerankProvider()

    def test_parse_clean_json(self):
        order = self.provider._parse_order("[2, 0, 1]", 3)
        self.assertEqual(order, [2, 0, 1])

    def test_parse_json_wrapped_in_prose(self):
        text = "Sure, here is the ranking:\n[1, 0, 2]\nHope that helps!"
        order = self.provider._parse_order(text, 3)
        self.assertEqual(order, [1, 0, 2])

    def test_parse_invalid_permutation_falls_back(self):
        # Not a permutation of range(3) - missing index 2, duplicate 0.
        order = self.provider._parse_order("[0, 0, 1]", 3)
        self.assertEqual(order, [0, 1, 2])

    def test_parse_garbage_falls_back(self):
        order = self.provider._parse_order("I cannot help with that.", 3)
        self.assertEqual(order, [0, 1, 2])


class TestLLMRerankProviderRerank(unittest.TestCase):
    """Test the end-to-end rerank() call with a mocked LLM."""

    def setUp(self):
        with patch("lahmajo.llm.get_llm", return_value=MagicMock()):
            self.provider = LLMRerankProvider()

    def test_rerank_reorders_by_llm_response(self):
        docs = _make_docs(3)
        mock_response = MagicMock()
        mock_response.content = "[2, 0, 1]"
        self.provider.llm.invoke.return_value = mock_response

        results = self.provider.rerank("some query", docs, top_k=3)

        self.assertEqual([doc.metadata["source"] for doc, _ in results], ["doc2", "doc0", "doc1"])

    def test_rerank_respects_top_k(self):
        docs = _make_docs(5)
        mock_response = MagicMock()
        mock_response.content = "[4, 3, 2, 1, 0]"
        self.provider.llm.invoke.return_value = mock_response

        results = self.provider.rerank("some query", docs, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual([doc.metadata["source"] for doc, _ in results], ["doc4", "doc3"])

    def test_rerank_falls_back_on_llm_exception(self):
        docs = _make_docs(3)
        self.provider.llm.invoke.side_effect = RuntimeError("connection refused")

        results = self.provider.rerank("some query", docs, top_k=3)

        # Falls back to original order rather than raising.
        self.assertEqual([doc.metadata["source"] for doc, _ in results], ["doc0", "doc1", "doc2"])

    def test_rerank_single_candidate_short_circuits(self):
        docs = _make_docs(1)
        results = self.provider.rerank("some query", docs, top_k=1)
        self.assertEqual(len(results), 1)
        self.provider.llm.invoke.assert_not_called()

    def test_rerank_empty_candidates(self):
        results = self.provider.rerank("some query", [], top_k=5)
        self.assertEqual(results, [])


class TestGetRerankProvider(unittest.TestCase):
    """Test the rerank provider factory."""

    def test_default_is_none(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(get_rerank_provider())

    def test_explicit_none(self):
        with patch.dict("os.environ", {RERANK_PROVIDER_ENV: "none"}):
            self.assertIsNone(get_rerank_provider())

    def test_llm_provider_returns_instance(self):
        with patch.dict("os.environ", {RERANK_PROVIDER_ENV: "llm"}):
            with patch("lahmajo.llm.get_llm", return_value=MagicMock()):
                provider = get_rerank_provider()
                self.assertIsInstance(provider, LLMRerankProvider)

    def test_unknown_provider_raises(self):
        with patch.dict("os.environ", {RERANK_PROVIDER_ENV: "bogus"}):
            with self.assertRaises(ValueError):
                get_rerank_provider()


if __name__ == '__main__':
    unittest.main()
