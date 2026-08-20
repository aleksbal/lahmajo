# tests/test_retrieval_service.py
"""Unit tests for retrieval service."""
import unittest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from lahmajo.services.retrieval_service import retrieve_context
from lahmajo.indexes.state import add_documents, reset_vector_index


class TestRetrievalService(unittest.TestCase):
    """Test retrieval service."""
    
    def setUp(self):
        """Reset state before each test."""
        reset_vector_index()
    
    @patch('lahmajo.services.retrieval_service.get_vector_index')
    def test_retrieve_context_no_documents(self, mock_get_index):
        """Test retrieval when no documents are indexed."""
        mock_index = MagicMock()
        mock_index.similarity_search.return_value = []
        mock_get_index.return_value = mock_index
        
        serialized, docs = retrieve_context("test query")
        
        self.assertEqual(serialized, "No relevant documents found in the knowledge base for this query.")
        self.assertEqual(docs, [])
    
    @patch('lahmajo.services.retrieval_service.HybridRetriever')
    @patch('lahmajo.services.retrieval_service.get_vector_index')
    def test_retrieve_context_with_documents(self, mock_get_index, mock_hybrid_class):
        """Test retrieval with documents using hybrid search."""
        # Setup mocks
        mock_index = MagicMock()
        mock_get_index.return_value = mock_index
        
        # Content must be >= 100 chars: retrieve_context() filters out shorter chunks.
        doc1 = Document(page_content="Test content 1. " * 8, metadata={"source": "test1"})
        doc2 = Document(page_content="Test content 2. " * 8, metadata={"source": "test2"})
        add_documents([doc1, doc2])
        
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            (doc1, 0.9),
            (doc2, 0.8)
        ]
        mock_hybrid_class.return_value = mock_retriever
        
        # Test retrieval
        serialized, docs = retrieve_context("test query", k=2)
        
        # Verify hybrid search was called
        mock_hybrid_class.assert_called_once()
        mock_retriever.search.assert_called_once()
        
        # Verify results
        self.assertEqual(len(docs), 2)
        self.assertIn("Test content 1", serialized)
        self.assertIn("Test content 2", serialized)

    @patch('lahmajo.services.retrieval_service.HybridRetriever')
    @patch('lahmajo.services.retrieval_service.get_vector_index')
    def test_use_hybrid_false_skips_hybrid_retriever(self, mock_get_index, mock_hybrid_class):
        """use_hybrid=False should go straight to vector-only search, even with documents indexed."""
        mock_index = MagicMock()
        doc = Document(page_content="Vector-only content. " * 8, metadata={"source": "test1"})
        mock_index.similarity_search_with_score.return_value = [(doc, 0.9)]
        mock_get_index.return_value = mock_index

        add_documents([Document(page_content="Some indexed doc. " * 8, metadata={"source": "test1"})])

        serialized, docs = retrieve_context("test query", use_hybrid=False)

        mock_hybrid_class.assert_not_called()
        mock_index.similarity_search_with_score.assert_called_once()
        self.assertEqual(len(docs), 1)

    @patch('lahmajo.services.retrieval_service.LLMRerankProvider')
    @patch('lahmajo.services.retrieval_service.get_rerank_provider')
    @patch('lahmajo.services.retrieval_service.get_vector_index')
    def test_use_rerank_true_forces_rerank_even_when_disabled_globally(
        self, mock_get_index, mock_get_rerank_provider, mock_llm_rerank_class
    ):
        """use_rerank=True should force reranking on even when RERANK_PROVIDER=none globally."""
        mock_index = MagicMock()
        doc = Document(page_content="Some content. " * 8, metadata={"source": "test1"})
        mock_index.similarity_search_with_score.return_value = [(doc, 0.9)]
        mock_get_index.return_value = mock_index

        # Simulate RERANK_PROVIDER=none (the global default).
        mock_get_rerank_provider.return_value = None

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [(doc, 1.0)]
        mock_llm_rerank_class.return_value = mock_reranker

        serialized, docs = retrieve_context("test query", use_rerank=True)

        mock_reranker.rerank.assert_called_once()
        self.assertEqual(len(docs), 1)

    @patch('lahmajo.services.retrieval_service.get_rerank_provider')
    @patch('lahmajo.services.retrieval_service.get_vector_index')
    def test_use_rerank_false_disables_rerank_even_when_configured_globally(
        self, mock_get_index, mock_get_rerank_provider
    ):
        """use_rerank=False should skip reranking even if RERANK_PROVIDER is configured globally."""
        mock_index = MagicMock()
        doc = Document(page_content="Some content. " * 8, metadata={"source": "test1"})
        mock_index.similarity_search_with_score.return_value = [(doc, 0.9)]
        mock_get_index.return_value = mock_index

        # Simulate RERANK_PROVIDER=llm being configured globally.
        mock_reranker = MagicMock()
        mock_get_rerank_provider.return_value = mock_reranker

        serialized, docs = retrieve_context("test query", use_rerank=False)

        mock_reranker.rerank.assert_not_called()
        self.assertEqual(len(docs), 1)


if __name__ == '__main__':
    unittest.main()
