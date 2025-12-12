# tests/test_retrieval_service.py
"""Unit tests for retrieval service."""
import unittest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.services.retrieval_service import retrieve_context
from app.core.vector_store import add_documents, reset_vector_store


class TestRetrievalService(unittest.TestCase):
    """Test retrieval service."""
    
    def setUp(self):
        """Reset state before each test."""
        reset_vector_store()
    
    @patch('app.services.retrieval_service.get_vector_store')
    def test_retrieve_context_no_documents(self, mock_get_store):
        """Test retrieval when no documents are indexed."""
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []
        mock_get_store.return_value = mock_store
        
        serialized, docs = retrieve_context("test query")
        
        self.assertEqual(serialized, "No relevant documents found in the knowledge base for this query.")
        self.assertEqual(docs, [])
    
    @patch('app.services.retrieval_service.HybridRetriever')
    @patch('app.services.retrieval_service.get_vector_store')
    def test_retrieve_context_with_documents(self, mock_get_store, mock_hybrid_class):
        """Test retrieval with documents using hybrid search."""
        # Setup mocks
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        
        doc1 = Document(page_content="Test content 1", metadata={"source": "test1"})
        doc2 = Document(page_content="Test content 2", metadata={"source": "test2"})
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


if __name__ == '__main__':
    unittest.main()
