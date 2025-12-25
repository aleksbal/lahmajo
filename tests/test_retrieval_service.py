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
    
    @patch('lahmajo.services.retrieval_service.get_all_documents')
    @patch('lahmajo.services.retrieval_service.HybridRetriever')
    @patch('lahmajo.services.retrieval_service.get_vector_index')
    def test_retrieve_context_with_documents(self, mock_get_index, mock_hybrid_class, mock_get_all_docs):
        """Test retrieval with documents using hybrid search."""
        # Setup mocks
        mock_index = MagicMock()
        mock_get_index.return_value = mock_index
        
        # Create documents with content longer than MIN_CHARS (100) to pass filtering
        doc1_content = "Test content 1 " * 10  # Make it longer than 100 chars
        doc2_content = "Test content 2 " * 10  # Make it longer than 100 chars
        
        doc1 = Document(
            page_content=doc1_content,
            metadata={"source": "test1"}
        )
        doc2 = Document(
            page_content=doc2_content,
            metadata={"source": "test2"}
        )
        
        # Mock get_all_documents to return the test documents
        mock_get_all_docs.return_value = [doc1, doc2]
        
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
        self.assertIn(doc1_content, serialized)
        self.assertIn(doc2_content, serialized)


if __name__ == '__main__':
    unittest.main()
