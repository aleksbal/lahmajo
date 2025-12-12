# tests/test_vector_store.py
"""Unit tests for vector store management."""
import unittest
from unittest.mock import patch, MagicMock

from app.core.vector_store import (
    get_vector_store,
    get_all_documents,
    add_documents,
    reset_vector_store
)
from langchain_core.documents import Document


class TestVectorStore(unittest.TestCase):
    """Test vector store management."""
    
    def setUp(self):
        """Reset state before each test."""
        reset_vector_store()
    
    @patch('app.core.vector_store.build_vector_store')
    def test_get_vector_store_lazy_init(self, mock_build):
        """Test that vector store is initialized lazily."""
        mock_store = MagicMock()
        mock_build.return_value = mock_store
        
        # First call should initialize
        store = get_vector_store()
        self.assertEqual(store, mock_store)
        mock_build.assert_called_once()
        
        # Second call should return same instance
        store2 = get_vector_store()
        self.assertEqual(store, store2)
        # Should not call build again
        self.assertEqual(mock_build.call_count, 1)
    
    def test_get_all_documents_empty(self):
        """Test getting documents when empty."""
        docs = get_all_documents()
        self.assertEqual(docs, [])
    
    def test_add_documents(self):
        """Test adding documents to index."""
        doc1 = Document(page_content="Test 1", metadata={"source": "test1"})
        doc2 = Document(page_content="Test 2", metadata={"source": "test2"})
        
        add_documents([doc1, doc2])
        
        docs = get_all_documents()
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].page_content, "Test 1")
        self.assertEqual(docs[1].page_content, "Test 2")
    
    def test_reset_vector_store(self):
        """Test resetting vector store."""
        # Add some documents
        doc = Document(page_content="Test", metadata={})
        add_documents([doc])
        
        # Reset
        reset_vector_store()
        
        # Should be empty
        docs = get_all_documents()
        self.assertEqual(docs, [])


if __name__ == '__main__':
    unittest.main()
