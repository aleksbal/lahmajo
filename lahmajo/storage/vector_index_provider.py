# lahmajo/storage/vector_index_provider.py
"""Vector index provider factory - supports multiple vector store implementations."""
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional

from langchain_core.documents import Document


class VectorIndexProvider(ABC):
    """
    Abstract base class for vector index providers.
    
    All vector index implementations must implement this interface.
    """
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector index.
        
        Args:
            documents: List of documents to add
        """
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of documents
        """
        pass
    
    @abstractmethod
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with scores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        pass


class InMemoryVectorIndexProvider(VectorIndexProvider):
    """
    Vector index implementation using LangChain's InMemoryVectorStore (default).
    
    This is a simple in-memory vector store suitable for development and small datasets.
    """
    
    def __init__(self):
        """Initialize the in-memory vector index provider."""
        from langchain_core.vectorstores import InMemoryVectorStore
        from lahmajo.llm import get_embeddings
        
        self.InMemoryVectorStore = InMemoryVectorStore
        self.embeddings = get_embeddings()
        self.vector_store = None
        
        # Initialize the vector store
        self.vector_store = InMemoryVectorStore(self.embeddings)
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the in-memory vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        self.vector_store.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search using in-memory vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Search with scores using in-memory vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        return self.vector_store.similarity_search_with_score(query, k=k)


class ElasticsearchVectorIndexProvider(VectorIndexProvider):
    """
    Vector index implementation using Elasticsearch.
    
    Suitable for production environments with large datasets.
    """
    
    def __init__(self):
        """Initialize the Elasticsearch vector index provider."""
        try:
            from langchain_elasticsearch import ElasticsearchStore
        except ImportError:
            raise ImportError(
                "langchain-elasticsearch is required for Elasticsearch vector index. "
                "Install it with: pip install langchain-elasticsearch"
            )
        
        from lahmajo.llm import get_embeddings
        
        self.ElasticsearchStore = ElasticsearchStore
        self.embeddings = get_embeddings()
        self.vector_store = None
        
        # Get Elasticsearch configuration from environment
        es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
        index_name = os.getenv("ELASTICSEARCH_INDEX", "lahmajo_vectors")
        
        # Initialize Elasticsearch store
        self.vector_store = ElasticsearchStore(
            embedding=self.embeddings,
            es_url=es_url,
            index_name=index_name,
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to Elasticsearch."""
        if self.vector_store is None:
            raise ValueError("Elasticsearch store not initialized")
        self.vector_store.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search using Elasticsearch."""
        if self.vector_store is None:
            raise ValueError("Elasticsearch store not initialized")
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Search with scores using Elasticsearch."""
        if self.vector_store is None:
            raise ValueError("Elasticsearch store not initialized")
        return self.vector_store.similarity_search_with_score(query, k=k)


# Environment variable keys
VECTOR_INDEX_PROVIDER_ENV = "VECTOR_INDEX_PROVIDER"  # "in_memory" (default), "elasticsearch", etc.


def get_vector_index_provider() -> VectorIndexProvider:
    """
    Get a vector index provider instance based on environment configuration.
    
    Supported providers:
    - in_memory: LangChain's InMemoryVectorStore (default)
    - elasticsearch: Elasticsearch vector store
    
    Environment variables:
    - VECTOR_INDEX_PROVIDER: Provider name (default: "in_memory")
    - ELASTICSEARCH_URL: Elasticsearch URL (default: "http://localhost:9200")
    - ELASTICSEARCH_INDEX: Elasticsearch index name (default: "lahmajo_vectors")
    
    Returns:
        VectorIndexProvider instance
    """
    provider = os.getenv(VECTOR_INDEX_PROVIDER_ENV, "in_memory").lower()
    
    if provider == "in_memory":
        return InMemoryVectorIndexProvider()
    
    elif provider == "elasticsearch":
        return ElasticsearchVectorIndexProvider()
    
    # Future providers can be added here:
    # elif provider == "pinecone":
    #     return PineconeVectorIndexProvider()
    # elif provider == "weaviate":
    #     return WeaviateVectorIndexProvider()
    
    else:
        raise ValueError(
            f"Unknown vector index provider: {provider}. "
            f"Supported providers: in_memory, elasticsearch. "
            f"Set {VECTOR_INDEX_PROVIDER_ENV} environment variable."
        )

