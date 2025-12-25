# lahmajo/indexes/vector_provider.py
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
    Stores documents with both 'text' field (for BM25) and 'embedding' field (for vector search).
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
        self.es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
        self.index_name = os.getenv("ELASTICSEARCH_INDEX", "lahmajo_vectors")
        
        # Get embedding dimension for index mapping
        try:
            test_embedding = self.embeddings.embed_query("test")
            self.embedding_dim = len(test_embedding)
        except Exception as e:
            raise ValueError(f"Failed to get embedding dimension: {e}")
        
        # Initialize Elasticsearch store with proper index mapping
        self.vector_store = ElasticsearchStore(
            embedding=self.embeddings,
            es_url=self.es_url,
            index_name=self.index_name,
        )
        
        # Ensure index mapping includes both text and embedding fields
        self._ensure_index_mapping()
    
    def _ensure_index_mapping(self):
        """Ensure ES index has proper mapping with both text and dense_vector fields."""
        try:
            from elasticsearch import Elasticsearch
            
            es_client = Elasticsearch([self.es_url])
            
            # Check if index exists
            if not es_client.indices.exists(index=self.index_name):
                # Create index with proper mapping
                mapping = {
                    "mappings": {
                        "properties": {
                            "text": {
                                "type": "text",
                                "analyzer": "standard"
                            },
                            "embedding": {
                                "type": "dense_vector",
                                "dims": self.embedding_dim,
                                "index": True,
                                "similarity": "cosine"
                            },
                            "metadata": {
                                "type": "object",
                                "enabled": True
                            }
                        }
                    }
                }
                es_client.indices.create(index=self.index_name, body=mapping)
            else:
                # Update existing index mapping if needed
                try:
                    current_mapping = es_client.indices.get_mapping(index=self.index_name)
                    props = current_mapping[self.index_name]["mappings"].get("properties", {})
                    
                    # Check if text field exists, if not add it
                    if "text" not in props:
                        es_client.indices.put_mapping(
                            index=self.index_name,
                            body={
                                "properties": {
                                    "text": {
                                        "type": "text",
                                        "analyzer": "standard"
                                    }
                                }
                            }
                        )
                except Exception:
                    # If mapping update fails, continue - langchain will handle it
                    pass
        except ImportError:
            # elasticsearch client not available, langchain will handle mapping
            pass
        except Exception:
            # If mapping setup fails, continue - langchain will handle it
            pass
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to Elasticsearch with both text and embedding fields."""
        if self.vector_store is None:
            raise ValueError("Elasticsearch store not initialized")
        
        # Ensure documents have text field for BM25 search
        enhanced_docs = []
        for doc in documents:
            # Create enhanced document with text field
            enhanced_metadata = doc.metadata.copy()
            enhanced_metadata["text"] = doc.page_content
            enhanced_doc = Document(
                page_content=doc.page_content,
                metadata=enhanced_metadata
            )
            enhanced_docs.append(enhanced_doc)
        
        self.vector_store.add_documents(enhanced_docs)
    
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
    
    def supports_native_hybrid(self) -> bool:
        """Check if this provider supports native ES hybrid search."""
        return True
    
    def get_es_client(self):
        """Get Elasticsearch client for direct ES operations."""
        try:
            from elasticsearch import Elasticsearch
            return Elasticsearch([self.es_url])
        except ImportError:
            raise ImportError(
                "elasticsearch Python client is required. "
                "Install it with: pip install elasticsearch"
            )
    
    def get_index_name(self) -> str:
        """Get the Elasticsearch index name."""
        return self.index_name


# Environment variable keys
VECTOR_INDEX_PROVIDER_ENV = "VECTOR_INDEX_PROVIDER"  # "in_memory" (default), "elasticsearch", etc.


def get_vector_index_provider() -> VectorIndexProvider:
    """
    Get a vector index provider instance based on environment configuration.
    
    Supported providers:
    - in_memory: LangChain's InMemoryVectorStore (default)
    - elasticsearch: Elasticsearch vector store
    
    When both VECTOR_INDEX_PROVIDER=elasticsearch and BM25_PROVIDER=elasticsearch,
    consider using ElasticsearchHybridProvider for native hybrid search.
    
    Environment variables:
    - VECTOR_INDEX_PROVIDER: Provider name (default: "in_memory")
    - BM25_PROVIDER: BM25 provider name (checked for ES native hybrid)
    - ELASTICSEARCH_URL: Elasticsearch URL (default: "http://localhost:9200")
    - ELASTICSEARCH_INDEX: Elasticsearch index name (default: "lahmajo_vectors")
    - ELASTICSEARCH_USE_NATIVE_HYBRID: Force use of native hybrid (auto-detected if both providers are ES)
    
    Returns:
        VectorIndexProvider instance
    """
    provider = os.getenv(VECTOR_INDEX_PROVIDER_ENV, "in_memory").lower()
    bm25_provider = os.getenv("BM25_PROVIDER", "rank_bm25").lower()
    use_native_hybrid = os.getenv("ELASTICSEARCH_USE_NATIVE_HYBRID", "").lower() == "true"
    
    # Check if we should use ES native hybrid provider
    if provider == "elasticsearch" and bm25_provider == "elasticsearch":
        if use_native_hybrid:
            try:
                from lahmajo.indexes.elasticsearch_hybrid_provider import ElasticsearchHybridProvider
                return ElasticsearchHybridProvider()
            except ImportError:
                # Fall back to regular ES vector provider if hybrid not available
                pass
    
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

