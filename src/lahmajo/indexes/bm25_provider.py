# lahmajo/indexes/bm25_provider.py
"""BM25 provider factory - supports multiple BM25/keyword search implementations."""
import os
from abc import ABC, abstractmethod
from typing import List, Tuple

from langchain_core.documents import Document


class BM25Provider(ABC):
    """
    Abstract base class for BM25/keyword search providers.
    
    All BM25 implementations must implement this interface.
    """
    
    @abstractmethod
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents for keyword search.
        
        Args:
            documents: List of documents to index
        """
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        """
        Search documents by query.
        
        Args:
            query: Search query string
            top_k: Number of top results to return (None = return all)
            
        Returns:
            List of (document, score) tuples, sorted by score (higher is better)
        """
        pass


class RankBM25Provider(BM25Provider):
    """
    BM25 implementation using rank-bm25 library (default).
    
    This is the standard Python BM25 implementation.
    """
    
    def __init__(self):
        """Initialize the rank-bm25 provider."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 is required for BM25 provider. "
                "Install it with: pip install rank-bm25"
            )
        self.BM25Okapi = BM25Okapi
        self.bm25 = None
        self.documents = []
    
    def index_documents(self, documents: List[Document]) -> None:
        """Index documents using rank-bm25."""
        import re
        
        self.documents = documents
        
        # Tokenize documents for BM25
        def tokenize(text: str) -> List[str]:
            """Simple tokenization for BM25."""
            tokens = re.findall(r'\w+', text.lower())
            return tokens
        
        tokenized_docs = [tokenize(doc.page_content) for doc in documents]
        self.bm25 = self.BM25Okapi(tokenized_docs)
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        """Search using rank-bm25."""
        if self.bm25 is None:
            raise ValueError("Documents must be indexed before searching. Call index_documents() first.")
        
        import re
        
        # Tokenize query
        def tokenize(text: str) -> List[str]:
            tokens = re.findall(r'\w+', text.lower())
            return tokens
        
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Create list of (document, score) tuples
        results = [(doc, score) for doc, score in zip(self.documents, scores)]
        
        # Sort by score (higher is better)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k if specified
        if top_k is not None:
            return results[:top_k]
        
        return results


class ElasticsearchBM25Provider(BM25Provider):
    """
    BM25 implementation using Elasticsearch native BM25 scoring.
    
    Uses ES native match queries with BM25 scoring algorithm.
    Suitable for production environments with large datasets.
    """
    
    def __init__(self):
        """Initialize the Elasticsearch BM25 provider."""
        try:
            from elasticsearch import Elasticsearch
        except ImportError:
            raise ImportError(
                "elasticsearch Python client is required for Elasticsearch BM25 provider. "
                "Install it with: pip install elasticsearch"
            )
        
        # Get Elasticsearch configuration from environment
        es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
        self.index_name = os.getenv("ELASTICSEARCH_INDEX", "lahmajo_vectors")
        
        self.es_client = Elasticsearch([es_url])
        
        # Verify connection
        try:
            if not self.es_client.ping():
                raise ConnectionError("Cannot connect to Elasticsearch. Ensure ES is running.")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Elasticsearch: {e}")
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents to Elasticsearch text field for BM25 search.
        
        Note: Documents should already be indexed by ElasticsearchVectorIndexProvider.
        This method ensures the text field is properly set for BM25 queries.
        """
        # Documents are indexed via vector provider, but we ensure text field exists
        # This is a no-op if documents are already indexed with text field
        # The actual indexing happens in the vector provider's add_documents method
        pass
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        """Search using Elasticsearch native BM25 scoring via match query."""
        try:
            # Build ES match query for BM25 search
            search_body = {
                "query": {
                    "match": {
                        "text": {
                            "query": query,
                            "operator": "or"  # Match any term (OR logic)
                        }
                    }
                },
                "size": top_k if top_k is not None else 100,
                "_source": ["text", "metadata"]
            }
            
            # Execute search
            response = self.es_client.search(
                index=self.index_name,
                body=search_body
            )
            
            # Convert ES results to Document objects
            results = []
            for hit in response["hits"]["hits"]:
                score = hit["_score"]
                source = hit["_source"]
                
                # Extract text content
                # ES stores text in 'text' field, but langchain might use 'page_content'
                text_content = source.get("text", "") or source.get("page_content", "")
                
                # If still empty, skip this document
                if not text_content:
                    continue
                
                # Extract metadata
                metadata = source.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                
                # Create Document
                doc = Document(
                    page_content=text_content,
                    metadata=metadata
                )
                
                results.append((doc, float(score)))
            
            # Results are already sorted by ES score (descending)
            return results
            
        except Exception as e:
            raise RuntimeError(f"Elasticsearch BM25 search failed: {e}")
    
    def get_index_name(self) -> str:
        """Get the Elasticsearch index name."""
        return self.index_name


# Environment variable keys
BM25_PROVIDER_ENV = "BM25_PROVIDER"  # "rank_bm25" (default), "elasticsearch", "whoosh", etc.


def get_bm25_provider() -> BM25Provider:
    """
    Get a BM25 provider instance based on environment configuration.
    
    Supported providers:
    - rank_bm25: rank-bm25 library (default)
    - Additional providers can be added here
    
    Environment variables:
    - BM25_PROVIDER: Provider name (default: "rank_bm25")
    
    Returns:
        BM25Provider instance
    """
    provider = os.getenv(BM25_PROVIDER_ENV, "rank_bm25").lower()
    
    if provider == "rank_bm25":
        return RankBM25Provider()
    
    elif provider == "elasticsearch":
        return ElasticsearchBM25Provider()
    
    # Future providers can be added here:
    # elif provider == "whoosh":
    #     return WhooshBM25Provider()
    
    else:
        raise ValueError(
            f"Unknown BM25 provider: {provider}. "
            f"Supported providers: rank_bm25, elasticsearch. "
            f"Set {BM25_PROVIDER_ENV} environment variable."
        )

