# lahmajo/indexes/elasticsearch_hybrid_provider.py
"""Elasticsearch native hybrid search provider - combines BM25 and vector search in single ES query."""
import os
from typing import List, Tuple

from langchain_core.documents import Document

from lahmajo.indexes.vector_provider import VectorIndexProvider
from lahmajo.indexes.bm25_provider import BM25Provider
from lahmajo.llm import get_embeddings


class ElasticsearchHybridProvider(VectorIndexProvider, BM25Provider):
    """
    Unified Elasticsearch provider that uses native ES hybrid search.
    
    Combines BM25 (keyword) and vector (semantic) search in a single ES query,
    leveraging ES's native scoring and optimization capabilities.
    
    This is more efficient than separate queries because:
    - Single network round trip
    - ES native score combination
    - Better performance at scale
    """
    
    def __init__(self):
        """Initialize the Elasticsearch hybrid provider."""
        try:
            from elasticsearch import Elasticsearch
            from langchain_elasticsearch import ElasticsearchStore
        except ImportError:
            raise ImportError(
                "elasticsearch and langchain-elasticsearch are required. "
                "Install with: pip install elasticsearch langchain-elasticsearch"
            )
        
        self.embeddings = get_embeddings()
        
        # Get Elasticsearch configuration from environment
        self.es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
        self.index_name = os.getenv("ELASTICSEARCH_INDEX", "lahmajo_vectors")
        
        # Initialize ES client
        self.es_client = Elasticsearch([self.es_url])
        
        # Verify connection
        try:
            if not self.es_client.ping():
                raise ConnectionError("Cannot connect to Elasticsearch. Ensure ES is running.")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Elasticsearch: {e}")
        
        # Initialize ElasticsearchStore for vector operations
        self.vector_store = ElasticsearchStore(
            embedding=self.embeddings,
            es_url=self.es_url,
            index_name=self.index_name,
        )
        
        # Get embedding dimension
        try:
            test_embedding = self.embeddings.embed_query("test")
            self.embedding_dim = len(test_embedding)
        except Exception as e:
            raise ValueError(f"Failed to get embedding dimension: {e}")
        
        # Ensure index mapping
        self._ensure_index_mapping()
    
    def _ensure_index_mapping(self):
        """Ensure ES index has proper mapping with both text and dense_vector fields."""
        try:
            # Check if index exists
            if not self.es_client.indices.exists(index=self.index_name):
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
                self.es_client.indices.create(index=self.index_name, body=mapping)
            else:
                # Update existing index mapping if needed
                try:
                    current_mapping = self.es_client.indices.get_mapping(index=self.index_name)
                    props = current_mapping[self.index_name]["mappings"].get("properties", {})
                    
                    # Check if text field exists, if not add it
                    if "text" not in props:
                        self.es_client.indices.put_mapping(
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
        except Exception:
            # If mapping setup fails, continue - langchain will handle it
            pass
    
    # VectorIndexProvider interface methods
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to Elasticsearch with both text and embedding fields."""
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
        """Search using Elasticsearch vector search only."""
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Search using Elasticsearch vector search only, with scores."""
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    # BM25Provider interface methods
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents for BM25 search.
        
        Note: Documents are indexed via add_documents() which handles both
        vector and text fields. This is a no-op for compatibility.
        """
        # Documents are already indexed via add_documents()
        pass
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        """Search using Elasticsearch native BM25 scoring only."""
        try:
            search_body = {
                "query": {
                    "match": {
                        "text": {
                            "query": query,
                            "operator": "or"
                        }
                    }
                },
                "size": top_k if top_k is not None else 100,
                "_source": ["text", "metadata"]
            }
            
            response = self.es_client.search(
                index=self.index_name,
                body=search_body
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                score = hit["_score"]
                source = hit["_source"]
                
                text_content = source.get("text", "") or source.get("page_content", "")
                
                # Skip if no text content
                if not text_content:
                    continue
                
                metadata = source.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                
                doc = Document(
                    page_content=text_content,
                    metadata=metadata
                )
                
                results.append((doc, float(score)))
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Elasticsearch BM25 search failed: {e}")
    
    # Native hybrid search method
    
    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6
    ) -> List[Tuple[Document, float]]:
        """
        Perform native ES hybrid search combining BM25 and vector search in single query.
        
        This uses ES's native hybrid query combining:
        - match query for BM25 keyword search
        - knn query for vector similarity search
        
        Args:
            query: Search query
            k: Number of results to return
            bm25_weight: Weight for BM25 scores (0-1)
            vector_weight: Weight for vector scores (0-1)
            
        Returns:
            List of (document, combined_score) tuples, sorted by score
        """
        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Build native ES hybrid query
            # ES 8.0+ supports combining bool query with knn in single request
            # The knn query is at top level, and query is used for filtering/boosting
            search_body = {
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": k * 2,  # Get more candidates for better combination
                    "num_candidates": k * 3,  # ES recommendation: 3x k
                    "boost": vector_weight
                },
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "text": {
                                        "query": query,
                                        "operator": "or",
                                        "boost": bm25_weight
                                    }
                                }
                            }
                        ]
                    }
                },
                "size": k,
                "_source": ["text", "metadata"]
            }
            
            # Execute hybrid search
            response = self.es_client.search(
                index=self.index_name,
                body=search_body
            )
            
            # Convert ES results to Document objects
            results = []
            for hit in response["hits"]["hits"]:
                score = hit["_score"]
                source = hit["_source"]
                
                text_content = source.get("text", "") or source.get("page_content", "")
                
                # Skip if no text content
                if not text_content:
                    continue
                
                metadata = source.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                
                doc = Document(
                    page_content=text_content,
                    metadata=metadata
                )
                
                results.append((doc, float(score)))
            
            return results
            
        except Exception as e:
            # Fallback to separate queries if native hybrid fails
            # (e.g., older ES version that doesn't support knn in search)
            try:
                # Try vector search
                vector_results = self.similarity_search_with_score(query, k=k)
                # Try BM25 search
                bm25_results = self.search(query, top_k=k)
                
                # Combine results (simple approach)
                combined = {}
                for doc, score in vector_results:
                    key = doc.page_content
                    combined[key] = (doc, vector_weight * score)
                
                for doc, score in bm25_results:
                    key = doc.page_content
                    if key in combined:
                        combined[key] = (doc, combined[key][1] + bm25_weight * score)
                    else:
                        combined[key] = (doc, bm25_weight * score)
                
                results = list(combined.values())
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:k]
                
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Elasticsearch hybrid search failed: {e}. "
                    f"Fallback also failed: {fallback_error}"
                )
    
    def supports_native_hybrid(self) -> bool:
        """Check if this provider supports native ES hybrid search."""
        return True
    
    def get_index_name(self) -> str:
        """Get the Elasticsearch index name."""
        return self.index_name

