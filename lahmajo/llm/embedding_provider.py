# lahmajo/llm/embedding_provider.py
"""Embedding provider factory - supports multiple embedding backends."""
import os
from typing import Any

# Environment variable keys
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"  # "ollama_local", "ollama_cloud", "openai"
EMBEDDING_MODEL_ENV = "EMBEDDING_MODEL"  # Model name (e.g., "embeddinggemma", "text-embedding-ada-002")
EMBEDDING_BASE_URL_ENV = "EMBEDDING_BASE_URL"  # Base URL for Ollama (default: http://127.0.0.1:11434)
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"  # OpenAI API key (shared with LLM)


def get_embeddings() -> Any:
    """
    Get an embeddings instance based on environment configuration.
    
    Supported providers:
    - ollama_local: Local Ollama (default)
    - ollama_cloud: Ollama via cloud API
    - openai: OpenAI API
    
    Environment variables:
    - EMBEDDING_PROVIDER: Provider name (default: "ollama_local")
    - EMBEDDING_MODEL: Model name (default: "embeddinggemma" for Ollama, "text-embedding-ada-002" for OpenAI)
    - EMBEDDING_BASE_URL: Base URL for Ollama (default: "http://127.0.0.1:11434")
    - OPENAI_API_KEY: Required for OpenAI provider
    
    Returns:
        LangChain embeddings instance
    """
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "ollama_local").lower()
    model = os.getenv(EMBEDDING_MODEL_ENV)
    base_url = os.getenv(EMBEDDING_BASE_URL_ENV, "http://127.0.0.1:11434")
    
    if provider == "ollama_local" or provider == "ollama_cloud":
        from langchain_ollama import OllamaEmbeddings
        
        # Default model for Ollama
        if not model:
            model = "embeddinggemma"
        
        return OllamaEmbeddings(
            model=model,
            base_url=base_url,
        )
    
    elif provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            raise ImportError(
                f"langchain-openai is required for OpenAI embeddings. "
                f"Install it with: pip install langchain-openai"
            )
        
        # Check for API key
        api_key = os.getenv(OPENAI_API_KEY_ENV)
        if not api_key:
            raise ValueError(
                f"OPENAI_API_KEY environment variable is required for OpenAI embeddings. "
                f"Set it with: export OPENAI_API_KEY='your-key-here'"
            )
        
        # Default model for OpenAI
        if not model:
            model = "text-embedding-ada-002"
        
        return OpenAIEmbeddings(
            model=model,
            api_key=api_key,
        )
    
    else:
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            f"Supported providers: ollama_local, ollama_cloud, openai. "
            f"Set {EMBEDDING_PROVIDER_ENV} environment variable."
        )

