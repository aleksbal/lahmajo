# lahmajo/llm/llm_provider.py
"""LLM provider factory - supports multiple LLM backends."""
import os
from typing import Any

# Environment variable keys
LLM_PROVIDER_ENV = "LLM_PROVIDER"  # "ollama_local", "ollama_cloud", "openai"
LLM_MODEL_ENV = "LLM_MODEL"  # Model name (e.g., "gpt-oss:120b-cloud", "gpt-4", "llama3")
LLM_BASE_URL_ENV = "LLM_BASE_URL"  # Base URL for Ollama (default: http://127.0.0.1:11434)
LLM_TEMPERATURE_ENV = "LLM_TEMPERATURE"  # Temperature (default: 0.1)
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"  # OpenAI API key


def get_llm() -> Any:
    """
    Get an LLM instance based on environment configuration.
    
    Supported providers:
    - ollama_local: Local Ollama (default)
    - ollama_cloud: Ollama via cloud API
    - openai: OpenAI API
    
    Environment variables:
    - LLM_PROVIDER: Provider name (default: "ollama_local")
    - LLM_MODEL: Model name (default: "gpt-oss:120b-cloud" for Ollama, "gpt-4" for OpenAI)
    - LLM_BASE_URL: Base URL for Ollama (default: "http://127.0.0.1:11434")
    - LLM_TEMPERATURE: Temperature (default: 0.1)
    - OPENAI_API_KEY: Required for OpenAI provider
    
    Returns:
        LangChain LLM instance
    """
    provider = os.getenv(LLM_PROVIDER_ENV, "ollama_local").lower()
    model = os.getenv(LLM_MODEL_ENV)
    base_url = os.getenv(LLM_BASE_URL_ENV, "http://127.0.0.1:11434")
    temperature = float(os.getenv(LLM_TEMPERATURE_ENV, "0.1"))
    
    if provider == "ollama_local" or provider == "ollama_cloud":
        from langchain_ollama import ChatOllama
        
        # Default model for Ollama
        if not model:
            model = "gpt-oss:120b-cloud"
        
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
        )
    
    elif provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                f"langchain-openai is required for OpenAI provider. "
                f"Install it with: pip install langchain-openai"
            )
        
        # Check for API key
        api_key = os.getenv(OPENAI_API_KEY_ENV)
        if not api_key:
            raise ValueError(
                f"OPENAI_API_KEY environment variable is required for OpenAI provider. "
                f"Set it with: export OPENAI_API_KEY='your-key-here'"
            )
        
        # Default model for OpenAI
        if not model:
            model = "gpt-4"
        
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )
    
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Supported providers: ollama_local, ollama_cloud, openai. "
            f"Set {LLM_PROVIDER_ENV} environment variable."
        )

