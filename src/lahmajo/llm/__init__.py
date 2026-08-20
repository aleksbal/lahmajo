# lahmajo/llm/__init__.py
"""LLM and embedding provider abstractions."""
from lahmajo.llm.llm_provider import get_llm
from lahmajo.llm.embedding_provider import get_embeddings

__all__ = ["get_llm", "get_embeddings"]

