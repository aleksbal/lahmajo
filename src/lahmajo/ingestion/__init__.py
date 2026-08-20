# lahmajo/ingestion/__init__.py
"""Document ingestion pipeline - loading, chunking, and embedding."""
from lahmajo.ingestion.processing import (
    ingest_documents,
    load_from_url,
    load_from_file,
    build_vector_store,
)

__all__ = [
    "ingest_documents",
    "load_from_url",
    "load_from_file",
    "build_vector_store",
]

