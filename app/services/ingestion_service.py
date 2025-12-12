# app/services/ingestion_service.py
"""Ingestion service - handles document ingestion workflow."""
import os
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from app.core.vector_store import get_vector_store, add_documents
from app.indexing import ingest_documents


def ingest_documents_from_files(
    urls: Optional[List[str]] = None,
    file_paths: Optional[List[str]] = None,
    use_semantic: bool = False,
    show_progress: bool = False
) -> Tuple[int, List[Document]]:
    """
    Ingest documents from URLs and/or file paths.
    
    Args:
        urls: List of URLs to load
        file_paths: List of file paths to load
        use_semantic: Whether to use semantic chunking
        show_progress: Whether to show progress indicators
        
    Returns:
        Tuple of (chunks_added, ingested_documents)
    """
    vector_store = get_vector_store()
    
    chunks_added, ingested_docs = ingest_documents(
        vector_store,
        urls=urls,
        file_paths=file_paths,
        use_semantic=use_semantic,
        show_progress=show_progress
    )
    
    # Add documents to hybrid search index
    add_documents(ingested_docs)
    
    return chunks_added, ingested_docs


async def save_uploaded_files(files: List) -> Tuple[List[str], str]:
    """
    Save uploaded files to temporary directory.
    
    Args:
        files: List of uploaded file objects
        
    Returns:
        Tuple of (file_paths, temp_dir)
    """
    file_paths = []
    temp_dir = None
    
    for file in files:
        if file.filename:
            # Validate file extension
            ext = Path(file.filename).suffix.lower()
            if ext not in [".txt", ".pdf", ".md"]:
                raise ValueError(f"Unsupported file type: {ext}. Supported: .txt, .pdf, .md")
            
            # Create temp directory only when needed
            if temp_dir is None:
                temp_dir = tempfile.mkdtemp()
            
            # Save file
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            file_paths.append(file_path)
    
    return file_paths, temp_dir


def cleanup_temp_files(file_paths: List[str], temp_dir: str):
    """Clean up temporary files and directory."""
    if temp_dir:
        for file_path in file_paths:
            try:
                os.remove(file_path)
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass
