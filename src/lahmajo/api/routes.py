# lahmajo/api/routes.py
"""FastAPI routes - API endpoints only."""
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from lahmajo.services.rag_service import ask_question
from lahmajo.services.ingestion_service import (
    ingest_documents_from_files,
    save_uploaded_files,
    cleanup_temp_files
)
from lahmajo.indexes.state import get_vector_index, get_all_documents
from lahmajo.search.hybrid_search import HybridRetriever
from lahmajo.search.rerank_provider import get_rerank_provider, LLMRerankProvider
from lahmajo.services.retrieval_service import _dedupe_chunks

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RAG Web UI")


class AskRequest(BaseModel):
    question: str
    use_hybrid: bool = True
    use_rerank: Optional[bool] = None  # None = use RERANK_PROVIDER env var default


class AskResponse(BaseModel):
    answer: str


STATIC_DIR_ENV = "LAHMAJO_STATIC_DIR"


def _default_static_dir() -> Path:
    # __file__ is src/lahmajo/api/routes.py - parents[3] is the repo root
    # (api -> lahmajo -> src -> repo root). Correct for how this project is
    # actually installed/run (editable install - `pip install -e .` - both
    # locally and in docker/Dockerfile), but NOT for a normal, non-editable
    # `pip install .`: that copies the package into site-packages without
    # static/ alongside it, since static/ lives outside src/lahmajo/ and isn't
    # packaged. LAHMAJO_STATIC_DIR below is the escape hatch for that case.
    return Path(__file__).resolve().parents[3] / "static"


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serve the HTML UI."""
    static_dir = Path(os.environ[STATIC_DIR_ENV]) if STATIC_DIR_ENV in os.environ else _default_static_dir()
    html_path = static_dir / "index.html"
    if html_path.exists():
        # Explicit encoding: read_text() defaults to locale.getpreferredencoding(),
        # which is cp1252 on Windows - static/index.html is UTF-8 and contains
        # bytes invalid in cp1252, which crashed this route with a 500 once the
        # path lookup above actually started finding the file.
        return html_path.read_text(encoding="utf-8")
    else:
        # Fallback HTML if file doesn't exist
        return """
        <!DOCTYPE html>
        <html>
        <head><title>RAG Web UI</title></head>
        <body><h1>RAG Web UI</h1><p>Please create static/index.html</p></body>
        </html>
        """


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest):
    """Ask a question using the RAG pipeline."""
    try:
        logging.info(f"Question received: {request.question} (use_hybrid={request.use_hybrid}, use_rerank={request.use_rerank})")
        answer = ask_question(
            request.question,
            show_progress=False,
            use_hybrid=request.use_hybrid,
            use_rerank=request.use_rerank,
        )
        logging.info(f"Answer length: {len(answer)} characters")
        return AskResponse(answer=answer)
    except Exception as e:
        logging.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.get("/debug/search")
async def debug_search(query: str, use_hybrid: bool = True, use_rerank: bool = False):
    """Debug endpoint to test vector store search directly."""
    try:
        vector_index = get_vector_index()
        all_docs = get_all_documents()

        # Fetch a larger candidate pool when reranking so it has more to choose from;
        # the final results list is still capped at 15 either way.
        RESULT_COUNT = 15
        candidate_k = 20 if use_rerank else RESULT_COUNT

        # Use hybrid search if available and requested
        if use_hybrid and all_docs and len(all_docs) > 0:
            try:
                # Both vector_index and BM25 are indexes - vector_index is the semantic index,
                # and BM25 index is created from the provider factory
                hybrid_retriever = HybridRetriever(vector_index, all_docs)
                # Combined via Reciprocal Rank Fusion (RRF) for the Python-side path (see
                # HybridRetriever.search()); bm25_weight/vector_weight are only consulted
                # when ES native hybrid search is active.
                results = hybrid_retriever.search(query, k=candidate_k)
                filtered_results = [(doc, score) for doc, score in results if len(doc.page_content.strip()) >= 100]
                search_type = "hybrid"
            except Exception as e:
                logging.warning(f"Hybrid search failed: {e}, falling back to vector")
                use_hybrid = False

        if not use_hybrid or not all_docs:
            # Fallback to vector-only search
            try:
                results_with_scores = vector_index.similarity_search_with_score(query, k=candidate_k)
                filtered_results = [(doc, float(score)) for doc, score in results_with_scores if len(doc.page_content.strip()) >= 100]
                search_type = "vector"
            except:
                results = vector_index.similarity_search(query, k=candidate_k)
                filtered_results = [(doc, None) for doc in results if len(doc.page_content.strip()) >= 100]
                search_type = "vector"

        # Collapse near-duplicate chunks (overlapping adaptive-chunking windows) before
        # reranking/display, same as retrieve_context() does for the real answer path.
        deduped_docs = _dedupe_chunks([doc for doc, _ in filtered_results])
        deduped_ids = {id(doc) for doc in deduped_docs}
        filtered_results = [(doc, score) for doc, score in filtered_results if id(doc) in deduped_ids]

        # Rerank if requested. Uses RERANK_PROVIDER if one is configured, otherwise
        # explicitly uses the LLM reranker for this call regardless of global config -
        # this is a debug toggle, so it should let you preview reranking even when
        # RERANK_PROVIDER=none globally.
        reranked = False
        if use_rerank:
            try:
                rerank_provider = get_rerank_provider() or LLMRerankProvider()
                candidates = [doc for doc, _ in filtered_results]
                filtered_results = rerank_provider.rerank(query, candidates, top_k=RESULT_COUNT)
                reranked = True
            except Exception as e:
                logging.warning(f"Reranking failed: {e}, using original order")

        # Get all unique sources
        all_sources = set()
        for doc, _ in filtered_results:
            source = doc.metadata.get("source", "unknown")
            all_sources.add(source)

        return JSONResponse({
            "query": query,
            "search_type": search_type,
            "reranked": reranked,
            "results_count": len(filtered_results),
            "sources_found": sorted(list(all_sources)),
            "results": [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "content_preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "content_length": len(doc.page_content),
                    "score": score,
                    "metadata": doc.metadata
                }
                for doc, score in filtered_results
            ]
        })
    except Exception as e:
        logging.error(f"Debug search error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/documents")
async def get_documents():
    """Get all ingested documents grouped by source."""
    try:
        all_docs = get_all_documents()
        
        # Group documents by source
        documents_by_source = {}
        for doc in all_docs:
            source = doc.metadata.get("source", "unknown")
            if source not in documents_by_source:
                documents_by_source[source] = {
                    "source": source,
                    "chunks": [],
                    "total_chunks": 0,
                    "total_chars": 0
                }
            
            chunk_length = len(doc.page_content)
            documents_by_source[source]["chunks"].append({
                "length": chunk_length,
                "preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            })
            documents_by_source[source]["total_chunks"] += 1
            documents_by_source[source]["total_chars"] += chunk_length
        
        # Convert to list and sort by source name
        documents_list = list(documents_by_source.values())
        documents_list.sort(key=lambda x: x["source"])
        
        return JSONResponse({
            "status": "success",
            "total_documents": len(documents_list),
            "total_chunks": len(all_docs),
            "documents": documents_list
        })
    except Exception as e:
        logging.error(f"Error getting documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting documents: {str(e)}")


@app.post("/ingest")
async def ingest_endpoint(
    url: Optional[str] = Form(None),
    files: Optional[list[UploadFile]] = File(None),
    chunking_strategy: str = Form("recursive"),  # "recursive" or "semantic"
):
    """Ingest documents from URL and/or uploaded files."""
    try:
        # Validate chunking strategy
        if chunking_strategy not in ["recursive", "semantic"]:
            raise HTTPException(
                status_code=400,
                detail="chunking_strategy must be 'recursive' or 'semantic'"
            )
        
        use_semantic = chunking_strategy == "semantic"
        logging.info(f"Ingesting - URL: {url}, Files: {[f.filename for f in (files or [])]}, Strategy: {chunking_strategy}")
        
        urls = [url] if url and url.strip() else []
        file_paths = []
        temp_dir = None
        
        # Save uploaded files temporarily
        if files:
            try:
                file_paths, temp_dir = await save_uploaded_files(files)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        if not urls and not file_paths:
            raise HTTPException(
                status_code=400,
                detail="Please provide either a URL or upload at least one file"
            )
        
        # Ingest documents
        chunks_added, ingested_docs = ingest_documents_from_files(
            urls=urls,
            file_paths=file_paths,
            use_semantic=use_semantic,
            show_progress=False
        )
        
        # Clean up temporary files
        cleanup_temp_files(file_paths, temp_dir)
        
        strategy_name = "Semantic" if use_semantic else "Recursive"
        return JSONResponse({
            "status": "success",
            "chunks_added": chunks_added,
            "chunking_strategy": chunking_strategy,
            "message": f"Successfully ingested {chunks_added} chunks using {strategy_name} chunking"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error ingesting documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error ingesting documents: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
