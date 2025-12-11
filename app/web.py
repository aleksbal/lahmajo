# app/web.py
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from app.agent import ask_question, get_vector_store, add_documents_to_index
from app.indexing import ingest_documents
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RAG Web UI")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serve the HTML UI."""
    html_path = Path(__file__).parent.parent / "static" / "index.html"
    if html_path.exists():
        return html_path.read_text()
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
        logging.info(f"Question received: {request.question}")
        answer = ask_question(request.question, show_progress=False)
        logging.info(f"Answer length: {len(answer)} characters")
        return AskResponse(answer=answer)
    except Exception as e:
        logging.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.get("/debug/search")
async def debug_search(query: str, use_hybrid: bool = True):
    """Debug endpoint to test vector store search directly."""
    try:
        from app.agent import get_all_documents
        from app.hybrid_search import HybridRetriever
        
        vector_store = get_vector_store()
        all_docs = get_all_documents()
        
        # Use hybrid search if available and requested
        if use_hybrid and all_docs and len(all_docs) > 0:
            try:
                hybrid_retriever = HybridRetriever(vector_store, all_docs)
                query_lower = query.lower()
                is_name_query = any(word.isupper() or len(word) > 8 for word in query.split())
                
                if is_name_query or len(query.split()) <= 3:
                    bm25_weight, vector_weight = 0.6, 0.4
                else:
                    bm25_weight, vector_weight = 0.4, 0.6
                
                results = hybrid_retriever.search(query, k=15, bm25_weight=bm25_weight, vector_weight=vector_weight)
                filtered_results = [(doc, score) for doc, score in results if len(doc.page_content.strip()) >= 100]
                search_type = "hybrid"
            except Exception as e:
                logging.warning(f"Hybrid search failed: {e}, falling back to vector")
                use_hybrid = False
        
        if not use_hybrid or not all_docs:
            # Fallback to vector-only search
            try:
                results_with_scores = vector_store.similarity_search_with_score(query, k=15)
                filtered_results = [(doc, float(score)) for doc, score in results_with_scores if len(doc.page_content.strip()) >= 100]
                search_type = "vector"
            except:
                results = vector_store.similarity_search(query, k=15)
                filtered_results = [(doc, None) for doc in results if len(doc.page_content.strip()) >= 100]
                search_type = "vector"
        
        # Get all unique sources
        all_sources = set()
        for doc, _ in filtered_results:
            source = doc.metadata.get("source", "unknown")
            all_sources.add(source)
        
        return JSONResponse({
            "query": query,
            "search_type": search_type,
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
        vector_store = get_vector_store()
        
        urls = [url] if url and url.strip() else []
        file_paths = []
        temp_dir = None
        
        # Save uploaded files temporarily
        if files:
            for file in files:
                if file.filename:
                    # Validate file extension
                    ext = Path(file.filename).suffix.lower()
                    if ext not in [".txt", ".pdf", ".md"]:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Unsupported file type: {ext}. Supported: .txt, .pdf, .md"
                        )
                    
                    # Create temp directory only when needed
                    if temp_dir is None:
                        temp_dir = tempfile.mkdtemp()
                    
                    # Save file
                    file_path = os.path.join(temp_dir, file.filename)
                    with open(file_path, "wb") as f:
                        content = await file.read()
                        f.write(content)
                    file_paths.append(file_path)
        
        if not urls and not file_paths:
            raise HTTPException(
                status_code=400,
                detail="Please provide either a URL or upload at least one file"
            )
        
        # Ingest documents
        chunks_added, ingested_docs = ingest_documents(
            vector_store,
            urls=urls,
            file_paths=file_paths,
            use_semantic=use_semantic,
            show_progress=False
        )
        
        # Add documents to hybrid search index
        add_documents_to_index(ingested_docs)
        
        # Clean up temporary files
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
        raise HTTPException(status_code=500, detail=f"Error ingesting documents: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
