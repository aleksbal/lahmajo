# rag_ollama_demo/agent_app.py
from typing import Iterable, List

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

from app.indexing import build_vector_store
from app.hybrid_search import HybridRetriever


# Lazy initialization - build vector store only when first needed
_vector_store = None
_all_documents = []  # Store all documents for hybrid search


def get_vector_store():
    """Get or build the vector store (lazy initialization)."""
    global _vector_store, _all_documents
    if _vector_store is None:
        _vector_store = build_vector_store(show_progress=True)
        # Initialize documents list - we'll populate it as documents are added
        _all_documents = []
    return _vector_store


def get_all_documents() -> List:
    """Get all documents in the vector store for hybrid search."""
    global _all_documents
    # This will be populated as documents are added
    return _all_documents


def add_documents_to_index(documents: List):
    """Add documents to the global index for hybrid search."""
    global _all_documents
    _all_documents.extend(documents)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query. Always use this tool to search the knowledge base before answering."""
    import logging
    logger = logging.getLogger(__name__)
    
    vector_store = get_vector_store()
    all_docs = get_all_documents()
    
    # Use hybrid search if we have documents indexed, otherwise fall back to vector only
    if all_docs and len(all_docs) > 0:
        try:
            # Hybrid search: BM25 (keyword) + Vector (semantic)
            # This is the industry standard approach
            hybrid_retriever = HybridRetriever(vector_store, all_docs)
            
            # For name queries, give more weight to BM25 (keyword matching)
            # For semantic queries, give more weight to vector search
            query_lower = query.lower()
            is_name_query = any(word.isupper() or len(word) > 8 for word in query.split())
            
            if is_name_query or len(query.split()) <= 3:
                # Name or short query - prioritize keyword matching
                bm25_weight, vector_weight = 0.6, 0.4
            else:
                # Longer semantic query - balance both
                bm25_weight, vector_weight = 0.4, 0.6
            
            results = hybrid_retriever.search(query, k=10, bm25_weight=bm25_weight, vector_weight=vector_weight)
            top_docs = [doc for doc, score in results]
            
            logger.info(f"Hybrid search - Query: {query}, BM25 weight: {bm25_weight}, Vector weight: {vector_weight}")
            for i, (doc, score) in enumerate(results[:5]):
                logger.info(f"Doc {i+1} source: {doc.metadata.get('source', 'unknown')}, hybrid_score: {score:.4f}")
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to vector search: {e}")
            # Fallback to vector search
            try:
                retrieved_with_scores = vector_store.similarity_search_with_score(query, k=10)
                top_docs = [doc for doc, score in retrieved_with_scores]
            except:
                top_docs = vector_store.similarity_search(query, k=10)
    else:
        # Fallback to vector-only search
        try:
            retrieved_with_scores = vector_store.similarity_search_with_score(query, k=10)
            top_docs = [doc for doc, score in retrieved_with_scores]
        except:
            top_docs = vector_store.similarity_search(query, k=10)
    
    # Filter out very small chunks
    MIN_CHARS = 100
    filtered_docs = [doc for doc in top_docs if len(doc.page_content.strip()) >= MIN_CHARS]
    
    # Take top 8
    top_docs = filtered_docs[:8]
    
    # Debug logging
    logger.info(f"Retrieval query: {query}")
    logger.info(f"Retrieved {len(top_docs)} documents")
    for i, doc in enumerate(top_docs[:5]):
        logger.info(f"Doc {i+1} source: {doc.metadata.get('source', 'unknown')}")
        logger.info(f"Doc {i+1} preview: {doc.page_content[:100]}...")
    
    # If no results, return a message indicating no context found
    if not top_docs:
        return "No relevant documents found in the knowledge base for this query.", []
    
    # Format results with source information
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
        for doc in top_docs
    )
    # (text sent to the model, artifacts kept for you)
    return serialized, top_docs


def create_rag_agent():
    # Local Ollama chat model
    model = ChatOllama(
        model="mistral",  # or llama3
        temperature=0.1,
        base_url="http://127.0.0.1:11434",
    )

    tools = [retrieve_context]

    system_prompt = (
        "You have access to a tool that retrieves context from a knowledge base containing documents. "
        "ALWAYS use the retrieve_context tool FIRST to search for relevant information before answering any question. "
        "Only answer based on the retrieved context. "
        "If the tool does not return helpful context, say so explicitly. "
        "Do not make up information or use your training data - only use information from the retrieved context."
    )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent


def ask_question(query: str, show_progress: bool = False) -> str:
    """Ask a question and return the answer."""
    agent = create_rag_agent()

    final_answer = None
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        msg = step["messages"][-1]
        final_answer = msg

    content = getattr(final_answer, "content", "")
    return content.strip() if content else ""

