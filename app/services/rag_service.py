# app/services/rag_service.py
"""RAG service - orchestrates the RAG agent for question answering."""
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

from app.services.retrieval_service import retrieve_context


@tool(response_format="content_and_artifact")
def retrieve_context_tool(query: str):
    """
    Retrieve information to help answer a query. 
    Always use this tool to search the knowledge base before answering.
    """
    serialized, docs = retrieve_context(query)
    return serialized, docs


def create_rag_agent():
    """Create a RAG agent with retrieval tool."""
    # Local Ollama chat model
    model = ChatOllama(
        model="mistral",  # or llama3
        temperature=0.1,
        base_url="http://127.0.0.1:11434",
    )

    tools = [retrieve_context_tool]

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
    """
    Ask a question and return the answer using the RAG pipeline.
    
    Args:
        query: Question to ask
        show_progress: Whether to show progress (currently unused, kept for compatibility)
        
    Returns:
        Answer string
    """
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
