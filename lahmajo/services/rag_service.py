# lahmajo/services/rag_service.py
"""RAG service - orchestrates the RAG agent for question answering."""
from langchain.agents import create_agent
from langchain.tools import tool

from lahmajo.llm import get_llm
from lahmajo.services.retrieval_service import retrieve_context


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
    # Get LLM from provider factory (configurable via environment variables)
    model = get_llm()

    tools = [retrieve_context_tool]

    system_prompt = (
        """
        You are an assistant with access to a retrieval tool that can fetch relevant excerpts from a knowledge base.

        When the user’s question is about specific documents, facts, procedures, or project-specific information, call retrieve_context first.
        When the user’s question is general (small talk, generic explanations, brainstorming, or purely mathematical reasoning), 
        you may answer without retrieval.
        
        Use the retrieved context as the source of truth. Do not introduce factual claims that are not supported by the retrieved context.
        If the context is missing, unclear, or conflicting, say so and ask a targeted follow-up question or explain what cannot be determined.
        
        In your answer, quote or cite which retrieved excerpt(s) you used (e.g., [source 1], [source 2]).
        """
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
