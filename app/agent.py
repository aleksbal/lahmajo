# rag_ollama_demo/agent_app.py
from typing import Iterable

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

from app.indexing import build_vector_store


# Lazy initialization - build vector store only when first needed
_vector_store = None


def get_vector_store():
    """Get or build the vector store (lazy initialization)."""
    global _vector_store
    if _vector_store is None:
        _vector_store = build_vector_store(show_progress=True)
    return _vector_store


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    vector_store = get_vector_store()
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    # (text sent to the model, artifacts kept for you)
    return serialized, retrieved_docs


def create_rag_agent():
    # Local Ollama chat model
    model = ChatOllama(
        model="mistral",  # or llama3
        temperature=0.1,
        base_url="http://127.0.0.1:11434",
    )

    tools = [retrieve_context]

    system_prompt = (
        "You have access to a tool that retrieves context from a blog post. "
        "Use the tool to help answer user queries. "
        "If the tool does not return helpful context, say so explicitly."
    )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent


def ask_question(query: str, show_progress: bool = True) -> str:
    """Convenience wrapper: stream to console & return final answer."""
    agent = create_rag_agent()

    final_answer = None
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        msg = step["messages"][-1]
        # Only show progress if requested (for cleaner output)
        if show_progress:
            msg.pretty_print()
        final_answer = msg

    # `final_answer` is the last AIMessage in the stream
    return getattr(final_answer, "content", "")

