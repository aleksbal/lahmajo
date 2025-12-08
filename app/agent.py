# rag_ollama_demo/agent_app.py
from typing import Iterable

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

from app.indexing import build_vector_store


# Build vector store once per process
vector_store = build_vector_store()


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
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


def ask_question(query: str) -> str:
    """Convenience wrapper: stream to console & return final answer."""
    agent = create_rag_agent()

    final_answer = None
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        msg = step["messages"][-1]
        # Pretty print streaming, like in the docs
        msg.pretty_print()  # prints Human/AI/Tool messages as they happen
        final_answer = msg

    # `final_answer` is the last AIMessage in the stream
    return getattr(final_answer, "content", "")

