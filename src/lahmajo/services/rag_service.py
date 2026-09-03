# lahmajo/services/rag_service.py
"""RAG service - orchestrates the RAG agent for question answering."""
import logging
from typing import Any, List, Optional, Tuple

from langchain.agents import create_agent
from langchain.tools import tool

from lahmajo.llm import get_llm
from lahmajo.services.retrieval_service import SourceRef, retrieve_context_with_sources

logger = logging.getLogger(__name__)


def create_rag_agent(use_hybrid: bool = True, use_rerank: Optional[bool] = None):
    """
    Create a RAG agent with a retrieval tool.

    Args:
        use_hybrid: If False, the retrieval tool uses vector-only search instead of hybrid.
        use_rerank: Per-call reranking override passed through to retrieve_context()
            (None = use RERANK_PROVIDER env var, True/False = force on/off for this agent).
    """
    # Get LLM from provider factory (configurable via environment variables)
    model = get_llm()

    # Built per agent (not module-level) so use_hybrid/use_rerank can vary per request -
    # e.g. from the GUI/API, for comparing retrieval techniques without a server restart.
    @tool(response_format="content_and_artifact")
    def retrieve_context_tool(query: str):
        """
        Retrieve information to help answer a query.
        Always use this tool to search the knowledge base before answering.
        """
        serialized, _docs, sources = retrieve_context_with_sources(
            query, use_hybrid=use_hybrid, use_rerank=use_rerank
        )
        # The artifact carries the SourceRefs rather than the raw documents: it is
        # what ask_question_with_sources() reads back off the ToolMessage to report
        # real references, and the documents' text is already in `serialized`.
        return serialized, sources

    tools = [retrieve_context_tool]

    system_prompt = (
        """
        You are an assistant with access to a retrieval tool that can fetch relevant excerpts from a knowledge base.

        When the user’s question is about specific documents, facts, procedures, or project-specific information, call retrieve_context first.
        When the user’s question is general (small talk, generic explanations, brainstorming, or purely mathematical reasoning),
        you may answer without retrieval.

        Use the retrieved context as the source of truth. Do not introduce factual claims that are not supported by the retrieved context.
        If the context is missing, unclear, or conflicting, say so and ask a targeted follow-up question or explain what cannot be determined.

        Each retrieved excerpt is labelled with its own marker, "[source 1]", "[source 2]", and so on, followed by the file it came from.
        Cite the excerpt you used for a claim by writing that exact marker inline, e.g. "the deadline is 30 June [source 2]".
        Cite every factual claim you take from the context, and only use markers that actually appear in the retrieved excerpts.
        """
    )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent


def _collect_sources(messages: List[Any]) -> List[SourceRef]:
    """Pull the SourceRefs the retrieval tool attached to its ToolMessage artifacts.

    Today the agent performs at most one retrieval per answer (it either calls the
    tool once or not at all - see issue #8), so in practice this returns one call's
    references, numbered to match the `[source N]` markers in the context that call
    produced. If multi-hop retrieval lands, several calls will each carry their own
    1-based numbering and the markers will no longer be globally unique; warn rather
    than silently emit colliding indices.
    """
    artifacts = [
        message.artifact
        for message in messages
        if getattr(message, "artifact", None)
    ]
    source_lists = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, list) and all(isinstance(ref, SourceRef) for ref in artifact)
    ]

    if len(source_lists) > 1:
        logger.warning(
            f"{len(source_lists)} retrieval calls in one answer - [source N] markers are "
            "numbered per call and may collide across them."
        )

    return [ref for refs in source_lists for ref in refs]


def ask_question_with_sources(
    query: str,
    show_progress: bool = False,
    use_hybrid: bool = True,
    use_rerank: Optional[bool] = None,
) -> Tuple[str, List[SourceRef]]:
    """
    Ask a question and return the answer together with the excerpts it could cite.

    Args:
        query: Question to ask
        show_progress: Whether to show progress (currently unused, kept for compatibility)
        use_hybrid: If False, retrieval uses vector-only search instead of hybrid.
        use_rerank: Per-call reranking override (None = RERANK_PROVIDER env var,
            True/False = force reranking on/off for this call).

    Returns:
        Tuple of (answer, source_refs). source_refs is empty when the agent answered
        without retrieving anything, or when retrieval found nothing.
    """
    agent = create_rag_agent(use_hybrid=use_hybrid, use_rerank=use_rerank)

    final_answer = None
    final_messages: List[Any] = []
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        final_messages = step["messages"]
        final_answer = final_messages[-1]

    content = getattr(final_answer, "content", "")
    answer = content.strip() if content else ""
    return answer, _collect_sources(final_messages)


def ask_question(
    query: str,
    show_progress: bool = False,
    use_hybrid: bool = True,
    use_rerank: Optional[bool] = None,
) -> str:
    """
    Ask a question and return the answer using the RAG pipeline.

    Thin wrapper over ask_question_with_sources() for callers that only need the
    answer text.

    Args:
        query: Question to ask
        show_progress: Whether to show progress (currently unused, kept for compatibility)
        use_hybrid: If False, retrieval uses vector-only search instead of hybrid.
        use_rerank: Per-call reranking override (None = RERANK_PROVIDER env var,
            True/False = force reranking on/off for this call).

    Returns:
        Answer string
    """
    answer, _sources = ask_question_with_sources(
        query,
        show_progress=show_progress,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
    )
    return answer
