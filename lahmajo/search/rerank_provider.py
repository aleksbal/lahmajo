# lahmajo/search/rerank_provider.py
"""Rerank provider factory - supports pluggable reranking of retrieved candidates.

Reranking is an optional second pass over the candidates that hybrid search (BM25 +
vector, combined via RRF) already produced: it re-scores a larger candidate pool with
something more accurate than rank fusion, then keeps the best top_k for the LLM.
Disabled by default (RERANK_PROVIDER=none) so existing behavior is unchanged unless
explicitly opted into.
"""
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Max characters of each candidate's content included in the rerank prompt, to keep
# the prompt a reasonable size when there are many candidates.
MAX_CHARS_PER_CANDIDATE = 400


class RerankProvider(ABC):
    """
    Abstract base class for rerank providers.

    All rerank implementations must implement this interface.
    """

    @abstractmethod
    def rerank(self, query: str, candidates: List[Document], top_k: int) -> List[Tuple[Document, float]]:
        """
        Re-score and reorder candidates by relevance to the query.

        Args:
            query: The search query
            candidates: Candidate documents to rerank (already retrieved, e.g. via hybrid search)
            top_k: Number of top results to return

        Returns:
            List of (document, score) tuples, best first, length <= top_k.
            Score is provider-specific and only meaningful for ordering within this call.
        """
        pass


class LLMRerankProvider(RerankProvider):
    """
    Rerank implementation that asks the already-configured LLM (see llm/llm_provider.py)
    to order candidates by relevance. Reuses whatever LLM_PROVIDER is already set up
    (Ollama local/cloud or OpenAI) - no new dependencies or services required.
    """

    def __init__(self):
        """Initialize the LLM rerank provider."""
        from lahmajo.llm import get_llm
        self.llm = get_llm()

    def rerank(self, query: str, candidates: List[Document], top_k: int) -> List[Tuple[Document, float]]:
        """Rerank candidates by asking the LLM for a best-to-worst index order."""
        if not candidates:
            return []

        # A single candidate needs no reranking.
        if len(candidates) == 1:
            return [(candidates[0], 1.0)]

        prompt = self._build_prompt(query, candidates)

        try:
            response = self.llm.invoke(prompt)
            content = getattr(response, "content", response)
            order = self._parse_order(str(content), len(candidates))
        except Exception as e:
            logger.warning(f"LLM rerank failed, falling back to original order: {e}")
            order = list(range(len(candidates)))

        # Pseudo-scores: purely for ordering (1.0 for best, decreasing), not a
        # calibrated relevance score.
        ranked = [(candidates[i], 1.0 - (rank / len(order))) for rank, i in enumerate(order)]
        return ranked[:top_k]

    def _build_prompt(self, query: str, candidates: List[Document]) -> str:
        """Build a prompt listing numbered candidate excerpts for the LLM to rank."""
        lines = [
            "You are ranking search results by relevance to a query.",
            f"Query: {query}",
            "",
            "Candidates:",
        ]
        for i, doc in enumerate(candidates):
            excerpt = doc.page_content.strip().replace("\n", " ")[:MAX_CHARS_PER_CANDIDATE]
            lines.append(f"[{i}] {excerpt}")

        lines.extend([
            "",
            f"Return ONLY a JSON array of the {len(candidates)} candidate indices above, "
            "ordered from most to least relevant to the query. Include every index exactly once.",
            "Example response: [2, 0, 4, 1, 3]",
        ])
        return "\n".join(lines)

    def _parse_order(self, text: str, num_candidates: int) -> List[int]:
        """
        Parse the LLM's response into a valid, complete index order.

        Defensive by design: local/small LLMs can wrap the JSON in prose or markdown
        fences, so we extract the first bracketed list rather than requiring an exact
        JSON-only response. Falls back to the original order if parsing fails or the
        result isn't a valid permutation of range(num_candidates).
        """
        match = re.search(r"\[[\d,\s]*\]", text)
        if match:
            try:
                indices = json.loads(match.group(0))
                if (
                    isinstance(indices, list)
                    and sorted(indices) == list(range(num_candidates))
                ):
                    return indices
            except (json.JSONDecodeError, TypeError):
                pass

        logger.warning(
            f"Could not parse a valid rerank order from LLM response, using original order: {text[:200]!r}"
        )
        return list(range(num_candidates))


# Environment variable keys
RERANK_PROVIDER_ENV = "RERANK_PROVIDER"  # "none" (default), "llm"


def get_rerank_provider() -> Optional[RerankProvider]:
    """
    Get a rerank provider instance based on environment configuration.

    Supported providers:
    - none: No reranking (default) - hybrid search's own ordering is used as-is
    - llm: Reuses the configured LLM (LLM_PROVIDER) to rank candidates

    Environment variables:
    - RERANK_PROVIDER: Provider name (default: "none")

    Returns:
        RerankProvider instance, or None if reranking is disabled
    """
    provider = os.getenv(RERANK_PROVIDER_ENV, "none").lower()

    if provider == "none":
        return None

    elif provider == "llm":
        return LLMRerankProvider()

    # Future providers can be added here:
    # elif provider == "cross_encoder":
    #     return CrossEncoderRerankProvider()

    else:
        raise ValueError(
            f"Unknown rerank provider: {provider}. "
            f"Supported providers: none, llm. "
            f"Set {RERANK_PROVIDER_ENV} environment variable."
        )
