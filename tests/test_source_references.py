"""Tests for citable source references.

Covers the serialized-context format the LLM is asked to cite against, and the
SourceRef payload that carries those references back out to the API and GUI.
"""
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from lahmajo.services.rag_service import _collect_sources, create_rag_agent
from lahmajo.services.retrieval_service import (
    NO_CONTEXT_MESSAGE,
    PREVIEW_CHARS,
    SourceRef,
    _build_source_refs,
    _serialize_context,
    retrieve_context,
    retrieve_context_with_sources,
)


def _doc(content: str, source: str = "doc.txt") -> Document:
    return Document(page_content=content, metadata={"source": source})


class TestSerializedContext(unittest.TestCase):
    def test_excerpts_are_numbered_from_one(self):
        serialized = _serialize_context([(_doc("alpha", "a.txt"), 0.5), (_doc("beta", "b.txt"), 0.25)])

        self.assertIn("[source 1] a.txt", serialized)
        self.assertIn("[source 2] b.txt", serialized)
        self.assertIn("alpha", serialized)
        self.assertIn("beta", serialized)

    def test_score_is_included_when_available(self):
        self.assertIn("(score: 0.8312)", _serialize_context([(_doc("x"), 0.83124)]))

    def test_score_is_omitted_when_missing(self):
        serialized = _serialize_context([(_doc("x"), None)])
        self.assertIn("[source 1]", serialized)
        self.assertNotIn("score", serialized)

    def test_start_index_continues_numbering(self):
        # A second retrieval call in the same answer must not restart at [source 1].
        serialized = _serialize_context(
            [(_doc("gamma", "c.txt"), 0.5), (_doc("delta", "d.txt"), 0.25)], start_index=3
        )

        self.assertIn("[source 3] c.txt", serialized)
        self.assertIn("[source 4] d.txt", serialized)
        self.assertNotIn("[source 1]", serialized)


class TestSourceRefs(unittest.TestCase):
    def test_indices_are_one_based_and_match_serialization(self):
        scored = [(_doc("alpha", "a.txt"), 0.5), (_doc("beta", "b.txt"), 0.25)]

        refs = _build_source_refs(scored)
        serialized = _serialize_context(scored)

        self.assertEqual([r.index for r in refs], [1, 2])
        for ref in refs:
            self.assertIn(f"[source {ref.index}] {ref.source}", serialized)

    def test_start_index_offsets_refs_to_match_serialization(self):
        scored = [(_doc("gamma", "c.txt"), 0.5), (_doc("delta", "d.txt"), 0.25)]

        refs = _build_source_refs(scored, start_index=3)
        serialized = _serialize_context(scored, start_index=3)

        self.assertEqual([r.index for r in refs], [3, 4])
        for ref in refs:
            self.assertIn(f"[source {ref.index}] {ref.source}", serialized)

    def test_long_preview_is_truncated_but_length_is_the_full_chunk(self):
        content = "x" * (PREVIEW_CHARS + 50)

        ref = _build_source_refs([(_doc(content), None)])[0]

        self.assertTrue(ref.preview.endswith("..."))
        self.assertEqual(ref.length, PREVIEW_CHARS + 50)

    def test_missing_score_is_none_not_zero(self):
        # A null score must not be confused with a score of 0.0.
        self.assertIsNone(_build_source_refs([(_doc("x"), None)])[0].score)

    def test_to_dict_round_trips_all_fields(self):
        ref = SourceRef(index=1, source="a.txt", score=0.5, length=10, preview="hello")
        self.assertEqual(
            ref.to_dict(),
            {"index": 1, "source": "a.txt", "score": 0.5, "length": 10, "preview": "hello"},
        )


class TestRetrieveContextWithSources(unittest.TestCase):
    @patch("lahmajo.services.retrieval_service.HybridRetriever")
    @patch("lahmajo.services.retrieval_service.get_vector_index")
    @patch("lahmajo.services.retrieval_service.get_all_documents")
    def test_scores_survive_from_hybrid_search(self, mock_all_docs, mock_get_index, mock_hybrid_class):
        doc_a = _doc("A" * 150, "a.txt")
        doc_b = _doc("B" * 150, "b.txt")
        mock_all_docs.return_value = [doc_a, doc_b]
        mock_get_index.return_value = MagicMock()
        retriever = MagicMock()
        retriever.search.return_value = [(doc_a, 0.9), (doc_b, 0.4)]
        mock_hybrid_class.return_value = retriever

        serialized, docs, sources = retrieve_context_with_sources("q", k=2)

        self.assertEqual([s.score for s in sources], [0.9, 0.4])
        self.assertEqual([s.source for s in sources], ["a.txt", "b.txt"])
        self.assertEqual(len(docs), 2)
        self.assertIn("[source 1] a.txt", serialized)

    @patch("lahmajo.services.retrieval_service.get_vector_index")
    @patch("lahmajo.services.retrieval_service.get_all_documents")
    def test_no_documents_returns_empty_sources(self, mock_all_docs, mock_get_index):
        mock_all_docs.return_value = []
        index = MagicMock()
        index.similarity_search_with_score.return_value = []
        mock_get_index.return_value = index

        serialized, docs, sources = retrieve_context_with_sources("q")

        self.assertEqual(serialized, NO_CONTEXT_MESSAGE)
        self.assertEqual(docs, [])
        self.assertEqual(sources, [])

    @patch("lahmajo.services.retrieval_service.HybridRetriever")
    @patch("lahmajo.services.retrieval_service.get_vector_index")
    @patch("lahmajo.services.retrieval_service.get_all_documents")
    def test_retrieve_context_wrapper_keeps_its_two_tuple_contract(
        self, mock_all_docs, mock_get_index, mock_hybrid_class
    ):
        doc_a = _doc("A" * 150, "a.txt")
        mock_all_docs.return_value = [doc_a]
        mock_get_index.return_value = MagicMock()
        retriever = MagicMock()
        retriever.search.return_value = [(doc_a, 0.9)]
        mock_hybrid_class.return_value = retriever

        result = retrieve_context("q")

        self.assertEqual(len(result), 2)
        serialized, docs = result
        # The wrapper returns the original document objects, not enriched copies.
        self.assertIs(docs[0], doc_a)
        self.assertIn("[source 1]", serialized)


class TestAgentReferenceNumbering(unittest.TestCase):
    """The retrieval tool numbers references continuously across calls.

    Nothing stops the agent from retrieving twice for one answer, and if each call
    restarted at 1 the answer's `[source 1]` would resolve to two different chunks.
    """

    @staticmethod
    def _fake_retrieve(query, use_hybrid=True, use_rerank=None, start_index=1):
        refs = [
            SourceRef(index=start_index, source=f"{query}-1.txt", score=None, length=5, preview="x"),
            SourceRef(index=start_index + 1, source=f"{query}-2.txt", score=None, length=5, preview="y"),
        ]
        return "ctx", [], refs

    @contextmanager
    def _retrieval_tool(self, side_effect):
        """Yield (retrieve_fn, retrieval_mock) for one agent, with retrieval stubbed.

        The patches must stay live while the tool is *called*, not just while the
        agent is built - the tool resolves retrieve_context_with_sources at call
        time. The tool itself only exists inside create_rag_agent's closure, so it
        is recovered from the create_agent call; its wrapped .func is driven
        directly, since invoking the StructuredTool would drop the artifact under
        test here.
        """
        with patch("lahmajo.services.rag_service.create_agent") as mock_create_agent,                 patch("lahmajo.services.rag_service.get_llm"),                 patch("lahmajo.services.rag_service.retrieve_context_with_sources") as mock_retrieve:
            mock_retrieve.side_effect = side_effect
            create_rag_agent()
            yield mock_create_agent.call_args.kwargs["tools"][0].func, mock_retrieve

    def test_second_call_does_not_restart_at_one(self):
        with self._retrieval_tool(self._fake_retrieve) as (retrieve, _mock):
            first_context, first_refs = retrieve("a")
            _second_context, second_refs = retrieve("b")

        self.assertEqual([r.index for r in first_refs], [1, 2])
        self.assertEqual([r.index for r in second_refs], [3, 4])
        self.assertIn("ctx", first_context)

    def test_flattened_refs_have_unique_indices(self):
        with self._retrieval_tool(self._fake_retrieve) as (retrieve, _mock):
            _, first_refs = retrieve("a")
            _, second_refs = retrieve("b")

        message_a, message_b = MagicMock(), MagicMock()
        message_a.artifact, message_b.artifact = first_refs, second_refs

        indices = [ref.index for ref in _collect_sources([message_a, message_b])]

        self.assertEqual(indices, [1, 2, 3, 4])
        self.assertEqual(len(set(indices)), len(indices))

    def test_empty_retrieval_does_not_consume_indices(self):
        # A call that found nothing returns no refs; the next call still starts at 1.
        side_effect = [(NO_CONTEXT_MESSAGE, [], []), self._fake_retrieve("b")]

        with self._retrieval_tool(side_effect) as (retrieve, mock_retrieve):
            retrieve("a")
            _, refs = retrieve("b")
            second_call = mock_retrieve.call_args_list[1]

        self.assertEqual([r.index for r in refs], [1, 2])
        self.assertEqual(second_call.kwargs["start_index"], 1)

    def test_each_agent_starts_its_own_numbering(self):
        # The counter is per agent, and agents are built per question.
        with self._retrieval_tool(self._fake_retrieve) as (retrieve, _mock):
            retrieve("a")
        with self._retrieval_tool(self._fake_retrieve) as (retrieve, _mock):
            _, refs = retrieve("a")

        self.assertEqual([r.index for r in refs], [1, 2])


class TestCollectSources(unittest.TestCase):
    def test_collects_refs_from_tool_artifacts(self):
        refs = [SourceRef(index=1, source="a.txt", score=0.5, length=10, preview="hi")]
        tool_message = MagicMock()
        tool_message.artifact = refs
        plain_message = MagicMock()
        plain_message.artifact = None

        self.assertEqual(_collect_sources([plain_message, tool_message]), refs)

    def test_ignores_artifacts_that_are_not_source_refs(self):
        # The artifact slot previously carried Documents; don't mistake those for refs.
        message = MagicMock()
        message.artifact = [_doc("not a ref")]

        self.assertEqual(_collect_sources([message]), [])

    def test_no_artifacts_yields_no_sources(self):
        message = MagicMock()
        message.artifact = None

        self.assertEqual(_collect_sources([message]), [])


if __name__ == "__main__":
    unittest.main()
