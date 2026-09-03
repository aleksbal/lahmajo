"""Tests for the CLI argument parsing and the search subcommand.

retrieve_context()/ask_question() are mocked throughout - these tests cover the
CLI's own behaviour (argument wiring, output formatting, backwards compatibility),
not retrieval itself, which tests/test_retrieval_service.py covers.
"""
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from langchain_core.documents import Document

from lahmajo.cli import _format_results, _normalize_argv, build_parser, main


def _doc(content: str, source: str = "doc.txt") -> Document:
    return Document(page_content=content, metadata={"source": source})


class TestArgvNormalization(unittest.TestCase):
    """The pre-subcommand invocation form must keep working."""

    def test_bare_question_gets_implicit_ask(self):
        self.assertEqual(_normalize_argv(["what is this?"]), ["ask", "what is this?"])

    def test_known_subcommand_is_left_alone(self):
        self.assertEqual(_normalize_argv(["search", "query"]), ["search", "query"])
        self.assertEqual(_normalize_argv(["ask", "question"]), ["ask", "question"])

    def test_options_are_left_alone(self):
        # Otherwise `lahmajo --help` would become `lahmajo ask --help`.
        self.assertEqual(_normalize_argv(["--help"]), ["--help"])

    def test_empty_argv_is_left_alone(self):
        self.assertEqual(_normalize_argv([]), [])


class TestParser(unittest.TestCase):
    def test_search_defaults(self):
        args = build_parser().parse_args(["search", "my query"])
        self.assertEqual(args.command, "search")
        self.assertEqual(args.query, "my query")
        self.assertEqual(args.k, 8)
        self.assertTrue(args.use_hybrid)
        self.assertFalse(args.as_json)
        # None means "defer to RERANK_PROVIDER", not "off".
        self.assertIsNone(args.use_rerank)

    def test_search_flags(self):
        args = build_parser().parse_args(
            ["search", "q", "--k", "3", "--no-hybrid", "--rerank", "--json"]
        )
        self.assertEqual(args.k, 3)
        self.assertFalse(args.use_hybrid)
        self.assertTrue(args.use_rerank)
        self.assertTrue(args.as_json)

    def test_no_rerank_is_distinct_from_unset(self):
        self.assertIs(build_parser().parse_args(["search", "q", "--no-rerank"]).use_rerank, False)
        self.assertIsNone(build_parser().parse_args(["search", "q"]).use_rerank)

    def test_rerank_and_no_rerank_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["search", "q", "--rerank", "--no-rerank"])

    def test_k_must_be_positive(self):
        # A negative k would reach retrieve_context()'s `[:k]` slice and quietly
        # drop the last chunk(s) instead of being rejected.
        for bad in ("-1", "0"):
            with self.subTest(k=bad), self.assertRaises(SystemExit):
                build_parser().parse_args(["search", "q", "--k", bad])

    def test_k_must_be_a_number(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["search", "q", "--k", "many"])


class TestSearchCommand(unittest.TestCase):
    @patch("lahmajo.cli.retrieve_context")
    def test_search_passes_flags_through(self, mock_retrieve):
        mock_retrieve.return_value = ("serialized", [_doc("some content")])

        with redirect_stdout(io.StringIO()):
            exit_code = main(["search", "my query", "--k", "3", "--no-hybrid", "--rerank"])

        self.assertEqual(exit_code, 0)
        mock_retrieve.assert_called_once_with(
            "my query", k=3, use_hybrid=False, use_rerank=True
        )

    @patch("lahmajo.cli.retrieve_context")
    def test_search_human_output_lists_ranked_sources(self, mock_retrieve):
        mock_retrieve.return_value = (
            "serialized",
            [_doc("first chunk", "a.txt"), _doc("second chunk", "b.txt")],
        )

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main(["search", "q"])
        output = buffer.getvalue()

        self.assertIn("[1] a.txt", output)
        self.assertIn("[2] b.txt", output)
        self.assertIn("2 chunk(s) from 2 document(s)", output)

    @patch("lahmajo.cli.retrieve_context")
    def test_search_json_output_is_valid_and_ranked(self, mock_retrieve):
        mock_retrieve.return_value = (
            "serialized",
            [_doc("first chunk", "a.txt"), _doc("second chunk", "b.txt")],
        )

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main(["search", "q", "--json"])
        payload = json.loads(buffer.getvalue())

        self.assertEqual(payload["query"], "q")
        self.assertEqual(payload["count"], 2)
        self.assertEqual(payload["sources"], ["a.txt", "b.txt"])
        self.assertEqual([r["rank"] for r in payload["results"]], [1, 2])
        self.assertEqual(payload["results"][0]["content"], "first chunk")
        # Unset --rerank must serialize as null, not false.
        self.assertIsNone(payload["use_rerank"])

    @patch("lahmajo.cli.retrieve_context")
    def test_empty_results_explain_the_in_process_index(self, mock_retrieve):
        mock_retrieve.return_value = ("No relevant documents found...", [])

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = main(["search", "q"])

        self.assertEqual(exit_code, 0)
        self.assertIn("in_memory", buffer.getvalue())

    def test_format_results_truncates_long_previews(self):
        output = _format_results("q", [_doc("x" * 500)])
        self.assertIn("...", output)
        self.assertIn("(500 chars)", output)


class TestAskCommand(unittest.TestCase):
    @patch("lahmajo.cli.ask_question")
    def test_bare_question_still_works(self, mock_ask):
        mock_ask.return_value = "  the answer  "

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = main(["what is this?"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(buffer.getvalue().strip(), "the answer")
        mock_ask.assert_called_once_with(
            "what is this?", show_progress=False, use_hybrid=True, use_rerank=None
        )

    @patch("lahmajo.cli.ask_question")
    def test_explicit_ask_accepts_retrieval_flags(self, mock_ask):
        mock_ask.return_value = "answer"

        with redirect_stdout(io.StringIO()):
            main(["ask", "question", "--no-hybrid", "--no-rerank"])

        mock_ask.assert_called_once_with(
            "question", show_progress=False, use_hybrid=False, use_rerank=False
        )

    @patch("lahmajo.cli.ask_question")
    @patch("builtins.input", side_effect=["a question", "quit"])
    def test_bare_invocation_enters_repl(self, _mock_input, mock_ask):
        mock_ask.return_value = "answer"

        with redirect_stdout(io.StringIO()):
            exit_code = main([])

        self.assertEqual(exit_code, 0)
        mock_ask.assert_called_once()


if __name__ == "__main__":
    unittest.main()
