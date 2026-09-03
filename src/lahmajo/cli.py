# lahmajo/cli.py
"""CLI interface for RAG system."""
import argparse
import json
import sys
from typing import List, Optional

from langchain_core.documents import Document

from lahmajo.services.rag_service import ask_question
from lahmajo.services.retrieval_service import retrieve_context

# Subcommand names. Anything else in first position is treated as a bare question,
# so the pre-subcommand form `lahmajo "some question"` keeps working.
SUBCOMMANDS = ("ask", "search")

PREVIEW_CHARS = 200

# retrieve_context() returns documents only - the (doc, score) pairs that hybrid
# search and reranking produce are dropped inside it. Rather than invent a score
# column, `search` prints rank order and says where the ordering came from.
NO_SCORE_NOTE = (
    "Results are in retrieval rank order. Relevance scores are not surfaced by "
    "retrieve_context(); use GET /debug/search for per-result scores."
)

EMPTY_INDEX_NOTE = (
    "No documents matched. Note that with the default in-process index "
    "(VECTOR_INDEX_PROVIDER=in_memory) the knowledge base only contains what was "
    "ingested in this same process, so a CLI search sees nothing unless an external "
    "backend (e.g. VECTOR_INDEX_PROVIDER=elasticsearch) is configured."
)


def _positive_int(value: str) -> int:
    """argparse type for --k.

    Without this, `--k -1` is accepted and reaches retrieve_context(), whose
    `filtered_docs[:k]` slice then drops the last chunk and returns the rest -
    silently answering a different question than the one asked.
    """
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or greater, got {number}")
    return number


def _add_retrieval_flags(parser: argparse.ArgumentParser) -> None:
    """Add the retrieval toggles shared by `ask` and `search`.

    These mirror the per-request overrides the API already exposes on /ask and
    /debug/search, so the CLI and the HTTP surface offer the same knobs.
    """
    parser.add_argument(
        "--no-hybrid",
        dest="use_hybrid",
        action="store_false",
        default=True,
        help="Use vector-only search instead of hybrid BM25 + vector.",
    )
    rerank = parser.add_mutually_exclusive_group()
    # default=None means "defer to RERANK_PROVIDER", matching retrieve_context()'s
    # Optional[bool] contract - which is why this cannot be a single store_true.
    rerank.add_argument(
        "--rerank",
        dest="use_rerank",
        action="store_true",
        default=None,
        help="Force reranking on for this call, even if RERANK_PROVIDER=none.",
    )
    rerank.add_argument(
        "--no-rerank",
        dest="use_rerank",
        action="store_false",
        help="Force reranking off for this call, even if RERANK_PROVIDER is set.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="lahmajo",
        description="RAG Agent CLI. Run without arguments for an interactive prompt.",
    )
    subparsers = parser.add_subparsers(dest="command")

    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question and print the answer.",
        description="Ask a question and print the answer.",
    )
    ask_parser.add_argument("question", nargs="?", default=None, help="Question to ask")
    _add_retrieval_flags(ask_parser)

    search_parser = subparsers.add_parser(
        "search",
        help="Retrieve chunks for a query without generating an answer.",
        description=(
            "Run retrieval only and print the chunks that would be handed to the "
            "LLM - the CLI counterpart of GET /debug/search."
        ),
    )
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--k",
        type=_positive_int,
        default=8,
        help=(
            "Maximum number of chunks to return (default: 8, matching "
            "retrieve_context()). Fewer come back if the corpus has less to offer."
        ),
    )
    _add_retrieval_flags(search_parser)
    search_parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit machine-readable JSON, for scripting or diffing two configurations.",
    )

    return parser


def _normalize_argv(argv: List[str]) -> List[str]:
    """Insert an implicit `ask` for the pre-subcommand invocation form.

    `lahmajo "some question"` predates subcommands and must keep working, so a
    first argument that is neither a known subcommand nor an option is treated as
    a question. Options (-h, --help) are left alone for the top-level parser.
    """
    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith("-"):
        return ["ask", *argv]
    return argv


def _source_of(doc: Document) -> str:
    return doc.metadata.get("source", "unknown")


def _format_results(query: str, docs: List[Document]) -> str:
    """Render retrieved chunks for human reading."""
    if not docs:
        return EMPTY_INDEX_NOTE

    sources = sorted({_source_of(doc) for doc in docs})
    lines = [
        "Query: " + query,
        "{} chunk(s) from {} document(s): {}".format(
            len(docs), len(sources), ", ".join(sources)
        ),
        NO_SCORE_NOTE,
        "",
    ]
    for i, doc in enumerate(docs, start=1):
        content = doc.page_content.strip()
        preview = content[:PREVIEW_CHARS].replace("\n", " ")
        if len(content) > PREVIEW_CHARS:
            preview += "..."
        lines.append("[{}] {} ({} chars)".format(i, _source_of(doc), len(doc.page_content)))
        lines.append("    " + preview)
        lines.append("")
    return "\n".join(lines).rstrip()


def _results_as_json(
    query: str,
    docs: List[Document],
    use_hybrid: bool,
    use_rerank: Optional[bool],
    k: int,
) -> str:
    """Render retrieved chunks as JSON."""
    return json.dumps(
        {
            "query": query,
            "k": k,
            "use_hybrid": use_hybrid,
            "use_rerank": use_rerank,
            "count": len(docs),
            "sources": sorted({_source_of(doc) for doc in docs}),
            "results": [
                {
                    "rank": i,
                    "source": _source_of(doc),
                    "length": len(doc.page_content),
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for i, doc in enumerate(docs, start=1)
            ],
        },
        indent=2,
        default=str,
    )


def run_search(args: argparse.Namespace) -> int:
    """Handle `lahmajo search`."""
    _, docs = retrieve_context(
        args.query,
        k=args.k,
        use_hybrid=args.use_hybrid,
        use_rerank=args.use_rerank,
    )

    if args.as_json:
        print(_results_as_json(args.query, docs, args.use_hybrid, args.use_rerank, args.k))
    else:
        print(_format_results(args.query, docs))
    return 0


def run_ask(args: argparse.Namespace) -> int:
    """Handle `lahmajo ask`, including the interactive prompt."""
    if args.question:
        answer = ask_question(
            args.question,
            show_progress=False,
            use_hybrid=args.use_hybrid,
            use_rerank=args.use_rerank,
        )
        print(answer.strip())
        return 0

    return run_repl(args)


def run_repl(args: argparse.Namespace) -> int:
    """Interactive question prompt."""
    print("RAG Agent - Type 'quit' to exit\n")

    while True:
        try:
            question = input("Question: ").strip()
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                break

            print("Processing...")
            answer = ask_question(
                question,
                show_progress=False,
                use_hybrid=args.use_hybrid,
                use_rerank=args.use_rerank,
            )
            print("\nAnswer:")
            print(answer.strip())
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            return 0
        except EOFError:
            print("\nGoodbye!")
            return 0

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(_normalize_argv(raw_argv))

    if args.command == "search":
        return run_search(args)

    if args.command == "ask":
        return run_ask(args)

    # Bare `lahmajo` - interactive prompt, with retrieval defaults.
    defaults = argparse.Namespace(question=None, use_hybrid=True, use_rerank=None)
    return run_repl(defaults)


if __name__ == "__main__":
    sys.exit(main())
