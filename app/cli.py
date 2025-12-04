# rag_ollama_demo/cli.py
import argparse

from app.agent import ask_question


def main():
    parser = argparse.ArgumentParser(
        description="Tiny LangChain + Ollama RAG demo."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="What is task decomposition?",
        help="Question to ask the RAG agent.",
    )

    args = parser.parse_args()
    answer = ask_question(args.question)

    print("\n=== FINAL ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    main()
