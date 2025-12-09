# rag_ollama_demo/cli.py
import argparse
import sys

from app.agent import ask_question


def main():
    parser = argparse.ArgumentParser(
        description="Tiny LangChain + Ollama RAG demo. Interactive mode by default."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Question to ask the RAG agent (optional, will enter interactive mode if not provided).",
    )

    args = parser.parse_args()

    # If question provided as argument, answer it and exit
    if args.question:
        print("Processing your question...\n")
        answer = ask_question(args.question, show_progress=False)
        print("\n=== FINAL ANSWER ===\n")
        print(answer)
        return

    # Interactive mode: wait for user input
    print("RAG Agent - Interactive Mode")
    print("Ask questions about the blog post. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            print("\nProcessing...\n")
            answer = ask_question(question, show_progress=False)
            print("\n=== FINAL ANSWER ===\n")
            print(answer)
            print()  # Empty line before next prompt

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            sys.exit(0)
        except EOFError:
            print("\n\nGoodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
