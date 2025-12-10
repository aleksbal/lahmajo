# rag_ollama_demo/cli.py
import argparse
import sys

from app.agent import ask_question


def main():
    parser = argparse.ArgumentParser(description="RAG Agent CLI")
    parser.add_argument("question", nargs="?", default=None, help="Question to ask")

    args = parser.parse_args()

    if args.question:
        answer = ask_question(args.question, show_progress=False)
        print(answer.strip())
        return

    print("RAG Agent - Type 'quit' to exit\n")

    while True:
        try:
            question = input("Question: ").strip()
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                break

            print("Processing...")
            answer = ask_question(question, show_progress=False)
            print("\nAnswer:")
            print(answer.strip())
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)
        except EOFError:
            print("\nGoodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
