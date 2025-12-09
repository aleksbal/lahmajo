# rag_ollama_demo/cli.py
import argparse
import sys
import termios
import tty

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
        print("🤔 Processing your question...\n")
        answer = ask_question(args.question)
        print("\n=== FINAL ANSWER ===\n")
        print(answer)
        return

    # Interactive mode: wait for user input
    print("RAG Agent - Interactive Mode")
    print("Ask questions about the blog post. Type 'quit' or 'exit' to stop.\n")

    old_settings = None  # Track terminal settings for cleanup
    
    while True:
        try:
            question = input("Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            # Disable input during processing
            print("\n🤔 Processing... (please wait, input disabled during processing)\n")
            
            # Disable terminal echo and input to prevent garbage input
            old_settings = None
            if sys.stdin.isatty():
                try:
                    old_settings = termios.tcgetattr(sys.stdin)
                    # Set terminal to raw mode to prevent input buffering
                    tty.setraw(sys.stdin.fileno())
                except (termios.error, OSError):
                    # If terminal operations fail, continue anyway
                    old_settings = None
            
            try:
                answer = ask_question(question)
            finally:
                # Re-enable terminal
                if old_settings is not None:
                    try:
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                        old_settings = None  # Reset after restore
                    except (termios.error, OSError):
                        pass  # Ignore errors when restoring
            
            print("\n=== FINAL ANSWER ===\n")
            print(answer)
            print()  # Empty line before next prompt

        except KeyboardInterrupt:
            # Restore terminal if interrupted
            if old_settings is not None:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                except (termios.error, OSError):
                    pass
            print("\n\nGoodbye!")
            sys.exit(0)
        except EOFError:
            if old_settings is not None:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                except (termios.error, OSError):
                    pass
            print("\n\nGoodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
