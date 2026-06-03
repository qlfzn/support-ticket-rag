import argparse
from rag import run_rag_pipeline, run_interactive_chat

def main():
    parser = argparse.ArgumentParser(
        prog="RAG System for Support Ticket Response"
    )
    parser.add_argument(
        "--mode",
        choices=["pipeline", "chat"],
        default="chat",
        help="'pipeline' to initialize embeddings, 'chat' for interactive mode"
    )

    args = parser.parse_args()

    if args.mode == "pipeline":
        print("Running RAG pipeline\n")
        run_rag_pipeline()
    elif args.mode == "chat":
        print("Initialising input field for new ticket\n")
        run_interactive_chat()

if __name__ == "__main__":
    main()