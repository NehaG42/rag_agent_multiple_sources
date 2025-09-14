"""
Conversational CLI app for the multi-source RAG agent.

This is a copy of the previous app.py so CLI usage remains available.

Usage:
  python -m rag_agent.cli_app
"""

import os
from typing import Dict

from agent import agent_executor
from tools.document_tools import doc_index, doc_state

from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


# In-memory chat history store keyed by session_id
_store: Dict[str, BaseChatMessageHistory] = {}


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


# Wrap the agent with chat history support
conversational_agent = RunnableWithMessageHistory(
    agent_executor,
    _get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


WELCOME = (
    "\n=== Conversational RAG Agent ===\n"
    "Type your questions to chat.\n"
    "Commands:\n"
    "  :index <files_csv> [| <urls_csv>]  - Index local files and/or URLs\n"
    "  :status                            - Check indexing status\n"
    "  :reset                             - Reset conversation memory\n"
    "  :exit                              - Quit\n"
)


def _handle_index(cmd: str) -> None:
    """Parse and run the :index command.

    Syntax:
      :index <files_csv> [| <urls_csv>]

    Examples:
      :index documents/Albert_Malvino.txt,documents/deep-learning-in-python-prerequisites.txt
      :index ./a.pdf,./b.docx | https://example.com/page1,https://example.com/page2
    """
    # Remove leading ':index' and surrounding spaces
    rest = cmd[len(":index"):].strip()
    if not rest:
        print("❌ Provide file paths and/or URLs.")
        print("Example: :index documents/Albert_Malvino.txt,documents/deep-learning-in-python-prerequisites.txt")
        return

    files_csv = rest
    urls_csv = ""
    if "|" in rest:
        parts = rest.split("|", 1)
        files_csv = parts[0].strip()
        urls_csv = parts[1].strip()

    print(f"🔄 Indexing documents...")
    print(f"Files: {files_csv}")
    if urls_csv:
        print(f"URLs: {urls_csv}")

    try:
        result = doc_index(files_csv=files_csv, urls_csv=urls_csv)
        print(f"✅ {result}")

        # Check if indexing was successful
        if doc_state.indexed:
            print(f"📊 Total indexed chunks: {doc_state.chunk_count}")
        else:
            print("⚠️  Indexing completed but no documents were indexed.")

    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        import traceback
        traceback.print_exc()


def _handle_status() -> None:
    """Show current indexing status."""
    if doc_state.indexed and doc_state.vdb is not None:
        print(f"✅ Documents indexed: {doc_state.chunk_count} chunks")
        print("You can now ask questions about the indexed documents!")
    else:
        print("❌ No documents indexed yet.")
        print("Use :index to index some documents first.")


def main() -> None:
    session_id = "default"
    print(WELCOME)

    # Get the current directory and suggest sample docs
    current_dir = os.getcwd()
    sample_docs = "documents/Albert_Malvino.txt,documents/deep-learning-in-python-prerequisites.txt"

    # Check if we're in the rag_agent directory
    if "rag_agent" in current_dir:
        print(f"💡 Tip: Try: :index {sample_docs}")
    else:
        print(f"💡 Tip: Navigate to rag_agent directory and try: :index {sample_docs}")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting.")
            break

        if not user:
            continue

        if user.lower() in {":exit", "exit", ":quit", "quit"}:
            print("👋 Goodbye!")
            break
        if user.startswith(":index"):
            _handle_index(user)
            continue
        if user in {":status", ":info"}:
            _handle_status()
            continue
        if user in {":reset", ":clear"}:
            _store[session_id] = ChatMessageHistory()  # reset memory
            print("🧹 Conversation memory cleared.")
            continue

        try:
            result = conversational_agent.invoke(
                {"input": user},
                config={"configurable": {"session_id": session_id}},
            )
            # AgentExecutor returns a dict with an "output" key
            output = result.get("output", result)
            print(f"🤖 Assistant: {output}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

