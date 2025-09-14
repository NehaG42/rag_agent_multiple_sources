"""
Streamlit app for the multi-source RAG agent with conversation memory.

Run:
  streamlit run rag_agent/app.py

Notes:
- Requires OPENAI_API_KEY in environment (see config.py)
- For web search, set BRAVE_SEARCH_API_KEY
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict
import hashlib

import streamlit as st

# Ensure imports work when running via `streamlit run rag_agent/app.py`
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: F401  # ensure .env is loaded
from agent import agent_executor
from tools.document_tools import doc_index, doc_state

from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


def get_session_history(session_id: str) -> ChatMessageHistory:
    histories: Dict[str, ChatMessageHistory] = st.session_state.setdefault("histories", {})
    if session_id not in histories:
        histories[session_id] = ChatMessageHistory()
    return histories[session_id]


def build_agent_with_memory() -> RunnableWithMessageHistory:
    return RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


def list_available_documents() -> Dict[str, Path]:
    docs_dir = ROOT / "documents"
    docs_dir.mkdir(exist_ok=True)
    allowed = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".html", ".htm"}
    files = {}
    for p in sorted(docs_dir.glob("*")):
        if p.suffix.lower() in allowed and p.is_file():
            files[p.name] = p
    return files


def sidebar():
    st.sidebar.header("Index Documents")

    # 1) Upload new documents into rag_agent/documents
    docs_dir = ROOT / "documents"
    docs_dir.mkdir(exist_ok=True)
    allowed_ext = ["pdf", "docx", "doc", "txt", "md", "csv", "html", "htm"]
    uploads = st.sidebar.file_uploader(
        "Upload documents",
        type=allowed_ext,
        accept_multiple_files=True,
        help="Files are saved to rag_agent/documents",
    )

    def _resolve_collision(path: Path) -> Path:
        if not path.exists():
            return path
        stem, suffix = path.stem, path.suffix
        i = 1
        while True:
            candidate = path.with_name(f"{stem}-{i}{suffix}")
            if not candidate.exists():
                return candidate
            i += 1

    # Track saved uploads to avoid re-saving on each rerun
    saved_hashes = st.session_state.setdefault("uploaded_hashes", set())
    recent_saved_paths = []

    if uploads:
        for f in uploads:
            try:
                data = f.getvalue()
            except Exception:
                data = f.read()
            file_sig = hashlib.sha1(data).hexdigest() + "|" + f.name
            if file_sig in saved_hashes:
                continue  # already saved in this session
            dest = _resolve_collision(docs_dir / f.name)
            try:
                with open(dest, "wb") as out:
                    out.write(data)
                saved_hashes.add(file_sig)
                recent_saved_paths.append(str(dest))
            except Exception as e:
                st.sidebar.warning(f"Failed to save {f.name}: {e}")
        if recent_saved_paths:
            names = [Path(p).name for p in recent_saved_paths]
            st.sidebar.success("Saved: " + ", ".join(names))
            # Keep for easy indexing if user clicks immediately
            st.session_state["recent_saved_paths"] = recent_saved_paths
            st.sidebar.info("Select uploaded files below, then click 'Index Selected'.")

    # 2) Select existing documents for indexing
    files = list_available_documents()
    choices = st.sidebar.multiselect(
        "Select documents to index",
        options=list(files.keys()),
        help="Files from rag_agent/documents",
    )
    urls_csv = st.sidebar.text_input(
        "URLs (optional)",
        placeholder="https://example.com/page1, https://example.com/page2",
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        do_index = st.button("Index Selected", use_container_width=True)
    with col2:
        reset_mem = st.button("Reset Memory", use_container_width=True)

    if reset_mem:
        sid = st.session_state.setdefault("session_id", "streamlit")
        st.session_state.get("histories", {}).pop(sid, None)
        st.success("Conversation memory cleared.")

    if do_index:
        selected_paths = [str(files[name]) for name in choices]
        # If nothing selected, fall back to most recently uploaded in this run
        if not selected_paths:
            selected_paths = st.session_state.get("recent_saved_paths", [])
        files_csv = ",".join(selected_paths)
        try:
            res = doc_index(files_csv=files_csv, urls_csv=urls_csv)
            st.session_state["index_result"] = res
            # Quick status hint
            if doc_state.indexed:
                st.sidebar.info(f"Indexed chunks: {doc_state.chunk_count}")
            else:
                st.sidebar.warning("No documents indexed.")
        except Exception as e:
            st.session_state["index_result"] = f"Indexing failed: {e}"


def render_chat():
    st.title("🤖 RAG Agent — Chat")
    st.caption("Ask about your docs, Wikipedia, ArXiv, LangSmith docs, or the web.")

    if msg := st.session_state.get("index_result"):
        if msg.lower().startswith("indexed"):
            st.success(msg)
        else:
            st.warning(msg)

    sid = st.session_state.setdefault("session_id", "streamlit")
    agent = build_agent_with_memory()

    # Display history
    history = get_session_history(sid)
    for m in history.messages:
        role = "user" if m.type == "human" else "assistant"
        with st.chat_message(role):
            st.write(m.content)

    # Input
    prompt = st.chat_input("Type your message")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        try:
            result = agent.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": sid}},
            )
            output = result.get("output", result)
        except Exception as e:
            output = f"Error: {e}"
        with st.chat_message("assistant"):
            st.write(output)


def main():
    st.set_page_config(page_title="RAG Agent", page_icon="🤖", layout="centered")
    sidebar()
    render_chat()


if __name__ == "__main__":
    main()
