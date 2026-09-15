"""Chat with your PDF - retrieval as an agent loop, not a single lookup.

Classic RAG does one pass: embed the question, fetch the nearest chunks, answer
from them. It works until the answer is phrased differently from the question -
ask about "the penalty for filing late" and a section headed "Remedies" is
never retrieved, so the model answers "not in the document" about a document
that plainly covers it.

Here retrieval is a *tool* instead. The agent searches, reads what came back,
notices the gap, and searches again using the document's own vocabulary. It can
also calculate with what it finds, and it reviews its own draft before
answering - so an unsupported claim gets caught rather than shipped.

The pipeline underneath is the same five steps, and they are still worth
knowing: CHUNK, EMBED, STORE, RETRIEVE, ANSWER. They live in
``gemini_agent/rag/``.

Run:  streamlit run rag_app.py
"""

from __future__ import annotations

import hashlib
import os

import streamlit as st
from dotenv import load_dotenv

from gemini_agent import AgentConfig, AgentSession, ConfigError, EventRecorder, fan_out
from gemini_agent.render import StatusRenderer

load_dotenv()

st.set_page_config(page_title="Chat with your PDF", page_icon="P", layout="centered")

DOCUMENT_RULES = """This conversation is about the document the user loaded.

Ground every factual claim in passages from search_documents, and cite the page
you took each figure from. Search more than once when the first passages are
thin - rephrase using the document's own vocabulary rather than the user's.
When the document genuinely does not cover something, say so plainly instead of
reasoning around the gap; if you then add what you know from outside the
document, label it clearly as outside the document."""


@st.cache_resource(show_spinner=False)
def get_shared_client():
    """One HTTP client per server process - safe to share between visitors."""
    from google import genai

    return genai.Client()


def get_session() -> AgentSession:
    """One agent per *browser session*.

    Deliberately not ``@st.cache_resource``: that caches per server process, so
    every visitor would share one conversation, one set of notes and one loaded
    document. Only the transport client below is shared.
    """
    if "agent" not in st.session_state:
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            raise ConfigError(
                "GEMINI_API_KEY is not set. Put it in a .env file next to this "
                "script (see .env.example)."
            )
        st.session_state.agent = AgentSession.create(
            config=AgentConfig.from_env(),
            name="the document assistant",
            client=get_shared_client(),
            enable_search=False,
            enable_delegation=False,
            extra_instructions=DOCUMENT_RULES,
        )
    return st.session_state.agent


st.title("Chat with your PDF")

try:
    session = get_session()
except ConfigError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Could not start: {type(exc).__name__}: {exc}")
    st.stop()

st.caption(
    f"Embeddings: {session.config.embed_model} - answers: {session.config.model} - "
    f"retrieval runs as a tool, so the agent can search repeatedly"
)

if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []
if "rag_file" not in st.session_state:
    st.session_state.rag_file = None
if "rag_last_audio" not in st.session_state:
    st.session_state.rag_last_audio = None

# ---------------------------------------------------------------- INDEXING
uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded is not None:
    file_id = f"{uploaded.name}-{uploaded.size}"
    if st.session_state.rag_file != file_id:
        with st.status(f"Indexing {uploaded.name}...", expanded=True) as status:
            status.write("Reading the PDF and splitting it into chunks...")
            bar = st.progress(0.0)

            def progress(done: int, total: int) -> None:
                bar.progress(done / total, text=f"Embedded {done}/{total} chunks")

            try:
                session.unload_documents()
                added = session.load_pdf(
                    uploaded, uploaded.name, progress, file_id=file_id
                )
            except Exception as exc:
                status.update(label="Indexing failed", state="error")
                st.error(
                    f"{exc}\n\nOn the free tier this is usually the embedding "
                    "rate limit. Wait a minute and re-upload, or try a smaller "
                    "PDF."
                )
                st.stop()

            st.session_state.rag_file = file_id
            st.session_state.rag_messages = []
            session.clear()
            label = (
                f"Loaded {added} cached chunks from {uploaded.name}"
                if session.documents.cache_hit
                else f"Indexed {added} chunks from {uploaded.name}"
            )
            status.update(label=label, state="complete", expanded=False)

        if session.documents.index.truncated:
            st.warning(
                f"That PDF was large, so only the first {session.config.max_chunks} "
                "chunks were indexed - later pages are not searchable. For a "
                "whole book, a real vector database (ChromaDB, pgvector) is the "
                "next step."
            )

# ------------------------------------------------------------------- CHAT
if session.documents is None or session.documents.is_empty:
    st.info("Upload a PDF above to get started.")
    st.stop()

st.caption(f"Loaded: {session.documents.describe()}")

st.toggle(
    "Voice replies",
    key="rag_voice_on",
    help="Read the answer aloud, using Gemini text-to-speech.",
)
recording = st.audio_input("Or ask your question out loud")

for message in st.session_state.rag_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        if message.get("sources"):
            with st.expander("Passages the answer was built from"):
                for source in message["sources"]:
                    st.markdown(f"**{source['label']}**")
                    st.caption(source["text"])

# Typed question wins; a recording is transcribed once, keyed by content hash.
question = st.chat_input("Ask a question about the document...")

if not question and recording is not None:
    audio_bytes = recording.getvalue()
    fingerprint = hashlib.sha256(audio_bytes).hexdigest()
    if fingerprint != st.session_state.rag_last_audio:
        st.session_state.rag_last_audio = fingerprint
        with st.spinner("Transcribing..."):
            try:
                question = session.transcribe(
                    audio_bytes,
                    mime_type=getattr(recording, "type", None) or "audio/wav",
                )
            except Exception as exc:
                st.warning(f"Could not transcribe that recording: {exc}")
        if not question:
            st.info("I could not make out any words in that recording.")

if question:
    st.session_state.rag_messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        recorder = EventRecorder()
        with st.status("Searching the document...", expanded=True) as status:
            sink = fan_out(recorder, StatusRenderer(status, show_thoughts=False))
            result = session.ask(question, sink=sink)
            status.update(
                label=f"Answered after {result.tool_calls} search(es)"
                if result.ok
                else "Something went wrong",
                state="complete" if result.ok else "error",
                expanded=False,
            )

        st.markdown(result.answer)

        # Show the passages the agent actually retrieved, in the order it saw them.
        sources = []
        for event in recorder.of("tool_finished"):
            if event.name == "search_documents" and event.ok:
                query = next(
                    (
                        e.args.get("query", "")
                        for e in recorder.of("tool_started")
                        if e.call_id == event.call_id
                    ),
                    "",
                )
                sources.append({"label": f'search: "{query}"', "text": event.preview})

        if sources:
            with st.expander("Passages the answer was built from"):
                for source in sources:
                    st.markdown(f"**{source['label']}**")
                    st.caption(source["text"])

        if st.session_state.get("rag_voice_on"):
            with st.spinner("Generating voice..."):
                audio = session.speak(result.answer)
            if audio:
                st.audio(audio, format="audio/wav", autoplay=True)
            else:
                st.caption("Voice reply unavailable right now.")

    st.session_state.rag_messages.append(
        {"role": "assistant", "text": result.answer, "sources": sources}
    )
