"""
Chat with your PDF - a standalone RAG app (Gemini embeddings + NumPy search)

RAG = Retrieval Augmented Generation:
  1. CHUNK    - split the pdf text into small pieces
  2. EMBED    - turn each chunk into a vector (numbers capturing meaning)
  3. STORE    - keep those vectors in memory
  4. RETRIEVE - embed the question, find the most similar chunks (semantic search)
  5. ANSWER   - hand those chunks to the model so it answers FROM the pdf

This version is hardened for LARGE PDFs: it retries on rate limits, paces the
requests, shows progress, and caps very large books so indexing stays sane.

Run:  streamlit run rag_app.py
"""

import os
import json
import time
import hashlib

import numpy as np
import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from dotenv import load_dotenv

import agent_core as core  # shared Gemini voice helpers (STT/TTS)

load_dotenv()

# CONFIGURATION
CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

# Large-PDF controls
EMBED_BATCH = 10          # chunks per embedding request (smaller = safer)
INTER_BATCH_DELAY = 0.3   # seconds to pause between batches (eases rate limits)
EMBED_RETRIES = 5         # attempts per batch on transient errors
# Safety cap on how many chunks we EMBED. The brute-force NumPy search handles
# far more than this fine — the cap exists purely to keep embedding within
# free-tier quota. Raise it (RAG_MAX_CHUNKS) if your quota allows.
MAX_CHUNKS = int(os.getenv("RAG_MAX_CHUNKS", "1200"))

# On-disk cache so re-uploading the same PDF skips re-embedding (slow + quota).
CACHE_DIR = os.getenv("RAG_CACHE_DIR", ".rag_cache")

SYSTEM_RULE = (
    "You answer questions about an uploaded document. Use ONLY the context "
    "provided. If the answer is not in the context, say you could not find it "
    "in the document. Be clear and concise, and quote figures exactly."
)


@st.cache_resource
def get_client():
    return genai.Client()


# 1. CHUNK
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


# 2. EMBED (hardened: retry + backoff + pacing + progress)
def _embed_batch_with_retry(client, batch, task_type):
    """Embed one batch, retrying on transient rate-limit / overload errors."""
    last_error = None
    for attempt in range(1, EMBED_RETRIES + 1):
        try:
            resp = client.models.embed_content(
                model=EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=EMBED_DIM
                ),
            )
            return [e.values for e in resp.embeddings]
        except genai_errors.APIError as e:
            msg = str(e)
            transient = (
                "429" in msg or "RESOURCE_EXHAUSTED" in msg
                or "rate" in msg.lower() or "quota" in msg.lower()
                or "503" in msg or "UNAVAILABLE" in msg
                or "overloaded" in msg.lower()
            )
            if not transient:
                raise  # a real error (bad request / key) - don't retry
            last_error = e
            if attempt < EMBED_RETRIES:
                time.sleep(2 ** (attempt - 1))  # 1s, 2s, 4s, 8s...
    raise last_error


def embed_texts(client, texts, task_type, progress=None):
    """Embed a list of strings into a NumPy matrix. `progress` is an optional
    callback(done, total) used to drive a Streamlit progress bar."""
    vectors = []
    total = len(texts)
    for i in range(0, total, EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        vectors.extend(_embed_batch_with_retry(client, batch, task_type))
        if progress is not None:
            progress(min(i + EMBED_BATCH, total), total)
        if i + EMBED_BATCH < total:
            time.sleep(INTER_BATCH_DELAY)
    return np.array(vectors, dtype="float32")


# 4. RETRIEVE
def top_k_chunks(query_vec, doc_matrix, k=TOP_K):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    d = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-10)
    sims = d @ q
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]


# 5. ANSWER
def answer_question(client, question, chunks, embeddings):
    q_vec = embed_texts(client, [question], task_type="RETRIEVAL_QUERY")[0]
    idx, scores = top_k_chunks(q_vec, embeddings)
    context = "\n\n---\n\n".join(chunks[i] for i in idx)
    prompt = (
        f"{SYSTEM_RULE}\n\n"
        f"Context from the document:\n{context}\n\n"
        f"Question: {question}"
    )
    resp = client.models.generate_content(
        model=CHAT_MODEL,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
    )
    return resp.text, idx, scores


# 6. EMBEDDING CACHE (so re-uploading the same PDF is instant)
# The key folds in the file id AND the indexing parameters, so changing the
# embed model, dimension, or chunk size correctly invalidates a stale cache.
def _cache_key(file_id):
    raw = f"{file_id}|{EMBED_MODEL}|{EMBED_DIM}|{CHUNK_SIZE}|{CHUNK_OVERLAP}|{MAX_CHUNKS}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_cache(file_id, cache_dir=CACHE_DIR):
    """Return (chunks, embeddings) for this file if cached, else None."""
    base = os.path.join(cache_dir, _cache_key(file_id))
    chunks_path, emb_path = base + ".chunks.json", base + ".emb.npy"
    if not (os.path.exists(chunks_path) and os.path.exists(emb_path)):
        return None
    try:
        with open(chunks_path, encoding="utf-8") as f:
            chunks = json.load(f)
        embeddings = np.load(emb_path)
        return chunks, embeddings
    except Exception:
        return None  # corrupt/partial cache — treat as a miss and re-embed


def save_cache(file_id, chunks, embeddings, cache_dir=CACHE_DIR):
    """Persist chunks + embeddings; best-effort (cache failures never block use)."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        base = os.path.join(cache_dir, _cache_key(file_id))
        with open(base + ".chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f)
        np.save(base + ".emb.npy", embeddings)
    except Exception:
        pass


# UI
# Wrapped in main() so the pure functions above (chunk_text, top_k_chunks, ...)
# can be imported by tests without executing any Streamlit code. `streamlit run
# rag_app.py` runs this file as __main__, so the app still launches normally.
def main():
    st.set_page_config(page_title="Chat with your PDF", page_icon="P", layout="centered")
    st.title("Chat with your PDF")
    st.caption(f"RAG demo - embeddings: {EMBED_MODEL} - answers: {CHAT_MODEL}")

    if not os.getenv("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY not found in .env")
        st.stop()

    client = get_client()

    for key in ("rag_chunks", "rag_emb", "rag_file", "rag_msgs"):
        if key not in st.session_state:
            st.session_state[key] = None if key != "rag_msgs" else []
    if "rag_last_audio_id" not in st.session_state:
        st.session_state.rag_last_audio_id = None  # dedupe recordings

    uploaded = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded is not None:
        file_id = f"{uploaded.name}-{uploaded.size}"
        if st.session_state.rag_file != file_id:
            with st.status("Indexing your PDF...", expanded=True) as status:
                capped = False

                # Fast path: this exact PDF was indexed before — load from disk.
                cached = load_cache(file_id)
                if cached is not None:
                    status.write("Found a cached index — loading instead of re-embedding...")
                    chunks, embeddings = cached
                else:
                    status.write("Reading text from the PDF...")
                    reader = PdfReader(uploaded)
                    raw = "\n".join((page.extract_text() or "") for page in reader.pages)

                    if not raw.strip():
                        status.update(label="No selectable text found", state="error")
                        st.warning(
                            "This PDF has no extractable text (it may be scanned images). "
                            "Try a text-based PDF."
                        )
                        st.stop()

                    status.write("Splitting into chunks...")
                    chunks = chunk_text(raw)

                    # Cap very large documents so embedding stays within free-tier quota.
                    if len(chunks) > MAX_CHUNKS:
                        capped = True
                        chunks = chunks[:MAX_CHUNKS]

                    status.write(f"Embedding {len(chunks)} chunks (this can take a while)...")
                    bar = st.progress(0.0)

                    def _update(done, total):
                        bar.progress(done / total, text=f"Embedded {done}/{total} chunks")

                    try:
                        embeddings = embed_texts(
                            client, chunks, task_type="RETRIEVAL_DOCUMENT", progress=_update
                        )
                    except genai_errors.APIError as e:
                        status.update(label="Embedding failed", state="error")
                        st.error(
                            f"Embedding error after retries: {e}\n\n"
                            "This usually means the free-tier rate limit was hit. "
                            "Try a smaller PDF, or wait a minute and re-upload."
                        )
                        st.stop()

                    save_cache(file_id, chunks, embeddings)

                st.session_state.rag_chunks = chunks
                st.session_state.rag_emb = embeddings
                st.session_state.rag_file = file_id
                st.session_state.rag_msgs = []

                label = f"Indexed {len(chunks)} chunks from {uploaded.name}"
                if capped:
                    label += f" (capped at {MAX_CHUNKS}; later pages not indexed)"
                status.update(label=label, state="complete", expanded=False)

            if capped:
                st.warning(
                    f"This PDF was large, so only the first {MAX_CHUNKS} chunks were "
                    "indexed. Questions about later pages may not be answerable. For a "
                    "full book, a vector database (e.g. ChromaDB) is the next step."
                )

    if st.session_state.rag_chunks is not None:
        st.toggle("🔊 Voice replies", key="rag_voice_on",
                  help="Speak the answer aloud (Gemini text-to-speech).")
        audio_in = st.audio_input("🎙️ Or ask out loud", key="rag_mic")

        for m in st.session_state.rag_msgs:
            with st.chat_message(m["role"]):
                st.markdown(m["text"])

        # Resolve the question from the text box or the microphone. Typed text
        # wins; a new recording is transcribed once (deduped by content hash).
        typed = st.chat_input("Ask a question about the document...")
        q = typed
        if not typed and audio_in is not None:
            audio_bytes = audio_in.getvalue()
            audio_id = hashlib.sha256(audio_bytes).hexdigest()
            if audio_id != st.session_state.rag_last_audio_id:
                st.session_state.rag_last_audio_id = audio_id
                with st.spinner("Transcribing your voice..."):
                    try:
                        spoken = core.transcribe_audio(
                            client, audio_bytes,
                            mime_type=getattr(audio_in, "type", None) or "audio/wav",
                        )
                    except Exception as e:
                        spoken = ""
                        st.warning(f"Could not transcribe the recording: {e}")
                if spoken:
                    q = spoken
                else:
                    st.info("I couldn't make out any words — try again.")

        if q:
            st.session_state.rag_msgs.append({"role": "user", "text": q})
            with st.chat_message("user"):
                st.markdown(q)

            with st.chat_message("assistant"):
                with st.spinner("Searching the document..."):
                    try:
                        answer, idx, scores = answer_question(
                            client, q,
                            st.session_state.rag_chunks,
                            st.session_state.rag_emb,
                        )
                    except genai_errors.APIError as e:
                        answer, idx, scores = f"API error: {e}", [], []

                st.markdown(answer)

                # Speak the answer if voice replies are on (never break on TTS).
                if st.session_state.get("rag_voice_on") and not answer.startswith("API error"):
                    with st.spinner("Generating voice..."):
                        wav = core.synthesize_speech(client, answer)
                    if wav:
                        st.audio(wav, format="audio/wav", autoplay=True)
                    else:
                        st.caption("🔇 Voice reply unavailable right now.")

                if len(idx) > 0:
                    with st.expander("Sources used (retrieved chunks)"):
                        for i, s in zip(idx, scores):
                            st.markdown(f"**Chunk {int(i)}** - similarity {s:.3f}")
                            snippet = st.session_state.rag_chunks[int(i)]
                            st.caption(snippet[:400] + ("..." if len(snippet) > 400 else ""))

            st.session_state.rag_msgs.append({"role": "assistant", "text": answer})
    else:
        st.info("Upload a PDF above to get started.")


if __name__ == "__main__":
    main()