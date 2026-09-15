"""Angel - the same agent as chatbot.py, behind a browser UI.

Streamlit re-runs this file top to bottom on every interaction, so everything
that must survive a rerun lives in ``st.session_state`` - including the
``AgentSession``, which owns the conversation, the toolbox and any indexed
documents.

The agent is not reimplemented here. It emits the same event stream the CLI
renders; ``StatusRenderer`` writes those events into the status panel, and the
full trace is kept per message so it can be reopened after the fact.

Run:  streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from gemini_agent import AgentSession, ConfigError, EventRecorder, fan_out
from gemini_agent.render import StatusRenderer

load_dotenv()

st.set_page_config(
    page_title="Angel",
    page_icon="A",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------
# STYLING
# --------------------------------------------------------------------------
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=Outfit:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
      :root {
        --angel-ink:    #2b2a3d;
        --angel-soft:   #6b6a85;
        --angel-glow:   #c8b6ff;
        --angel-sky:    #e7f0ff;
      }
      .stApp {
        background:
          radial-gradient(1100px 500px at 50% -8%, #f3ecff 0%, rgba(243,236,255,0) 60%),
          radial-gradient(900px 500px at 100% 0%, #e7f0ff 0%, rgba(231,240,255,0) 55%),
          radial-gradient(800px 600px at 0% 20%, #fff0f6 0%, rgba(255,240,246,0) 50%),
          linear-gradient(180deg, #fbfaff 0%, #f6f4ff 100%);
      }
      #MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; }
      [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
      }
      .block-container { padding-top: 2.2rem; max-width: 780px; }
      .stApp, .stMarkdown, p, li {
        font-family: 'Outfit', sans-serif; color: var(--angel-ink);
      }
      .angel-hero { text-align: center; margin: 0.5rem 0 1.2rem; }
      .angel-wing {
        font-size: 2.4rem; line-height: 1;
        filter: drop-shadow(0 4px 14px rgba(200,182,255,0.7));
      }
      .angel-title {
        font-family: 'Cormorant Garamond', serif;
        font-weight: 700; font-size: 4.4rem; line-height: 1;
        letter-spacing: 0.5px; margin: 0.2rem 0 0.1rem;
        background: linear-gradient(100deg, #8a6cff 0%, #c86dd7 45%, #ff9bc7 100%);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 6px 30px rgba(200,150,255,0.25);
        animation: rise 0.9s cubic-bezier(.2,.8,.2,1) both;
      }
      .angel-tag {
        font-family: 'Outfit', sans-serif; font-weight: 300; font-size: 1.02rem;
        color: var(--angel-soft); letter-spacing: 2.5px; text-transform: uppercase;
        animation: rise 0.9s 0.1s cubic-bezier(.2,.8,.2,1) both;
      }
      .angel-rule {
        width: 90px; height: 2px; margin: 1rem auto 0;
        background: linear-gradient(90deg, transparent, var(--angel-glow), transparent);
      }
      @keyframes rise {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
      }
      [data-testid="stChatMessage"] {
        border-radius: 20px; padding: 0.4rem; margin-bottom: 0.5rem;
        box-shadow: 0 6px 24px rgba(140,120,200,0.10);
        border: 1px solid rgba(200,182,255,0.25);
        backdrop-filter: blur(6px); background: rgba(255,255,255,0.55);
        animation: rise 0.45s ease both;
      }
      [data-testid="stChatInput"] {
        border-radius: 18px; border: 1px solid rgba(200,182,255,0.5);
        box-shadow: 0 8px 30px rgba(160,130,230,0.15);
        background: rgba(255,255,255,0.75);
      }
      [data-testid="stChatInput"] textarea { font-family: 'Outfit', sans-serif; }
      .stButton button {
        border-radius: 14px; border: 1px solid rgba(200,182,255,0.6);
        background: rgba(255,255,255,0.7); color: var(--angel-ink);
        font-family: 'Outfit', sans-serif; font-weight: 500;
        transition: all 0.2s ease;
      }
      .stButton button:hover {
        border-color: var(--angel-glow);
        box-shadow: 0 4px 16px rgba(160,130,230,0.3);
        transform: translateY(-1px);
      }
      .angel-cards {
        display: flex; gap: 0.7rem; justify-content: center;
        flex-wrap: wrap; margin: 0.2rem 0 1.2rem;
      }
      .angel-card {
        flex: 1 1 170px; max-width: 250px;
        background: rgba(255,255,255,0.6);
        border: 1px solid rgba(200,182,255,0.35);
        border-radius: 18px; padding: 0.9rem 1.1rem;
        box-shadow: 0 6px 22px rgba(140,120,200,0.10);
        backdrop-filter: blur(6px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
      }
      .angel-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 28px rgba(160,130,230,0.22);
      }
      .angel-card-title {
        font-family: 'Cormorant Garamond', serif; font-weight: 600;
        font-size: 1.2rem; margin-bottom: 0.15rem;
      }
      .angel-card-desc {
        font-family: 'Outfit', sans-serif; font-weight: 300;
        font-size: 0.88rem; color: var(--angel-soft);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="angel-hero">
      <div class="angel-wing">&#128330;</div>
      <div class="angel-title">Angel</div>
      <div class="angel-tag">plans &middot; acts &middot; checks its work</div>
      <div class="angel-rule"></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# SESSION
# --------------------------------------------------------------------------
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
            name="Angel", client=get_shared_client()
        )
    return st.session_state.agent


try:
    session = get_session()
except ConfigError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # a bad key surfaces here, not as a stack trace
    st.error(f"Could not start Angel: {type(exc).__name__}: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    st.session_state.indexed = None


# --------------------------------------------------------------------------
# CAPABILITY CARDS
# --------------------------------------------------------------------------
CARD_COPY = {
    "web_search": ("Web search", "Live results for anything current."),
    "fetch_page": ("Read a page", "Opens a source and reads it properly."),
    "calculate": ("Exact maths", "Every figure calculated, never estimated."),
    "current_datetime": ("Today's date", "Knows what day it actually is."),
    "search_documents": ("Your document", "Searches the PDF you upload."),
    "remember": ("Memory", "Keeps what matters as the chat grows."),
    "delegate": ("Sub-agents", "Hands big subtasks to a focused helper."),
}

cards = [CARD_COPY[name] for name in session.tool_names if name in CARD_COPY][:4]
st.markdown(
    '<div class="angel-cards">'
    + "".join(
        f'<div class="angel-card"><div class="angel-card-title">{title}</div>'
        f'<div class="angel-card-desc">{description}</div></div>'
        for title, description in cards
    )
    + "</div>",
    unsafe_allow_html=True,
)

if session.search_client is None:
    st.info("No TAVILY_API_KEY, so Angel cannot search the web this session.")


# --------------------------------------------------------------------------
# CONTROLS
# --------------------------------------------------------------------------
with st.expander("Document and settings", expanded=False):
    uploaded = st.file_uploader(
        "Give Angel a PDF to search", type="pdf",
        help="Retrieval becomes one of Angel's tools, so it can search the "
             "document, calculate with what it finds, and check the web too.",
    )

    if uploaded is not None:
        file_id = f"{uploaded.name}-{uploaded.size}"
        if st.session_state.indexed != file_id:
            with st.status(f"Indexing {uploaded.name}...", expanded=True) as status:
                bar = st.progress(0.0)

                def progress(done: int, total: int) -> None:
                    bar.progress(done / total, text=f"Embedded {done}/{total} chunks")

                try:
                    added = session.load_pdf(uploaded, uploaded.name, progress)
                except Exception as exc:
                    status.update(label="Indexing failed", state="error")
                    st.error(
                        f"{exc}\n\nOn the free tier this is usually the embedding "
                        "rate limit - wait a minute and try again, or use a "
                        "smaller PDF."
                    )
                else:
                    st.session_state.indexed = file_id
                    status.update(
                        label=f"Indexed {added} chunks from {uploaded.name}",
                        state="complete", expanded=False,
                    )

    if session.documents is not None and not session.documents.is_empty:
        st.caption(f"Loaded: {session.documents.describe()}")
        if st.button("Remove document", use_container_width=True):
            session.unload_documents()
            st.session_state.indexed = None
            st.rerun()

    st.divider()

    left, right = st.columns(2)
    with left:
        planning = st.selectbox(
            "Planning",
            ["auto", "always", "never"],
            index=["auto", "always", "never"].index(session.config.planning),
            help="How often Angel drafts an explicit plan before acting.",
        )
    with right:
        reflections = st.slider(
            "Review passes", 0, 3, session.config.max_reflections,
            help="How many times Angel may reject its own draft and try again.",
        )

    if (
        planning != session.config.planning
        or reflections != session.config.max_reflections
    ):
        session.reconfigure(planning=planning, max_reflections=reflections)

    st.caption(
        f"Model {session.config.model} - tools: {', '.join(session.tool_names)}"
    )

_left, _mid, _right = st.columns([1, 2, 1])
with _mid:
    if st.button("Clear conversation", use_container_width=True):
        session.clear()
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------
# CHAT
# --------------------------------------------------------------------------
def render_trace(events) -> None:
    """Replay a finished turn's event stream inside an expander."""
    for event in events:
        if event.kind in ("plan_ready", "replanned"):
            label = "Re-planned" if event.kind == "replanned" else "Planned"
            st.markdown(f"**{label}:** {event.goal}")
            for step in event.steps:
                st.markdown(f"- {step}")
        elif event.kind == "tool_started":
            arguments = ", ".join(f"{k}={v}" for k, v in event.args.items())
            st.markdown(
                f"**{event.name}** `{arguments[:200]}`" if arguments
                else f"**{event.name}**"
            )
        elif event.kind == "tool_finished" and not event.ok:
            st.markdown(f":red[failed] {event.preview}")
        elif event.kind == "reflection":
            if event.verdict == "accept":
                st.markdown(f":green[Review passed] ({event.confidence:.0%})")
            else:
                st.markdown(f":orange[Revised:] {'; '.join(event.issues[:3])}")
        elif event.kind == "turn_finished":
            st.caption(
                f"{event.steps} steps - {event.tool_calls} tool calls - "
                f"{event.elapsed:.1f}s"
            )


if not st.session_state.messages:
    st.markdown(
        "<p style='text-align:center;color:#6b6a85;font-size:1.05rem;'>"
        "Ask about today's news, a calculation worth getting right, or upload "
        "a PDF and ask about that.</p>",
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        if message.get("events"):
            with st.expander("How Angel worked this out"):
                render_trace(message["events"])

if prompt := st.chat_input("Message Angel..."):
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        recorder = EventRecorder()
        with st.status("Angel is thinking...", expanded=True) as status:
            sink = fan_out(recorder, StatusRenderer(status))
            result = session.ask(prompt, sink=sink)
            status.update(
                label="Done" if result.ok else "Something went wrong",
                state="complete" if result.ok else "error",
                expanded=False,
            )
        st.markdown(result.answer)

        # Render the trace for this turn too, not just for past ones - otherwise
        # the message you just sent is the only one you cannot inspect.
        if recorder.events:
            with st.expander("How Angel worked this out"):
                render_trace(recorder.events)

    st.session_state.messages.append(
        {"role": "assistant", "text": result.answer, "events": recorder.events}
    )
