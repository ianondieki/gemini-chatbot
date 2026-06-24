"""
Angel - Gemini agent with web search + calculator (Streamlit UI)

Same agent as chatbot.py, with a polished browser interface.
Streamlit re-runs this file top-to-bottom on every interaction, so
conversation state lives in st.session_state. The agentic loop
(model -> tool -> feed back -> repeat) is identical to the CLI version.

Run:  streamlit run app.py
"""

import os

import streamlit as st
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from tavily import TavilyClient
from dotenv import load_dotenv

import agent_core as core

load_dotenv()

# CONFIGURATION
APP_NAME = "Angel"

SYSTEM_PROMPT = """You are Angel, a helpful, friendly AI assistant with two tools:
- web_search: use it for recent news, current events, live prices, or anything
  that may have changed after your training cutoff.
- calculate: use it for any arithmetic, so you never guess at numbers.
For general knowledge you already know well, just answer directly.
Always be clear and concise."""

# The agent itself (tools, calculator, search, retry/fallback, agentic loop)
# lives in agent_core and is shared with the CLI front-end in chatbot.py.
CONFIG = core.build_config(SYSTEM_PROMPT)


@st.cache_resource
def get_clients():
    return genai.Client(), TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# PAGE CONFIG + STYLING
st.set_page_config(page_title=APP_NAME, page_icon="A", layout="centered",
                   initial_sidebar_state="collapsed")

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
        --angel-blush:  #ffd6e8;
        --angel-sky:    #e7f0ff;
        --angel-cloud:  #faf8ff;
      }
      .stApp {
        background:
          radial-gradient(1100px 500px at 50% -8%, #f3ecff 0%, rgba(243,236,255,0) 60%),
          radial-gradient(900px 500px at 100% 0%, #e7f0ff 0%, rgba(231,240,255,0) 55%),
          radial-gradient(800px 600px at 0% 20%, #fff0f6 0%, rgba(255,240,246,0) 50%),
          linear-gradient(180deg, #fbfaff 0%, #f6f4ff 100%);
      }
      #MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; }
      .block-container { padding-top: 2.2rem; max-width: 760px; }
      .angel-hero { text-align: center; margin: 0.5rem 0 1.6rem; }
      .angel-wing { font-size: 2.4rem; line-height: 1;
        filter: drop-shadow(0 4px 14px rgba(200,182,255,0.7)); }
      .angel-title {
        font-family: 'Cormorant Garamond', serif;
        font-weight: 700; font-size: 4.4rem; line-height: 1;
        letter-spacing: 0.5px; margin: 0.2rem 0 0.1rem;
        color: #a06bff;  /* fallback if background-clip:text is unsupported */
        text-shadow: 0 6px 30px rgba(200,150,255,0.25);
        animation: rise 0.9s cubic-bezier(.2,.8,.2,1) both;
      }
      /* Only make the text transparent where the gradient can actually be
         clipped to it — otherwise the title would vanish (e.g. older Firefox). */
      @supports ((-webkit-background-clip: text) or (background-clip: text)) {
        .angel-title {
          background: linear-gradient(100deg, #8a6cff 0%, #c86dd7 45%, #ff9bc7 100%);
          -webkit-background-clip: text; background-clip: text;
          -webkit-text-fill-color: transparent;
        }
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
      .stApp, .stMarkdown, p, li { font-family: 'Outfit', sans-serif; color: var(--angel-ink); }
      [data-testid="stChatMessage"] {
        border-radius: 20px; padding: 0.4rem 0.4rem; margin-bottom: 0.5rem;
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
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f5f0ff 0%, #eef3ff 100%);
        border-right: 1px solid rgba(200,182,255,0.35);
      }
      [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h1 {
        font-family: 'Cormorant Garamond', serif; color: var(--angel-ink);
      }
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
      /* Hide the sidebar entirely (mobile-friendly: nothing to toggle) */
      [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }
      /* Ability cards in the main area */
      .angel-cards {
        display: flex; gap: 0.8rem; justify-content: center;
        flex-wrap: wrap; margin: 0.2rem 0 1.4rem;
      }
      .angel-card {
        flex: 1 1 200px; max-width: 300px;
        background: rgba(255,255,255,0.6);
        border: 1px solid rgba(200,182,255,0.35);
        border-radius: 18px; padding: 1rem 1.2rem;
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
        font-size: 1.25rem; color: var(--angel-ink); margin-bottom: 0.2rem;
      }
      .angel-card-desc {
        font-family: 'Outfit', sans-serif; font-weight: 300;
        font-size: 0.92rem; color: var(--angel-soft);
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
      <div class="angel-tag">your celestial assistant</div>
      <div class="angel-rule"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# GUARDS + CLIENTS
if not os.getenv("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY not found in .env")
    st.stop()
if not os.getenv("TAVILY_API_KEY"):
    st.error("TAVILY_API_KEY not found in .env")
    st.stop()

client, tavily = get_clients()

if "history" not in st.session_state:
    st.session_state.history = []
if "display" not in st.session_state:
    st.session_state.display = []

# Ability cards in the main area (replaces the sidebar; always visible on mobile)
st.markdown(
    """
    <div class="angel-cards">
      <div class="angel-card">
        <div class="angel-card-title">Web search</div>
        <div class="angel-card-desc">Live results from across the web for anything current.</div>
      </div>
      <div class="angel-card">
        <div class="angel-card-title">Newton&#39;s Brain</div>
        <div class="angel-card-desc">Exact math, calculated with Newton-grade precision.</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Centered Clear button under the cards.
_left, _mid, _right = st.columns([1, 2, 1])
with _mid:
    if st.button("Clear conversation", use_container_width=True, key="clear_main"):
        st.session_state.history = []
        st.session_state.display = []
        st.rerun()

# CHAT
if not st.session_state.display:
    st.markdown(
        "<p style='text-align:center;color:#6b6a85;font-size:1.05rem;'>"
        "Ask me about today's news, a tricky calculation, or anything at all.</p>",
        unsafe_allow_html=True,
    )

for msg in st.session_state.display:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

if prompt := st.chat_input("Message Angel..."):
    st.session_state.display.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.history.append(
        types.Content(role="user", parts=[types.Part(text=prompt)])
    )
    core.trim_history(st.session_state.history)

    with st.chat_message("assistant"):
        with st.status("Angel is thinking...", expanded=True) as status:
            try:
                answer = core.run_agent(
                    client, tavily, st.session_state.history, CONFIG,
                    notify=status.write,
                )
                status.update(label="Done", state="complete", expanded=False)
            except genai_errors.APIError as e:
                answer = f"API error: {e}"
                status.update(label="Error", state="error")
            except Exception as e:
                answer = f"Error: {e}"
                status.update(label="Error", state="error")
        st.markdown(answer)

    st.session_state.display.append({"role": "assistant", "text": answer})