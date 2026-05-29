"""
Angel - Gemini agent with web search + calculator (Streamlit UI)

Same agent as chatbot.py, with a polished browser interface.
Streamlit re-runs this file top-to-bottom on every interaction, so
conversation state lives in st.session_state. The agentic loop
(model -> tool -> feed back -> repeat) is identical to the CLI version.

Run:  streamlit run app.py
"""

import os
import ast
import time
import operator

import streamlit as st
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# CONFIGURATION
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FALLBACK_MODEL = "gemini-2.5-flash-lite"  # tried if the main model is overloaded
MAX_RETRIES = 4                            # attempts per model on transient errors
MAX_HISTORY = 20
MAX_TOOL_ROUNDS = 6  # safety cap: stop the agent looping forever
APP_NAME = "Angel"

SYSTEM_PROMPT = """You are Angel, a helpful, friendly AI assistant with two tools:
- web_search: use it for recent news, current events, live prices, or anything
  that may have changed after your training cutoff.
- calculate: use it for any arithmetic, so you never guess at numbers.
For general knowledge you already know well, just answer directly.
Always be clear and concise."""


@st.cache_resource
def get_clients():
    return genai.Client(), TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# TOOL DEFINITIONS
TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="web_search",
            description=(
                "Search the web for current, real-time information: recent news, "
                "current events, live prices, sports scores, weather, or anything "
                "that may have changed after your training cutoff."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="The search query to look up",
                    )
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="calculate",
            description=(
                "Evaluate a math expression and return the exact result. Use this "
                "for any arithmetic. Supports + - * / ** % and parentheses."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "expression": types.Schema(
                        type=types.Type.STRING,
                        description="A pure math expression, e.g. '2400 * 0.15'",
                    )
                },
                required=["expression"],
            ),
        ),
    ]
)

CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[TOOLS],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


# TOOL 1: WEB SEARCH
def do_search(tavily, query: str) -> str:
    try:
        results = tavily.search(query=query, max_results=5, include_answer="basic")
        parts = []
        if results.get("answer"):
            parts.append(f"Quick answer: {results['answer']}\n")
        for r in results.get("results", []):
            parts.append(
                f"Title:   {r.get('title', '')}\n"
                f"URL:     {r.get('url', '')}\n"
                f"Summary: {r.get('content', '')}\n"
            )
        return "\n---\n".join(parts) if parts else "No results found."
    except Exception as e:
        return f"Search error: {e}"


# TOOL 2: CALCULATOR (safe - no eval())
_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Only basic arithmetic is allowed")


def do_calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        return f"{expression} = {_safe_eval(tree.body)}"
    except Exception as e:
        return f"Calculation error: could not evaluate '{expression}' ({e})"



# RESILIENT MODEL CALL (handles 503 "overloaded" with retries + fallback)
def generate_with_retry(client, history, status=None):
    """
    Call Gemini, retrying on transient errors (503/429/UNAVAILABLE).
    Backs off a bit each try, and falls back to a lighter model if the
    primary one stays overloaded. Raises the last error only if all fail.
    """
    models_to_try = [MODEL, FALLBACK_MODEL]
    last_error = None

    for model_name in models_to_try:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return client.models.generate_content(
                    model=model_name, contents=history, config=CONFIG
                )
            except genai_errors.APIError as e:
                msg = str(e)
                transient = (
                    "503" in msg or "UNAVAILABLE" in msg
                    or "429" in msg or "overloaded" in msg.lower()
                    or "high demand" in msg.lower()
                )
                if not transient:
                    raise  # a real error (bad key, bad request) - don't retry
                last_error = e
                if attempt < MAX_RETRIES:
                    wait = 2 ** (attempt - 1)  # 1s, 2s, 4s...
                    note = f"Model busy, retrying in {wait}s (try {attempt}/{MAX_RETRIES})..."
                    if status is not None:
                        status.write(note)
                    else:
                        print(f"  {note}", flush=True)
                    time.sleep(wait)
            # after the inner loop: this model failed all retries -> try fallback
        if status is not None:
            status.write(f"Switching to backup model: {FALLBACK_MODEL}")
        else:
            print(f"  Switching to backup model: {FALLBACK_MODEL}", flush=True)

    raise last_error  # everything failed


# AGENTIC TURN
def run_agent(client, tavily, history, status):
    rounds = 0
    while True:
        rounds += 1
        if rounds > MAX_TOOL_ROUNDS:
            return "Stopped: too many tool calls in one turn."

        response = generate_with_retry(client, history, status)
        model_content = response.candidates[0].content
        history.append(model_content)

        if response.function_calls:
            tool_result_parts = []
            for call in response.function_calls:
                args = dict(call.args or {})
                if call.name == "web_search":
                    q = args.get("query", "")
                    status.write(f"Searching the web - *{q}*")
                    result = do_search(tavily, q)
                elif call.name == "calculate":
                    expr = args.get("expression", "")
                    status.write(f"Calculating - *{expr}*")
                    result = do_calculate(expr)
                else:
                    result = f"Unknown tool: {call.name}"
                tool_result_parts.append(
                    types.Part.from_function_response(
                        name=call.name, response={"result": result}
                    )
                )
            history.append(types.Content(role="user", parts=tool_result_parts))
        else:
            return response.text


# PAGE CONFIG + STYLING
st.set_page_config(page_title=APP_NAME, page_icon="A", layout="centered",
                   initial_sidebar_state="expanded")

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

# Main-area Clear button: always visible, including on mobile where the
# sidebar is collapsed. Centered under the hero using a middle column.
_left, _mid, _right = st.columns([1, 2, 1])
with _mid:
    if st.button("Clear conversation", use_container_width=True, key="clear_main"):
        st.session_state.history = []
        st.session_state.display = []
        st.rerun()

# SIDEBAR (extra info; optional on mobile)
with st.sidebar:
    st.markdown("## Angel")
    st.caption(f"Model - {MODEL}")
    st.markdown("**Abilities**")
    st.markdown("Web search\n\nCalculator")
    st.divider()
    st.caption("Angel can search the live web and do exact math. Ask anything.")

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
    while len(st.session_state.history) > MAX_HISTORY:
        st.session_state.history.pop(0)

    with st.chat_message("assistant"):
        with st.status("Angel is thinking...", expanded=True) as status:
            try:
                answer = run_agent(client, tavily, st.session_state.history, status)
                status.update(label="Done", state="complete", expanded=False)
            except genai_errors.APIError as e:
                answer = f"API error: {e}"
                status.update(label="Error", state="error")
            except Exception as e:
                answer = f"Error: {e}"
                status.update(label="Error", state="error")
        st.markdown(answer)

    st.session_state.display.append({"role": "assistant", "text": answer})