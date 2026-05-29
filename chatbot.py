"""
CLI Chatbot — Gemini + Web Search + Calculator (Tavily)

A terminal chatbot powered by Google's Gemini (free tier) with TWO tools:
  - web_search : live web results via Tavily (for current info)
  - calculate  : safe arithmetic (for math)

This version shows the real agentic skill: with more than one tool, the model
must CHOOSE which tool fits the question (or answer directly). Your code then
ROUTES the request to the matching handler — that routing is the heart of an
agent.

Run:  python chatbot.py
"""

import os
import ast
import time
import operator

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FALLBACK_MODEL = "gemini-2.5-flash-lite"  # tried if the main model is overloaded
MAX_RETRIES = 4                            # attempts per model on transient errors

SYSTEM_PROMPT = """You are a helpful, friendly AI assistant with two tools:
- web_search: use it for recent news, current events, live prices, or anything
  that may have changed after your training cutoff.
- calculate: use it for any arithmetic, so you never guess at numbers.
For general knowledge you already know well, just answer directly.
Always be clear and concise."""

CHATBOT_NAME = "Gemini"
USER_NAME = "You"
MAX_HISTORY = 20

client = genai.Client()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# ─────────────────────────────────────────
# TOOL DEFINITIONS — what the model is allowed to ask for
# ─────────────────────────────────────────
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
                "for any arithmetic instead of computing it yourself. Supports "
                "+ - * / ** % and parentheses, e.g. '15/100 * 2400' or '(3+4)**2'."
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

conversation_history = []


# ─────────────────────────────────────────
# TOOL 1: WEB SEARCH
# ─────────────────────────────────────────
def do_search(query: str) -> str:
    """Run a Tavily web search and return formatted results."""
    print(f"\n  🔍 Searching: {query}", flush=True)
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


# ─────────────────────────────────────────
# TOOL 2: CALCULATOR (safe — no eval())
# ─────────────────────────────────────────
# We parse the expression into a syntax tree and walk it ourselves, allowing
# ONLY arithmetic. This is why we never use Python's eval(), which would let a
# crafted string run arbitrary code.
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
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
    """Safely evaluate an arithmetic expression."""
    print(f"\n  🧮 Calculating: {expression}", flush=True)
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: could not evaluate '{expression}' ({e})"


# ─────────────────────────────────────────
# TOOL ROUTER — sends each request to the right handler
# ─────────────────────────────────────────
def handle_tool_call(name: str, args: dict) -> str:
    """The agent's dispatcher: pick the handler that matches the tool name."""
    if name == "web_search":
        return do_search(args.get("query", ""))
    if name == "calculate":
        return do_calculate(args.get("expression", ""))
    return f"Unknown tool: {name}"


# ─────────────────────────────────────────
# HISTORY MANAGEMENT
# ─────────────────────────────────────────
def _is_function_response(content) -> bool:
    for part in content.parts or []:
        if getattr(part, "function_response", None) is not None:
            return True
    return False


def _is_valid_first_message(content) -> bool:
    return content.role == "user" and not _is_function_response(content)


def trim_history() -> None:
    while len(conversation_history) > MAX_HISTORY:
        conversation_history.pop(0)
    while conversation_history and not _is_valid_first_message(conversation_history[0]):
        conversation_history.pop(0)



# ─────────────────────────────────────────
# RESILIENT MODEL CALL (handles 503 with retries + fallback)
# ─────────────────────────────────────────
def generate_with_retry(history):
    """Call Gemini, retrying on transient 503/429 errors and falling back
    to a lighter model if the primary one stays overloaded."""
    last_error = None
    for model_name in (MODEL, FALLBACK_MODEL):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return client.models.generate_content(
                    model=model_name, contents=history, config=CONFIG
                )
            except genai_errors.APIError as e:
                msg = str(e)
                transient = ("503" in msg or "UNAVAILABLE" in msg or "429" in msg
                             or "overloaded" in msg.lower() or "high demand" in msg.lower())
                if not transient:
                    raise
                last_error = e
                if attempt < MAX_RETRIES:
                    wait = 2 ** (attempt - 1)
                    print(f"  ⏳ Model busy, retrying in {wait}s "
                          f"(try {attempt}/{MAX_RETRIES})...", flush=True)
                    time.sleep(wait)
        print(f"  ↪ Switching to backup model: {FALLBACK_MODEL}", flush=True)
    raise last_error


# ─────────────────────────────────────────
# CHAT (agentic loop)
# ─────────────────────────────────────────
def chat(user_message: str) -> None:
    conversation_history.append(
        types.Content(role="user", parts=[types.Part(text=user_message)])
    )
    trim_history()
    checkpoint = len(conversation_history)

    try:
        while True:
            response = generate_with_retry(conversation_history)

            model_content = response.candidates[0].content
            conversation_history.append(model_content)

            if response.function_calls:
                tool_result_parts = []
                for call in response.function_calls:
                    result = handle_tool_call(call.name, dict(call.args or {}))
                    tool_result_parts.append(
                        types.Part.from_function_response(
                            name=call.name,
                            response={"result": result},
                        )
                    )
                conversation_history.append(
                    types.Content(role="user", parts=tool_result_parts)
                )
                # Loop again — Gemini reads the results and continues.
            else:
                print(f"\n{CHATBOT_NAME}: {response.text}\n")
                return

    except genai_errors.APIError as e:
        print(f"\n❌ API error: {e}")
        del conversation_history[checkpoint:]
    except Exception as e:
        print(f"\n❌ Error: {e}")
        del conversation_history[checkpoint:]


# ─────────────────────────────────────────
# CLI HELPERS
# ─────────────────────────────────────────
def print_welcome() -> None:
    print("\n" + "=" * 52)
    print("   🤖  CLI Chatbot — Gemini + Search + Calculator")
    print("=" * 52)
    print(f"  Model: {MODEL}")
    print("  Tools: web_search, calculate")
    print("  Commands:")
    print("    'quit' or 'exit'  → End the chat")
    print("    'clear'           → Clear conversation history")
    print("    'history'         → Show conversation so far")
    print("=" * 52 + "\n")


def _render_message(content) -> str:
    role = USER_NAME if content.role == "user" else CHATBOT_NAME
    pieces = []
    for part in content.parts or []:
        if getattr(part, "text", None):
            pieces.append(part.text)
        elif getattr(part, "function_call", None) is not None:
            fc = part.function_call
            arg_str = ", ".join(f"{k}={v}" for k, v in (fc.args or {}).items())
            pieces.append(f"[called {fc.name}: {arg_str}]")
        elif getattr(part, "function_response", None) is not None:
            pieces.append("[tool result]")
    return f"{role}: " + " ".join(p for p in pieces if p)


def print_history() -> None:
    if not conversation_history:
        print("\n[No conversation history yet]\n")
        return
    print("\n--- Conversation History ---")
    for content in conversation_history:
        print(_render_message(content))
    print("----------------------------\n")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in .env")
        return
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ TAVILY_API_KEY not found in .env")
        return

    print_welcome()

    while True:
        try:
            user_input = input(f"{USER_NAME}: ").strip()
            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd in ("quit", "exit"):
                print("\n👋 Goodbye!\n")
                break
            elif cmd == "clear":
                conversation_history.clear()
                print("\n🗑️  History cleared.\n")
                continue
            elif cmd == "history":
                print_history()
                continue

            chat(user_input)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()