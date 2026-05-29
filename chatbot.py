"""
CLI Chatbot — Gemini + Web Search (Tavily)

A terminal chatbot powered by Google's Gemini (free tier). Gemini answers
directly when it can, and searches the web (via Tavily) when it needs
current information.

This version wires up function-calling MANUALLY (the agentic loop) instead
of letting the SDK auto-run tools — so you can see the core pattern of
agentic AI: model asks for a tool -> you run it -> you feed the result back
-> model reasons again.

Run:  python chatbot.py
"""

import os

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
# A fast, free-tier-friendly model. Override in .env with GEMINI_MODEL.
# Other options: "gemini-2.5-flash-lite" (lighter/cheaper), "gemini-2.5-pro" (stronger).
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = """You are a helpful, friendly AI assistant with access to web search.
When asked about recent events, current data, news, prices, or anything that
may have changed recently — call the web_search tool to find up-to-date information.
For general knowledge questions you already know well, answer directly.
Always be clear and concise."""

CHATBOT_NAME = "Gemini"
USER_NAME = "You"
MAX_HISTORY = 20  # max number of turns kept in memory

# The SDK picks up GEMINI_API_KEY from the environment automatically.
client = genai.Client()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# ─────────────────────────────────────────
# TOOL DEFINITION — tells Gemini it can search
# ─────────────────────────────────────────
SEARCH_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="web_search",
            description=(
                "Search the web for current, real-time information. "
                "Use this for recent news, current events, live prices, "
                "sports scores, weather, or anything that may have changed "
                "after your training cutoff."
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
        )
    ]
)

# Build the request config once and reuse it.
CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[SEARCH_TOOL],
    # We run tools ourselves, so turn off the SDK's auto-runner.
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

# History is a list of types.Content objects (roles: "user" / "model").
conversation_history = []


# ─────────────────────────────────────────
# WEB SEARCH
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
# HISTORY MANAGEMENT
# ─────────────────────────────────────────
def _is_function_response(content) -> bool:
    """True if this Content carries tool results (not a normal user message)."""
    for part in content.parts or []:
        if getattr(part, "function_response", None) is not None:
            return True
    return False


def _is_valid_first_message(content) -> bool:
    """History must start with a real 'user' message, not orphaned tool output."""
    return content.role == "user" and not _is_function_response(content)


def trim_history() -> None:
    """Keep history bounded while staying valid for the next API call."""
    while len(conversation_history) > MAX_HISTORY:
        conversation_history.pop(0)
    while conversation_history and not _is_valid_first_message(conversation_history[0]):
        conversation_history.pop(0)


# ─────────────────────────────────────────
# CHAT (agentic loop)
# ─────────────────────────────────────────
def chat(user_message: str) -> None:
    """Send a message, run the agentic loop, and print the reply."""
    conversation_history.append(
        types.Content(role="user", parts=[types.Part(text=user_message)])
    )
    trim_history()

    # Checkpoint so we can roll back cleanly if this turn errors out.
    checkpoint = len(conversation_history)

    try:
        while True:
            response = client.models.generate_content(
                model=MODEL,
                contents=conversation_history,
                config=CONFIG,
            )

            # Save the model's full turn (text and/or function-call parts).
            model_content = response.candidates[0].content
            conversation_history.append(model_content)

            # ── Did Gemini ask to use a tool? ──────────────────────────
            if response.function_calls:
                tool_result_parts = []
                for call in response.function_calls:
                    if call.name == "web_search":
                        query = call.args.get("query", "")
                        result = do_search(query)
                    else:
                        result = f"Unknown tool: {call.name}"
                    tool_result_parts.append(
                        types.Part.from_function_response(
                            name=call.name,
                            response={"result": result},
                        )
                    )
                # Feed results back as the next 'user' turn, then loop.
                conversation_history.append(
                    types.Content(role="user", parts=tool_result_parts)
                )
                # Loop again — Gemini reads the results and continues.
            else:
                # ── Final text answer ──────────────────────────────────
                print(f"\n{CHATBOT_NAME}: {response.text}\n")
                return

    except genai_errors.APIError as e:
        print(f"\n❌ API error: {e}")
        del conversation_history[checkpoint:]  # drop this turn's partial state
    except Exception as e:
        print(f"\n❌ Error: {e}")
        del conversation_history[checkpoint:]


# ─────────────────────────────────────────
# CLI HELPERS
# ─────────────────────────────────────────
def print_welcome() -> None:
    print("\n" + "=" * 52)
    print("   🤖  CLI Chatbot — Gemini + Web Search")
    print("=" * 52)
    print(f"  Model: {MODEL}")
    print("  Ask anything! I search the web when needed.")
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
            query = part.function_call.args.get("query", "")
            pieces.append(f"[searched: {query}]")
        elif getattr(part, "function_response", None) is not None:
            pieces.append("[search results]")
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