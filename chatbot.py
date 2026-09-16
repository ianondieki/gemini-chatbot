"""
CLI Chatbot — Gemini + Web Search + Calculator (Tavily)

A terminal chatbot powered by Google's Gemini (free tier) with TWO tools:
  - web_search : live web results via Tavily (for current info)
  - calculate  : safe arithmetic (for math)

This version shows the real agentic skill: with more than one tool, the model
must CHOOSE which tool fits the question (or answer directly). The shared
agent_core then ROUTES the request to the matching handler — that routing is
the heart of an agent. This file is just the terminal front-end; the agent
itself lives in agent_core.py (and is shared with the Streamlit UI in app.py).

Run:  python chatbot.py
"""

import os

from google import genai
from google.genai import errors as genai_errors
from tavily import TavilyClient
from dotenv import load_dotenv

import agent_core as core

load_dotenv()

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
CHATBOT_NAME = "Gemini"
USER_NAME = "You"

CONFIG = core.build_config()  # default system prompt
conversation_history = []

client = genai.Client()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def _notify(message: str) -> None:
    """Front-end progress reporter passed into the shared agent loop."""
    print(f"  {message}", flush=True)


# ─────────────────────────────────────────
# CHAT (delegates the agentic loop to agent_core)
# ─────────────────────────────────────────
def chat(user_message: str) -> None:
    conversation_history.append(
        core.types.Content(role="user", parts=[core.types.Part(text=user_message)])
    )
    core.trim_history(conversation_history)
    checkpoint = len(conversation_history)

    try:
        answer = core.run_agent(
            client, tavily, conversation_history, CONFIG, notify=_notify
        )
        print(f"\n{CHATBOT_NAME}: {answer}\n")
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
    print("   🤖  CLI Chatbot — Gemini + Search + Newton's Brain")
    print("=" * 52)
    print(f"  Model: {core.MODEL}")
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
