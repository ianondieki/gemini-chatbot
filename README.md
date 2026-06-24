# Gemini CLI Chatbot (with Web Search)

A terminal chatbot powered by Google's **Gemini** API (free tier). Gemini
answers directly when it can, and searches the web (via Tavily) when it needs
current information.

This version implements **manual function-calling** — the core agentic loop:
the model requests a tool, your code runs it, you feed the result back, the
model reasons again. That pattern is the foundation of agentic AI.

## What you need

- **Python 3.10+**
- A **Gemini API key** — https://aistudio.google.com/ → "Get API key" (FREE, no card)
- A **Tavily API key** — https://app.tavily.com/ (free tier)

Check your Python version:

```bash
python --version
```

## Setup (Windows, VS Code terminal)

```powershell
# 1. create and activate a virtual environment
python -m venv venv
venv\Scripts\Activate.ps1          # PowerShell
# venv\Scripts\activate            # if your terminal is cmd

# 2. install dependencies
pip install -r requirements.txt

# 3. create your .env (next to chatbot.py) and add your keys
notepad .env
```

(macOS / Linux: `source venv/bin/activate`)

Your `.env` should contain:

```
GEMINI_API_KEY=your-real-gemini-key
TAVILY_API_KEY=tvly-your-real-key
GEMINI_MODEL=gemini-2.5-flash
```

No spaces around `=`, no quotes, and the file must be named exactly `.env`.

## Run

```powershell
python chatbot.py
```

## Other front-ends

The same agent powers three interfaces. The shared logic — tool definitions,
the safe calculator, web search, retry/fallback, and the agentic loop — lives
in `agent_core.py`, so the CLI and the web UI stay in lock-step.

| File | What it is | Run |
| ---- | ---------- | --- |
| `chatbot.py` | Terminal chatbot (Gemini + search + calculator) | `python chatbot.py` |
| `app.py` | "Angel" — the same agent with a Streamlit web UI | `streamlit run app.py` |
| `rag_app.py` | "Chat with your PDF" — a standalone RAG demo | `streamlit run rag_app.py` |

`chatbot.py` and `app.py` need both `GEMINI_API_KEY` and `TAVILY_API_KEY`.
`rag_app.py` only needs `GEMINI_API_KEY`.

## Voice (Angel web UI)

`app.py` has an optional voice layer, built entirely on Gemini — no extra
dependencies or keys:

- **Speak instead of type** — the 🎙️ recorder transcribes your question with
  Gemini's audio understanding, then runs it through the normal agent.
- **Hear the reply** — toggle **🔊 Voice replies** to have Angel's answers
  spoken back via Gemini text-to-speech.

The voice and model are configurable in `.env` (`GEMINI_VOICE`,
`GEMINI_TTS_MODEL`). If the TTS model isn't enabled for your key, the text
reply still works and voice output degrades quietly.

## Tests

Pure, no-network unit tests cover the security-critical calculator, history
trimming, and the RAG chunking/retrieval math:

```bash
pip install -r requirements.txt
pytest
```

## Commands (inside the chat)

| Command          | Action                       |
| ---------------- | ---------------------------- |
| `quit` / `exit`  | End the chat                 |
| `clear`          | Clear conversation history   |
| `history`        | Show the conversation so far |

## Choosing a model

Default is `gemini-2.5-flash` (fast, free-tier friendly). Change `GEMINI_MODEL`
in `.env`:

- `gemini-2.5-flash` — fast, balanced default
- `gemini-2.5-flash-lite` — lighter / cheapest
- `gemini-2.5-pro` — stronger reasoning

## Notes

- Gemini's free tier has daily limits but no card required — plenty for testing.
- Tavily search uses its own free tier.
- `.env` holds secrets — never commit it. The included `.gitignore` already
  blocks it.
- Conversation history is kept only in memory and resets when you quit.

## Going further

The agentic loop here is provider-agnostic in spirit: model -> tool request ->
run tool -> feed back -> repeat. Once you have an Anthropic API key, the same
loop maps directly onto Claude's tool-use API — a good next exercise.