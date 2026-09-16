"""
agent_core — shared building blocks for the Gemini agent.

Both the CLI (chatbot.py) and the Streamlit UI (app.py) are the *same* agent
with different front-ends, so the parts that must stay in lock-step live here:

  - the tool declarations the model is allowed to call
  - the safe calculator (parses to an AST and walks it — never eval())
  - the Tavily web-search formatter
  - generate_with_retry: 503/429 handling with backoff + model fallback
  - history trimming that keeps the conversation valid for Gemini
  - run_agent: the model -> tool -> feed-back -> repeat loop

The handlers (do_search / do_calculate) are pure: they return strings and do
no printing, so each front-end can report progress its own way via a `notify`
callback.
"""

import os
import io
import ast
import wave
import time
import operator

from google.genai import types
from google.genai import errors as genai_errors

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FALLBACK_MODEL = "gemini-2.5-flash-lite"  # tried if the main model is overloaded
MAX_RETRIES = 4                            # attempts per model on transient errors
MAX_HISTORY = 20                           # cap conversation turns kept in memory
MAX_TOOL_ROUNDS = 6                        # safety cap: stop the agent looping forever

# Voice (Gemini-native: same client/key, no extra dependencies)
TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
TTS_VOICE = os.getenv("GEMINI_VOICE", "Kore")  # any Gemini prebuilt voice name
TTS_SAMPLE_RATE = 24000  # Gemini TTS returns 24 kHz, 16-bit, mono PCM

DEFAULT_SYSTEM_PROMPT = """You are a helpful, friendly AI assistant with two tools:
- web_search: use it for recent news, current events, live prices, or anything
  that may have changed after your training cutoff.
- calculate: use it for any arithmetic, so you never guess at numbers.
For general knowledge you already know well, just answer directly.
Always be clear and concise."""


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


def build_config(system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> types.GenerateContentConfig:
    """Build the generate-content config. The system prompt differs per front-end
    (the Streamlit UI gives the agent the persona "Angel"), so it's a parameter."""
    return types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[TOOLS],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


# ─────────────────────────────────────────
# TOOL 1: WEB SEARCH
# ─────────────────────────────────────────
SEARCH_ERROR_PREFIX = "Search error:"


def is_search_error(result: str) -> bool:
    """True if a do_search result represents a failure rather than results."""
    return result.startswith(SEARCH_ERROR_PREFIX)


def do_search(tavily, query: str) -> str:
    """Run a Tavily web search and return formatted results (pure: no printing).

    On failure the result is prefixed with SEARCH_ERROR_PREFIX and instructs the
    model to tell the user the search failed — so a transient Tavily outage
    can't be silently turned into a confidently-wrong "live" answer."""
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
        return (
            f"{SEARCH_ERROR_PREFIX} {e}. The web search did not return results. "
            "Tell the user the live search failed and that you could not retrieve "
            "current information — do not invent an answer from memory."
        )


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
    """Safely evaluate an arithmetic expression (pure: no printing)."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: could not evaluate '{expression}' ({e})"


# ─────────────────────────────────────────
# HISTORY MANAGEMENT
# ─────────────────────────────────────────
# Gemini rejects a conversation that begins with a model turn or with an
# orphaned tool-result (a function_response whose matching call was trimmed
# away). trim_history caps the length AND repairs the leading edge so the
# history always starts with a genuine user message.
def _is_function_response(content) -> bool:
    for part in content.parts or []:
        if getattr(part, "function_response", None) is not None:
            return True
    return False


def _is_valid_first_message(content) -> bool:
    return content.role == "user" and not _is_function_response(content)


def trim_history(history: list, max_history: int = MAX_HISTORY) -> None:
    """Trim `history` IN PLACE: drop oldest turns past the cap, then drop any
    leading turns that aren't a valid first user message."""
    while len(history) > max_history:
        history.pop(0)
    while history and not _is_valid_first_message(history[0]):
        history.pop(0)


# ─────────────────────────────────────────
# RESILIENT MODEL CALL (handles 503/429 with retries + fallback)
# ─────────────────────────────────────────
def _is_transient(msg: str) -> bool:
    low = msg.lower()
    return (
        "503" in msg or "UNAVAILABLE" in msg or "429" in msg
        or "overloaded" in low or "high demand" in low
    )


def generate_with_retry(client, history, config, notify=None):
    """Call Gemini, retrying on transient 503/429 errors and falling back to a
    lighter model if the primary one stays overloaded. `notify` is an optional
    callable(str) used to report retry/fallback status. Raises the last error
    only if every model and attempt fails."""
    last_error = None
    for model_name in (MODEL, FALLBACK_MODEL):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return client.models.generate_content(
                    model=model_name, contents=history, config=config
                )
            except genai_errors.APIError as e:
                if not _is_transient(str(e)):
                    raise  # a real error (bad key, bad request) — don't retry
                last_error = e
                if attempt < MAX_RETRIES:
                    wait = 2 ** (attempt - 1)  # 1s, 2s, 4s...
                    if notify:
                        notify(f"Model busy, retrying in {wait}s "
                               f"(try {attempt}/{MAX_RETRIES})...")
                    time.sleep(wait)
        if notify:
            notify(f"Switching to backup model: {FALLBACK_MODEL}")
    raise last_error


# ─────────────────────────────────────────
# AGENTIC LOOP — model -> tool -> feed back -> repeat
# ─────────────────────────────────────────
def run_agent(client, tavily, history, config, notify=None,
              max_rounds=MAX_TOOL_ROUNDS):
    """Drive the agent until it produces a text answer. Mutates `history` in
    place (appending model turns and tool results) and returns the answer text.

    `notify(str)` reports tool activity and retry status. Capped at
    `max_rounds` tool rounds so a misbehaving model can't loop forever.
    """
    rounds = 0
    while True:
        rounds += 1
        if rounds > max_rounds:
            return "Stopped: too many tool calls in one turn."

        response = generate_with_retry(client, history, config, notify)
        model_content = response.candidates[0].content
        history.append(model_content)

        if not response.function_calls:
            return response.text

        tool_result_parts = []
        for call in response.function_calls:
            args = dict(call.args or {})
            if call.name == "web_search":
                q = args.get("query", "")
                if notify:
                    notify(f"Searching the web — {q}")
                result = do_search(tavily, q)
                if notify and is_search_error(result):
                    notify("⚠️ Web search failed — answering without live results.")
            elif call.name == "calculate":
                expr = args.get("expression", "")
                if notify:
                    notify(f"Newton's Brain calculating — {expr}")
                result = do_calculate(expr)
            else:
                result = f"Unknown tool: {call.name}"
            tool_result_parts.append(
                types.Part.from_function_response(
                    name=call.name, response={"result": result}
                )
            )
        history.append(types.Content(role="user", parts=tool_result_parts))


# ─────────────────────────────────────────
# VOICE (Gemini-native speech-to-text and text-to-speech)
# ─────────────────────────────────────────
def pcm_to_wav(pcm: bytes, sample_rate: int = TTS_SAMPLE_RATE,
               channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw little-endian PCM samples in a WAV container so a browser can
    play them. Pure / stdlib-only — Gemini TTS hands back headerless PCM."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def transcribe_audio(client, audio_bytes: bytes, mime_type: str = "audio/wav",
                     model: str = MODEL) -> str:
    """Speech-to-text via Gemini's audio understanding. Returns the transcript
    (stripped); raises on API error so the caller can report it like any other
    model call."""
    response = client.models.generate_content(
        model=model,
        contents=[
            "Transcribe this audio to text verbatim. Return ONLY the words "
            "spoken, with no commentary, labels, or punctuation you did not hear.",
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
        ],
    )
    return (response.text or "").strip()


def synthesize_speech(client, text: str, voice: str = TTS_VOICE,
                      model: str = TTS_MODEL):
    """Text-to-speech via Gemini. Returns WAV bytes, or None if synthesis is
    unavailable (e.g. the TTS model isn't enabled for this key) — voice output
    is a nicety, so a failure must never break the text reply."""
    try:
        response = client.models.generate_content(
            model=model,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice
                        )
                    )
                ),
            ),
        )
        pcm = response.candidates[0].content.parts[0].inline_data.data
        return pcm_to_wav(pcm) if pcm else None
    except Exception:
        return None
