"""Conversation memory that survives being trimmed.

The obvious way to cap history is ``history.pop(0)`` in a loop. It is also
wrong: Gemini rejects a transcript whose first message is a function
*response* with no matching call, and dropping a model's ``function_call`` while
keeping its response corrupts the turn. The old code papered over this by
popping until the head looked plausible, which silently threw away good context
and could still strand a call/response pair in the middle.

``WorkingMemory`` fixes both halves:

* it only ever cuts at a **safe boundary** — a real user message that is not a
  tool result — so call/response pairs stay together, and
* instead of deleting the old turns it **summarises** them into a running
  digest that is prepended to the next request, so the agent remembers what
  happened three topics ago rather than forgetting it outright.

It also carries a scratchpad of notes the agent writes to itself, which is what
makes ``remember``/``recall`` work across turns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence

from google.genai import types

from .config import AgentConfig
from .llm import LLMClient, user_text

_SUMMARY_INSTRUCTION = (
    "You compress conversation history. Rewrite the exchange below as a terse "
    "factual digest in under 150 words: what the user wanted, what was found "
    "(keep names, numbers, URLs and dates exactly), and anything still open. "
    "No preamble, no commentary — just the digest."
)


@dataclass
class Note:
    """One fact the agent chose to remember, with an optional topic label."""

    text: str
    topic: str = "general"
    at: float = field(default_factory=time.time)

    def render(self) -> str:
        return f"[{self.topic}] {self.text}"


def is_tool_result(content: Any) -> bool:
    """True when this content carries function responses rather than prose."""
    for part in getattr(content, "parts", None) or []:
        if getattr(part, "function_response", None) is not None:
            return True
    return False


def has_tool_call(content: Any) -> bool:
    """True when the model asked for a tool in this content."""
    for part in getattr(content, "parts", None) or []:
        if getattr(part, "function_call", None) is not None:
            return True
    return False


def is_safe_boundary(content: Any) -> bool:
    """A transcript may start here: a genuine user message, not a tool result."""
    return getattr(content, "role", None) == "user" and not is_tool_result(content)


def estimate_tokens(contents: Iterable[Any]) -> int:
    """Rough token estimate (~4 chars/token) — enough to decide when to compact.

    Deliberately local: asking the API to count tokens on every turn would cost
    a round trip to answer a question we only need approximately.
    """
    chars = 0
    for content in contents:
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                chars += len(text)
            call = getattr(part, "function_call", None)
            if call is not None:
                chars += len(str(getattr(call, "args", "") or "")) + 24
            response = getattr(part, "function_response", None)
            if response is not None:
                chars += len(str(getattr(response, "response", "") or ""))
    return chars // 4


class WorkingMemory:
    """The transcript, the running summary and the scratchpad for one session."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.transcript: List[types.Content] = []
        self.summary: str = ""
        self.notes: List[Note] = []

    # --- transcript ----------------------------------------------------
    def append(self, content: types.Content) -> None:
        if content is not None:
            self.transcript.append(content)

    def append_user(self, text: str) -> None:
        self.append(user_text(text))

    def extend(self, contents: Sequence[types.Content]) -> None:
        for content in contents:
            self.append(content)

    def __len__(self) -> int:
        return len(self.transcript)

    def clear(self) -> None:
        self.transcript.clear()
        self.summary = ""
        self.notes.clear()

    def rollback_to(self, marker: int) -> None:
        """Undo everything appended since ``marker``.

        Used when a turn fails: a half-finished exchange (a tool call with no
        response) would poison every later request.
        """
        del self.transcript[marker:]

    def marker(self) -> int:
        return len(self.transcript)

    # --- notes ---------------------------------------------------------
    def remember(self, text: str, topic: str = "general") -> Note:
        note = Note(text=text.strip(), topic=topic.strip() or "general")
        self.notes.append(note)
        return note

    def recall(self, topic: Optional[str] = None, limit: int = 20) -> List[Note]:
        if topic:
            needle = topic.strip().lower()
            hits = [
                n for n in self.notes
                if needle in n.topic.lower() or needle in n.text.lower()
            ]
        else:
            hits = list(self.notes)
        return hits[-limit:]

    # --- request assembly ----------------------------------------------
    def contents_for_model(
        self, preamble: Optional[str] = None
    ) -> List[types.Content]:
        """The transcript plus any digest/notes, ready to send.

        The digest and notes ride in front as a user message rather than in the
        system prompt so they stay cache-friendly and obviously non-authoritative.
        """
        head: List[types.Content] = []
        context_blocks: List[str] = []

        if self.summary:
            context_blocks.append(f"Summary of earlier conversation:\n{self.summary}")
        if self.notes:
            noted = "\n".join(f"- {n.render()}" for n in self.notes[-20:])
            context_blocks.append(f"Notes you saved earlier:\n{noted}")
        if preamble:
            context_blocks.append(preamble)

        if context_blocks:
            head.append(user_text("\n\n".join(context_blocks)))

        return head + list(self.transcript)

    def estimated_tokens(self) -> int:
        return estimate_tokens(self.contents_for_model())

    # --- compaction -----------------------------------------------------
    def needs_compaction(self) -> bool:
        return (
            self.config.compaction_enabled
            and len(self.transcript) > self.config.max_history_messages
        )

    def compact(self, llm: Optional[LLMClient] = None) -> bool:
        """Summarise the oldest turns away. Returns True if anything changed."""
        if not self.needs_compaction():
            return False

        cut = self._safe_cut_index(
            len(self.transcript) - self.config.keep_recent_messages
        )
        if cut <= 0:
            return False

        older = self.transcript[:cut]
        digest = self._summarise(older, llm)
        if digest:
            self.summary = (
                f"{self.summary}\n{digest}".strip() if self.summary else digest
            )
        del self.transcript[:cut]
        return True

    def _safe_cut_index(self, desired: int) -> int:
        """The first index at or after ``desired`` where the transcript may start.

        Walking *forward* to a safe boundary is the whole trick: it guarantees
        the kept history opens with a real user message and never begins with a
        function response whose call we just deleted.
        """
        if desired <= 0:
            return 0
        for index in range(desired, len(self.transcript)):
            if is_safe_boundary(self.transcript[index]):
                return index
        return 0  # no boundary ahead — keep everything rather than corrupt it

    def _summarise(
        self, contents: Sequence[types.Content], llm: Optional[LLMClient]
    ) -> str:
        transcript = render_transcript(contents)
        if not transcript.strip():
            return ""
        if llm is None:
            return _fallback_digest(contents)
        try:
            response = llm.generate(
                [user_text(f"{_SUMMARY_INSTRUCTION}\n\n{transcript}")],
                temperature=0.0,
                thinking_budget=0,
                include_thoughts=False,
            )
            return response.text.strip() or _fallback_digest(contents)
        except Exception:
            # A failed summary must never cost the user their turn.
            return _fallback_digest(contents)


def _fallback_digest(contents: Sequence[types.Content]) -> str:
    """A deterministic digest for when no model is available (tests, outages)."""
    lines: List[str] = []
    for content in contents:
        role = getattr(content, "role", "?")
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text and not getattr(part, "thought", False):
                flat = " ".join(text.split())
                lines.append(f"{role}: {flat[:200]}")
            call = getattr(part, "function_call", None)
            if call is not None:
                lines.append(f"{role}: called {call.name}")
    return "\n".join(lines[-12:])


def render_transcript(contents: Sequence[types.Content], limit: int = 400) -> str:
    """Flatten contents into readable text — for summaries, logs and `history`."""
    lines: List[str] = []
    for content in contents:
        role = getattr(content, "role", "?")
        pieces: List[str] = []
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "thought", False):
                continue
            text = getattr(part, "text", None)
            if text:
                pieces.append(text.strip())
                continue
            call = getattr(part, "function_call", None)
            if call is not None:
                args = ", ".join(f"{k}={v!r}" for k, v in (call.args or {}).items())
                pieces.append(f"[calls {call.name}({args})]")
                continue
            response = getattr(part, "function_response", None)
            if response is not None:
                payload = (response.response or {}).get("result", "")
                flat = " ".join(str(payload).split())
                pieces.append(
                    f"[{response.name} returned: "
                    f"{flat[:limit]}{'…' if len(flat) > limit else ''}]"
                )
        if pieces:
            lines.append(f"{role}: " + " ".join(pieces))
    return "\n".join(lines)
