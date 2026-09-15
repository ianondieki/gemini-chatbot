"""A typed event stream describing what the agent is doing, as it happens.

The loop never prints. It emits events, and a *sink* decides what to do with
them — the CLI draws them as a live trace, Streamlit writes them into a status
panel, and tests collect them into a list and assert on the sequence. That one
indirection is what lets a single agent implementation serve three front ends.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class AgentEvent:
    """Base event. ``kind`` is a stable string so sinks can switch on it."""

    kind: str = field(init=False, default="event")
    at: float = field(default_factory=time.time, init=False, repr=False)


@dataclass
class TurnStarted(AgentEvent):
    user_message: str
    depth: int = 0

    def __post_init__(self) -> None:
        self.kind = "turn_started"


@dataclass
class PlanReady(AgentEvent):
    goal: str
    complexity: str
    steps: List[str]
    success_criteria: List[str]
    replanned: bool = False

    def __post_init__(self) -> None:
        self.kind = "replanned" if self.replanned else "plan_ready"


@dataclass
class Thought(AgentEvent):
    """A thought summary emitted by the model's extended thinking."""

    text: str
    step: int

    def __post_init__(self) -> None:
        self.kind = "thought"


@dataclass
class ModelMessage(AgentEvent):
    """Prose the model produced — a draft answer, or commentary before a tool."""

    text: str
    step: int
    is_draft: bool = False

    def __post_init__(self) -> None:
        self.kind = "model_message"


@dataclass
class ToolStarted(AgentEvent):
    call_id: str
    name: str
    args: Dict[str, Any]
    step: int

    def __post_init__(self) -> None:
        self.kind = "tool_started"


@dataclass
class ToolFinished(AgentEvent):
    call_id: str
    name: str
    ok: bool
    preview: str
    duration: float
    step: int
    cached: bool = False

    def __post_init__(self) -> None:
        self.kind = "tool_finished"


@dataclass
class Reflection(AgentEvent):
    verdict: str
    confidence: float
    issues: List[str]
    next_actions: List[str]
    iteration: int

    def __post_init__(self) -> None:
        self.kind = "reflection"


@dataclass
class BudgetWarning(AgentEvent):
    reason: str

    def __post_init__(self) -> None:
        self.kind = "budget_warning"


@dataclass
class AgentErrorEvent(AgentEvent):
    message: str
    recoverable: bool = True

    def __post_init__(self) -> None:
        self.kind = "error"


@dataclass
class TurnFinished(AgentEvent):
    answer: str
    steps: int
    tool_calls: int
    reflections: int
    elapsed: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0

    def __post_init__(self) -> None:
        self.kind = "turn_finished"


Sink = Callable[[AgentEvent], None]


def null_sink(_event: AgentEvent) -> None:
    """Drop every event. The default when a caller does not want a trace."""


class EventRecorder:
    """Collect events in a list. Handy in tests and for 'show me the trace' UIs."""

    def __init__(self, forward: Optional[Sink] = None) -> None:
        self.events: List[AgentEvent] = []
        self._forward = forward

    def __call__(self, event: AgentEvent) -> None:
        self.events.append(event)
        if self._forward is not None:
            self._forward(event)

    def kinds(self) -> List[str]:
        return [e.kind for e in self.events]

    def of(self, kind: str) -> List[AgentEvent]:
        return [e for e in self.events if e.kind == kind]

    def clear(self) -> None:
        self.events.clear()


def fan_out(*sinks: Optional[Sink]) -> Sink:
    """Combine several sinks into one; ``None`` entries are ignored."""
    live = [s for s in sinks if s is not None]

    def _emit(event: AgentEvent) -> None:
        for sink in live:
            sink(event)

    return _emit
