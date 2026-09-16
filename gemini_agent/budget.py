"""Budgets that keep an agentic loop from running forever.

An agent that can call tools in a loop needs hard stops, and it needs them in
more than one dimension: a cheap tool called 200 times is as bad as a loop that
never terminates. ``Budget`` tracks steps, tool calls, reflection rounds and
wall-clock time together and reports *why* it ran out, so the loop can tell the
model "wrap up, you are out of time" instead of dying with a stack trace.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .config import AgentConfig


@dataclass
class Budget:
    max_steps: int
    max_tool_calls: int
    max_reflections: int
    wall_clock_seconds: float

    steps_used: int = field(default=0, init=False)
    tool_calls_used: int = field(default=0, init=False)
    reflections_used: int = field(default=0, init=False)
    started_at: float = field(default_factory=time.monotonic, init=False)

    @classmethod
    def from_config(cls, config: AgentConfig) -> "Budget":
        return cls(
            max_steps=config.max_steps,
            max_tool_calls=config.max_tool_calls,
            max_reflections=config.max_reflections,
            wall_clock_seconds=config.wall_clock_seconds,
        )

    def start(self) -> None:
        """Reset the clock and all counters for a fresh user turn."""
        self.steps_used = 0
        self.tool_calls_used = 0
        self.reflections_used = 0
        self.started_at = time.monotonic()

    # --- clock ---------------------------------------------------------
    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.wall_clock_seconds - self.elapsed)

    # --- checks --------------------------------------------------------
    def exhausted(self) -> Optional[str]:
        """Return a human-readable reason if any limit is spent, else ``None``."""
        if self.steps_used >= self.max_steps:
            return f"reached the {self.max_steps}-step limit for one turn"
        if self.tool_calls_used >= self.max_tool_calls:
            return f"reached the {self.max_tool_calls}-tool-call limit for one turn"
        if self.remaining_seconds <= 0:
            return f"ran out of time after {self.wall_clock_seconds:.0f}s"
        return None

    def tool_calls_left(self) -> int:
        return max(0, self.max_tool_calls - self.tool_calls_used)

    def can_reflect(self) -> bool:
        return (
            self.reflections_used < self.max_reflections
            and self.remaining_seconds > 0
        )

    # --- spending ------------------------------------------------------
    def spend_step(self) -> None:
        self.steps_used += 1

    def spend_tool_calls(self, count: int = 1) -> None:
        self.tool_calls_used += count

    def spend_reflection(self) -> None:
        self.reflections_used += 1

    def summary(self) -> str:
        return (
            f"{self.steps_used}/{self.max_steps} steps, "
            f"{self.tool_calls_used}/{self.max_tool_calls} tool calls, "
            f"{self.elapsed:.1f}s"
        )
