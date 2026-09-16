"""Drawing the event stream in a terminal.

The agent's reasoning is the interesting part, and it is invisible unless
something renders it. This turns the event stream into a live trace: the plan,
each tool call and its result, the critic's verdict, and a one-line summary of
what the turn cost. Colour is used only when stdout is a terminal, so piping
the CLI into a file produces clean text.
"""

from __future__ import annotations

import os
import sys
import textwrap
from typing import Optional

from .events import (
    AgentErrorEvent, AgentEvent, BudgetWarning, ModelMessage, PlanReady,
    Reflection, Thought, ToolFinished, ToolStarted, TurnFinished, TurnStarted,
)


class Style:
    """ANSI colours, or empty strings when colour would be noise."""

    def __init__(self, enabled: Optional[bool] = None) -> None:
        if enabled is None:
            enabled = (
                sys.stdout.isatty()
                and os.environ.get("TERM", "") != "dumb"
                and not os.environ.get("NO_COLOR")
            )
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def magenta(self, text: str) -> str:
        return self._wrap("35", text)


class ConsoleRenderer:
    """An event sink that draws the agent's trace as it happens."""

    def __init__(
        self,
        verbose: bool = True,
        show_thoughts: bool = True,
        width: int = 88,
        stream=None,
    ) -> None:
        self.verbose = verbose
        self.show_thoughts = show_thoughts
        self.width = width
        self.stream = stream or sys.stdout
        self.style = Style()

    # ------------------------------------------------------------------
    def __call__(self, event: AgentEvent) -> None:
        handler = getattr(self, f"_on_{event.kind}", None)
        if handler is not None:
            handler(event)

    def _write(self, text: str = "") -> None:
        print(text, file=self.stream, flush=True)

    def _indent(self, text: str, prefix: str = "      ") -> str:
        return textwrap.indent(
            textwrap.fill(text, width=self.width - len(prefix)), prefix
        )

    # ------------------------------------------------------------------
    def _on_plan_ready(self, event: PlanReady) -> None:
        if not self.verbose:
            return
        s = self.style
        self._write(s.magenta(f"  [plan] {event.goal}  ({event.complexity})"))
        for step in event.steps:
            self._write(s.dim(f"      {step}"))
        if event.success_criteria:
            self._write(s.dim("      done when:"))
            for criterion in event.success_criteria:
                self._write(s.dim(f"        - {criterion}"))

    def _on_replanned(self, event: PlanReady) -> None:
        if not self.verbose:
            return
        self._write(self.style.magenta(f"  [replan] {event.goal}"))
        for step in event.steps:
            self._write(self.style.dim(f"      {step}"))

    def _on_thought(self, event: Thought) -> None:
        if not (self.verbose and self.show_thoughts):
            return
        first = event.text.strip().split("\n")[0]
        self._write(self.style.dim(f"  [thinking] {first[:200]}"))

    def _on_model_message(self, event: ModelMessage) -> None:
        if not self.verbose or not event.is_draft:
            return
        self._write(self.style.dim("  [draft written, reviewing it...]"))

    def _on_tool_started(self, event: ToolStarted) -> None:
        if not self.verbose:
            return
        arguments = ", ".join(
            f"{k}={_shorten(v)}" for k, v in event.args.items()
        )
        self._write(
            self.style.cyan(f"  [tool] {event.name}({arguments})")
        )

    def _on_tool_finished(self, event: ToolFinished) -> None:
        if not self.verbose:
            return
        s = self.style
        if event.cached:
            marker, colour = "cached", s.dim
        elif event.ok:
            marker, colour = f"{event.duration:.1f}s", s.green
        else:
            marker, colour = "failed", s.red
        self._write(colour(f"      -> [{marker}] {event.preview}"))

    def _on_reflection(self, event: Reflection) -> None:
        if not self.verbose:
            return
        s = self.style
        if event.verdict == "accept":
            self._write(
                s.green(f"  [review] accepted (confidence {event.confidence:.0%})")
            )
            return
        self._write(s.yellow(f"  [review] revising - pass {event.iteration}"))
        for issue in event.issues[:4]:
            self._write(s.dim(f"      issue: {issue}"))
        for action in event.next_actions[:4]:
            self._write(s.dim(f"      next:  {action}"))

    def _on_budget_warning(self, event: BudgetWarning) -> None:
        self._write(self.style.yellow(f"  [budget] {event.reason} - wrapping up"))

    def _on_error(self, event: AgentErrorEvent) -> None:
        colour = self.style.yellow if event.recoverable else self.style.red
        self._write(colour(f"  [error] {event.message}"))

    def _on_turn_finished(self, event: TurnFinished) -> None:
        if not self.verbose:
            return
        bits = [
            f"{event.steps} step{'s' if event.steps != 1 else ''}",
            f"{event.tool_calls} tool call{'s' if event.tool_calls != 1 else ''}",
            f"{event.elapsed:.1f}s",
        ]
        if event.reflections:
            bits.append(f"{event.reflections} revision(s)")
        if event.thought_tokens:
            bits.append(f"{event.thought_tokens} thinking tokens")
        self._write(self.style.dim("  [" + ", ".join(bits) + "]"))


def _shorten(value: object, limit: int = 60) -> str:
    text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "..."


class StatusRenderer:
    """An event sink for Streamlit's ``st.status`` panel.

    Kept here rather than in ``app.py`` so the Streamlit front end stays a thin
    layer over the same event stream the CLI uses.
    """

    def __init__(self, status, show_thoughts: bool = True) -> None:
        self.status = status
        self.show_thoughts = show_thoughts

    def __call__(self, event: AgentEvent) -> None:
        write = self.status.write

        if event.kind in ("plan_ready", "replanned"):
            label = "Re-planning" if event.kind == "replanned" else "Planning"
            lines = [f"**{label}:** {event.goal}"]
            lines += [f"- {step}" for step in event.steps]
            write("\n".join(lines))

        elif event.kind == "thought" and self.show_thoughts:
            first = event.text.strip().split("\n")[0]
            write(f"*{first[:220]}*")

        elif event.kind == "tool_started":
            arguments = ", ".join(f"{k}={_shorten(v, 80)}" for k, v in event.args.items())
            # A no-argument tool would otherwise render an empty code span.
            write(f"**{event.name}** &nbsp; `{arguments}`" if arguments else f"**{event.name}**")

        elif event.kind == "tool_finished":
            if not event.ok:
                write(f":red[failed] {event.preview}")
            elif event.cached:
                write(":gray[reused an earlier result]")

        elif event.kind == "model_message" and event.is_draft:
            write("Draft written - reviewing it...")

        elif event.kind == "reflection":
            if event.verdict == "accept":
                write(f":green[Review passed] ({event.confidence:.0%} confidence)")
            else:
                issues = "; ".join(event.issues[:3]) or "needs more evidence"
                write(f":orange[Revising:] {issues}")

        elif event.kind == "budget_warning":
            write(f":orange[{event.reason} - wrapping up]")

        elif event.kind == "error":
            write(f":red[{event.message}]")
