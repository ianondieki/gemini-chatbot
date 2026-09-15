"""Shared fixtures: a scripted fake model that needs no API key and no network.

The whole point of ``LLMClient`` being a Protocol is that the agent loop can be
driven by a script. ``FakeLLM`` takes a list of turns - text, tool calls, or a
JSON payload for planning/reflection - and hands them out in order, recording
every request it saw. That makes assertions about *loop behaviour* possible:
that a tool failure is retried, that a critique triggers a second pass, that a
budget stop still produces an answer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gemini_agent.config import AgentConfig          # noqa: E402
from gemini_agent.llm import LLMResponse, ToolCall, Usage  # noqa: E402


def text_turn(text: str) -> Dict[str, Any]:
    return {"text": text}


def tool_turn(*calls: Sequence) -> Dict[str, Any]:
    """``tool_turn(("calculate", {"expression": "2+2"}))``."""
    return {"calls": [(name, dict(args)) for name, args in calls]}


def json_turn(payload: Any) -> Dict[str, Any]:
    import json as _json

    return {"text": _json.dumps(payload), "json": True}


class FakeLLM:
    """A scripted stand-in for :class:`gemini_agent.llm.GeminiLLM`."""

    def __init__(self, script: Optional[List[Dict[str, Any]]] = None) -> None:
        self.script: List[Dict[str, Any]] = list(script or [])
        self.requests: List[Dict[str, Any]] = []
        self.total_usage = Usage()
        self.default_text = "I have finished."

    def queue(self, *turns: Dict[str, Any]) -> "FakeLLM":
        self.script.extend(turns)
        return self

    # --- LLMClient -----------------------------------------------------
    def generate(
        self,
        contents: Sequence[types.Content],
        *,
        system_instruction: Optional[str] = None,
        tools: Optional[types.Tool] = None,
        response_schema: Optional[types.Schema] = None,
        temperature: Optional[float] = None,
        thinking_budget: Optional[int] = None,
        include_thoughts: Optional[bool] = None,
    ) -> LLMResponse:
        self.requests.append(
            {
                "contents": list(contents),
                "system_instruction": system_instruction,
                "tools": tools,
                "response_schema": response_schema,
                "structured": response_schema is not None,
            }
        )

        turn = self._next(
            structured=response_schema is not None, allow_calls=tools is not None
        )
        usage = Usage(prompt_tokens=10, output_tokens=5, total_tokens=15, calls=1)
        self.total_usage.add(usage)

        parts: List[types.Part] = []
        calls: List[ToolCall] = []

        for index, (name, args) in enumerate(turn.get("calls", [])):
            parts.append(
                types.Part(function_call=types.FunctionCall(name=name, args=args))
            )
            calls.append(ToolCall(id=f"call_{len(self.requests)}_{index}", name=name, args=args))

        text = turn.get("text", "")
        if text:
            parts.append(types.Part(text=text))

        thoughts = turn.get("thoughts", [])
        for thought in thoughts:
            parts.insert(0, types.Part(text=thought, thought=True))

        return LLMResponse(
            text=text,
            thoughts=list(thoughts),
            tool_calls=calls,
            content=types.Content(role="model", parts=parts or [types.Part(text="")]),
            usage=usage,
            model="fake",
        )

    def generate_json(
        self,
        contents: Sequence[types.Content],
        *,
        schema: types.Schema,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        thinking_budget: Optional[int] = None,
    ) -> Any:
        from gemini_agent.llm import parse_json_loosely

        response = self.generate(
            contents,
            system_instruction=system_instruction,
            response_schema=schema,
            temperature=temperature,
        )
        return parse_json_loosely(response.text)

    # --- internals -----------------------------------------------------
    def _next(self, structured: bool, allow_calls: bool = True) -> Dict[str, Any]:
        while self.script:
            turn = self.script.pop(0)
            # A structured call consumes only a structured turn, and vice
            # versa, so a script stays readable when planning is on.
            if bool(turn.get("json")) != structured:
                if structured:
                    self.script.insert(0, turn)
                    return {"text": "{}", "json": True}
                continue
            # Gemini cannot return a function call when no tools are declared,
            # so neither may the fake - otherwise a forced-answer call (which
            # passes tools=None) would be served a scripted tool turn.
            if turn.get("calls") and not allow_calls:
                continue
            return turn
        return {"text": "{}" if structured else self.default_text}


@pytest.fixture
def config() -> AgentConfig:
    """A fast, deterministic config: no planning, no reflection, tight budgets."""
    return AgentConfig(
        planning="never",
        max_reflections=0,
        max_steps=6,
        max_tool_calls=8,
        wall_clock_seconds=30.0,
        parallel_tools=False,
        max_history_messages=10,
        keep_recent_messages=4,
    )


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()
