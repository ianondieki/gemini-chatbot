"""A thin, testable wrapper around the Gemini API.

Three things live here that the rest of the package should not have to think
about:

1. **Resilience** — transient 429/503s are retried with exponential backoff,
   then the call falls back to a lighter model rather than failing the turn.
2. **Normalisation** — a raw ``GenerateContentResponse`` becomes an
   ``LLMResponse`` with text, thought summaries, tool calls and token usage
   already pulled apart, so callers never poke at ``candidates[0].content.parts``.
3. **Structured output** — ``generate_json`` asks for a JSON schema and parses
   the reply tolerantly, which is how planning and reflection stay machine
   readable.

``LLMClient`` is a Protocol, so tests substitute a scripted fake and exercise
the whole agent loop with no network and no API key.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from .config import AgentConfig
from .errors import LLMError, ModelBlocked

log = logging.getLogger(__name__)

_TRANSIENT_MARKERS = (
    "429", "503", "500", "unavailable", "overloaded", "high demand",
    "resource_exhausted", "deadline", "timeout", "internal error",
)

_THINKING_MARKERS = ("thinking", "thought")


@dataclass
class ToolCall:
    """One function call the model asked for, normalised out of the response."""

    id: str
    name: str
    args: Dict[str, Any]

    def signature(self) -> str:
        """A stable key for caching and repeat-failure detection."""
        return f"{self.name}({json.dumps(self.args, sort_keys=True, default=str)})"


@dataclass
class Usage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.output_tokens += other.output_tokens
        self.thought_tokens += other.thought_tokens
        self.total_tokens += other.total_tokens
        self.calls += other.calls


@dataclass
class LLMResponse:
    text: str = ""
    thoughts: List[str] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    content: Optional[types.Content] = None
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    finish_reason: str = ""

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(Protocol):
    """The surface the agent depends on. Implemented by GeminiLLM and by fakes."""

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
        ...


def _is_transient(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


def _mentions_thinking(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in _THINKING_MARKERS)


def _extract(response: Any, model: str) -> LLMResponse:
    """Pull a raw Gemini response apart into the shape the agent wants."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        feedback = getattr(response, "prompt_feedback", None)
        reason = getattr(feedback, "block_reason", None) or "no candidates returned"
        raise ModelBlocked(str(reason))

    candidate = candidates[0]
    content = getattr(candidate, "content", None)
    finish = getattr(candidate, "finish_reason", "") or ""

    texts: List[str] = []
    thoughts: List[str] = []
    calls: List[ToolCall] = []

    for index, part in enumerate(getattr(content, "parts", None) or []):
        call = getattr(part, "function_call", None)
        if call is not None:
            calls.append(
                ToolCall(
                    id=getattr(call, "id", None) or f"call_{index}_{call.name}",
                    name=call.name or "",
                    args=dict(call.args or {}),
                )
            )
            continue
        text = getattr(part, "text", None)
        if not text:
            continue
        if getattr(part, "thought", False):
            thoughts.append(text.strip())
        else:
            texts.append(text)

    meta = getattr(response, "usage_metadata", None)
    usage = Usage(
        prompt_tokens=getattr(meta, "prompt_token_count", 0) or 0,
        output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
        thought_tokens=getattr(meta, "thoughts_token_count", 0) or 0,
        total_tokens=getattr(meta, "total_token_count", 0) or 0,
        calls=1,
    )

    return LLMResponse(
        text="".join(texts).strip(),
        thoughts=thoughts,
        tool_calls=calls,
        content=content,
        usage=usage,
        model=model,
        finish_reason=str(finish),
    )


def parse_json_loosely(text: str) -> Optional[Any]:
    """Parse JSON that may be wrapped in prose or a ``` fence.

    Structured-output mode usually returns clean JSON, but the fallback model
    sometimes wraps it. Losing a whole plan to a stray fence would be silly.
    """
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


class GeminiLLM:
    """Gemini-backed :class:`LLMClient` with retries, fallback and usage totals."""

    def __init__(
        self,
        config: AgentConfig,
        client: Optional[Any] = None,
        sleep=time.sleep,
    ) -> None:
        self.config = config
        self.client = client if client is not None else genai.Client()
        self.total_usage = Usage()
        self._sleep = sleep
        # Models that rejected a thinking config once; we stop sending it.
        self._no_thinking: set = set()

    # --- public API ----------------------------------------------------
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
        last_error: Optional[Exception] = None

        for model_name in self._models():
            for attempt in range(1, self.config.max_retries + 1):
                try:
                    config = self._build_config(
                        model_name,
                        system_instruction=system_instruction,
                        tools=tools,
                        response_schema=response_schema,
                        temperature=temperature,
                        thinking_budget=thinking_budget,
                        include_thoughts=include_thoughts,
                    )
                    raw = self.client.models.generate_content(
                        model=model_name, contents=list(contents), config=config
                    )
                    result = _extract(raw, model_name)
                    self.total_usage.add(result.usage)
                    return result
                except genai_errors.APIError as exc:
                    if not _is_transient(exc):
                        # A thinking config the model will not accept is worth
                        # one silent retry without it, rather than a dead turn.
                        if _mentions_thinking(exc) and model_name not in self._no_thinking:
                            log.info("model %s rejected thinking config", model_name)
                            self._no_thinking.add(model_name)
                            continue
                        raise
                    last_error = exc
                    if attempt < self.config.max_retries:
                        self._sleep(self._backoff(attempt))
                except ModelBlocked:
                    raise
                except Exception as exc:  # transport hiccups, not APIError
                    if not _is_transient(exc):
                        raise
                    last_error = exc
                    if attempt < self.config.max_retries:
                        self._sleep(self._backoff(attempt))
            log.warning("model %s exhausted its retries", model_name)

        raise LLMError(
            f"every model was unavailable after {self.config.max_retries} retries "
            f"({last_error})"
        ) from last_error

    def generate_json(
        self,
        contents: Sequence[types.Content],
        *,
        schema: types.Schema,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        thinking_budget: Optional[int] = None,
    ) -> Optional[Any]:
        """Ask for schema-constrained JSON and return it parsed, or ``None``.

        Planning and reflection are advisory: if the model returns something
        unparseable the agent should degrade to plain reasoning, not crash.
        """
        response = self.generate(
            contents,
            system_instruction=system_instruction,
            response_schema=schema,
            temperature=temperature,
            thinking_budget=thinking_budget,
            include_thoughts=False,
        )
        return parse_json_loosely(response.text)

    # --- internals -----------------------------------------------------
    def _models(self) -> List[str]:
        models = [self.config.model]
        if self.config.fallback_model and self.config.fallback_model != self.config.model:
            models.append(self.config.fallback_model)
        return models

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter, so retries do not synchronise."""
        base = self.config.retry_base_delay * (2 ** (attempt - 1))
        return min(30.0, base) * (0.75 + random.random() * 0.5)

    def _build_config(
        self,
        model_name: str,
        *,
        system_instruction: Optional[str],
        tools: Optional[types.Tool],
        response_schema: Optional[types.Schema],
        temperature: Optional[float],
        thinking_budget: Optional[int],
        include_thoughts: Optional[bool],
    ) -> types.GenerateContentConfig:
        kwargs: Dict[str, Any] = {
            "temperature": (
                self.config.temperature if temperature is None else temperature
            ),
        }
        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        if tools is not None:
            kwargs["tools"] = [tools]
            # We drive the loop ourselves; the SDK must not call tools for us.
            kwargs["automatic_function_calling"] = (
                types.AutomaticFunctionCallingConfig(disable=True)
            )

        if response_schema is not None:
            # Structured output and tools are mutually exclusive on Gemini.
            kwargs.pop("tools", None)
            kwargs.pop("automatic_function_calling", None)
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = response_schema

        budget = (
            self.config.thinking_budget if thinking_budget is None else thinking_budget
        )
        thoughts = (
            self.config.include_thoughts
            if include_thoughts is None
            else include_thoughts
        )
        if model_name not in self._no_thinking and budget != 0:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=budget,
                include_thoughts=thoughts,
            )

        return types.GenerateContentConfig(**kwargs)


def user_text(text: str) -> types.Content:
    """Shorthand for a plain user message."""
    return types.Content(role="user", parts=[types.Part(text=text)])


def model_text(text: str) -> types.Content:
    """Shorthand for a plain model message."""
    return types.Content(role="model", parts=[types.Part(text=text)])
