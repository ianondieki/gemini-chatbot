"""Configuration for the agent, read once from the environment.

Everything tunable lives here so no module reaches for ``os.getenv`` on its
own. ``AgentConfig.from_env()`` is the single place environment variables are
interpreted, which makes the rest of the package trivially testable: tests just
build an ``AgentConfig(...)`` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Literal

from .errors import ConfigError

PlanningMode = Literal["auto", "always", "never"]

_PLANNING_MODES = ("auto", "always", "never")


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be a whole number, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class AgentConfig:
    """Every knob the agent exposes, with free-tier-friendly defaults."""

    # --- models -------------------------------------------------------
    model: str = "gemini-2.5-flash"
    fallback_model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.3

    # Extended thinking. -1 lets the model decide how long to think, 0 turns
    # thinking off, a positive number caps the thinking tokens.
    thinking_budget: int = -1
    include_thoughts: bool = True

    # Planning and reflection get their own (cheaper, shorter) thinking budget
    # because they are short structured calls, not open-ended reasoning.
    meta_thinking_budget: int = 0

    # --- loop shape ---------------------------------------------------
    planning: PlanningMode = "auto"
    max_steps: int = 12          # model turns inside one user turn
    max_tool_calls: int = 24     # individual tool executions per user turn
    max_reflections: int = 2     # critic rounds after a draft answer
    wall_clock_seconds: float = 180.0

    # --- tool execution ----------------------------------------------
    parallel_tools: bool = True
    max_parallel_tools: int = 4
    tool_timeout_seconds: float = 45.0
    observation_char_limit: int = 6000
    max_repeat_failures: int = 2   # identical failing call before we tell it to stop

    # --- delegation ---------------------------------------------------
    max_delegation_depth: int = 1
    delegate_max_steps: int = 6
    delegate_max_tool_calls: int = 8

    # --- transport resilience ----------------------------------------
    max_retries: int = 4
    retry_base_delay: float = 1.0

    # --- memory -------------------------------------------------------
    max_history_messages: int = 40
    keep_recent_messages: int = 16
    compaction_enabled: bool = True

    # --- tools --------------------------------------------------------
    search_max_results: int = 5
    search_depth: str = "basic"     # "basic" or "advanced"
    fetch_char_limit: int = 8000

    # --- retrieval ----------------------------------------------------
    embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 4
    embed_batch: int = 10
    embed_batch_delay: float = 0.3
    embed_retries: int = 5
    max_chunks: int = 1200

    # Re-uploading the same PDF should not re-pay the embedding cost.
    cache_enabled: bool = True
    cache_dir: str = ".rag_cache"

    # --- voice ---------------------------------------------------------
    stt_model: str = ""            # blank falls back to `model`
    tts_model: str = "gemini-2.5-flash-preview-tts"
    tts_voice: str = "Kore"        # any Gemini prebuilt voice name
    tts_sample_rate: int = 24_000  # Gemini TTS returns 24 kHz 16-bit mono

    def __post_init__(self) -> None:
        if self.planning not in _PLANNING_MODES:
            raise ConfigError(
                f"planning must be one of {_PLANNING_MODES}, got {self.planning!r}"
            )
        if self.max_steps < 1:
            raise ConfigError("max_steps must be at least 1")
        if self.max_tool_calls < 0:
            raise ConfigError("max_tool_calls cannot be negative")
        if self.max_reflections < 0:
            raise ConfigError("max_reflections cannot be negative")
        if self.keep_recent_messages >= self.max_history_messages:
            raise ConfigError(
                "keep_recent_messages must be smaller than max_history_messages"
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError("chunk_overlap must be smaller than chunk_size")
        if self.max_parallel_tools < 1:
            raise ConfigError("max_parallel_tools must be at least 1")

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Build a config from environment variables, falling back to defaults."""
        return cls(
            model=_env_str("GEMINI_MODEL", cls.model),
            fallback_model=_env_str("GEMINI_FALLBACK_MODEL", cls.fallback_model),
            temperature=_env_float("AGENT_TEMPERATURE", cls.temperature),
            thinking_budget=_env_int("AGENT_THINKING_BUDGET", cls.thinking_budget),
            include_thoughts=_env_bool("AGENT_INCLUDE_THOUGHTS", cls.include_thoughts),
            meta_thinking_budget=_env_int(
                "AGENT_META_THINKING_BUDGET", cls.meta_thinking_budget
            ),
            planning=_env_str("AGENT_PLANNING", cls.planning),  # type: ignore[arg-type]
            max_steps=_env_int("AGENT_MAX_STEPS", cls.max_steps),
            max_tool_calls=_env_int("AGENT_MAX_TOOL_CALLS", cls.max_tool_calls),
            max_reflections=_env_int("AGENT_MAX_REFLECTIONS", cls.max_reflections),
            wall_clock_seconds=_env_float(
                "AGENT_WALL_CLOCK_SECONDS", cls.wall_clock_seconds
            ),
            parallel_tools=_env_bool("AGENT_PARALLEL_TOOLS", cls.parallel_tools),
            max_parallel_tools=_env_int(
                "AGENT_MAX_PARALLEL_TOOLS", cls.max_parallel_tools
            ),
            tool_timeout_seconds=_env_float(
                "AGENT_TOOL_TIMEOUT", cls.tool_timeout_seconds
            ),
            max_delegation_depth=_env_int(
                "AGENT_MAX_DELEGATION_DEPTH", cls.max_delegation_depth
            ),
            max_retries=_env_int("AGENT_MAX_RETRIES", cls.max_retries),
            max_history_messages=_env_int(
                "AGENT_MAX_HISTORY", cls.max_history_messages
            ),
            keep_recent_messages=_env_int(
                "AGENT_KEEP_RECENT", cls.keep_recent_messages
            ),
            compaction_enabled=_env_bool(
                "AGENT_COMPACTION", cls.compaction_enabled
            ),
            search_max_results=_env_int("SEARCH_MAX_RESULTS", cls.search_max_results),
            search_depth=_env_str("SEARCH_DEPTH", cls.search_depth),
            embed_model=_env_str("GEMINI_EMBED_MODEL", cls.embed_model),
            retrieval_top_k=_env_int("RAG_TOP_K", cls.retrieval_top_k),
            max_chunks=_env_int("RAG_MAX_CHUNKS", cls.max_chunks),
            cache_enabled=_env_bool("RAG_CACHE", cls.cache_enabled),
            cache_dir=_env_str("RAG_CACHE_DIR", cls.cache_dir),
            stt_model=_env_str("GEMINI_STT_MODEL", cls.stt_model),
            tts_model=_env_str("GEMINI_TTS_MODEL", cls.tts_model),
            tts_voice=_env_str("GEMINI_VOICE", cls.tts_voice),
        )

    def for_delegate(self) -> "AgentConfig":
        """A tighter budget for sub-agents so delegation cannot run away."""
        return replace(
            self,
            planning="never",
            max_steps=self.delegate_max_steps,
            max_tool_calls=self.delegate_max_tool_calls,
            max_reflections=0,
            wall_clock_seconds=min(self.wall_clock_seconds, 90.0),
        )
