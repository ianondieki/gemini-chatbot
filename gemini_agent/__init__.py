"""A deliberate agent on top of Gemini: plan, act, observe, critique, revise.

Quick start::

    from gemini_agent import AgentSession

    session = AgentSession.create()
    result = session.ask("What changed in the news today, and why does it matter?")
    print(result.answer)

Add a trace by passing a sink::

    from gemini_agent import ConsoleRenderer
    session.ask("...", sink=ConsoleRenderer())
"""

from .budget import Budget
from .config import AgentConfig
from .errors import AgentError, ConfigError, LLMError, ToolError
from .events import AgentEvent, EventRecorder, Sink, fan_out, null_sink
from .llm import GeminiLLM, LLMClient, LLMResponse, ToolCall, Usage
from .loop import Agent, ToolContext, TurnResult
from .memory import WorkingMemory
from .prompts import build_system_prompt
from .reasoning import Critique, Plan
from .registry import ToolRegistry, ToolResult, ToolSpec
from .render import ConsoleRenderer
from .session import AgentSession
from .tools import build_registry

__version__ = "2.0.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentError",
    "AgentEvent",
    "AgentSession",
    "Budget",
    "ConfigError",
    "ConsoleRenderer",
    "Critique",
    "EventRecorder",
    "GeminiLLM",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "Plan",
    "Sink",
    "ToolCall",
    "ToolContext",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "TurnResult",
    "Usage",
    "WorkingMemory",
    "build_registry",
    "build_system_prompt",
    "fan_out",
    "null_sink",
]
