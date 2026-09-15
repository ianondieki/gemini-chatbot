"""Exception types used across the agent package."""

from __future__ import annotations


class AgentError(Exception):
    """Base class for every error this package raises deliberately."""


class ConfigError(AgentError):
    """Configuration is missing or nonsensical (bad env var, absent API key)."""


class LLMError(AgentError):
    """The model could not be called, or returned nothing usable."""


class ModelBlocked(LLMError):
    """The model returned no candidate — usually a safety or recitation block."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"The model returned no answer ({reason}).")
        self.reason = reason


class ToolError(AgentError):
    """A tool failed in a way the model should see and can recover from."""


class ToolNotFound(ToolError):
    """The model asked for a tool that is not registered."""


class InvalidToolArguments(ToolError):
    """The model called a tool with arguments that do not match its schema."""


class DelegationDepthExceeded(AgentError):
    """A sub-agent tried to spawn a sub-agent past the configured depth."""
