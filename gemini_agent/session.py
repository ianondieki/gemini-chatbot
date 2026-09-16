"""One wired-up session: model, tools, memory and agent, ready to talk to.

Front ends (the CLI, Streamlit, a test) should not each have to know how the
pieces fit together. ``AgentSession.create()`` builds the lot from the
environment, and ``session.ask(message)`` runs a turn. Loading a document
rewires the toolbox and the system prompt so retrieval appears as a real tool
rather than a separate mode.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, List, Optional

from .config import AgentConfig
from .errors import ConfigError
from .events import Sink
from .llm import GeminiLLM, LLMClient
from .loop import Agent, ToolContext, TurnResult
from .memory import WorkingMemory, render_transcript
from .prompts import build_system_prompt
from .rag import DocumentLibrary, Embedder
from .registry import ToolRegistry
from .tools import build_registry
from .tools.search import make_search_client
from . import voice


class AgentSession:
    """A conversation: persistent memory, a live toolbox, and an agent."""

    def __init__(
        self,
        llm: LLMClient,
        config: AgentConfig,
        *,
        name: str = "Angel",
        search_client: Any = None,
        documents: Optional[DocumentLibrary] = None,
        enable_delegation: bool = True,
        extra_instructions: str = "",
    ) -> None:
        self.llm = llm
        self.config = config
        self.name = name
        self.search_client = search_client
        self.documents = documents
        self.enable_delegation = enable_delegation
        self.extra_instructions = extra_instructions

        self.memory = WorkingMemory(config)
        self.registry: ToolRegistry = ToolRegistry()
        self.system_prompt = ""
        self.agent: Agent = None  # type: ignore[assignment]
        self._rebuild()

    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        config: Optional[AgentConfig] = None,
        *,
        name: str = "Angel",
        enable_search: bool = True,
        enable_documents: bool = True,
        enable_delegation: bool = True,
        extra_instructions: str = "",
        client: Any = None,
    ) -> "AgentSession":
        """Build a session from the environment, failing loudly on a missing key."""
        config = config or AgentConfig.from_env()

        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            raise ConfigError(
                "GEMINI_API_KEY is not set. Put it in a .env file next to this "
                "script (see .env.example), or export it in your shell."
            )

        llm = GeminiLLM(config, client=client)
        search_client = (
            make_search_client(os.getenv("TAVILY_API_KEY")) if enable_search else None
        )
        documents = (
            DocumentLibrary(Embedder(llm.client, config), config)
            if enable_documents
            else None
        )
        return cls(
            llm,
            config,
            name=name,
            search_client=search_client,
            documents=documents,
            enable_delegation=enable_delegation,
            extra_instructions=extra_instructions,
        )

    # ------------------------------------------------------------------
    def _rebuild(self) -> None:
        """Re-assemble the toolbox and prompt to match current capabilities."""
        has_documents = self.documents is not None and not self.documents.is_empty
        self.registry = build_registry(
            search_client=self.search_client,
            with_documents=has_documents,
            with_delegation=self.enable_delegation,
        )
        self.system_prompt = build_system_prompt(
            self.registry, name=self.name, extra=self.extra_instructions or None
        )

        context = ToolContext(
            config=self.config,
            memory=self.memory,
            search_client=self.search_client,
            documents=self.documents,
        )
        self.agent = Agent(
            llm=self.llm,
            registry=self.registry,
            config=self.config,
            memory=self.memory,
            system_prompt=self.system_prompt,
            context=context,
        )

    # ------------------------------------------------------------------
    def ask(self, message: str, sink: Optional[Sink] = None) -> TurnResult:
        """Run one user turn."""
        return self.agent.run(message, sink=sink)

    def load_pdf(
        self, source: Any, name: str, progress=None, file_id: Optional[str] = None
    ) -> int:
        """Index a PDF and switch the retrieval tools on.

        ``file_id`` enables the embedding cache, so re-uploading the same file
        skips the slow part entirely.
        """
        if self.documents is None:
            raise ConfigError("this session was created without document support")
        added = self.documents.add_pdf(source, name, progress, file_id=file_id)
        self._rebuild()
        return added

    def load_text(self, text: str, name: str, progress=None) -> int:
        """Index plain text and switch the retrieval tools on."""
        if self.documents is None:
            raise ConfigError("this session was created without document support")
        added = self.documents.add_text(text, name, progress)
        self._rebuild()
        return added

    def unload_documents(self) -> None:
        if self.documents is not None:
            self.documents.clear()
            self._rebuild()

    def clear(self) -> None:
        """Forget the conversation, keeping any loaded documents."""
        self.memory.clear()

    def reconfigure(self, **changes: Any) -> AgentConfig:
        """Change settings mid-session and rewire everything that holds a config.

        ``AgentConfig`` is frozen, so this replaces it and pushes the new one
        into the objects that captured a reference - otherwise a toggled
        setting would apply in some places and not others.
        """
        self.config = replace(self.config, **changes)
        self.memory.config = self.config
        if hasattr(self.llm, "config"):
            self.llm.config = self.config
        if self.documents is not None:
            self.documents.config = self.config
            self.documents.embedder.config = self.config
        self._rebuild()
        return self.config

    # --- voice ---------------------------------------------------------
    def transcribe(self, audio: bytes, mime_type: str = "audio/wav") -> str:
        """Turn spoken audio into text. Raises if the model call fails."""
        return voice.transcribe(self.llm.client, audio, self.config, mime_type)

    def speak(self, text: str) -> Optional[bytes]:
        """Render an answer as WAV audio, or ``None`` if TTS is unavailable."""
        if voice.is_error_answer(text):
            return None
        return voice.synthesize(self.llm.client, text, self.config)

    # ------------------------------------------------------------------
    @property
    def tool_names(self) -> List[str]:
        return self.registry.names

    def transcript(self) -> str:
        return render_transcript(self.memory.transcript)

    def stats(self) -> dict:
        usage = getattr(self.llm, "total_usage", None)
        return {
            "model": self.config.model,
            "tools": len(self.registry),
            "messages": len(self.memory),
            "notes": len(self.memory.notes),
            "documents": (
                self.documents.describe() if self.documents else "not enabled"
            ),
            "summarised": bool(self.memory.summary),
            "api_calls": getattr(usage, "calls", 0),
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "thought_tokens": getattr(usage, "thought_tokens", 0),
        }
