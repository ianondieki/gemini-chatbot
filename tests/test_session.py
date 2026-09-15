"""End to end: a whole AgentSession driven by a fake Gemini client.

Nothing is stubbed above the transport, so these exercise the real wiring the
front ends rely on - prompt assembly, the toolbox changing when a document is
loaded, tool results reaching the model, and usage accounting.
"""

from __future__ import annotations

import os
from unittest import mock

import numpy as np
import pytest
from google.genai import types

from gemini_agent import AgentSession
from gemini_agent.config import AgentConfig
from gemini_agent.errors import ConfigError


class FakeGeminiClient:
    """Implements just enough of ``genai.Client`` for a session to run."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        outer = self

        class Models:
            def generate_content(self, *, model, contents, config):
                outer.seen.append({"model": model, "contents": contents, "config": config})
                parts = outer.replies.pop(0) if outer.replies else [types.Part(text="done")]

                class Response:
                    candidates = [
                        types.Candidate(
                            content=types.Content(role="model", parts=parts),
                            finish_reason="STOP",
                        )
                    ]
                    usage_metadata = types.GenerateContentResponseUsageMetadata(
                        prompt_token_count=10, candidates_token_count=4,
                        total_token_count=14,
                    )

                return Response()

            def embed_content(self, *, model, contents, config):
                class Embedding:
                    def __init__(self, text):
                        # Deterministic pseudo-embedding, stable per text.
                        rng = np.random.default_rng(abs(hash(text)) % (2**32))
                        self.values = rng.random(config.output_dimensionality).tolist()

                class Response:
                    embeddings = [Embedding(t) for t in contents]

                return Response()

        self.models = Models()


def call(name, **args):
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


@pytest.fixture(autouse=True)
def api_key():
    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
        yield


@pytest.fixture
def config():
    return AgentConfig(
        planning="never", max_reflections=0, max_steps=5,
        chunk_size=200, chunk_overlap=40, embed_batch=4, embed_batch_delay=0,
    )


def build(replies, config, **kwargs) -> AgentSession:
    return AgentSession.create(
        config=config, client=FakeGeminiClient(replies), enable_search=False, **kwargs
    )


class TestWiring:
    def test_a_missing_key_is_a_clear_error_not_a_stack_trace(self, config):
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
                AgentSession.create(config=config)

    def test_the_toolbox_reflects_what_is_configured(self, config):
        session = build([], config)
        assert "calculate" in session.tool_names
        assert "web_search" not in session.tool_names   # no Tavily key
        assert "search_documents" not in session.tool_names  # no document yet

    def test_the_system_prompt_only_promises_real_tools(self, config):
        session = build([], config)
        assert "web_search" not in session.system_prompt
        assert "calculate" in session.system_prompt

    def test_the_tools_sent_to_gemini_match_the_registry(self, config):
        session = build([[types.Part(text="hi")]], config)
        session.ask("hello")
        declared = session.llm.client.seen[0]["config"].tools[0].function_declarations
        assert {d.name for d in declared} == set(session.tool_names)


class TestConversation:
    def test_a_tool_runs_and_its_result_reaches_the_model(self, config):
        session = build(
            [
                [call("calculate", expression="15/100 * 2400")],
                [types.Part(text="That is 360.")],
            ],
            config,
        )
        result = session.ask("What is 15% of 2400?")
        assert result.answer == "That is 360."
        assert "15/100 * 2400 = 360" in str(session.llm.client.seen[1]["contents"])

    def test_history_persists_across_turns(self, config):
        session = build(
            [[types.Part(text="Paris.")], [types.Part(text="About 2.1 million.")]],
            config,
        )
        session.ask("Capital of France?")
        session.ask("And its population?")
        assert "Capital of France?" in str(session.llm.client.seen[1]["contents"])
        assert len(session.memory) == 4

    def test_notes_survive_into_later_turns(self, config):
        session = build(
            [
                [call("remember", fact="The user lives in Nairobi", topic="location")],
                [types.Part(text="Noted.")],
                [types.Part(text="Then it is 3pm for you.")],
            ],
            config,
        )
        session.ask("I live in Nairobi")
        session.ask("What time is it for me?")
        assert "Nairobi" in str(session.llm.client.seen[-1]["contents"])

    def test_clear_forgets_the_conversation(self, config):
        session = build([[types.Part(text="hi")]], config)
        session.ask("hello")
        session.clear()
        assert len(session.memory) == 0

    def test_usage_accumulates_across_the_session(self, config):
        session = build(
            [[call("calculate", expression="1+1")], [types.Part(text="2")]], config
        )
        session.ask("1+1?")
        stats = session.stats()
        assert stats["api_calls"] == 2
        assert stats["prompt_tokens"] == 20


class TestDocuments:
    TEXT = (
        "The registration fee is 4,500 shillings. "
        "Late applications attract a surcharge of 10 percent. "
        "Appeals close 30 days after the decision. "
    ) * 12

    def test_loading_a_document_adds_the_retrieval_tools(self, config):
        session = build([], config)
        assert "search_documents" not in session.tool_names
        session.load_text(self.TEXT, "rules.txt")
        assert "search_documents" in session.tool_names
        assert "search_documents" in session.system_prompt

    def test_the_agent_can_search_the_loaded_document(self, config):
        session = build(
            [
                [call("search_documents", query="registration fee")],
                [types.Part(text="It is 4,500 shillings.")],
            ],
            config,
        )
        session.load_text(self.TEXT, "rules.txt")
        result = session.ask("What is the fee?")
        assert result.answer == "It is 4,500 shillings."
        assert "rules.txt" in str(session.llm.client.seen[-1]["contents"])

    def test_unloading_removes_the_tools_again(self, config):
        session = build([], config)
        session.load_text(self.TEXT, "rules.txt")
        session.unload_documents()
        assert "search_documents" not in session.tool_names


class TestReconfiguration:
    def test_changing_a_setting_reaches_every_holder(self, config):
        """A frozen config replaced in one place only would half-apply."""
        session = build([], config)
        session.reconfigure(planning="always", max_reflections=2)
        assert session.config.planning == "always"
        assert session.agent.config.planning == "always"
        assert session.agent.budget.max_reflections == 2
        assert session.memory.config.max_reflections == 2
        assert session.llm.config.planning == "always"
