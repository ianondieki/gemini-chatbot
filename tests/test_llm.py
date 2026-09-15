"""The transport layer: retries, fallback, normalisation and JSON parsing."""

from __future__ import annotations

import pytest
from google.genai import errors as genai_errors
from google.genai import types

from gemini_agent.config import AgentConfig
from gemini_agent.errors import LLMError, ModelBlocked
from gemini_agent.llm import GeminiLLM, parse_json_loosely


def api_error(status: int, message: str) -> genai_errors.APIError:
    return genai_errors.APIError(status, {"error": {"message": message}})


def response(parts, usage=None, finish="STOP"):
    class Response:
        candidates = [
            types.Candidate(
                content=types.Content(role="model", parts=parts), finish_reason=finish
            )
        ]
        usage_metadata = usage

    return Response()


class FakeModels:
    """Stands in for ``client.models``; replays a queue of results or errors."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.seen = []

    def generate_content(self, *, model, contents, config):
        self.seen.append({"model": model, "config": config, "contents": contents})
        outcome = self.outcomes.pop(0) if self.outcomes else response([types.Part(text="ok")])
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.models = FakeModels(outcomes)


def build(outcomes, **overrides) -> GeminiLLM:
    config = AgentConfig(max_retries=3, retry_base_delay=0.0, **overrides)
    return GeminiLLM(config, client=FakeClient(outcomes), sleep=lambda _s: None)


class TestNormalisation:
    def test_text_thoughts_and_calls_are_separated(self):
        llm = build(
            [
                response(
                    [
                        types.Part(text="my reasoning", thought=True),
                        types.Part(
                            function_call=types.FunctionCall(
                                name="calculate", args={"expression": "2+2"}
                            )
                        ),
                        types.Part(text="visible text"),
                    ]
                )
            ]
        )
        result = llm.generate([])
        assert result.thoughts == ["my reasoning"]
        assert result.text == "visible text"
        assert result.tool_calls[0].name == "calculate"
        assert result.tool_calls[0].args == {"expression": "2+2"}
        assert result.wants_tools

    def test_usage_is_read_and_accumulated(self):
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100, candidates_token_count=20,
            thoughts_token_count=7, total_token_count=127,
        )
        llm = build([response([types.Part(text="hi")], usage=usage)])
        result = llm.generate([])
        assert result.usage.prompt_tokens == 100
        assert result.usage.thought_tokens == 7
        assert llm.total_usage.total_tokens == 127
        assert llm.total_usage.calls == 1

    def test_a_blocked_response_raises_rather_than_returning_nothing(self):
        class Blocked:
            candidates = []
            prompt_feedback = types.GenerateContentResponsePromptFeedback(
                block_reason="SAFETY"
            )

        with pytest.raises(ModelBlocked):
            build([Blocked()]).generate([])


class TestResilience:
    def test_transient_errors_are_retried(self):
        llm = build([api_error(503, "overloaded"), response([types.Part(text="ok")])])
        assert llm.generate([]).text == "ok"
        assert len(llm.client.models.seen) == 2

    def test_permanent_errors_are_not_retried(self):
        llm = build([api_error(400, "API key not valid")])
        with pytest.raises(genai_errors.APIError):
            llm.generate([])
        assert len(llm.client.models.seen) == 1

    def test_it_falls_back_to_the_lighter_model(self):
        llm = build(
            [api_error(503, "overloaded")] * 3 + [response([types.Part(text="from backup")])]
        )
        assert llm.generate([]).text == "from backup"
        assert llm.client.models.seen[-1]["model"] == llm.config.fallback_model

    def test_it_gives_up_with_a_clear_error(self):
        llm = build([api_error(503, "overloaded")] * 6)
        with pytest.raises(LLMError, match="unavailable"):
            llm.generate([])

    def test_a_model_rejecting_thinking_is_retried_without_it(self):
        """Losing a whole turn because one model dislikes a thinking budget is waste."""
        llm = build(
            [
                api_error(400, "thinking_config is not supported"),
                response([types.Part(text="fine")]),
            ]
        )
        assert llm.generate([]).text == "fine"
        assert llm.client.models.seen[0]["config"].thinking_config is not None
        assert llm.client.models.seen[1]["config"].thinking_config is None


class TestRequestShaping:
    def test_tools_disable_automatic_function_calling(self):
        """The loop drives tools itself; the SDK must not also call them."""
        llm = build([response([types.Part(text="ok")])])
        tool = types.Tool(function_declarations=[])
        llm.generate([], tools=tool)
        config = llm.client.models.seen[0]["config"]
        assert config.automatic_function_calling.disable is True

    def test_structured_output_drops_tools(self):
        """Gemini rejects response_schema and tools together."""
        llm = build([response([types.Part(text="{}")])])
        llm.generate([], tools=types.Tool(function_declarations=[]),
                     response_schema=types.Schema(type=types.Type.OBJECT))
        config = llm.client.models.seen[0]["config"]
        assert config.tools is None
        assert config.response_mime_type == "application/json"

    def test_a_zero_thinking_budget_omits_the_config(self):
        llm = build([response([types.Part(text="ok")])])
        llm.generate([], thinking_budget=0)
        assert llm.client.models.seen[0]["config"].thinking_config is None

    def test_generate_json_parses_the_reply(self):
        llm = build([response([types.Part(text='{"verdict": "accept"}')])])
        parsed = llm.generate_json([], schema=types.Schema(type=types.Type.OBJECT))
        assert parsed == {"verdict": "accept"}

    def test_generate_json_returns_none_on_junk(self):
        llm = build([response([types.Part(text="sorry, I cannot")])])
        assert llm.generate_json([], schema=types.Schema(type=types.Type.OBJECT)) is None


class TestLooseJson:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ('{"a": 1}', {"a": 1}),
            ('```json\n{"a": 1}\n```', {"a": 1}),
            ('```\n{"a": 1}\n```', {"a": 1}),
            ('Here you go: {"a": 1} hope that helps', {"a": 1}),
            ("[1, 2, 3]", [1, 2, 3]),
        ],
    )
    def test_it_recovers_json_from_prose_and_fences(self, raw, expected):
        assert parse_json_loosely(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", "no json here", "{broken"])
    def test_it_returns_none_rather_than_raising(self, raw):
        assert parse_json_loosely(raw) is None
