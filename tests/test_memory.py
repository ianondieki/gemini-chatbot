"""Memory must be trimmable without corrupting the transcript."""

from __future__ import annotations

import pytest
from google.genai import types

from conftest import FakeLLM, text_turn

from gemini_agent.config import AgentConfig
from gemini_agent.memory import (
    WorkingMemory, estimate_tokens, is_safe_boundary, is_tool_result,
    render_transcript,
)


def user(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])


def model_call(name: str, **args) -> types.Content:
    return types.Content(
        role="model",
        parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))],
    )


def tool_response(name: str, result: str = "ok") -> types.Content:
    return types.Content(
        role="user",
        parts=[types.Part.from_function_response(name=name, response={"result": result})],
    )


@pytest.fixture
def memory() -> WorkingMemory:
    return WorkingMemory(
        AgentConfig(max_history_messages=9, keep_recent_messages=3)
    )


class TestClassification:
    def test_a_tool_response_is_not_a_safe_boundary(self):
        """Gemini rejects a transcript that opens with an orphaned tool result."""
        assert is_tool_result(tool_response("web_search"))
        assert not is_safe_boundary(tool_response("web_search"))

    def test_a_plain_user_message_is_a_safe_boundary(self):
        assert is_safe_boundary(user("hello"))

    def test_a_model_message_is_not_a_boundary(self):
        assert not is_safe_boundary(types.Content(role="model", parts=[types.Part(text="hi")]))


class TestCompaction:
    def _fill(self, memory, rounds=4):
        for i in range(rounds):
            memory.append(user(f"question {i}"))
            memory.append(model_call("web_search", q=str(i)))
            memory.append(tool_response("web_search", f"result {i}"))

    def test_it_only_triggers_over_the_limit(self, memory):
        memory.append(user("just one"))
        assert not memory.needs_compaction()
        assert memory.compact(llm=None) is False

    def test_the_kept_history_starts_at_a_safe_boundary(self, memory):
        """The bug this guards: a cut landing on a tool result poisons every
        later request with an orphaned function response."""
        self._fill(memory)
        assert memory.compact(llm=None) is True
        assert is_safe_boundary(memory.transcript[0])

    def test_call_and_response_pairs_are_never_split(self, memory):
        self._fill(memory)
        memory.compact(llm=None)
        for index, content in enumerate(memory.transcript):
            if is_tool_result(content):
                previous = memory.transcript[index - 1]
                assert previous.role == "model"

    def test_dropped_turns_become_a_summary(self, memory):
        self._fill(memory)
        memory.compact(llm=None)
        assert memory.summary
        assert "question 0" in memory.summary

    def test_the_summary_is_sent_with_the_next_request(self, memory):
        self._fill(memory)
        memory.compact(llm=None)
        rendered = str(memory.contents_for_model())
        assert "Summary of earlier conversation" in rendered

    def test_a_model_writes_the_summary_when_one_is_available(self, memory):
        self._fill(memory)
        llm = FakeLLM([text_turn("The user asked four questions about search.")])
        memory.compact(llm=llm)
        assert memory.summary == "The user asked four questions about search."

    def test_a_failing_summariser_does_not_lose_the_turn(self, memory):
        class Broken(FakeLLM):
            def generate(self, *args, **kwargs):
                raise RuntimeError("model down")

        self._fill(memory)
        assert memory.compact(llm=Broken()) is True
        assert memory.summary  # fell back to the deterministic digest

    def test_compaction_can_be_switched_off(self):
        memory = WorkingMemory(
            AgentConfig(max_history_messages=4, keep_recent_messages=2,
                        compaction_enabled=False)
        )
        for i in range(10):
            memory.append(user(str(i)))
        assert memory.compact(llm=None) is False
        assert len(memory) == 10

    def test_a_transcript_with_no_boundary_ahead_is_left_alone(self, memory):
        """Better an oversized transcript than a corrupted one."""
        memory.append(user("start"))
        for i in range(12):
            memory.append(model_call("t", i=i))
            memory.append(tool_response("t"))
        memory.compact(llm=None)
        assert is_safe_boundary(memory.transcript[0])


class TestNotes:
    def test_notes_survive_compaction(self, memory):
        memory.remember("the user is in Nairobi", topic="location")
        for i in range(30):
            memory.append(user(f"turn {i}"))
        memory.compact(llm=None)
        assert "Nairobi" in str(memory.contents_for_model())

    def test_recall_filters_by_topic_and_by_text(self, memory):
        memory.remember("likes metric units", topic="preferences")
        memory.remember("budget is 50k", topic="constraints")
        assert len(memory.recall("preferences")) == 1
        assert len(memory.recall("50k")) == 1
        assert len(memory.recall()) == 2

    def test_recall_of_an_unknown_topic_is_empty(self, memory):
        memory.remember("x", topic="a")
        assert memory.recall("nothing here") == []


class TestRollback:
    def test_rollback_removes_everything_after_the_marker(self, memory):
        memory.append(user("keep me"))
        marker = memory.marker()
        memory.append(user("drop me"))
        memory.append(model_call("t"))
        memory.rollback_to(marker)
        assert len(memory) == 1
        assert render_transcript(memory.transcript) == "user: keep me"


class TestRendering:
    def test_tool_calls_and_results_are_readable(self, memory):
        memory.append(user("what is 2+2?"))
        memory.append(model_call("calculate", expression="2+2"))
        memory.append(tool_response("calculate", "2+2 = 4"))
        rendered = render_transcript(memory.transcript)
        assert "calls calculate(expression='2+2')" in rendered
        assert "calculate returned: 2+2 = 4" in rendered

    def test_thoughts_are_left_out_of_the_transcript(self, memory):
        memory.append(
            types.Content(
                role="model",
                parts=[
                    types.Part(text="internal musing", thought=True),
                    types.Part(text="the answer"),
                ],
            )
        )
        rendered = render_transcript(memory.transcript)
        assert "internal musing" not in rendered and "the answer" in rendered

    def test_token_estimation_grows_with_content(self, memory):
        assert estimate_tokens([]) == 0
        memory.append(user("x" * 400))
        assert 80 <= memory.estimated_tokens() <= 140
