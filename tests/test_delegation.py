"""Delegation: a sub-agent gets a clean context and cannot run away."""

from __future__ import annotations

import pytest

from conftest import FakeLLM, text_turn, tool_turn

from gemini_agent.config import AgentConfig
from gemini_agent.loop import Agent, ToolContext
from gemini_agent.memory import WorkingMemory
from gemini_agent.registry import ToolRegistry
from gemini_agent.tools import delegate as delegate_tool


@pytest.fixture
def registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.tool()
    def echo(value: str) -> str:
        """Echo a value."""
        return f"echoed:{value}"

    @registry.tool()
    def secret() -> str:
        """A tool sub-agents should not get unless asked for."""
        return "classified"

    delegate_tool.register(registry)
    return registry


def build(llm, registry, config) -> Agent:
    memory = WorkingMemory(config)
    return Agent(
        llm=llm,
        registry=registry,
        config=config,
        memory=memory,
        system_prompt="parent",
        context=ToolContext(config=config, memory=memory),
    )


@pytest.fixture
def config() -> AgentConfig:
    return AgentConfig(
        planning="never", max_reflections=0, max_delegation_depth=1,
        delegate_max_steps=4, delegate_max_tool_calls=4,
    )


class TestDelegation:
    def test_the_subagent_answer_comes_back_as_the_tool_result(self, config, registry):
        llm = FakeLLM(
            [
                tool_turn(("delegate", {"task": "Find out what echo says about x",
                                        "tools": ["echo"]})),
                tool_turn(("echo", {"value": "x"})),        # the sub-agent's turn
                text_turn("The echo said x."),               # the sub-agent's answer
                text_turn("My sub-agent reported: the echo said x."),
            ]
        )
        result = build(llm, registry, config).run("delegate this")
        assert result.answer == "My sub-agent reported: the echo said x."
        parent_second_call = str(llm.requests[-1]["contents"])
        assert "The echo said x." in parent_second_call

    def test_the_subagent_starts_with_an_empty_memory(self, config, registry):
        """Isolation is the point: the parent's noise must not travel down."""
        llm = FakeLLM(
            [
                tool_turn(("delegate", {"task": "A task that stands alone here",
                                        "tools": ["echo"]})),
                text_turn("sub-agent answer"),
                text_turn("done"),
            ]
        )
        agent = build(llm, registry, config)
        agent.run("a very distinctive parent question about pelicans")
        sub_request = llm.requests[1]["contents"]
        assert "pelicans" not in str(sub_request)
        assert "A task that stands alone here" in str(sub_request)

    def test_the_subagent_toolbox_is_narrowed(self, config, registry):
        llm = FakeLLM(
            [
                tool_turn(("delegate", {"task": "Only echo is needed for this one",
                                        "tools": ["echo"]})),
                text_turn("sub answer"),
                text_turn("done"),
            ]
        )
        build(llm, registry, config).run("go")
        declared = llm.requests[1]["tools"].function_declarations
        assert [d.name for d in declared] == ["echo"]

    def test_a_subagent_cannot_delegate_further(self, config, registry):
        """Otherwise one turn can fork indefinitely."""
        llm = FakeLLM(
            [
                tool_turn(("delegate", {"task": "The outer subtask to work on",
                                        "tools": ["echo", "delegate"]})),
                text_turn("sub answer"),
                text_turn("done"),
            ]
        )
        build(llm, registry, config).run("go")
        declared = llm.requests[1]["tools"].function_declarations
        assert "delegate" not in [d.name for d in declared]

    def test_subagent_tool_calls_are_charged_to_the_parent(self, config, registry):
        llm = FakeLLM(
            [
                tool_turn(("delegate", {"task": "Do some real work with the echo tool",
                                        "tools": ["echo"]})),
                tool_turn(("echo", {"value": "a"})),
                tool_turn(("echo", {"value": "b"})),
                text_turn("sub done"),
                text_turn("parent done"),
            ]
        )
        result = build(llm, registry, config).run("go")
        assert result.tool_calls >= 3   # 1 delegate + 2 inside the sub-agent

    def test_a_depth_limited_agent_refuses_politely(self, config, registry):
        """The refusal is a tool result, so the model reads it and adapts."""
        memory = WorkingMemory(config)
        agent = Agent(
            llm=FakeLLM([text_turn("x")]),
            registry=registry,
            config=config,
            memory=memory,
            context=ToolContext(config=config, memory=memory, depth=1),
            depth=1,
        )
        message = agent._spawn_sub_agent("anything at all", ["echo"])
        assert "depth limit" in message

    def test_unknown_tool_names_are_reported(self, config, registry):
        llm = FakeLLM([text_turn("x")])
        agent = build(llm, registry, config)
        assert "none of the requested tools exist" in agent._spawn_sub_agent(
            "task", ["nonexistent"]
        )


class TestGuards:
    def test_a_task_too_short_to_stand_alone_is_refused(self, config, registry):
        result = registry.invoke(
            "delegate", {"task": "do it", "tools": ["echo"]},
            ctx=ToolContext(config=config, memory=WorkingMemory(config)),
        )
        assert "too short to stand alone" in result.content

    def test_an_empty_task_is_refused(self, config, registry):
        result = registry.invoke(
            "delegate", {"task": "   ", "tools": ["echo"]},
            ctx=ToolContext(config=config, memory=WorkingMemory(config)),
        )
        assert result.content.startswith("ERROR:")

    def test_delegation_is_absent_when_there_is_nothing_to_delegate(self):
        from gemini_agent.tools import build_registry

        assert "delegate" not in build_registry(search_client=None, with_notes=False)
