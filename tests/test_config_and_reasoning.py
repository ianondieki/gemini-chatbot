"""Config validation, budgets, prompt assembly and plan/critique parsing."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from gemini_agent.budget import Budget
from gemini_agent.config import AgentConfig
from gemini_agent.errors import ConfigError
from gemini_agent.prompts import build_system_prompt
from gemini_agent.reasoning import Critique, Plan
from gemini_agent.registry import ToolRegistry


class TestConfig:
    def test_defaults_are_coherent(self):
        config = AgentConfig()
        assert config.keep_recent_messages < config.max_history_messages
        assert config.chunk_overlap < config.chunk_size

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"planning": "sometimes"},
            {"max_steps": 0},
            {"max_reflections": -1},
            {"keep_recent_messages": 50, "max_history_messages": 10},
            {"chunk_size": 100, "chunk_overlap": 100},
            {"max_parallel_tools": 0},
        ],
    )
    def test_nonsense_is_rejected_at_construction(self, kwargs):
        with pytest.raises(ConfigError):
            AgentConfig(**kwargs)

    def test_environment_overrides_defaults(self):
        with mock.patch.dict(
            os.environ,
            {"GEMINI_MODEL": "gemini-2.5-pro", "AGENT_MAX_STEPS": "30",
             "AGENT_PLANNING": "always", "AGENT_PARALLEL_TOOLS": "false"},
        ):
            config = AgentConfig.from_env()
        assert config.model == "gemini-2.5-pro"
        assert config.max_steps == 30
        assert config.planning == "always"
        assert config.parallel_tools is False

    def test_a_non_numeric_env_var_is_reported_not_swallowed(self):
        with mock.patch.dict(os.environ, {"AGENT_MAX_STEPS": "lots"}):
            with pytest.raises(ConfigError, match="AGENT_MAX_STEPS"):
                AgentConfig.from_env()

    def test_blank_env_vars_fall_back_to_defaults(self):
        with mock.patch.dict(os.environ, {"GEMINI_MODEL": "  "}):
            assert AgentConfig.from_env().model == AgentConfig.model

    def test_the_delegate_config_is_strictly_tighter(self):
        config = AgentConfig()
        sub = config.for_delegate()
        assert sub.max_steps < config.max_steps
        assert sub.max_reflections == 0
        assert sub.planning == "never"


class TestBudget:
    def test_each_dimension_reports_its_own_reason(self):
        budget = Budget(max_steps=2, max_tool_calls=3, max_reflections=1,
                        wall_clock_seconds=100)
        budget.start()
        assert budget.exhausted() is None
        budget.spend_step(); budget.spend_step()
        assert "2-step limit" in budget.exhausted()

        budget.start()
        budget.spend_tool_calls(3)
        assert "3-tool-call limit" in budget.exhausted()

    def test_time_runs_out(self):
        budget = Budget(max_steps=99, max_tool_calls=99, max_reflections=1,
                        wall_clock_seconds=-1)
        budget.start()
        assert "ran out of time" in budget.exhausted()

    def test_reflections_are_capped(self):
        budget = Budget(max_steps=9, max_tool_calls=9, max_reflections=1,
                        wall_clock_seconds=100)
        budget.start()
        assert budget.can_reflect()
        budget.spend_reflection()
        assert not budget.can_reflect()

    def test_tool_calls_left_never_goes_negative(self):
        budget = Budget(max_steps=9, max_tool_calls=2, max_reflections=0,
                        wall_clock_seconds=100)
        budget.start()
        budget.spend_tool_calls(5)
        assert budget.tool_calls_left() == 0

    def test_start_resets_everything(self):
        budget = Budget.from_config(AgentConfig())
        budget.start(); budget.spend_step(); budget.spend_tool_calls(4)
        budget.start()
        assert budget.steps_used == 0 and budget.tool_calls_used == 0


class TestPrompt:
    def test_the_prompt_lists_the_tools_that_exist(self):
        registry = ToolRegistry()
        registry.tool(name="widget", description="Do a widget thing.")(lambda: "")
        prompt = build_system_prompt(registry, name="Angel")
        assert "Angel" in prompt
        assert "widget" in prompt and "Do a widget thing." in prompt

    def test_it_never_advertises_a_tool_that_is_absent(self):
        """The old static prompt promised web search whether or not it existed."""
        prompt = build_system_prompt(ToolRegistry())
        assert "web_search" not in prompt
        assert "no tools this session" in prompt

    def test_extra_guidance_is_appended(self):
        prompt = build_system_prompt(ToolRegistry(), extra="Speak only in Swahili.")
        assert prompt.endswith("Speak only in Swahili.")


class TestPlanParsing:
    def test_a_well_formed_plan_round_trips(self):
        plan = Plan.from_dict(
            {
                "complexity": "complex",
                "goal": "Compare two laptops",
                "steps": [{"description": "look up A", "tool": "web_search"}],
                "success_criteria": ["names a winner"],
                "risks": ["prices change"],
            }
        )
        assert plan.goal == "Compare two laptops"
        assert "look up A" in plan.render()
        assert "names a winner" in plan.render()
        assert not plan.is_direct

    def test_plain_string_steps_are_accepted(self):
        plan = Plan.from_dict(
            {"complexity": "simple", "goal": "g", "steps": ["do the thing"],
             "success_criteria": []}
        )
        assert plan.steps[0].description == "do the thing"

    def test_an_unknown_complexity_degrades_to_simple(self):
        plan = Plan.from_dict({"complexity": "wild", "goal": "g", "steps": []})
        assert plan.complexity == "simple"

    @pytest.mark.parametrize("bad", [None, "text", {}, {"goal": ""}, []])
    def test_junk_yields_no_plan(self, bad):
        assert Plan.from_dict(bad) is None

    def test_direct_is_recognised(self):
        plan = Plan.from_dict({"complexity": "direct", "goal": "answer it", "steps": []})
        assert plan.is_direct


class TestCritiqueParsing:
    def test_a_revise_verdict_becomes_actionable_guidance(self):
        critique = Critique.from_dict(
            {"verdict": "revise", "confidence": 0.3,
             "issues": ["no source"], "unmet_criteria": ["must cite"],
             "next_actions": ["search again"]}
        )
        guidance = critique.as_guidance()
        assert not critique.accepted
        assert "no source" in guidance
        assert "must cite" in guidance
        assert "search again" in guidance

    def test_confidence_is_clamped(self):
        assert Critique.from_dict({"verdict": "accept", "confidence": 7}).confidence == 1.0
        assert Critique.from_dict({"verdict": "accept", "confidence": -2}).confidence == 0.0
        assert Critique.from_dict({"verdict": "accept", "confidence": "x"}).confidence == 1.0

    def test_an_unknown_verdict_defaults_to_accepting(self):
        """A broken critic must not be able to block every answer forever."""
        assert Critique.from_dict({"verdict": "maybe"}).accepted

    @pytest.mark.parametrize("bad", [None, "text", 42])
    def test_junk_yields_no_critique(self, bad):
        assert Critique.from_dict(bad) is None
