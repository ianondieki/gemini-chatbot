"""The loop: planning, tool execution, self-repair, reflection and budgets.

These drive the real :class:`Agent` against a scripted model, so they assert on
behaviour the loop is responsible for rather than on anything Gemini does.
"""

from __future__ import annotations

import time

import pytest

from conftest import FakeLLM, json_turn, text_turn, tool_turn

from gemini_agent.config import AgentConfig
from gemini_agent.events import EventRecorder
from gemini_agent.loop import Agent, ToolContext
from gemini_agent.memory import WorkingMemory
from gemini_agent.registry import ToolRegistry


@pytest.fixture
def registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.calls = []  # type: ignore[attr-defined]

    @registry.tool()
    def echo(value: str) -> str:
        """Echo a value back."""
        registry.calls.append(("echo", value))  # type: ignore[attr-defined]
        return f"echoed:{value}"

    @registry.tool()
    def slow(seconds: float = 0.2) -> str:
        """Sleep, then report."""
        time.sleep(seconds)
        return f"slept {seconds}"

    @registry.tool()
    def flaky(mode: str) -> str:
        """Fail unless asked nicely."""
        if mode != "nice":
            raise RuntimeError("be nicer")
        return "worked"

    return registry


def build(llm, registry, config, sink=None) -> Agent:
    memory = WorkingMemory(config)
    return Agent(
        llm=llm,
        registry=registry,
        config=config,
        memory=memory,
        system_prompt="test",
        sink=sink,
        context=ToolContext(config=config, memory=memory),
    )


class TestBasicTurn:
    def test_a_plain_answer_costs_one_step(self, config, registry):
        llm = FakeLLM([text_turn("Paris.")])
        result = build(llm, registry, config).run("Capital of France?")
        assert result.answer == "Paris." and result.steps == 1
        assert result.tool_calls == 0 and result.ok

    def test_tool_result_is_fed_back_and_the_model_answers(self, config, registry):
        llm = FakeLLM([tool_turn(("echo", {"value": "hi"})), text_turn("It said hi.")])
        result = build(llm, registry, config).run("Echo hi")
        assert result.answer == "It said hi."
        assert result.steps == 2 and result.tool_calls == 1
        assert registry.calls == [("echo", "hi")]

    def test_the_observation_reaches_the_next_request(self, config, registry):
        llm = FakeLLM([tool_turn(("echo", {"value": "hi"})), text_turn("done")])
        build(llm, registry, config).run("go")
        second = llm.requests[1]["contents"]
        rendered = str([p for c in second for p in (c.parts or [])])
        assert "echoed:hi" in rendered

    def test_usage_is_attributed_to_the_turn(self, config, registry):
        llm = FakeLLM([text_turn("hi")])
        result = build(llm, registry, config).run("hello")
        assert result.usage.calls == 1 and result.usage.total_tokens == 15


class TestSelfRepair:
    def test_a_failing_tool_does_not_end_the_turn(self, config, registry):
        llm = FakeLLM(
            [
                tool_turn(("flaky", {"mode": "rude"})),
                tool_turn(("flaky", {"mode": "nice"})),
                text_turn("Fixed it."),
            ]
        )
        result = build(llm, registry, config).run("use flaky")
        assert result.answer == "Fixed it." and result.ok

    def test_the_failure_text_reaches_the_model(self, config, registry):
        llm = FakeLLM([tool_turn(("flaky", {"mode": "rude"})), text_turn("gave up")])
        build(llm, registry, config).run("go")
        rendered = str(llm.requests[1]["contents"])
        assert "be nicer" in rendered

    def test_repeating_a_failing_call_earns_an_instruction_to_stop(self, config, registry):
        """Without this the model can loop on the same broken call until budget dies."""
        llm = FakeLLM(
            [
                tool_turn(("flaky", {"mode": "rude"})),
                tool_turn(("flaky", {"mode": "rude"})),
                text_turn("giving up"),
            ]
        )
        build(llm, registry, config).run("go")
        rendered = str(llm.requests[2]["contents"])
        assert "Do not repeat it" in rendered

    def test_an_unknown_tool_is_reported_rather_than_raised(self, config, registry):
        llm = FakeLLM([tool_turn(("teleport", {})), text_turn("no such tool")])
        result = build(llm, registry, config).run("go")
        assert result.ok
        assert "no tool called 'teleport'" in str(llm.requests[1]["contents"])

    def test_two_empty_replies_stop_the_loop(self, config, registry):
        llm = FakeLLM([text_turn(""), text_turn(""), text_turn("recovered")])
        result = build(llm, registry, config).run("go")
        assert result.stopped_early is not None
        assert "empty response" in result.stopped_early


class TestToolExecution:
    def test_identical_calls_are_cached_within_a_turn(self, config, registry):
        llm = FakeLLM(
            [
                tool_turn(("echo", {"value": "x"})),
                tool_turn(("echo", {"value": "x"})),
                text_turn("done"),
            ]
        )
        result = build(llm, registry, config).run("go")
        assert registry.calls == [("echo", "x")]      # executed once
        assert result.tool_calls == 2                  # still charged twice

    def test_a_different_argument_is_not_a_cache_hit(self, config, registry):
        llm = FakeLLM(
            [
                tool_turn(("echo", {"value": "x"})),
                tool_turn(("echo", {"value": "y"})),
                text_turn("done"),
            ]
        )
        build(llm, registry, config).run("go")
        assert registry.calls == [("echo", "x"), ("echo", "y")]

    def test_independent_calls_run_in_parallel(self, registry):
        config = AgentConfig(
            planning="never", max_reflections=0, parallel_tools=True,
            max_parallel_tools=4,
        )
        llm = FakeLLM(
            [
                tool_turn(
                    ("slow", {"seconds": 0.3}),
                    ("slow", {"seconds": 0.3}),
                    ("slow", {"seconds": 0.3}),
                ),
                text_turn("done"),
            ]
        )
        started = time.monotonic()
        build(llm, registry, config).run("go")
        elapsed = time.monotonic() - started
        assert elapsed < 0.7, f"parallel calls took {elapsed:.2f}s - serialised?"

    def test_a_timeout_is_reported_as_a_tool_error(self, registry):
        config = AgentConfig(
            planning="never", max_reflections=0, tool_timeout_seconds=0.05
        )
        llm = FakeLLM([tool_turn(("slow", {"seconds": 1.0})), text_turn("done")])
        recorder = EventRecorder()
        build(llm, registry, config, sink=recorder).run("go")
        finished = recorder.of("tool_finished")[0]
        assert finished.ok is False and "timed out" in finished.preview

    def test_a_huge_observation_is_truncated(self, registry):
        config = AgentConfig(
            planning="never", max_reflections=0, observation_char_limit=100
        )
        registry.tool(name="huge", description="Return a lot.")(
            lambda: "x" * 5000
        )
        llm = FakeLLM([tool_turn(("huge", {})), text_turn("done")])
        build(llm, registry, config).run("go")
        assert "output truncated" in str(llm.requests[1]["contents"])


class TestBudgets:
    def test_running_out_of_steps_still_produces_an_answer(self, registry):
        """A budget stop must degrade to a real answer, not a stub string."""
        config = AgentConfig(planning="never", max_reflections=0, max_steps=3)
        llm = FakeLLM([tool_turn(("echo", {"value": str(i)})) for i in range(10)])
        llm.default_text = "Here is what I found so far."
        result = build(llm, registry, config).run("go")
        assert result.stopped_early and "3-step limit" in result.stopped_early
        assert result.answer == "Here is what I found so far."

    def test_the_tool_call_budget_is_enforced_per_turn(self, registry):
        """A batch larger than the remaining allowance is trimmed, not run."""
        config = AgentConfig(planning="never", max_reflections=0, max_tool_calls=2)
        llm = FakeLLM(
            [
                tool_turn(
                    ("echo", {"value": "a"}),
                    ("echo", {"value": "b"}),
                    ("echo", {"value": "c"}),
                ),
                text_turn("done"),
            ]
        )
        result = build(llm, registry, config).run("go")
        assert result.tool_calls == 2
        assert registry.calls == [("echo", "a"), ("echo", "b")]
        # The refused call still gets a response part, so the model is told why.
        assert "budget for this turn is spent" in str(llm.requests[1]["contents"])

    def test_the_wall_clock_stops_the_loop(self, registry):
        config = AgentConfig(
            planning="never", max_reflections=0, max_steps=50,
            wall_clock_seconds=0.2, tool_timeout_seconds=5,
        )
        # Distinct arguments, or the per-turn cache would serve one sleep 20 times.
        llm = FakeLLM(
            [tool_turn(("slow", {"seconds": 0.15 + i / 1000})) for i in range(20)]
        )
        result = build(llm, registry, config).run("go")
        assert result.stopped_early and "ran out of time" in result.stopped_early


class TestPlanning:
    def test_a_plan_is_requested_and_emitted(self, registry):
        config = AgentConfig(planning="always", max_reflections=0)
        llm = FakeLLM(
            [
                json_turn(
                    {
                        "complexity": "complex",
                        "goal": "Compare two things",
                        "steps": [{"description": "look up A", "tool": "echo"}],
                        "success_criteria": ["mentions both"],
                    }
                ),
                text_turn("Here is the comparison."),
            ]
        )
        recorder = EventRecorder()
        result = build(llm, registry, config, sink=recorder).run("Compare A and B")
        assert result.plan is not None and result.plan.goal == "Compare two things"
        assert recorder.of("plan_ready")

    def test_the_plan_is_visible_to_the_acting_model(self, registry):
        config = AgentConfig(planning="always", max_reflections=0)
        llm = FakeLLM(
            [
                json_turn(
                    {
                        "complexity": "complex",
                        "goal": "Find the price",
                        "steps": [{"description": "search", "tool": "echo"}],
                        "success_criteria": ["cites a source"],
                    }
                ),
                text_turn("done"),
            ]
        )
        build(llm, registry, config).run("price?")
        acting = [r for r in llm.requests if not r["structured"]][0]
        assert "Find the price" in str(acting["contents"])
        assert "cites a source" in str(acting["contents"])

    def test_a_direct_verdict_skips_the_plan_entirely(self, registry):
        """Over-planning a trivial question is a cost with no benefit."""
        config = AgentConfig(planning="always", max_reflections=0)
        llm = FakeLLM(
            [
                json_turn(
                    {
                        "complexity": "direct",
                        "goal": "Answer from knowledge",
                        "steps": [],
                        "success_criteria": ["is correct"],
                    }
                ),
                text_turn("Paris."),
            ]
        )
        result = build(llm, registry, config).run("Capital of France?")
        assert result.plan is None and result.answer == "Paris."

    def test_unparseable_plans_degrade_to_a_plain_loop(self, registry):
        config = AgentConfig(planning="always", max_reflections=0)
        llm = FakeLLM([{"text": "not json", "json": True}, text_turn("answered anyway")])
        result = build(llm, registry, config).run("something complicated")
        assert result.plan is None and result.answer == "answered anyway"

    def test_trivial_input_does_not_pay_for_a_planning_call(self, registry):
        config = AgentConfig(planning="auto", max_reflections=0)
        llm = FakeLLM([text_turn("Hello!")])
        build(llm, registry, config).run("hi")
        assert all(not r["structured"] for r in llm.requests)


class TestReflection:
    def test_an_accepted_draft_is_returned_unchanged(self, registry):
        config = AgentConfig(planning="never", max_reflections=1)
        llm = FakeLLM(
            [
                text_turn("The answer is 42."),
                json_turn({"verdict": "accept", "confidence": 0.9,
                           "issues": [], "next_actions": []}),
            ]
        )
        result = build(llm, registry, config).run("meaning of life?")
        assert result.answer == "The answer is 42." and not result.critiques

    def test_a_revise_verdict_drives_a_second_pass(self, registry):
        config = AgentConfig(planning="never", max_reflections=1)
        llm = FakeLLM(
            [
                text_turn("Probably about 40."),
                json_turn({"verdict": "revise", "confidence": 0.2,
                           "issues": ["the number was guessed"],
                           "next_actions": ["look it up"]}),
                text_turn("It is exactly 42, per the source."),
                json_turn({"verdict": "accept", "confidence": 0.95,
                           "issues": [], "next_actions": []}),
            ]
        )
        result = build(llm, registry, config).run("how many?")
        assert result.answer == "It is exactly 42, per the source."
        assert len(result.critiques) == 1

    def test_the_critique_is_handed_to_the_revising_model(self, registry):
        config = AgentConfig(planning="never", max_reflections=1)
        llm = FakeLLM(
            [
                text_turn("draft"),
                json_turn({"verdict": "revise", "confidence": 0.1,
                           "issues": ["no source cited"],
                           "next_actions": ["cite the source"]}),
                text_turn("revised"),
                json_turn({"verdict": "accept", "confidence": 1.0,
                           "issues": [], "next_actions": []}),
            ]
        )
        build(llm, registry, config).run("go")
        revising = [r for r in llm.requests if not r["structured"]][1]
        assert "no source cited" in str(revising["contents"])
        assert "cite the source" in str(revising["contents"])

    def test_reflection_rounds_are_capped(self, registry):
        """An endlessly dissatisfied critic must not hold the turn hostage."""
        config = AgentConfig(planning="never", max_reflections=2)
        llm = FakeLLM()
        for _ in range(10):
            llm.queue(
                text_turn("draft"),
                json_turn({"verdict": "revise", "confidence": 0.1,
                           "issues": ["still wrong"], "next_actions": ["try again"]}),
            )
        result = build(llm, registry, config).run("go")
        assert len(result.critiques) == 2

    def test_a_failed_critique_accepts_the_draft(self, registry):
        config = AgentConfig(planning="never", max_reflections=1)
        llm = FakeLLM([text_turn("draft"), {"text": "garbage", "json": True}])
        result = build(llm, registry, config).run("go")
        assert result.answer == "draft"


class TestFailureHandling:
    def test_a_model_error_rolls_the_transcript_back(self, config, registry):
        """A half-written turn would corrupt every later request."""

        class Exploding(FakeLLM):
            def generate(self, *args, **kwargs):
                raise RuntimeError("upstream is down")

        agent = build(Exploding(), registry, config)
        agent.memory.append_user("earlier message")
        before = len(agent.memory)
        result = agent.run("this will fail")
        assert not result.ok and "upstream is down" in result.answer
        assert len(agent.memory) == before

    def test_a_broken_event_sink_cannot_break_the_agent(self, config, registry):
        def hostile(_event):
            raise ValueError("renderer exploded")

        result = build(FakeLLM([text_turn("fine")]), registry, config, sink=hostile).run("go")
        assert result.answer == "fine"


class TestEventStream:
    def test_the_sequence_describes_the_turn(self, registry):
        config = AgentConfig(planning="always", max_reflections=1)
        llm = FakeLLM(
            [
                json_turn({"complexity": "complex", "goal": "G",
                           "steps": [{"description": "s"}], "success_criteria": ["c"]}),
                tool_turn(("echo", {"value": "x"})),
                text_turn("final"),
                json_turn({"verdict": "accept", "confidence": 1.0,
                           "issues": [], "next_actions": []}),
            ]
        )
        recorder = EventRecorder()
        build(llm, registry, config, sink=recorder).run("do a complex thing please")
        assert recorder.kinds() == [
            "turn_started", "plan_ready", "tool_started", "tool_finished",
            "model_message", "reflection", "turn_finished",
        ]

    def test_thoughts_are_surfaced(self, config, registry):
        llm = FakeLLM([{"text": "answer", "thoughts": ["I should check X"]}])
        recorder = EventRecorder()
        build(llm, registry, config, sink=recorder).run("go")
        assert recorder.of("thought")[0].text == "I should check X"


class TestTimeoutsAreReal:
    """A tool that never returns must not be able to hang the whole turn."""

    def test_a_hanging_tool_does_not_block_the_loop(self):
        import threading

        never = threading.Event()
        registry = ToolRegistry()

        @registry.tool()
        def hang() -> str:
            """Block until the test releases it."""
            never.wait(30)
            return "eventually"

        config = AgentConfig(
            planning="never", max_reflections=0, tool_timeout_seconds=0.1
        )
        llm = FakeLLM([tool_turn(("hang", {})), text_turn("moved on")])

        started = time.monotonic()
        try:
            result = build(llm, registry, config).run("go")
            elapsed = time.monotonic() - started
        finally:
            never.set()   # release the worker thread

        assert result.answer == "moved on"
        assert elapsed < 2.0, f"the turn took {elapsed:.1f}s despite a 0.1s timeout"

    def test_one_deadline_covers_the_whole_batch(self):
        """Eight queued calls must not each be allowed the full timeout."""
        registry = ToolRegistry()

        @registry.tool()
        def crawl(marker: int) -> str:
            """Take a while."""
            time.sleep(2)
            return str(marker)

        config = AgentConfig(
            planning="never", max_reflections=0, parallel_tools=False,
            tool_timeout_seconds=0.3,
        )
        llm = FakeLLM(
            [tool_turn(*[("crawl", {"marker": i}) for i in range(4)]),
             text_turn("gave up on those")]
        )
        started = time.monotonic()
        build(llm, registry, config).run("go")
        elapsed = time.monotonic() - started
        assert elapsed < 1.5, f"batch took {elapsed:.1f}s - deadline not shared"
