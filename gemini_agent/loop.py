"""The agentic loop: plan, act, observe, critique, revise.

The shape of one user turn:

    intake
      |
      v
    [ PLAN ]  goal, steps, checkable success criteria
      |
      v
    [ ACT ] --- wants tools? --yes--> [ OBSERVE ] --> back to ACT
      |                                run in parallel, cache repeats,
      no                               turn errors into usable hints
      |
      v
    draft answer
      |
      v
    [ CRITIQUE ]  re-read against the criteria and the evidence
      |
      +-- revise --> back to ACT, with the concrete fixes attached
      |
      +-- accept --> final answer

What each phase buys, over calling the model in a `while True`:

* **Plan** turns an implicit strategy into explicit success criteria, and
  labels trivial questions ``direct`` so they skip the machinery entirely.
* **Observe** runs independent tool calls in parallel, caches repeats within a
  turn, and turns failures into *instructions the model can act on* — so a bad
  argument gets repaired on the next step instead of ending the turn.
* **Critique** re-reads the draft against the criteria and the evidence, and
  can send it back for another pass with concrete fixes attached.
* **Budgets** are checked in four dimensions, and running out produces a real
  answer ("here is what I have, this part is unverified") rather than a stub.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from google.genai import types

from .budget import Budget
from .config import AgentConfig
from .errors import AgentError, LLMError, ModelBlocked
from .events import (
    AgentErrorEvent, AgentEvent, BudgetWarning, ModelMessage, PlanReady,
    Reflection, Sink, Thought, ToolFinished, ToolStarted, TurnFinished,
    TurnStarted, null_sink,
)
from .llm import LLMClient, ToolCall, Usage
from .memory import WorkingMemory
from .reasoning import Critique, Plan, critique_answer, make_plan
from .registry import ToolRegistry, ToolResult

log = logging.getLogger(__name__)

TRUNCATION_NOTE = "\n\n[… output truncated. Narrow the query if you need more.]"


@dataclass
class ToolContext:
    """What tools get handed at call time.

    Tools declare a parameter named ``ctx`` to receive this; it is stripped from
    the schema the model sees, so a tool can reach shared services (the search
    client, the document index, the agent's own memory, sub-agents) without any
    of it leaking into the function-calling interface.
    """

    config: AgentConfig
    memory: WorkingMemory
    depth: int = 0
    search_client: Any = None
    documents: Any = None
    llm: Optional[LLMClient] = None
    registry: Optional[ToolRegistry] = None
    spawn: Optional[Callable[[str, Optional[Sequence[str]]], str]] = None
    emit: Sink = null_sink


@dataclass
class TurnResult:
    """Everything one ``Agent.run`` produced, beyond the answer text."""

    answer: str
    plan: Optional[Plan] = None
    critiques: List[Critique] = field(default_factory=list)
    steps: int = 0
    tool_calls: int = 0
    elapsed: float = 0.0
    usage: Usage = field(default_factory=Usage)
    stopped_early: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Agent:
    """One agent: a model, a toolbox, a memory and a budget."""

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        config: AgentConfig,
        memory: Optional[WorkingMemory] = None,
        system_prompt: str = "",
        sink: Optional[Sink] = None,
        context: Optional[ToolContext] = None,
        depth: int = 0,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.config = config
        self.memory = memory if memory is not None else WorkingMemory(config)
        self.system_prompt = system_prompt
        self.sink: Sink = sink or null_sink
        self.depth = depth
        self.budget = Budget.from_config(config)
        self.context = context or ToolContext(
            config=config, memory=self.memory, depth=depth
        )
        self.context.llm = llm
        self.context.registry = registry
        self.context.emit = self._emit
        if self.context.spawn is None:
            self.context.spawn = self._spawn_sub_agent

        # Per-turn state, reset by _begin_turn.
        self._tool_cache: Dict[str, ToolResult] = {}
        self._failures: Dict[str, int] = {}
        self._empty_replies = 0
        self._turn_marker = 0

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------
    def run(self, user_message: str, sink: Optional[Sink] = None) -> TurnResult:
        """Run one full user turn and return the answer plus a trace summary."""
        previous_sink = self.sink
        if sink is not None:
            self.sink = sink
            self.context.emit = self._emit

        usage_before = self._usage_snapshot()
        self._begin_turn(user_message)
        self._emit(TurnStarted(user_message=user_message, depth=self.depth))

        try:
            result = self._run_turn(user_message)
        except (LLMError, ModelBlocked, AgentError) as exc:
            self.memory.rollback_to(self._turn_marker)
            message = str(exc)
            self._emit(AgentErrorEvent(message=message, recoverable=False))
            result = TurnResult(answer=f"I hit an error: {message}", error=message)
        except Exception as exc:  # pragma: no cover - genuinely unexpected
            log.exception("unexpected failure in agent turn")
            self.memory.rollback_to(self._turn_marker)
            message = f"{type(exc).__name__}: {exc}"
            self._emit(AgentErrorEvent(message=message, recoverable=False))
            result = TurnResult(answer=f"I hit an error: {message}", error=message)
        finally:
            self.sink = previous_sink if sink is not None else self.sink
            self.context.emit = self._emit

        result.steps = self.budget.steps_used
        result.tool_calls = self.budget.tool_calls_used
        result.elapsed = self.budget.elapsed
        result.usage = self._usage_delta(usage_before)

        self._emit(
            TurnFinished(
                answer=result.answer,
                steps=result.steps,
                tool_calls=result.tool_calls,
                reflections=len(result.critiques),
                elapsed=result.elapsed,
                prompt_tokens=result.usage.prompt_tokens,
                output_tokens=result.usage.output_tokens,
                thought_tokens=result.usage.thought_tokens,
            )
        )
        return result

    # ------------------------------------------------------------------
    # turn orchestration
    # ------------------------------------------------------------------
    def _begin_turn(self, user_message: str) -> None:
        self._tool_cache.clear()
        self._failures.clear()
        self._empty_replies = 0
        self.budget.start()

        if self.memory.needs_compaction():
            self.memory.compact(self.llm)

        self._turn_marker = self.memory.marker()
        self.memory.append_user(user_message)

    def _run_turn(self, user_message: str) -> TurnResult:
        plan = self._plan(user_message)
        critiques: List[Critique] = []
        evidence_marker = self.memory.marker()

        answer = ""
        stopped_early: Optional[str] = None

        while True:
            answer, stopped_early = self._act(plan)

            if stopped_early or not self.budget.can_reflect():
                break

            critique = self._reflect(
                user_message=user_message,
                draft=answer,
                plan=plan,
                evidence=self.memory.transcript[evidence_marker:],
            )
            if critique is None or critique.accepted:
                break

            critiques.append(critique)
            self.budget.spend_reflection()

            if critique.needs_replan and plan is not None:
                replan = self._plan(user_message, previous=plan, critique=critique)
                if replan is not None:
                    plan = replan

            # The critique enters the transcript as a real turn, so the model
            # answers it the way it would answer a user asking for a fix.
            self.memory.append_user(critique.as_guidance())

        return TurnResult(
            answer=answer,
            plan=plan,
            critiques=critiques,
            stopped_early=stopped_early,
        )

    # ------------------------------------------------------------------
    # phase 1: plan
    # ------------------------------------------------------------------
    def _plan(
        self,
        user_message: str,
        previous: Optional[Plan] = None,
        critique: Optional[Critique] = None,
    ) -> Optional[Plan]:
        if self.config.planning == "never":
            return None
        if self.config.planning == "auto" and previous is None:
            if not self._worth_planning(user_message):
                return None

        try:
            plan = make_plan(
                self.llm,
                self.config,
                user_message=user_message,
                tool_catalogue=self.registry.catalogue(),
                history=self.memory.transcript[:-1],
                previous=previous,
                critique=critique,
            )
        except Exception as exc:
            # Planning is advisory: a failure downgrades to a plain ReAct loop.
            log.info("planning failed, continuing without a plan: %s", exc)
            self._emit(
                AgentErrorEvent(message=f"planning skipped ({exc})", recoverable=True)
            )
            return None

        if plan is None:
            return None

        self._emit(
            PlanReady(
                goal=plan.goal,
                complexity=plan.complexity,
                steps=[s.render(i) for i, s in enumerate(plan.steps, 1)],
                success_criteria=plan.success_criteria,
                replanned=previous is not None,
            )
        )
        # A plan that says "no tools needed" has done its job by saying so.
        return None if plan.is_direct and previous is None else plan

    def _worth_planning(self, user_message: str) -> bool:
        """Cheap triage so 'hello' does not cost a planning round trip.

        Only a heuristic, and only a *lower* bound: anything it lets through
        still gets judged ``direct`` by the planner itself, which is the real
        decision. This just avoids paying for the call on obviously trivial input.
        """
        text = user_message.strip()
        if len(text) < 24 and "?" not in text:
            return False
        signals = (
            " and ", " then ", " compare", " versus", " vs ", " each ", " both ",
            " after ", " before ", " why ", " how many", " calculate", " plan ",
            " step", " list ", " summarise", " summarize", " difference",
        )
        lowered = f" {text.lower()} "
        if any(signal in lowered for signal in signals):
            return True
        return len(text) > 120 or text.count("?") > 1

    # ------------------------------------------------------------------
    # phase 2+3: act and observe
    # ------------------------------------------------------------------
    def _act(self, plan: Optional[Plan]) -> Tuple[str, Optional[str]]:
        """Drive model/tool rounds until there is a draft answer or a budget stop."""
        tools = self.registry.declarations()

        while True:
            exhausted = self.budget.exhausted()
            if exhausted:
                self._emit(BudgetWarning(reason=exhausted))
                return self._forced_answer(exhausted), exhausted

            self.budget.spend_step()
            step = self.budget.steps_used

            response = self.llm.generate(
                self.memory.contents_for_model(self._preamble(plan)),
                system_instruction=self.system_prompt,
                tools=tools,
            )
            self.memory.append(response.content)

            for thought in response.thoughts:
                self._emit(Thought(text=thought, step=step))

            if response.tool_calls:
                if response.text:
                    self._emit(ModelMessage(text=response.text, step=step))
                parts = self._observe(response.tool_calls, step)
                self.memory.append(types.Content(role="user", parts=parts))
                continue

            if response.text:
                self._empty_replies = 0
                self._emit(
                    ModelMessage(
                        text=response.text,
                        step=step,
                        is_draft=self.budget.can_reflect(),
                    )
                )
                return response.text, None

            # No text and no tool call. Nudge once, then stop guessing.
            self._empty_replies += 1
            if self._empty_replies >= 2:
                reason = "the model returned an empty response twice"
                self._emit(BudgetWarning(reason=reason))
                return self._forced_answer(reason), reason
            self.memory.append_user(
                "That reply was empty. Either call a tool or give the answer."
            )

    def _preamble(self, plan: Optional[Plan]) -> Optional[str]:
        """Per-turn context injected ahead of the transcript, not stored in it."""
        if plan is None:
            return None
        remaining = self.budget.tool_calls_left()
        return (
            "Your plan for this turn (adapt it if the evidence says otherwise):\n"
            f"{plan.render()}\n\n"
            f"Budget left this turn: {remaining} tool calls, "
            f"{self.budget.remaining_seconds:.0f}s."
        )

    def _observe(self, calls: Sequence[ToolCall], step: int) -> List[types.Part]:
        """Execute a batch of tool calls and package the results for the model."""
        allowance = self.budget.tool_calls_left()
        runnable = list(calls[:allowance]) if allowance else []
        refused = list(calls[allowance:])

        results = self._execute(runnable, step)
        self.budget.spend_tool_calls(len(runnable))

        parts: List[types.Part] = []
        for call, result in zip(runnable, results):
            parts.append(
                types.Part.from_function_response(
                    name=call.name,
                    response={"result": self._shape_observation(call, result)},
                )
            )
        for call in refused:
            parts.append(
                types.Part.from_function_response(
                    name=call.name,
                    response={
                        "result": (
                            "ERROR: the tool-call budget for this turn is spent. "
                            "Answer with what you already have."
                        )
                    },
                )
            )
        return parts

    def _execute(
        self, calls: Sequence[ToolCall], step: int
    ) -> List[ToolResult]:
        """Run calls, in parallel when that is safe, with per-call timeouts."""
        if not calls:
            return []

        specs = [
            self.registry.get(c.name) if c.name in self.registry else None
            for c in calls
        ]
        serial = (
            not self.config.parallel_tools
            or len(calls) == 1
            or any(spec is None or not spec.concurrent_safe for spec in specs)
        )
        workers = 1 if serial else min(self.config.max_parallel_tools, len(calls))

        results: List[Optional[ToolResult]] = [None] * len(calls)
        pending: Dict[Any, int] = {}

        # Deliberately not a `with` block: ThreadPoolExecutor.__exit__ waits for
        # every future, so a tool that hangs forever would block here and the
        # per-call timeout below would mean nothing. We shut down without
        # waiting instead, and let a stuck worker thread die with the process.
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            for index, call in enumerate(calls):
                self._emit(
                    ToolStarted(
                        call_id=call.id, name=call.name, args=call.args, step=step
                    )
                )
                cached = self._cached(call)
                if cached is not None:
                    results[index] = cached
                    self._emit(
                        ToolFinished(
                            call_id=call.id,
                            name=call.name,
                            ok=cached.ok,
                            preview=cached.preview(),
                            duration=0.0,
                            step=step,
                            cached=True,
                        )
                    )
                    continue

                future = pool.submit(
                    self.registry.invoke, call.name, call.args, self.context
                )
                pending[future] = index

            deadline = time.monotonic() + self.config.tool_timeout_seconds
            for future, index in pending.items():
                call = calls[index]
                try:
                    # One shared deadline for the batch: waiting the full
                    # timeout on each of eight queued calls in turn could
                    # otherwise blow the turn's time budget eight times over.
                    result = future.result(
                        timeout=max(0.0, deadline - time.monotonic())
                    )
                except FutureTimeout:
                    future.cancel()
                    result = ToolResult(
                        name=call.name,
                        ok=False,
                        content=(
                            f"ERROR: {call.name} timed out after "
                            f"{self.config.tool_timeout_seconds:.0f}s."
                        ),
                        duration=self.config.tool_timeout_seconds,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    result = ToolResult(
                        name=call.name,
                        ok=False,
                        content=f"ERROR: {call.name} failed: {exc}",
                    )

                results[index] = result
                self._record(call, result)
                self._emit(
                    ToolFinished(
                        call_id=call.id,
                        name=call.name,
                        ok=result.ok,
                        preview=result.preview(),
                        duration=result.duration,
                        step=step,
                    )
                )
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        # Never drop a slot: a missing result would silently misalign the
        # observations from the calls they answer.
        return [
            result
            if result is not None
            else ToolResult(
                name=calls[index].name,
                ok=False,
                content=f"ERROR: {calls[index].name} produced no result.",
            )
            for index, result in enumerate(results)
        ]

    def _cached(self, call: ToolCall) -> Optional[ToolResult]:
        spec = self.registry.get(call.name) if call.name in self.registry else None
        if spec is None or not spec.cacheable:
            return None
        hit = self._tool_cache.get(call.signature())
        if hit is None:
            return None
        return ToolResult(
            name=hit.name, ok=hit.ok, content=hit.content, duration=0.0, cached=True
        )

    def _record(self, call: ToolCall, result: ToolResult) -> None:
        signature = call.signature()
        if result.ok:
            spec = self.registry.get(call.name) if call.name in self.registry else None
            if spec is not None and spec.cacheable:
                self._tool_cache[signature] = result
            self._failures.pop(signature, None)
        else:
            self._failures[signature] = self._failures.get(signature, 0) + 1

    def _shape_observation(self, call: ToolCall, result: ToolResult) -> str:
        """Trim the payload and, on repeated failure, tell the model to move on."""
        content = result.content
        limit = self.config.observation_char_limit
        if len(content) > limit:
            content = content[:limit] + TRUNCATION_NOTE
        if result.cached:
            content = f"(identical call already made this turn — cached result)\n{content}"

        failures = self._failures.get(call.signature(), 0)
        if not result.ok and failures >= self.config.max_repeat_failures:
            content += (
                f"\n\nThis exact call has now failed {failures} times. Do not "
                "repeat it. Either change the arguments, use a different tool, "
                "or answer without it and say what you could not check."
            )
        return content

    # ------------------------------------------------------------------
    # phase 4: critique
    # ------------------------------------------------------------------
    def _reflect(
        self,
        *,
        user_message: str,
        draft: str,
        plan: Optional[Plan],
        evidence: Sequence[types.Content],
    ) -> Optional[Critique]:
        if not draft.strip():
            return None
        try:
            critique = critique_answer(
                self.llm,
                self.config,
                user_message=user_message,
                draft=draft,
                plan=plan,
                evidence=evidence,
            )
        except Exception as exc:
            log.info("reflection failed, accepting the draft: %s", exc)
            return None

        if critique is None:
            return None

        self._emit(
            Reflection(
                verdict=critique.verdict,
                confidence=critique.confidence,
                issues=critique.issues,
                next_actions=critique.next_actions,
                iteration=self.budget.reflections_used + 1,
            )
        )
        return critique

    # ------------------------------------------------------------------
    # graceful degradation
    # ------------------------------------------------------------------
    def _forced_answer(self, reason: str) -> str:
        """Out of budget: get a real answer from what we have, not a stub."""
        self.memory.append_user(
            f"Stop working now — you {reason}. Answer the user with what you "
            "already have. State plainly which parts you could not verify. Do "
            "not call any more tools."
        )
        try:
            response = self.llm.generate(
                self.memory.contents_for_model(),
                system_instruction=self.system_prompt,
                tools=None,
                thinking_budget=0,
                include_thoughts=False,
            )
        except Exception as exc:
            log.info("forced answer failed: %s", exc)
            return (
                f"I stopped because I {reason}, and could not compose a final "
                f"summary ({exc}). Try narrowing the question."
            )

        self.memory.append(response.content)
        text = response.text.strip()
        return text or f"I stopped because I {reason}, without reaching an answer."

    # ------------------------------------------------------------------
    # delegation
    # ------------------------------------------------------------------
    def _spawn_sub_agent(
        self, task: str, tool_names: Optional[Sequence[str]] = None
    ) -> str:
        """Run a focused sub-agent on one subtask with a fresh, narrow context.

        The point is isolation: the sub-agent gets its own memory, so a noisy
        20-result search never reaches the parent's transcript — only the
        distilled answer does.
        """
        if self.depth >= self.config.max_delegation_depth:
            return (
                "ERROR: delegation depth limit reached. Do this subtask yourself "
                "with the tools you already have."
            )

        names = list(tool_names) if tool_names else self.registry.names
        allowed = [n for n in names if n in self.registry and n != "delegate"]
        if not allowed:
            return (
                "ERROR: none of the requested tools exist. Available: "
                + ", ".join(n for n in self.registry.names if n != "delegate")
            )

        sub_config = self.config.for_delegate()
        sub_registry = self.registry.subset(allowed)
        sub_memory = WorkingMemory(sub_config)
        sub_context = ToolContext(
            config=sub_config,
            memory=sub_memory,
            depth=self.depth + 1,
            search_client=self.context.search_client,
            documents=self.context.documents,
        )
        sub_agent = Agent(
            llm=self.llm,
            registry=sub_registry,
            config=sub_config,
            memory=sub_memory,
            system_prompt=(
                "You are a focused sub-agent. You were given one subtask by "
                "another agent. Use your tools to settle it, then reply with "
                "the findings only — facts, figures and sources, no preamble "
                "and no offers of further help. If you cannot settle it, say "
                "exactly what is missing."
            ),
            sink=self.sink,
            context=sub_context,
            depth=self.depth + 1,
        )
        result = sub_agent.run(task)
        # Sub-agent spending counts against the parent's tool budget too.
        self.budget.spend_tool_calls(result.tool_calls)
        return result.answer

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------
    def _emit(self, event: AgentEvent) -> None:
        try:
            self.sink(event)
        except Exception:  # a broken renderer must not break the agent
            log.exception("event sink raised")

    def _usage_snapshot(self) -> Usage:
        total = getattr(self.llm, "total_usage", None)
        if total is None:
            return Usage()
        return Usage(
            prompt_tokens=total.prompt_tokens,
            output_tokens=total.output_tokens,
            thought_tokens=total.thought_tokens,
            total_tokens=total.total_tokens,
            calls=total.calls,
        )

    def _usage_delta(self, before: Usage) -> Usage:
        total = getattr(self.llm, "total_usage", None)
        if total is None:
            return Usage()
        return Usage(
            prompt_tokens=total.prompt_tokens - before.prompt_tokens,
            output_tokens=total.output_tokens - before.output_tokens,
            thought_tokens=total.thought_tokens - before.thought_tokens,
            total_tokens=total.total_tokens - before.total_tokens,
            calls=total.calls - before.calls,
        )
