"""Planning and self-critique — the two phases that make the loop deliberate.

A plain ReAct loop is reactive: it takes whatever step looks good next and
stops as soon as the model produces prose. Two extra phases make it think
harder about the *whole* task:

**Plan** (before acting) forces an explicit decomposition — goal, steps, and
concrete success criteria. The criteria are the valuable part: they give the
critic something objective to check against, rather than asking "is this a good
answer?" in the abstract.

**Critique** (after a draft) re-reads the draft against those criteria and the
evidence actually gathered, and returns a verdict. ``revise`` sends the draft
back into the acting loop with the critique attached, so the second pass is
informed rather than a blind retry.

Both calls are schema-constrained JSON with thinking turned down, because they
are short structured judgements, not open-ended reasoning. Both are advisory:
if either fails to parse, the agent carries on without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from google.genai import types

from .config import AgentConfig
from .llm import LLMClient, user_text
from .memory import render_transcript

# --- schemas ----------------------------------------------------------------

PLAN_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "complexity": types.Schema(
            type=types.Type.STRING,
            enum=["direct", "simple", "complex"],
            description=(
                "direct = answerable from knowledge with no tools; "
                "simple = one tool call; complex = several dependent steps"
            ),
        ),
        "goal": types.Schema(
            type=types.Type.STRING,
            description="One sentence: what a complete answer must deliver.",
        ),
        "steps": types.Schema(
            type=types.Type.ARRAY,
            description="Ordered steps. Empty for a direct answer.",
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "description": types.Schema(type=types.Type.STRING),
                    "tool": types.Schema(
                        type=types.Type.STRING,
                        description="Tool to use, or 'none' for pure reasoning.",
                    ),
                },
                required=["description"],
            ),
        ),
        "success_criteria": types.Schema(
            type=types.Type.ARRAY,
            description="Checkable conditions the final answer must satisfy.",
            items=types.Schema(type=types.Type.STRING),
        ),
        "risks": types.Schema(
            type=types.Type.ARRAY,
            description="What could make this answer wrong.",
            items=types.Schema(type=types.Type.STRING),
        ),
    },
    required=["complexity", "goal", "steps", "success_criteria"],
)

CRITIQUE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "verdict": types.Schema(
            type=types.Type.STRING,
            enum=["accept", "revise"],
            description="accept = ready to send; revise = more work needed.",
        ),
        "confidence": types.Schema(
            type=types.Type.NUMBER,
            description="0.0-1.0 confidence that the draft is correct and complete.",
        ),
        "issues": types.Schema(
            type=types.Type.ARRAY,
            description="Concrete defects: unsupported claims, gaps, stale data.",
            items=types.Schema(type=types.Type.STRING),
        ),
        "unmet_criteria": types.Schema(
            type=types.Type.ARRAY,
            description="Success criteria the draft does not yet satisfy.",
            items=types.Schema(type=types.Type.STRING),
        ),
        "next_actions": types.Schema(
            type=types.Type.ARRAY,
            description="Specific actions that would fix the issues.",
            items=types.Schema(type=types.Type.STRING),
        ),
        "needs_replan": types.Schema(
            type=types.Type.BOOLEAN,
            description="True when the plan itself was wrong, not just the draft.",
        ),
    },
    required=["verdict", "issues", "next_actions"],
)


# --- data ------------------------------------------------------------------


@dataclass
class PlanStep:
    description: str
    tool: str = "none"

    def render(self, index: int) -> str:
        suffix = f"  (tool: {self.tool})" if self.tool and self.tool != "none" else ""
        return f"{index}. {self.description}{suffix}"


@dataclass
class Plan:
    goal: str
    complexity: str = "simple"
    steps: List[PlanStep] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)

    @property
    def is_direct(self) -> bool:
        return self.complexity == "direct" and not self.steps

    @classmethod
    def from_dict(cls, data: Any) -> Optional["Plan"]:
        if not isinstance(data, dict):
            return None
        goal = str(data.get("goal", "")).strip()
        if not goal:
            return None
        steps = []
        for raw in data.get("steps") or []:
            if isinstance(raw, dict):
                description = str(raw.get("description", "")).strip()
                if description:
                    steps.append(
                        PlanStep(description, str(raw.get("tool", "none") or "none"))
                    )
            elif isinstance(raw, str) and raw.strip():
                steps.append(PlanStep(raw.strip()))
        complexity = str(data.get("complexity", "simple")).strip().lower()
        if complexity not in ("direct", "simple", "complex"):
            complexity = "simple"
        return cls(
            goal=goal,
            complexity=complexity,
            steps=steps,
            success_criteria=[str(c) for c in (data.get("success_criteria") or []) if c],
            risks=[str(r) for r in (data.get("risks") or []) if r],
        )

    def render(self) -> str:
        lines = [f"Goal: {self.goal}"]
        if self.steps:
            lines.append("Plan:")
            lines.extend(s.render(i) for i, s in enumerate(self.steps, 1))
        if self.success_criteria:
            lines.append("The answer is only complete when:")
            lines.extend(f"- {c}" for c in self.success_criteria)
        if self.risks:
            lines.append("Watch out for:")
            lines.extend(f"- {r}" for r in self.risks)
        return "\n".join(lines)


@dataclass
class Critique:
    verdict: str = "accept"
    confidence: float = 1.0
    issues: List[str] = field(default_factory=list)
    unmet_criteria: List[str] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)
    needs_replan: bool = False

    @property
    def accepted(self) -> bool:
        return self.verdict == "accept"

    @classmethod
    def from_dict(cls, data: Any) -> Optional["Critique"]:
        if not isinstance(data, dict):
            return None
        verdict = str(data.get("verdict", "accept")).strip().lower()
        if verdict not in ("accept", "revise"):
            verdict = "accept"
        try:
            confidence = float(data.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        return cls(
            verdict=verdict,
            confidence=max(0.0, min(1.0, confidence)),
            issues=[str(i) for i in (data.get("issues") or []) if i],
            unmet_criteria=[str(c) for c in (data.get("unmet_criteria") or []) if c],
            next_actions=[str(a) for a in (data.get("next_actions") or []) if a],
            needs_replan=bool(data.get("needs_replan", False)),
        )

    def as_guidance(self) -> str:
        """The critique rendered as an instruction for the next acting pass."""
        lines = [
            "A reviewer checked your draft and it is NOT ready to send. "
            "Fix it before answering again."
        ]
        if self.issues:
            lines.append("Problems found:")
            lines.extend(f"- {i}" for i in self.issues)
        if self.unmet_criteria:
            lines.append("Success criteria not yet met:")
            lines.extend(f"- {c}" for c in self.unmet_criteria)
        if self.next_actions:
            lines.append("Do this next:")
            lines.extend(f"- {a}" for a in self.next_actions)
        lines.append(
            "Use tools where the fix needs evidence. When everything above is "
            "addressed, reply with the corrected final answer only."
        )
        return "\n".join(lines)


# --- the two calls ----------------------------------------------------------

_PLANNER_ROLE = (
    "You are the planning stage of an AI agent. You do not answer the user; you "
    "decide how the answer will be produced.\n"
    "Judge the complexity honestly — most questions are 'direct' and need no "
    "tools at all, and over-planning a simple question wastes the user's time.\n"
    "Reserve 'complex' for tasks that genuinely need several dependent steps, "
    "where a later step needs a result from an earlier one.\n"
    "Success criteria must be checkable by reading the final answer."
)

_CRITIC_ROLE = (
    "You are the review stage of an AI agent. A draft answer is in front of you, "
    "along with the evidence that was gathered to write it.\n"
    "Be strict but proportionate. Answer 'revise' only for defects that would "
    "mislead the user: a claim the evidence does not support, a number that was "
    "guessed instead of calculated, a missing part of a multi-part question, or "
    "stale information where the question demanded current facts.\n"
    "Do NOT ask for revisions over style, length, tone, or extra detail nobody "
    "asked for. If the draft answers the question correctly, accept it."
)


def make_plan(
    llm: LLMClient,
    config: AgentConfig,
    *,
    user_message: str,
    tool_catalogue: str,
    history: Sequence[types.Content] = (),
    previous: Optional[Plan] = None,
    critique: Optional[Critique] = None,
) -> Optional[Plan]:
    """Draft (or redraft) a plan. Returns ``None`` if the model gave nothing usable."""
    sections = [f"Tools available to the agent:\n{tool_catalogue or '(none)'}"]

    context = render_transcript(history[-6:]) if history else ""
    if context:
        sections.append(f"Conversation so far:\n{context}")

    sections.append(f"The user now asks:\n{user_message}")

    if previous is not None and critique is not None:
        sections.append(
            "Your previous plan did not work:\n"
            f"{previous.render()}\n\n"
            "The reviewer said:\n"
            + "\n".join(f"- {i}" for i in critique.issues + critique.next_actions)
            + "\n\nProduce a better plan that addresses this."
        )

    data = llm.generate_json(
        [user_text("\n\n".join(sections))],
        schema=PLAN_SCHEMA,
        system_instruction=_PLANNER_ROLE,
        thinking_budget=config.meta_thinking_budget,
    )
    return Plan.from_dict(data)


def critique_answer(
    llm: LLMClient,
    config: AgentConfig,
    *,
    user_message: str,
    draft: str,
    plan: Optional[Plan],
    evidence: Sequence[types.Content] = (),
) -> Optional[Critique]:
    """Review a draft answer against the plan and the evidence gathered."""
    sections = [f"The user asked:\n{user_message}"]

    if plan is not None:
        sections.append(f"The agent planned:\n{plan.render()}")

    evidence_text = render_transcript(evidence, limit=800) if evidence else ""
    sections.append(
        f"Evidence gathered by tools:\n{evidence_text}"
        if evidence_text.strip()
        else "No tools were used — the draft comes from the model's own knowledge."
    )

    sections.append(f"Draft answer:\n{draft}")
    sections.append(
        "Decide: is this ready to send to the user? Check every part of the "
        "question is answered, every factual claim is supported by the evidence "
        "above or is uncontroversial general knowledge, and no number was "
        "guessed where it should have been calculated or looked up."
    )

    data = llm.generate_json(
        [user_text("\n\n".join(sections))],
        schema=CRITIQUE_SCHEMA,
        system_instruction=_CRITIC_ROLE,
        thinking_budget=config.meta_thinking_budget,
    )
    return Critique.from_dict(data)
