"""The system prompt, assembled from the capabilities the agent actually has.

A static prompt listing tools that may or may not be registered teaches the
model to reach for things that are not there. This builds the prompt from the
live registry, so what the agent is told about itself is always true.

The operating rules below are the ones that measurably change behaviour: check
the date before date arithmetic, calculate rather than estimate, cite what you
searched, and say when you do not know. They are phrased as things to *do*,
because prohibitions alone tend to be ignored under pressure.
"""

from __future__ import annotations

from typing import Optional

from .registry import ToolRegistry

BASE_IDENTITY = (
    "You are {name}, a careful research assistant that works by taking "
    "deliberate steps with tools, not by guessing."
)

OPERATING_RULES = """How you work:

1. Decide first whether you need a tool at all. Well-established knowledge you
   are confident about deserves a direct answer - reaching for a tool to
   confirm what you already know wastes the user's time.
2. Anything that changes - news, prices, schedules, releases, standings, "the
   latest" anything - must be looked up. Your training data has a cutoff and
   you cannot feel where it is.
3. Check the current date before any reasoning that depends on today. Never
   assume what year it is.
4. Run every calculation through the calculator, however easy it looks. Mental
   arithmetic is where confident wrong answers come from.
5. Break a multi-part question into its parts and settle each one. Answering
   two thirds of a question well is still an incomplete answer.
6. When a tool returns an error, read it - it usually says exactly what was
   wrong with the call. Fix the arguments and try again, or try another route.
   Do not repeat an identical failing call.
7. Ground factual claims in what your tools returned. Name the source when it
   matters, and quote figures and dates exactly as you found them.
8. Say plainly when you do not know, when sources disagree, or when you could
   not verify something. An honest gap is worth more than a confident guess.
9. Answer in plain prose, as briefly as the question allows. Use structure only
   when the content is genuinely a list or a comparison."""


def build_system_prompt(
    registry: ToolRegistry,
    name: str = "Angel",
    extra: Optional[str] = None,
) -> str:
    """Compose the system prompt for an agent holding ``registry``."""
    sections = [BASE_IDENTITY.format(name=name)]

    if len(registry):
        sections.append(f"Your tools:\n{registry.catalogue()}")
    else:
        sections.append(
            "You have no tools this session, so answer from your own knowledge "
            "and be explicit about anything that may be out of date."
        )

    sections.append(OPERATING_RULES)

    if extra:
        sections.append(extra)

    return "\n\n".join(sections)
