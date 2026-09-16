"""Delegation: letting the agent spawn a focused sub-agent.

Two things go wrong when one agent does everything in one context. Research is
*noisy* - comparing five options means five searches whose raw output swamps
the transcript and buries the reasoning. And it is *serial* - each subtask
drags the whole accumulated history along with it.

``delegate`` gives a subtask its own agent, with its own empty memory and a
narrowed toolbox. The sub-agent burns through the noise and returns only its
findings; the parent's transcript gains a paragraph instead of ten thousand
characters of search results. Depth is capped by config, and the sub-agent's
tool calls are charged to the parent's budget, so this cannot become a fork
bomb.
"""

from __future__ import annotations

from typing import List

DESCRIPTION = (
    "Hand one self-contained subtask to a focused sub-agent and get back just "
    "its findings. Worth it when a subtask needs several tool calls of its own "
    "- researching one option out of several to compare, or digging through a "
    "document for one specific figure. Not worth it for anything you can do in "
    "a single tool call: the sub-agent starts with no knowledge of this "
    "conversation, so the task must be written to stand completely alone."
)


def delegate(ctx, task: str, tools: List[str]) -> str:
    """Give a self-contained subtask to a focused sub-agent."""
    task = task.strip()
    if not task:
        return "ERROR: 'task' was empty."
    if len(task) < 15:
        return (
            "ERROR: that task is too short to stand alone. The sub-agent sees "
            "none of this conversation, so spell out everything it needs."
        )

    spawn = getattr(ctx, "spawn", None)
    if spawn is None:
        return "ERROR: delegation is not available in this configuration."

    return spawn(task, tools or None)


def register(registry) -> None:
    registry.tool(
        name="delegate",
        description=DESCRIPTION,
        describe={
            "task": (
                "The complete, self-contained subtask. Include every detail "
                "the sub-agent needs - it cannot see this conversation."
            ),
            "tools": (
                "Which tools the sub-agent may use, e.g. ['web_search', "
                "'fetch_page']. Keep it to what the subtask genuinely needs."
            ),
        },
        cacheable=False,
    )(delegate)
