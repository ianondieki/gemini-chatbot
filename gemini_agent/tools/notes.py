"""A scratchpad the agent writes to itself.

Everything else the agent learns lives in the transcript, and the transcript is
compacted away as a conversation grows. Notes survive that: they are stored
separately and re-injected ahead of every request, so a constraint the user
mentioned twenty turns ago ("I'm in Nairobi", "budget is 50k") is still in
front of the model when it matters.

Writing a note is also a small act of reasoning - deciding what is worth
keeping is the same judgement that makes a long conversation coherent.
"""

from __future__ import annotations

REMEMBER_DESCRIPTION = (
    "Save a durable fact for the rest of this conversation: a user preference, "
    "a constraint, a figure you will need again. Notes survive history "
    "trimming, so save anything you would be annoyed to forget. Keep each note "
    "to one sentence. Do not save things you can recompute or look up again."
)

RECALL_DESCRIPTION = (
    "Read back the notes you saved earlier, optionally filtered by topic or "
    "keyword. Use this when the user refers to something from earlier in the "
    "conversation and you are not certain of the detail."
)


def remember(ctx, fact: str, topic: str = "general") -> str:
    """Save a fact to remember for the rest of the conversation."""
    fact = fact.strip()
    if not fact:
        return "ERROR: nothing to remember - 'fact' was empty."
    if len(fact) > 500:
        return "ERROR: that note is too long. Keep it to one sentence."

    existing = {n.text.lower() for n in ctx.memory.notes}
    if fact.lower() in existing:
        return f"Already noted: {fact}"

    note = ctx.memory.remember(fact, topic)
    return f"Noted under '{note.topic}': {note.text}"


def recall(ctx, topic: str = "") -> str:
    """Read back previously saved notes, optionally filtered."""
    notes = ctx.memory.recall(topic or None)
    if not notes:
        return (
            f"No notes match '{topic}'." if topic else "No notes saved yet."
        )
    return "\n".join(f"- {note.render()}" for note in notes)


def register(registry) -> None:
    registry.tool(
        name="remember",
        description=REMEMBER_DESCRIPTION,
        describe={
            "fact": "The single fact to save, written as a full sentence.",
            "topic": "A short label to file it under, e.g. 'preferences'.",
        },
        concurrent_safe=False,
        cacheable=False,
    )(remember)

    registry.tool(
        name="recall",
        description=RECALL_DESCRIPTION,
        describe={"topic": "Topic or keyword to filter by. Empty returns everything."},
        cacheable=False,
    )(recall)
