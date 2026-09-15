"""The current date and time.

Small, but it removes a whole class of quiet errors. A model asked "how many
days until the conference?" will otherwise anchor on its training cutoff and
answer with total confidence. Giving it a clock — and an explicit instruction
to check it before any date arithmetic — turns a wrong answer into a right one.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

DESCRIPTION = (
    "Get the current date and time. Call this before any reasoning that "
    "depends on today's date - ages, deadlines, 'how long until', 'the latest' "
    "or 'this year' - because your training data is older than today."
)


def current_datetime(offset_hours: float = 0.0) -> str:
    """Get the current date and time in UTC, optionally offset to a local zone."""
    if abs(offset_hours) > 14:
        return "ERROR: offset_hours must be between -14 and +14."

    now = datetime.now(timezone.utc)
    shifted = now + timedelta(hours=offset_hours)
    label = "UTC" if offset_hours == 0 else f"UTC{offset_hours:+g}"

    return (
        f"{shifted.strftime('%A, %d %B %Y, %H:%M')} {label} "
        f"(ISO: {shifted.strftime('%Y-%m-%d')}, "
        f"day {shifted.timetuple().tm_yday} of {shifted.year})"
    )


def register(registry) -> None:
    registry.tool(
        name="current_datetime",
        description=DESCRIPTION,
        describe={
            "offset_hours": (
                "Hours to offset from UTC for a local time, e.g. 3 for EAT or "
                "-5 for EST. Leave at 0 for UTC."
            )
        },
        cacheable=False,
    )(current_datetime)
