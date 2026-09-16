"""Tool assembly: build the toolbox that matches what is actually available.

Registering a tool the agent cannot use is worse than not having it - the model
will plan around ``web_search``, call it, and get an error back. So the
registry is assembled from what is configured: no Tavily key means no web
tools, no loaded document means no retrieval tools.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..registry import ToolRegistry
from . import calc, clock, delegate, documents, notes, search


def build_registry(
    *,
    search_client=None,
    with_documents: bool = False,
    with_delegation: bool = True,
    with_notes: bool = True,
    only: Optional[Sequence[str]] = None,
) -> ToolRegistry:
    """Assemble a registry from the capabilities that are actually present."""
    registry = ToolRegistry()

    calc.register(registry)
    clock.register(registry)

    if search_client is not None:
        search.register(registry)
    if with_documents:
        documents.register(registry)
    if with_notes:
        notes.register(registry)
    # Delegation is pointless with nothing to delegate.
    if with_delegation and len(registry) > 2:
        delegate.register(registry)

    return registry.subset(only) if only else registry


__all__ = ["build_registry", "calc", "clock", "delegate", "documents", "notes", "search"]
