"""Retrieval as a tool, rather than as a separate application.

Classic RAG is one-shot: embed the question, fetch k chunks, answer. That fails
whenever the answer is not reachable from the question's own wording - a
question about "the penalty for late filing" will not retrieve a section headed
"Remedies" unless something rephrases the query.

Exposing retrieval as a *tool* hands that decision to the agent. It can search,
read what came back, notice the gap, and search again with the vocabulary the
document actually uses - multi-hop retrieval, for the cost of a tool
declaration. It can also combine retrieval with its other tools, which a
standalone RAG app cannot: look up a figure in the PDF, then calculate with it.
"""

from __future__ import annotations

DESCRIPTION = (
    "Search the document the user loaded and return the most relevant "
    "passages, with their page numbers. Search more than once if the first "
    "passages are not enough - rephrase using the vocabulary the document "
    "itself uses. Answer only from what comes back, and cite the pages."
)

SUMMARY_DESCRIPTION = (
    "List what documents are loaded and how large they are. Call this if you "
    "are unsure whether there is a document to search at all."
)


def search_documents(ctx, query: str, passages: int = 0) -> str:
    """Search the loaded document for passages relevant to a query."""
    library = getattr(ctx, "documents", None)
    if library is None or library.index.is_empty:
        return (
            "ERROR: no document is loaded. Tell the user to upload one, and "
            "answer from your own knowledge if you can."
        )

    query = query.strip()
    if not query:
        return "ERROR: the search query is empty."

    k = passages if passages > 0 else ctx.config.retrieval_top_k
    k = max(1, min(12, k))

    try:
        hits = library.search(query, k=k)
    except Exception as exc:
        return f"ERROR: the document search failed ({type(exc).__name__}: {exc})."

    if not hits:
        return f"No passages in the document matched '{query}'."

    # A top score this low means the document almost certainly does not cover
    # the question; saying so stops the model dressing up noise as an answer.
    if hits[0].score < 0.25:
        header = (
            f"WARNING: the closest passage scores only {hits[0].score:.2f}, so "
            "the document probably does not cover this. Say so rather than "
            "inferring an answer from the passages below.\n\n"
        )
    else:
        header = ""

    return header + "\n\n---\n\n".join(hit.render() for hit in hits)


def document_summary(ctx) -> str:
    """Report which documents are currently loaded and searchable."""
    library = getattr(ctx, "documents", None)
    if library is None or library.index.is_empty:
        return "No document is loaded."
    return library.index.describe()


def register(registry) -> None:
    """Add the document tools to a registry (only worth doing once one is loaded)."""
    registry.tool(
        name="search_documents",
        description=DESCRIPTION,
        describe={
            "query": (
                "What to look for. Use the words the document would use, not "
                "necessarily the user's words."
            ),
            "passages": "How many passages to return (1-12). 0 uses the default.",
        },
    )(search_documents)

    registry.tool(name="document_summary", description=SUMMARY_DESCRIPTION)(
        document_summary
    )
