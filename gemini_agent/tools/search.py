"""Web tools: search, and fetching one page in full.

Search alone is a shallow tool — it returns snippets, and an agent that only
ever sees snippets will confidently summarise a headline it never read. Pairing
``web_search`` with ``fetch_page`` gives the loop somewhere to go when the
snippets disagree or are too thin: find the source, then read it.

Results are deduplicated by URL and trimmed, because the model pays for every
character of an observation and duplicate hits are common across queries.
"""

from __future__ import annotations

import html
import re
import urllib.error
import urllib.request
from typing import Any, List, Optional

_TAG = re.compile(r"<[^>]+>")
_SCRIPT = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.DOTALL | re.I)
_WHITESPACE = re.compile(r"\n\s*\n\s*\n+")

SEARCH_DESCRIPTION = (
    "Search the web for current information: news, events, prices, schedules, "
    "releases, statistics, or anything that may have changed since your "
    "training data. Returns titles, URLs and summaries. Prefer several narrow "
    "searches over one broad one, and search again with different wording if "
    "the first results are thin."
)

FETCH_DESCRIPTION = (
    "Fetch the readable text of one web page, given its URL. Use this after "
    "web_search when a result looks authoritative and you need the detail "
    "behind the summary, or when the user gives you a link directly."
)


def _clean_html(raw: str, limit: int) -> str:
    text = _SCRIPT.sub(" ", raw)
    text = _TAG.sub("\n", text)
    text = html.unescape(text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _WHITESPACE.sub("\n\n", text).strip()
    if len(text) > limit:
        text = text[:limit] + "\n\n[... page truncated ...]"
    return text


def search_web(ctx, query: str, max_results: int = 0) -> str:
    """Search the web and return the most relevant results."""
    client = getattr(ctx, "search_client", None)
    if client is None:
        return (
            "ERROR: web search is not configured (no TAVILY_API_KEY). "
            "Answer from your own knowledge and say it may not be current."
        )

    query = query.strip()
    if not query:
        return "ERROR: the search query is empty."

    limit = max_results if max_results > 0 else ctx.config.search_max_results
    limit = max(1, min(10, limit))

    try:
        payload = client.search(
            query=query,
            max_results=limit,
            search_depth=ctx.config.search_depth,
            include_answer="basic",
        )
    except Exception as exc:
        return f"ERROR: the search failed ({type(exc).__name__}: {exc})."

    blocks: List[str] = []
    answer = (payload or {}).get("answer")
    if answer:
        blocks.append(f"Quick answer: {answer}")

    seen = set()
    for item in (payload or {}).get("results", []):
        url = (item.get("url") or "").strip()
        if url and url in seen:
            continue
        seen.add(url)
        content = " ".join((item.get("content") or "").split())
        if len(content) > 1200:
            content = content[:1200] + "..."
        blocks.append(
            f"Title:   {item.get('title', '(untitled)')}\n"
            f"URL:     {url or '(no url)'}\n"
            f"Summary: {content}"
        )

    if not blocks:
        return (
            f"No results for '{query}'. Try different wording, or fewer and "
            "more distinctive terms."
        )
    return "\n\n---\n\n".join(blocks)


def fetch_page(ctx, url: str) -> str:
    """Fetch one web page and return its readable text."""
    url = url.strip()
    if not url:
        return "ERROR: no URL given."
    if not url.startswith(("http://", "https://")):
        return f"ERROR: '{url}' is not an http(s) URL."

    limit = ctx.config.fetch_char_limit

    # Tavily's extractor strips navigation and ads far better than we can, so
    # try it first and keep the raw fetch as a fallback.
    client = getattr(ctx, "search_client", None)
    if client is not None and hasattr(client, "extract"):
        try:
            payload = client.extract(urls=[url])
            for item in (payload or {}).get("results", []):
                content = (item.get("raw_content") or "").strip()
                if content:
                    if len(content) > limit:
                        content = content[:limit] + "\n\n[... page truncated ...]"
                    return f"Content of {url}:\n\n{content}"
        except Exception:
            pass  # fall through to the plain fetch

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; gemini-agent/1.0)",
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            content_type = response.headers.get("Content-Type", "")
            body = response.read(2_000_000)
    except urllib.error.HTTPError as exc:
        return f"ERROR: {url} returned HTTP {exc.code} ({exc.reason})."
    except Exception as exc:
        return f"ERROR: could not fetch {url} ({type(exc).__name__}: {exc})."

    charset = "utf-8"
    if "charset=" in content_type:
        charset = content_type.split("charset=", 1)[1].split(";")[0].strip() or "utf-8"
    try:
        text = body.decode(charset, errors="replace")
    except LookupError:
        text = body.decode("utf-8", errors="replace")

    if "html" in content_type.lower() or text.lstrip()[:200].lower().startswith(
        ("<!doctype", "<html")
    ):
        text = _clean_html(text, limit)
    elif len(text) > limit:
        text = text[:limit] + "\n\n[... page truncated ...]"

    if not text.strip():
        return f"ERROR: {url} returned no readable text."
    return f"Content of {url}:\n\n{text}"


def register(registry, *, include_fetch: bool = True) -> None:
    """Add the web tools to a registry."""
    registry.tool(
        name="web_search",
        description=SEARCH_DESCRIPTION,
        describe={
            "query": "What to search for. Keywords work better than questions.",
            "max_results": "How many results to return (1-10). 0 uses the default.",
        },
    )(search_web)

    if include_fetch:
        registry.tool(
            name="fetch_page",
            description=FETCH_DESCRIPTION,
            describe={"url": "The full http(s) URL of the page to read."},
        )(fetch_page)


def make_search_client(api_key: Optional[str]) -> Any:
    """Build a Tavily client, or ``None`` when no key is configured.

    Imported lazily so the package works — minus web search — without
    tavily-python installed.
    """
    if not api_key:
        return None
    try:
        from tavily import TavilyClient
    except ImportError:
        return None
    return TavilyClient(api_key=api_key)
