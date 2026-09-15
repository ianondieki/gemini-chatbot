"""Splitting documents into retrievable chunks.

The original splitter cut every ``size`` characters flat, which routinely
sliced sentences — and often numbers — in half, so a chunk could end mid-figure
and the model would retrieve a fact it could not read. This version slides the
cut backwards to the nearest sentence or word boundary within a small window,
and records where each chunk came from so answers can cite a page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

_SENTENCE_END = re.compile(r"[.!?](?:[\"')\]]+)?\s")
_WHITESPACE = re.compile(r"[ \t\r\f\v]+")


@dataclass
class Chunk:
    """One retrievable piece of a document, with enough provenance to cite it."""

    text: str
    index: int
    page: Optional[int] = None
    source: str = ""

    def citation(self) -> str:
        bits = [self.source] if self.source else []
        if self.page is not None:
            bits.append(f"p.{self.page}")
        bits.append(f"chunk {self.index}")
        return " ".join(bits)


def normalise(text: str) -> str:
    """Collapse runs of spaces but keep paragraph breaks, which carry meaning."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def _best_cut(text: str, start: int, hard_end: int, window: int) -> int:
    """Find a boundary at or just before ``hard_end`` so chunks end cleanly."""
    if hard_end >= len(text):
        return len(text)

    window_start = max(start + 1, hard_end - window)
    segment = text[window_start:hard_end]

    paragraph = segment.rfind("\n\n")
    if paragraph != -1:
        return window_start + paragraph + 2

    sentence_end = -1
    for match in _SENTENCE_END.finditer(segment):
        sentence_end = match.end()
    if sentence_end != -1:
        return window_start + sentence_end

    space = segment.rfind(" ")
    if space != -1:
        return window_start + space + 1

    return hard_end


def chunk_text(
    text: str,
    size: int = 1000,
    overlap: int = 150,
    source: str = "",
    page_spans: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[Chunk]:
    """Split ``text`` into overlapping chunks that end on natural boundaries.

    ``page_spans`` is an optional list of ``(page_number, end_offset)`` pairs
    from the extractor, used to label each chunk with the page it started on.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    text = normalise(text)
    if not text:
        return []

    window = max(40, size // 8)
    chunks: List[Chunk] = []
    start = 0

    while start < len(text):
        hard_end = min(start + size, len(text))
        end = _best_cut(text, start, hard_end, window)
        if end <= start:                      # no boundary found; take the hard cut
            end = hard_end
        piece = text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    text=piece,
                    index=len(chunks),
                    page=_page_for(start, page_spans),
                    source=source,
                )
            )
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)

    return chunks


def _page_for(
    offset: int, page_spans: Optional[Sequence[Tuple[int, int]]]
) -> Optional[int]:
    if not page_spans:
        return None
    for page, end in page_spans:
        if offset < end:
            return page
    return page_spans[-1][0]
