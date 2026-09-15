"""Reading text (and page offsets) out of a PDF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple


@dataclass
class ExtractedText:
    text: str
    page_spans: List[Tuple[int, int]]
    page_count: int

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())


def extract_pdf(source: Any) -> ExtractedText:
    """Extract text from a PDF, tracking where each page ends.

    The page offsets are what let a retrieved chunk say "p.42" instead of
    "chunk 137", which is the difference between a citation a reader can check
    and one they cannot.
    """
    from pypdf import PdfReader

    reader = PdfReader(source)
    pieces: List[str] = []
    spans: List[Tuple[int, int]] = []
    length = 0

    for number, page in enumerate(reader.pages, start=1):
        try:
            content = page.extract_text() or ""
        except Exception:
            content = ""  # one unreadable page should not sink the document
        pieces.append(content)
        length += len(content) + 1
        spans.append((number, length))

    return ExtractedText(
        text="\n".join(pieces), page_spans=spans, page_count=len(reader.pages)
    )
