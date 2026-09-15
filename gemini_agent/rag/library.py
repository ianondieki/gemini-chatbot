"""The loaded-documents library: ingest once, search many times."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from ..config import AgentConfig
from . import cache
from .chunking import chunk_text
from .pdf import extract_pdf
from .store import DocumentIndex, Embedder, Hit, Progress

log = logging.getLogger(__name__)


class EmptyDocument(ValueError):
    """The file held no extractable text - usually a scan with no OCR layer."""


class DocumentLibrary:
    """Holds every indexed document and answers similarity queries over them."""

    def __init__(self, embedder: Embedder, config: AgentConfig) -> None:
        self.embedder = embedder
        self.config = config
        self.index = DocumentIndex()
        self.cache_hit = False   # whether the last load came from the cache

    # --- ingestion -----------------------------------------------------
    def add_pdf(
        self,
        source: Any,
        name: str,
        progress: Optional[Progress] = None,
        file_id: Optional[str] = None,
    ) -> int:
        """Index a PDF. Returns the number of chunks added.

        ``file_id`` identifies the file for caching - pass something that
        changes when the content does, such as ``f"{name}-{size}"``.
        """
        cached = self._from_cache(file_id, name)
        if cached is not None:
            return cached

        extracted = extract_pdf(source)
        if not extracted.has_text:
            raise EmptyDocument(
                f"'{name}' has no selectable text. It is probably a scan - "
                "run it through OCR first."
            )
        return self._ingest(
            extracted.text,
            name,
            progress,
            page_spans=extracted.page_spans,
            file_id=file_id,
        )

    def add_text(
        self, text: str, name: str, progress: Optional[Progress] = None
    ) -> int:
        """Index plain text. Returns the number of chunks added."""
        if not text.strip():
            raise EmptyDocument(f"'{name}' is empty.")
        return self._ingest(text, name, progress, page_spans=None)

    def _from_cache(self, file_id: Optional[str], name: str) -> Optional[int]:
        """Load a previously embedded copy of this file, if we have one."""
        if not file_id:
            return None
        hit = cache.load(file_id, self.config)
        if hit is None:
            return None
        chunks, vectors = hit
        self.index.add(chunks, vectors, name)
        self.cache_hit = True
        return len(chunks)

    def _ingest(self, text, name, progress, page_spans, file_id=None) -> int:
        chunks = chunk_text(
            text,
            size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            source=name,
            page_spans=page_spans,
        )
        if not chunks:
            raise EmptyDocument(f"'{name}' produced no chunks.")

        room = self.config.max_chunks - len(self.index)
        if room <= 0:
            raise EmptyDocument(
                f"the library is full ({self.config.max_chunks} chunks). "
                "Clear it before loading another document."
            )
        if len(chunks) > room:
            chunks = chunks[:room]
            self.index.truncated = True

        vectors = self.embedder.embed(
            [c.text for c in chunks],
            task_type="RETRIEVAL_DOCUMENT",
            progress=progress,
        )
        self.index.add(chunks, vectors, name)
        if file_id:
            cache.save(file_id, chunks, vectors, self.config)
        return len(chunks)

    # --- retrieval -----------------------------------------------------
    def search(self, query: str, k: Optional[int] = None) -> List[Hit]:
        if self.index.is_empty:
            return []
        vector = self.embedder.embed_query(query)
        return self.index.search(vector, k=k or self.config.retrieval_top_k)

    def clear(self) -> None:
        self.index.clear()
        self.cache_hit = False

    @property
    def is_empty(self) -> bool:
        return self.index.is_empty

    def describe(self) -> str:
        return self.index.describe()
