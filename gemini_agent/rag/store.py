"""An in-memory vector store: embed once, then search with cosine similarity.

Small enough to read in one sitting, which is the point — the retrieval half of
RAG is a normalised dot product, and hiding that behind a database makes it
look like magic. Vectors are L2-normalised at insert time so a query is one
matrix-vector multiply, and scores come back on a comparable 0-1 scale.

Above a few thousand chunks this should become a real vector database; the
``DocumentIndex`` interface is deliberately the shape you would keep.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

from .chunking import Chunk

log = logging.getLogger(__name__)

Progress = Callable[[int, int], None]


@dataclass
class Hit:
    chunk: Chunk
    score: float

    def render(self, limit: int = 1200) -> str:
        body = self.chunk.text
        if len(body) > limit:
            body = body[:limit] + "..."
        return f"[{self.chunk.citation()} | similarity {self.score:.3f}]\n{body}"


@dataclass
class DocumentIndex:
    """Chunks plus their embedding matrix, searchable by cosine similarity."""

    chunks: List[Chunk] = field(default_factory=list)
    matrix: Optional[np.ndarray] = None
    sources: List[str] = field(default_factory=list)
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        return not self.chunks or self.matrix is None

    def describe(self) -> str:
        if self.is_empty:
            return "no document loaded"
        names = ", ".join(self.sources) or "document"
        note = " (truncated)" if self.truncated else ""
        return f"{len(self.chunks)} chunks from {names}{note}"

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray, source: str) -> None:
        """Add chunks and their vectors, re-indexing so ``chunk.index`` stays unique."""
        if len(chunks) != len(vectors):
            raise ValueError("chunk count and vector count must match")
        if not len(chunks):
            return

        normalised = _normalise_rows(np.asarray(vectors, dtype="float32"))
        offset = len(self.chunks)
        for position, chunk in enumerate(chunks):
            chunk.index = offset + position
            chunk.source = chunk.source or source
            self.chunks.append(chunk)

        self.matrix = (
            normalised
            if self.matrix is None
            else np.vstack([self.matrix, normalised])
        )
        if source and source not in self.sources:
            self.sources.append(source)

    def search(self, query_vector: Sequence[float], k: int = 4) -> List[Hit]:
        """Return the ``k`` chunks closest to ``query_vector``."""
        if self.is_empty or k <= 0:
            return []
        query = _normalise_rows(
            np.asarray(query_vector, dtype="float32").reshape(1, -1)
        )[0]
        scores = self.matrix @ query
        k = min(k, len(scores))
        # argpartition finds the top k without sorting everything.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [Hit(chunk=self.chunks[i], score=float(scores[i])) for i in top]

    def clear(self) -> None:
        self.chunks.clear()
        self.matrix = None
        self.sources.clear()
        self.truncated = False


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-10)


class Embedder:
    """Batched embedding with backoff, pacing and progress reporting.

    Free-tier embedding endpoints rate-limit aggressively, and a 400-page PDF
    is thousands of requests. Batching, a short pause between batches and
    exponential backoff are what make indexing a real book finish instead of
    dying two thirds of the way through.
    """

    def __init__(self, client: Any, config: Any, sleep=time.sleep) -> None:
        self.client = client
        self.config = config
        self._sleep = sleep

    def embed(
        self,
        texts: Sequence[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        progress: Optional[Progress] = None,
    ) -> np.ndarray:
        from google.genai import types

        if not texts:
            return np.zeros((0, self.config.embed_dim), dtype="float32")

        vectors: List[Sequence[float]] = []
        total = len(texts)
        batch_size = max(1, self.config.embed_batch)

        for start in range(0, total, batch_size):
            batch = list(texts[start : start + batch_size])
            vectors.extend(
                self._embed_batch(
                    batch,
                    types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.config.embed_dim,
                    ),
                )
            )
            if progress is not None:
                progress(min(start + batch_size, total), total)
            if start + batch_size < total and self.config.embed_batch_delay > 0:
                self._sleep(self.config.embed_batch_delay)

        return np.array(vectors, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text], task_type="RETRIEVAL_QUERY")[0]

    def _embed_batch(self, batch: List[str], config: Any) -> List[Sequence[float]]:
        from google.genai import errors as genai_errors

        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.embed_retries + 1):
            try:
                response = self.client.models.embed_content(
                    model=self.config.embed_model, contents=batch, config=config
                )
                return [e.values for e in response.embeddings]
            except genai_errors.APIError as exc:
                message = str(exc).lower()
                transient = any(
                    marker in message
                    for marker in (
                        "429", "resource_exhausted", "rate", "quota",
                        "503", "unavailable", "overloaded", "500",
                    )
                )
                if not transient:
                    raise
                last_error = exc
                if attempt < self.config.embed_retries:
                    self._sleep(min(30.0, 2 ** (attempt - 1)))
        raise last_error if last_error else RuntimeError("embedding failed")
