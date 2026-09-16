"""An on-disk cache so re-uploading the same PDF is instant.

Embedding a 300-page document is the slowest and most quota-expensive thing
this project does, and people re-upload the same file constantly — a browser
refresh is enough. Caching it turns a two-minute wait into nothing.

The cache key covers the file *and* every parameter that would change the
result (model, dimension, chunk size, overlap, cap), so changing any of them
invalidates stale entries rather than silently serving vectors that no longer
match the settings. Ported from the earlier standalone implementation, with
chunk provenance (page numbers) now preserved through the round trip.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np

from ..config import AgentConfig
from .chunking import Chunk

log = logging.getLogger(__name__)


def cache_key(file_id: str, config: AgentConfig) -> str:
    """A key that changes whenever anything affecting the vectors changes."""
    raw = "|".join(
        str(part)
        for part in (
            file_id,
            config.embed_model,
            config.embed_dim,
            config.chunk_size,
            config.chunk_overlap,
            config.max_chunks,
        )
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _paths(file_id: str, config: AgentConfig) -> Tuple[str, str]:
    base = os.path.join(config.cache_dir, cache_key(file_id, config))
    return f"{base}.json", f"{base}.npy"


def load(file_id: str, config: AgentConfig) -> Optional[Tuple[List[Chunk], np.ndarray]]:
    """Return cached chunks and vectors, or ``None`` on any kind of miss.

    A corrupt or half-written cache is a miss, not an error — re-embedding is
    always a correct fallback, so there is never a reason to fail here.
    """
    if not config.cache_enabled:
        return None

    meta_path, vector_path = _paths(file_id, config)
    if not (os.path.exists(meta_path) and os.path.exists(vector_path)):
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        vectors = np.load(vector_path)
        chunks = [
            Chunk(
                text=item["text"],
                index=item["index"],
                page=item.get("page"),
                source=item.get("source", ""),
            )
            for item in payload["chunks"]
        ]
        if len(chunks) != len(vectors):
            return None
        return chunks, vectors
    except Exception as exc:
        log.info("ignoring unreadable cache entry for %s: %s", file_id, exc)
        return None


def save(
    file_id: str, chunks: List[Chunk], vectors: np.ndarray, config: AgentConfig
) -> bool:
    """Persist chunks and vectors. Best effort — a failure is never fatal."""
    if not config.cache_enabled:
        return False

    meta_path, vector_path = _paths(file_id, config)
    try:
        os.makedirs(config.cache_dir, exist_ok=True)
        payload = {
            "chunks": [
                {
                    "text": c.text,
                    "index": c.index,
                    "page": c.page,
                    "source": c.source,
                }
                for c in chunks
            ]
        }
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        np.save(vector_path, vectors)
        return True
    except Exception as exc:
        log.info("could not write cache entry for %s: %s", file_id, exc)
        return False


def clear(config: AgentConfig) -> int:
    """Delete every cache entry. Returns how many files were removed."""
    if not os.path.isdir(config.cache_dir):
        return 0
    removed = 0
    for name in os.listdir(config.cache_dir):
        if name.endswith((".json", ".npy")):
            try:
                os.remove(os.path.join(config.cache_dir, name))
                removed += 1
            except OSError:
                pass
    return removed
