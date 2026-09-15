"""Retrieval: chunking, embedding, and an in-memory vector index."""

from . import cache
from .chunking import Chunk, chunk_text, normalise
from .library import DocumentLibrary, EmptyDocument
from .pdf import ExtractedText, extract_pdf
from .store import DocumentIndex, Embedder, Hit

__all__ = [
    "Chunk",
    "cache",
    "DocumentLibrary",
    "EmptyDocument",
    "chunk_text",
    "normalise",
    "DocumentIndex",
    "Embedder",
    "Hit",
    "ExtractedText",
    "extract_pdf",
]
