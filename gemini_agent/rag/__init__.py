"""Retrieval: chunking, embedding, and an in-memory vector index."""

from .chunking import Chunk, chunk_text, normalise
from .library import DocumentLibrary, EmptyDocument
from .pdf import ExtractedText, extract_pdf
from .store import DocumentIndex, Embedder, Hit

__all__ = [
    "Chunk",
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
