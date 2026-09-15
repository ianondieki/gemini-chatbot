"""Chunking, the vector store, and retrieval-as-a-tool."""

from __future__ import annotations

import numpy as np
import pytest

from gemini_agent.config import AgentConfig
from gemini_agent.loop import ToolContext
from gemini_agent.memory import WorkingMemory
from gemini_agent.rag.chunking import chunk_text, normalise
from gemini_agent.rag.library import DocumentLibrary, EmptyDocument
from gemini_agent.rag.store import DocumentIndex
from gemini_agent.tools.documents import document_summary, search_documents

TEXT = (
    "The annual fee is 1,250 dollars. Late filing carries a penalty. "
    "The penalty is 5 percent of the outstanding amount per month. "
    "Appeals must be lodged within 30 days of the assessment notice. "
) * 6


class TestChunking:
    def test_normalise_keeps_paragraphs_but_collapses_spaces(self):
        assert normalise("a    b\n\n\n\nc") == "a b\n\nc"

    def test_chunks_cover_the_text(self):
        chunks = chunk_text(TEXT, size=120, overlap=20)
        assert chunks
        joined = " ".join(c.text for c in chunks)
        assert "1,250 dollars" in joined

    def test_chunks_end_on_a_boundary_not_mid_word(self):
        """A chunk ending mid-number is a fact the model cannot read."""
        for chunk in chunk_text(TEXT, size=150, overlap=30):
            assert chunk.text == chunk.text.strip()
            assert not chunk.text.endswith(("penalt", "assessme", "outstandin"))

    def test_chunks_overlap_so_facts_are_not_split_away(self):
        chunks = chunk_text("word " * 200, size=200, overlap=80)
        assert len(chunks) > 1
        assert chunks[0].text[-30:] in chunks[1].text or len(chunks[0].text) < 30

    def test_indices_are_sequential(self):
        chunks = chunk_text(TEXT, size=120, overlap=20)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_pages_are_attached_for_citation(self):
        chunks = chunk_text(
            "a" * 500, size=100, overlap=10, source="doc.pdf",
            page_spans=[(1, 200), (2, 400), (3, 600)],
        )
        assert chunks[0].page == 1
        assert chunks[-1].page == 3
        assert "doc.pdf p.1" in chunks[0].citation()

    def test_empty_input_yields_nothing(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\n  ") == []

    def test_overlap_must_be_smaller_than_size(self):
        with pytest.raises(ValueError):
            chunk_text("text", size=100, overlap=100)

    def test_it_terminates_on_text_with_no_boundaries(self):
        chunks = chunk_text("x" * 1000, size=100, overlap=20)
        assert len(chunks) > 1


class TestVectorStore:
    def _index(self) -> DocumentIndex:
        index = DocumentIndex()
        chunks = chunk_text("alpha. beta. gamma. delta.", size=8, overlap=0, source="d")
        vectors = np.eye(len(chunks), dtype="float32")
        index.add(chunks, vectors, "d")
        return index

    def test_search_ranks_by_cosine_similarity(self):
        index = self._index()
        query = np.zeros(len(index.chunks), dtype="float32")
        query[2] = 1.0
        hits = index.search(query, k=2)
        assert hits[0].chunk.index == 2
        assert hits[0].score == pytest.approx(1.0)
        assert hits[0].score >= hits[1].score

    def test_unnormalised_vectors_still_score_correctly(self):
        index = self._index()
        query = np.zeros(len(index.chunks), dtype="float32")
        query[1] = 17.0                       # magnitude must not affect ranking
        assert index.search(query, k=1)[0].chunk.index == 1

    def test_k_larger_than_the_corpus_is_safe(self):
        index = self._index()
        assert len(index.search(np.ones(len(index.chunks)), k=99)) == len(index.chunks)

    def test_an_empty_index_returns_nothing(self):
        assert DocumentIndex().search([1.0, 0.0], k=3) == []
        assert DocumentIndex().is_empty

    def test_adding_a_second_document_keeps_indices_unique(self):
        index = self._index()
        width = index.matrix.shape[1]
        more = chunk_text("epsilon. zeta.", size=8, overlap=0, source="e")
        index.add(more, np.ones((len(more), width), dtype="float32"), "e")
        assert len({c.index for c in index.chunks}) == len(index.chunks)
        assert index.sources == ["d", "e"]
        assert index.matrix.shape[0] == len(index.chunks)

    def test_mismatched_vector_count_is_rejected(self):
        index = DocumentIndex()
        chunks = chunk_text("alpha. beta.", size=8, overlap=0)
        with pytest.raises(ValueError):
            index.add(chunks, np.ones((1, 3), dtype="float32"), "d")


class FakeEmbedder:
    """Bag-of-words vectors: crude, deterministic, and enough to test retrieval.

    Text sharing no vocabulary with the query embeds to the zero vector and so
    scores 0 - which is what lets the weak-match path be tested.
    """

    VOCAB = ["fee", "penalty", "appeal", "days", "percent", "dollars", "quantum"]

    def embed(self, texts, task_type="RETRIEVAL_DOCUMENT", progress=None):
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append([float(lowered.count(word)) for word in self.VOCAB])
        if progress:
            progress(len(texts), len(texts))
        return np.array(rows, dtype="float32")

    def embed_query(self, text):
        return self.embed([text], task_type="RETRIEVAL_QUERY")[0]


class TestLibrary:
    def _library(self, max_chunks=1200):
        config = AgentConfig(chunk_size=120, chunk_overlap=20, retrieval_top_k=2,
                             max_chunks=max_chunks)
        return DocumentLibrary(FakeEmbedder(), config)

    def test_indexing_then_searching_finds_the_right_passage(self):
        library = self._library()
        assert library.add_text(TEXT, "rules.txt") > 0
        hits = library.search("what is the penalty for late filing?", k=1)
        assert "penalty" in hits[0].chunk.text.lower()

    def test_empty_input_is_rejected_clearly(self):
        with pytest.raises(EmptyDocument):
            self._library().add_text("   ", "blank.txt")

    def test_oversized_documents_are_capped_not_dropped(self):
        library = self._library(max_chunks=3)
        library.add_text(TEXT, "long.txt")
        assert len(library.index) == 3
        assert library.index.truncated is True

    def test_describe_names_the_source(self):
        library = self._library()
        library.add_text(TEXT, "rules.txt")
        assert "rules.txt" in library.describe()

    def test_clear_empties_it(self):
        library = self._library()
        library.add_text(TEXT, "rules.txt")
        library.clear()
        assert library.is_empty


class TestRetrievalTool:
    def _ctx(self, library=None):
        config = AgentConfig(retrieval_top_k=2)
        return ToolContext(
            config=config, memory=WorkingMemory(config), documents=library
        )

    def test_it_says_so_when_no_document_is_loaded(self):
        assert "no document is loaded" in search_documents(self._ctx(), "anything")
        assert document_summary(self._ctx()) == "No document is loaded."

    def test_results_carry_citations(self):
        config = AgentConfig(chunk_size=120, chunk_overlap=20, retrieval_top_k=2)
        library = DocumentLibrary(FakeEmbedder(), config)
        library.add_text(TEXT, "rules.txt")
        output = search_documents(self._ctx(library), "penalty")
        assert "rules.txt" in output and "similarity" in output

    def test_a_weak_match_is_flagged_rather_than_dressed_up(self):
        """Retrieval always returns *something*; saying so prevents invention."""
        config = AgentConfig(chunk_size=120, chunk_overlap=20, retrieval_top_k=1)
        library = DocumentLibrary(FakeEmbedder(), config)
        library.add_text("Nothing relevant here at all. " * 20, "other.txt")
        output = search_documents(self._ctx(library), "quantum chromodynamics")
        assert "WARNING" in output

    def test_an_empty_query_is_refused(self):
        assert search_documents(self._ctx(), "   ").startswith("ERROR:")
