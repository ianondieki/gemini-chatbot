"""The embedding cache. Ported from PR #1, extended to cover page provenance."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from gemini_agent.config import AgentConfig
from gemini_agent.rag import cache
from gemini_agent.rag.chunking import Chunk


@pytest.fixture
def config(tmp_path):
    return AgentConfig(cache_dir=str(tmp_path / "cache"))


@pytest.fixture
def entry():
    chunks = [
        Chunk(text="the fee is 4,500", index=0, page=3, source="rules.pdf"),
        Chunk(text="appeals close in 30 days", index=1, page=4, source="rules.pdf"),
    ]
    return chunks, np.array([[0.1, 0.2], [0.3, 0.4]], dtype="float32")


class TestRoundTrip:
    def test_a_miss_returns_none(self, config):
        assert cache.load("never-seen", config) is None

    def test_what_goes_in_comes_back(self, config, entry):
        chunks, vectors = entry
        assert cache.save("doc-1", chunks, vectors, config) is True

        loaded = cache.load("doc-1", config)
        assert loaded is not None
        got_chunks, got_vectors = loaded
        assert [c.text for c in got_chunks] == [c.text for c in chunks]
        np.testing.assert_allclose(got_vectors, vectors)

    def test_page_provenance_survives(self, config, entry):
        """Without this a cached answer loses its citations."""
        chunks, vectors = entry
        cache.save("doc-1", chunks, vectors, config)
        got, _ = cache.load("doc-1", config)
        assert [c.page for c in got] == [3, 4]
        assert got[0].source == "rules.pdf"
        assert got[0].citation() == "rules.pdf p.3 chunk 0"


class TestInvalidation:
    @pytest.mark.parametrize(
        "change",
        [
            {"embed_model": "other-model"},
            {"embed_dim": 256},
            {"chunk_size": 500},
            {"chunk_overlap": 10},
            {"max_chunks": 50},
        ],
    )
    def test_changing_a_parameter_invalidates_the_entry(self, config, entry, change):
        """Serving vectors that no longer match the settings would be silent rot."""
        chunks, vectors = entry
        cache.save("doc-1", chunks, vectors, config)
        assert cache.load("doc-1", dataclasses.replace(config, **change)) is None

    def test_a_different_file_is_a_different_entry(self, config, entry):
        chunks, vectors = entry
        cache.save("doc-1", chunks, vectors, config)
        assert cache.load("doc-2", config) is None


class TestResilience:
    def test_a_corrupt_entry_is_a_miss_not_a_crash(self, config, entry):
        chunks, vectors = entry
        cache.save("doc-1", chunks, vectors, config)
        meta, _ = cache._paths("doc-1", config)
        with open(meta, "w", encoding="utf-8") as handle:
            handle.write("{ not json")
        assert cache.load("doc-1", config) is None

    def test_a_length_mismatch_is_a_miss(self, config, entry):
        chunks, vectors = entry
        cache.save("doc-1", chunks, vectors, config)
        _, vector_path = cache._paths("doc-1", config)
        np.save(vector_path, np.array([[0.1, 0.2]], dtype="float32"))
        assert cache.load("doc-1", config) is None

    def test_an_unwritable_directory_does_not_raise(self, entry):
        chunks, vectors = entry
        config = AgentConfig(cache_dir="/proc/nope/not-writable")
        assert cache.save("doc-1", chunks, vectors, config) is False

    def test_disabling_the_cache_bypasses_it_entirely(self, entry):
        chunks, vectors = entry
        config = AgentConfig(cache_enabled=False)
        assert cache.save("doc-1", chunks, vectors, config) is False
        assert cache.load("doc-1", config) is None


class TestClear:
    def test_clear_removes_entries(self, config, entry):
        chunks, vectors = entry
        cache.save("doc-1", chunks, vectors, config)
        cache.save("doc-2", chunks, vectors, config)
        assert cache.clear(config) == 4        # two json + two npy
        assert cache.load("doc-1", config) is None

    def test_clearing_a_missing_directory_is_fine(self, config):
        assert cache.clear(config) == 0
