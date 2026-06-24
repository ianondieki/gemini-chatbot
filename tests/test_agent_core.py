"""
Unit tests for the pure, security/correctness-critical helpers in agent_core
and rag_app. These need no API keys or network — they exercise the calculator,
history trimming, chunking, and retrieval math directly.

Run:  pytest
"""

import math

import pytest

import agent_core as core
from agent_core import do_calculate, do_search, is_search_error, trim_history
from google.genai import types


# ─────────────────────────────────────────
# Calculator: correct arithmetic
# ─────────────────────────────────────────
@pytest.mark.parametrize("expr, expected", [
    ("2 + 2", "4"),
    ("2400 * 0.15", "360.0"),
    ("(3 + 4) ** 2", "49"),
    ("10 / 4", "2.5"),
    ("17 % 5", "2"),
    ("17 // 5", "3"),
    ("-5 + 3", "-2"),
    ("2 ** 10", "1024"),
])
def test_calculate_arithmetic(expr, expected):
    assert do_calculate(expr) == f"{expr} = {expected}"


# ─────────────────────────────────────────
# Calculator: must REJECT anything that isn't plain arithmetic.
# This is the security boundary — these would be code execution under eval().
# ─────────────────────────────────────────
@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')",
    "os.system('ls')",
    "open('/etc/passwd').read()",
    "1 + foo",                 # a name
    "len([1, 2, 3])",          # a function call
    "[].__class__",            # attribute access
    "lambda: 1",
    "1 if True else 2",        # ternary / control flow
    "1; 2",                    # multiple statements
])
def test_calculate_rejects_non_arithmetic(expr):
    result = do_calculate(expr)
    assert result.startswith("Calculation error")


def test_calculate_handles_syntax_error():
    assert do_calculate("2 +").startswith("Calculation error")


def test_calculate_division_by_zero_is_caught():
    assert do_calculate("1 / 0").startswith("Calculation error")


# ─────────────────────────────────────────
# Web search: success formatting and explicit failure signalling
# ─────────────────────────────────────────
class _FakeTavily:
    def __init__(self, payload=None, exc=None):
        self._payload, self._exc = payload, exc

    def search(self, **kwargs):
        if self._exc:
            raise self._exc
        return self._payload


def test_do_search_formats_results():
    tav = _FakeTavily(payload={
        "answer": "42",
        "results": [{"title": "T", "url": "http://x", "content": "body"}],
    })
    out = do_search(tav, "meaning of life")
    assert not is_search_error(out)
    assert "Quick answer: 42" in out
    assert "http://x" in out


def test_do_search_no_results():
    out = do_search(_FakeTavily(payload={"results": []}), "q")
    assert out == "No results found."
    assert not is_search_error(out)


def test_do_search_failure_is_flagged_and_instructive():
    out = do_search(_FakeTavily(exc=RuntimeError("network down")), "q")
    assert is_search_error(out)
    assert "network down" in out
    # the model must be told not to fabricate a live answer
    assert "do not invent" in out.lower()


# ─────────────────────────────────────────
# History trimming
# ─────────────────────────────────────────
def _user(text):
    return types.Content(role="user", parts=[types.Part(text=text)])


def _model(text):
    return types.Content(role="model", parts=[types.Part(text=text)])


def _tool_result(name="web_search"):
    return types.Content(
        role="user",
        parts=[types.Part.from_function_response(name=name, response={"result": "x"})],
    )


def test_trim_caps_length():
    history = [_user(f"msg {i}") for i in range(30)]
    trim_history(history, max_history=10)
    assert len(history) == 10


def test_trim_drops_leading_model_turn():
    # After capping, the history must not start with a model turn.
    history = [_model("orphan answer"), _user("real question"), _model("answer")]
    trim_history(history, max_history=10)
    assert history[0].role == "user"
    assert history[0].parts[0].text == "real question"


def test_trim_drops_leading_orphan_tool_result():
    # The bug this guards: a tool-result whose matching call was trimmed away
    # would make Gemini reject the request. It must be dropped from the front.
    history = [_tool_result(), _user("real question")]
    trim_history(history, max_history=10)
    assert history[0].role == "user"
    assert history[0].parts[0].text == "real question"


def test_trim_keeps_valid_history_intact():
    history = [_user("q1"), _model("a1"), _user("q2")]
    trim_history(history, max_history=10)
    assert len(history) == 3


def test_trim_empty_history_is_safe():
    history = []
    trim_history(history)
    assert history == []


# ─────────────────────────────────────────
# RAG: chunking and retrieval (imported lazily so a missing optional dep here
# skips rather than fails the whole suite)
# ─────────────────────────────────────────
def _rag():
    return pytest.importorskip("rag_app")


def test_chunk_text_overlap_and_coverage():
    rag = _rag()
    text = "abcdefghij" * 30  # 300 chars
    chunks = rag.chunk_text(text, size=100, overlap=20)
    assert all(len(c) <= 100 for c in chunks)
    # Stride is size - overlap = 80, so consecutive chunks must overlap by 20.
    assert chunks[0][-20:] == chunks[1][:20]


def test_chunk_text_empty_input():
    rag = _rag()
    assert rag.chunk_text("") == []
    assert rag.chunk_text("    ") == []


def test_chunk_text_normalizes_whitespace():
    rag = _rag()
    chunks = rag.chunk_text("hello    world\n\nfoo", size=100, overlap=0)
    assert chunks == ["hello world foo"]


def test_top_k_chunks_ranks_by_cosine_similarity():
    rag = _rag()
    np = pytest.importorskip("numpy")
    # Three doc vectors; the query points exactly along the second one.
    docs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype="float32")
    query = np.array([0.0, 2.0], dtype="float32")  # magnitude shouldn't matter
    idx, scores = rag.top_k_chunks(query, docs, k=2)
    assert idx[0] == 1                       # best match is the aligned vector
    assert math.isclose(scores[0], 1.0, rel_tol=1e-5)
    # scores are returned in descending order
    assert scores[0] >= scores[1]


# ─────────────────────────────────────────
# RAG: embedding cache roundtrip
# ─────────────────────────────────────────
def test_cache_miss_then_roundtrip(tmp_path):
    rag = _rag()
    np = pytest.importorskip("numpy")
    cache_dir = str(tmp_path / "cache")
    file_id = "book.pdf-12345"
    chunks = ["alpha", "beta", "gamma"]
    embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype="float32")

    # Cold cache -> miss.
    assert rag.load_cache(file_id, cache_dir=cache_dir) is None

    rag.save_cache(file_id, chunks, embeddings, cache_dir=cache_dir)

    loaded = rag.load_cache(file_id, cache_dir=cache_dir)
    assert loaded is not None
    loaded_chunks, loaded_emb = loaded
    assert loaded_chunks == chunks
    assert np.allclose(loaded_emb, embeddings)


def test_cache_key_invalidates_on_param_change(tmp_path, monkeypatch):
    rag = _rag()
    # Different MAX_CHUNKS must produce a different cache key, so a config
    # change never serves a stale index.
    monkeypatch.setattr(rag, "MAX_CHUNKS", 1200)
    key_a = rag._cache_key("book.pdf-1")
    monkeypatch.setattr(rag, "MAX_CHUNKS", 5000)
    key_b = rag._cache_key("book.pdf-1")
    assert key_a != key_b


def test_corrupt_cache_is_treated_as_miss(tmp_path):
    rag = _rag()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    key = rag._cache_key("x.pdf-1")
    (cache_dir / f"{key}.chunks.json").write_text("{not valid json")
    (cache_dir / f"{key}.emb.npy").write_bytes(b"garbage")
    assert rag.load_cache("x.pdf-1", cache_dir=str(cache_dir)) is None
