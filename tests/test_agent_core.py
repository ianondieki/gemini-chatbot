"""
Unit tests for the pure, security/correctness-critical helpers in agent_core
and rag_app. These need no API keys or network — they exercise the calculator,
history trimming, chunking, and retrieval math directly.

Run:  pytest
"""

import math

import pytest

import agent_core as core
from agent_core import do_calculate, trim_history
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
