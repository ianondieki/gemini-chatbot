"""
Render smoke tests for the two Streamlit apps using Streamlit's AppTest. They
run each script in-process with dummy API keys and assert it renders without an
uncaught exception AND that the voice widgets (toggle + microphone) are present.

No network or real keys are used — clients construct lazily and we never submit
a message, so nothing calls Gemini/Tavily. Skips cleanly if Streamlit's testing
harness or an app dependency isn't installed.

Run:  pytest tests/test_app_smoke.py
"""

import os
import sys

import pytest

# Skip the whole module unless the pieces these apps need are importable.
pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("tavily")   # app.py imports TavilyClient at module load
np = pytest.importorskip("numpy")

from streamlit.testing.v1 import AppTest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)  # so the apps can `import agent_core`


@pytest.fixture(autouse=True)
def _dummy_keys(monkeypatch):
    for key in ("GEMINI_API_KEY", "TAVILY_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(key, "dummy-key-for-render")


def _app(name):
    return os.path.join(REPO, name)


def test_angel_app_renders_with_voice_widgets():
    at = AppTest.from_file(_app("app.py"), default_timeout=30).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Voice replies" in t.label for t in at.toggle)
    assert len(at.chat_input) > 0
    assert len(at.get("audio_input")) == 1


def test_rag_app_renders_with_voice_widgets_when_indexed():
    at = AppTest.from_file(_app("rag_app.py"), default_timeout=30)
    # Pretend a PDF is already indexed so the chat + voice section renders.
    at.session_state["rag_chunks"] = ["alpha chunk", "beta chunk"]
    at.session_state["rag_emb"] = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    at.session_state["rag_file"] = "seed.pdf-1"
    at.session_state["rag_msgs"] = []
    at.session_state["rag_last_audio_id"] = None
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Voice replies" in t.label for t in at.toggle)
    assert len(at.chat_input) > 0
    assert len(at.get("audio_input")) == 1


def test_rag_app_renders_upload_prompt_before_indexing():
    # Before any PDF, the app should show the upload prompt and no chat input.
    at = AppTest.from_file(_app("rag_app.py"), default_timeout=30).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.chat_input) == 0
