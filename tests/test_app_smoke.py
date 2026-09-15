"""Render smoke tests for the two Streamlit apps, via Streamlit's AppTest.

Ported from PR #1 and pointed at the rebuilt apps. Each script runs in-process
with dummy keys and must render without an uncaught exception, with its voice
widgets present. No message is ever submitted, so nothing reaches Gemini or
Tavily.

These are cheap and they cover the seam the unit tests cannot: the apps import
the package, build a session, and render real widgets. A rename in
``gemini_agent`` that breaks a front end shows up here.
"""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("numpy")

from streamlit.testing.v1 import AppTest  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


@pytest.fixture(autouse=True)
def dummy_keys(monkeypatch, tmp_path):
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(key, "dummy-key-for-render")
    # Keep any cache writes inside the test's own directory.
    monkeypatch.setenv("RAG_CACHE_DIR", str(tmp_path / "cache"))


def run(name: str) -> AppTest:
    app = AppTest.from_file(os.path.join(REPO, name), default_timeout=60)
    app.run()
    return app


class TestAngelApp:
    def test_it_renders_without_raising(self):
        app = run("app.py")
        assert not app.exception, f"app.py raised: {app.exception}"

    def test_the_voice_widgets_are_present(self):
        app = run("app.py")
        labels = [t.label for t in app.toggle]
        assert any("Voice replies" in label for label in labels)

    def test_the_chat_input_is_present(self):
        app = run("app.py")
        assert len(app.chat_input) >= 1

    def test_the_title_renders(self):
        app = run("app.py")
        assert any("Angel" in str(m.value) for m in app.markdown)


class TestRagApp:
    def test_it_renders_without_raising(self):
        app = run("rag_app.py")
        assert not app.exception, f"rag_app.py raised: {app.exception}"

    def test_it_asks_for_a_pdf_before_anything_else(self):
        app = run("rag_app.py")
        assert any("Upload a PDF" in str(i.value) for i in app.info)
