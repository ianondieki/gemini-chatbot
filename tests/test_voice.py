"""Speech in and out. Ported from PR #1's voice tests onto the new structure."""

from __future__ import annotations

import io
import wave

import pytest
from google.genai import types

from gemini_agent.config import AgentConfig
from gemini_agent.voice import is_error_answer, pcm_to_wav, synthesize, transcribe


class FakeAudioClient:
    """Stands in for ``genai.Client`` for the two voice calls."""

    def __init__(self, text=None, pcm=None, raises=None):
        outer = self

        class Models:
            def generate_content(self, *, model, contents, config=None):
                outer.last = {"model": model, "contents": contents, "config": config}
                if raises is not None:
                    raise raises
                if pcm is not None:
                    part = types.Part(
                        inline_data=types.Blob(data=pcm, mime_type="audio/L16")
                    )
                    return _wrap([part])
                return _wrap([types.Part(text=text or "")], text=text or "")

        self.models = Models()
        self.last = None


def _wrap(parts, text=""):
    class Response:
        candidates = [
            types.Candidate(content=types.Content(role="model", parts=parts))
        ]

    Response.text = text
    return Response()


@pytest.fixture
def config():
    return AgentConfig(tts_voice="Kore", tts_sample_rate=24_000)


class TestPcmToWav:
    def test_it_produces_a_playable_wav_header(self):
        wav = pcm_to_wav(b"\x01\x02" * 100)
        assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"

    def test_frames_survive_the_round_trip(self):
        pcm = bytes(range(256)) * 4
        with wave.open(io.BytesIO(pcm_to_wav(pcm)), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == 24_000
            assert handle.readframes(handle.getnframes()) == pcm

    def test_the_sample_rate_is_honoured(self):
        with wave.open(io.BytesIO(pcm_to_wav(b"\x00" * 64, sample_rate=16_000))) as h:
            assert h.getframerate() == 16_000


class TestTranscribe:
    def test_it_returns_the_spoken_text(self, config):
        client = FakeAudioClient(text="  what is the weather  ")
        assert transcribe(client, b"audio", config) == "what is the weather"

    def test_empty_audio_short_circuits_without_a_call(self, config):
        client = FakeAudioClient(text="unused")
        assert transcribe(client, b"", config) == ""
        assert client.last is None

    def test_silence_yields_an_empty_string(self, config):
        assert transcribe(FakeAudioClient(text=""), b"audio", config) == ""

    def test_it_falls_back_to_the_main_model(self, config):
        client = FakeAudioClient(text="hi")
        transcribe(client, b"audio", config)
        assert client.last["model"] == config.model

    def test_an_explicit_stt_model_wins(self):
        config = AgentConfig(stt_model="gemini-2.5-pro")
        client = FakeAudioClient(text="hi")
        transcribe(client, b"audio", config)
        assert client.last["model"] == "gemini-2.5-pro"

    def test_a_failure_is_raised_because_there_is_no_question_without_it(self, config):
        with pytest.raises(RuntimeError):
            transcribe(FakeAudioClient(raises=RuntimeError("mic")), b"a", config)


class TestSynthesize:
    def test_it_wraps_the_returned_pcm(self, config):
        audio = synthesize(FakeAudioClient(pcm=b"\x01\x02" * 50), "hello", config)
        assert audio is not None and audio[:4] == b"RIFF"

    def test_the_configured_voice_is_requested(self, config):
        client = FakeAudioClient(pcm=b"\x01\x02")
        synthesize(client, "hello", config)
        speech = client.last["config"].speech_config
        assert speech.voice_config.prebuilt_voice_config.voice_name == "Kore"
        assert client.last["config"].response_modalities == ["AUDIO"]

    def test_an_unavailable_tts_model_degrades_quietly(self, config):
        """The text answer is already on screen - TTS must never raise."""
        client = FakeAudioClient(raises=RuntimeError("model not enabled"))
        assert synthesize(client, "hello", config) is None

    def test_a_response_with_no_audio_returns_none(self, config):
        assert synthesize(FakeAudioClient(text="sorry"), "hello", config) is None

    def test_empty_text_is_not_sent_to_the_model(self, config):
        client = FakeAudioClient(pcm=b"\x01")
        assert synthesize(client, "   ", config) is None
        assert client.last is None


class TestErrorAnswers:
    @pytest.mark.parametrize(
        "answer",
        ["I hit an error: boom", "API error: 503", "Error: something broke"],
    )
    def test_error_answers_are_not_worth_speaking(self, answer):
        assert is_error_answer(answer)

    def test_a_real_answer_is_spoken(self):
        assert not is_error_answer("The total is 42.")
