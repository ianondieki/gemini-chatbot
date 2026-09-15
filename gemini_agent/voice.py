"""Speech in and speech out, using the Gemini client the agent already has.

No extra dependency and no second API key: transcription rides on Gemini's
audio understanding, and synthesis on its TTS models. Ported from the
standalone helpers in the earlier agent_core, with the models and voice moved
into ``AgentConfig`` so they are configurable like everything else.

The asymmetry between the two is deliberate:

* ``transcribe`` **raises** on failure. If the microphone input cannot be read
  there is no question to answer, so the caller must surface that.
* ``synthesize`` **returns None** on failure. The text answer is already on
  screen; losing the audio is a downgrade, not an error, and it must never
  take the reply down with it.
"""

from __future__ import annotations

import io
import logging
import wave
from typing import Any, Optional

from google.genai import types

from .config import AgentConfig

log = logging.getLogger(__name__)

TRANSCRIBE_INSTRUCTION = (
    "Transcribe this audio to text verbatim. Return ONLY the words spoken, "
    "with no commentary, labels, or punctuation you did not hear."
)


def pcm_to_wav(
    pcm: bytes,
    sample_rate: int = 24_000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Wrap raw little-endian PCM in a WAV container so a browser can play it.

    Gemini TTS returns headerless PCM; browsers will not play that. Pure
    stdlib, so it stays trivially testable.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def transcribe(
    client: Any,
    audio: bytes,
    config: AgentConfig,
    mime_type: str = "audio/wav",
) -> str:
    """Transcribe spoken audio to text. Raises if the model call fails."""
    if not audio:
        return ""

    response = client.models.generate_content(
        model=config.stt_model or config.model,
        contents=[
            TRANSCRIBE_INSTRUCTION,
            types.Part.from_bytes(data=audio, mime_type=mime_type),
        ],
    )
    return (getattr(response, "text", None) or "").strip()


def synthesize(
    client: Any,
    text: str,
    config: AgentConfig,
) -> Optional[bytes]:
    """Speak ``text``, returning WAV bytes, or ``None`` if that is not possible.

    Never raises: the TTS model may not be enabled for a given key, and a
    missing voice reply must not disturb the answer already on screen.
    """
    if not text.strip():
        return None

    try:
        response = client.models.generate_content(
            model=config.tts_model,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=config.tts_voice
                        )
                    )
                ),
            ),
        )
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        parts = getattr(candidates[0].content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None) if inline is not None else None
            if data:
                return pcm_to_wav(data, sample_rate=config.tts_sample_rate)
        return None
    except Exception as exc:
        log.info("speech synthesis unavailable: %s", exc)
        return None


def is_error_answer(answer: str) -> bool:
    """True for the agent's own error strings, which are not worth speaking."""
    return answer.strip().startswith(("I hit an error", "API error", "Error:"))
