"""Text-to-speech integrations."""

from .base import TTSBackend
from .kokoro import KokoroTTS
from .vibevoice import VibeVoiceTTS


def get_tts_backend(name: str, *, voice: str | None = None) -> TTSBackend:
    """Return the configured TTS backend for the selected runtime."""
    backend_name = (name or "kokoro").lower()
    if backend_name == "vibevoice":
        return VibeVoiceTTS(voice=voice or "default")
    if backend_name == "kokoro":
        return KokoroTTS(voice=voice or "af_heart")
    raise ValueError(f"Unsupported TTS backend: {name}")


__all__ = ["KokoroTTS", "VibeVoiceTTS", "TTSBackend", "get_tts_backend"]
