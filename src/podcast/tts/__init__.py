"""Text-to-speech integrations."""

from .base import TTSBackend
from .cartesia import CartesiaTTS, get_voice_pair
from .kokoro import KokoroTTS
from .vibevoice import VibeVoiceTTS


def get_tts_backend(name: str, *, voice: str | None = None, language: str = "en") -> TTSBackend:
    """Return the configured TTS backend for the selected runtime."""
    backend_name = (name or "kokoro").lower()
    if backend_name == "cartesia":
        return CartesiaTTS(voice=voice or get_voice_pair(language)["HOST"]["id"], language=language)
    if backend_name == "vibevoice":
        return VibeVoiceTTS(voice=voice or "default")
    if backend_name == "kokoro":
        return KokoroTTS(voice=voice or "af_heart")
    if backend_name == "dia":
        from .dia import DiaTTS

        return DiaTTS(voice=voice or "dialogue")
    raise ValueError(f"Unsupported TTS backend: {name}")


__all__ = ["CartesiaTTS", "KokoroTTS", "VibeVoiceTTS", "TTSBackend", "get_tts_backend", "get_voice_pair"]
