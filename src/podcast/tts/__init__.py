"""Text-to-speech integrations."""

from .base import TTSBackend
from .cartesia import CartesiaTTS
from .kokoro import KokoroTTS
from .vibevoice import VibeVoiceTTS


def get_tts_backend(name: str, *, voice: str | None = None) -> TTSBackend:
    """Return the configured TTS backend for the selected runtime."""
    backend_name = (name or "kokoro").lower()
    if backend_name == "cartesia":
        return CartesiaTTS(voice=voice or "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4")
    if backend_name == "vibevoice":
        return VibeVoiceTTS(voice=voice or "default")
    if backend_name == "kokoro":
        return KokoroTTS(voice=voice or "af_heart")
    if backend_name == "dia":
        from .dia import DiaTTS

        return DiaTTS(voice=voice or "dialogue")
    raise ValueError(f"Unsupported TTS backend: {name}")


__all__ = ["CartesiaTTS", "KokoroTTS", "VibeVoiceTTS", "TTSBackend", "get_tts_backend"]
