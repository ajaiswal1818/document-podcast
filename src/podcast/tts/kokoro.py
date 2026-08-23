"""Kokoro-based TTS wrapper."""

from __future__ import annotations


class KokoroTTS:
    """Placeholder for Kokoro text-to-speech integration."""

    def __init__(self, voice: str = "bfemalo") -> None:
        self.voice = voice

    def synthesize(self, text: str, output_path: str) -> str:
        """Generate audio for a text string."""
        return output_path
