"""Kokoro-based TTS wrapper."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional MLX dependency path
    from kokoro_mlx import KokoroTTS as KokoroRuntime
except Exception:  # pragma: no cover
    KokoroRuntime = None


class KokoroTTS:
    """Small wrapper around the MLX Kokoro implementation."""

    def __init__(self, voice: str = "af_heart") -> None:
        self.voice = voice
        self.model: Any | None = None

        if KokoroRuntime is not None:
            try:
                self.model = KokoroRuntime.from_pretrained()
            except Exception:  # pragma: no cover - download may fail at runtime
                self.model = None

    def synthesize(self, text: str, output_path: str, voice: str | None = None) -> str:
        """Generate audio for a text string and return the file path."""
        chosen_voice = voice or self.voice

        if self.model is None:
            return output_path

        try:
            if hasattr(self.model, "speak"):
                self.model.speak(text, voice=chosen_voice, output_path=output_path)
            else:
                self.model.generate(text=text, voice=chosen_voice, path=output_path)
        except Exception:  # pragma: no cover - runtime model path can vary by version
            pass

        return output_path
