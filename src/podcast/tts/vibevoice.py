"""VibeVoice-backed TTS adapter for long-form conversational speech."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .base import TTSBackend


class VibeVoiceTTS(TTSBackend):
    """Placeholder implementation for a future VibeVoice backend.

    This keeps the TTS abstraction consistent while allowing Kokoro to remain the baseline.
    """

    def __init__(self, voice: str = "default") -> None:
        self.voice = voice
        self.model: Any | None = None

    def synthesize(self, text: str, output_path: str, voice: str | None = None) -> str:
        """Create a simple silent WAV placeholder for VibeVoice in local testing."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = 24000
        samples = np.zeros(int(sample_rate * max(0.2, min(len(text) / 25.0, 1.0))), dtype=np.float32)
        sf.write(str(target), samples, sample_rate)
        return str(target)
