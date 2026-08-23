"""Base interface for TTS backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class TTSBackend(ABC):
    """Common interface for text-to-speech systems used by the podcast pipeline."""

    @abstractmethod
    def synthesize(self, text: str, output_path: str, voice: str | None = None) -> str:
        """Convert a text payload into an audio file at output_path and return the path."""
        raise NotImplementedError

    @staticmethod
    def synthesize_conversation(dialogue: Sequence[str], output_path: str, *, backend: "TTSBackend") -> str:
        """Batch-convert a conversation block to audio."""
        return backend.synthesize("\n".join(dialogue), output_path)
