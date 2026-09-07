"""Cartesia Sonic text-to-speech backend."""

from __future__ import annotations

import os
import sys
import wave
from pathlib import Path

from cartesia import Cartesia

from .base import TTSBackend


CARTESIA_MODEL_ID = "sonic-3.6"
SAMPLE_RATE = 44100
MAX_CONTINUATION_CHARS = 600

# Cartesia's recommended stable English voices for production use.
HOST_VOICE = "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"  # Skylar (en-US)
EXPERT_VOICE = "47c38ca4-5f35-497b-b1a3-415245fb35e1"  # Daniel (en-US)


def _load_project_env() -> None:
    """Load simple KEY=VALUE entries from the repository's ignored .env file."""
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def has_cartesia_api_key() -> bool:
    """Return whether a Cartesia API key is available from the environment or .env."""
    _load_project_env()
    return bool(os.getenv("CARTESIA_API_KEY"))


class CartesiaTTS(TTSBackend):
    """Synthesize podcast turns with Cartesia contexts and continuations."""

    def __init__(
        self,
        voice: str = HOST_VOICE,
        *,
        api_key: str | None = None,
        model_id: str = CARTESIA_MODEL_ID,
    ) -> None:
        _load_project_env()
        self.api_key = api_key or os.getenv("CARTESIA_API_KEY")
        self.voice = voice
        self.model_id = model_id

    def synthesize(self, text: str, output_path: str, voice: str | None = None) -> str:
        """Generate a WAV clip, preserving prosody when a turn spans many chunks."""
        if not self.api_key:
            raise RuntimeError("Cartesia requires CARTESIA_API_KEY in the environment or .env file.")

        transcript = text.strip()
        if not transcript:
            raise ValueError("Cartesia cannot synthesize empty text.")

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        chunks = self._continuation_chunks(transcript)
        client = Cartesia(api_key=self.api_key)
        audio_bytes = 0
        try:
            with client.tts.websocket_connect() as websocket:
                context = websocket.context(
                    model_id=self.model_id,
                    voice=voice or self.voice,
                    output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": SAMPLE_RATE},
                    language="en",
                )
                for chunk in chunks:
                    context.push(chunk)
                context.no_more_inputs()

                with wave.open(str(destination), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(SAMPLE_RATE)
                    for response in context.receive():
                        if response.type == "chunk" and response.audio:
                            wav_file.writeframes(response.audio)
                            audio_bytes += len(response.audio)
                        elif response.type == "error":
                            detail = getattr(response, "message", None) or getattr(response, "title", "Unknown error")
                            raise RuntimeError(f"Cartesia synthesis failed: {detail}")
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Cartesia synthesis request failed: {exc}") from exc
        if not audio_bytes:
            raise RuntimeError("Cartesia synthesis returned no audio.")
        print(
            f"TTS: Cartesia completed model={self.model_id} voice_id={voice or self.voice} "
            f"context_chunks={len(chunks)} pcm_bytes={audio_bytes}",
            file=sys.stderr,
        )
        return str(destination)

    @staticmethod
    def _continuation_chunks(transcript: str) -> list[str]:
        """Split only at whitespace, retaining it so the context joins valid text."""
        if len(transcript) <= MAX_CONTINUATION_CHARS:
            return [transcript]

        chunks: list[str] = []
        start = 0
        while len(transcript) - start > MAX_CONTINUATION_CHARS:
            split_at = transcript.rfind(" ", start, start + MAX_CONTINUATION_CHARS + 1)
            if split_at <= start:
                split_at = start + MAX_CONTINUATION_CHARS
            else:
                split_at += 1  # The space belongs to this chunk; contexts concatenate verbatim.
            chunks.append(transcript[start:split_at])
            start = split_at
        chunks.append(transcript[start:])
        return chunks
