"""Dia conversational TTS backend via mlx-audio.

Dia (Nari Labs, 1.6B) synthesizes a whole two-speaker exchange in one pass
using [S1]/[S2] speaker tags, so prosody carries across turn boundaries and
nonverbal cues like (laughs) are supported. Requires the optional mlx-audio
dependency: uv add mlx-audio
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence

from podcast.conversation.knowledge import scrub_speech_text

from .base import TTSBackend

DEFAULT_MODEL = "mlx-community/Dia-1.6B-4bit"
# Dia degrades on very long inputs; synthesize the script in bounded chunks.
MAX_CHUNK_CHARS = 700


class DiaTTS(TTSBackend):
    """Whole-conversation synthesis with cross-turn prosody."""

    def __init__(self, model_id: str = DEFAULT_MODEL, voice: str = "dialogue") -> None:
        try:
            from mlx_audio.tts.generate import generate_audio
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "mlx-audio is required for the Dia backend. Install it with: uv add mlx-audio"
            ) from exc
        self._generate_audio = generate_audio
        self.model_id = model_id
        self.voice = voice

    def _run_generation(self, text: str, output_path: Path) -> str:
        with tempfile.TemporaryDirectory() as workdir:
            prefix = str(Path(workdir) / "segment")
            self._generate_audio(
                text=text,
                model_path=self.model_id,
                file_prefix=prefix,
                audio_format="wav",
                join_audio=True,
                verbose=False,
            )
            produced = sorted(Path(workdir).glob("segment*.wav"))
            if not produced:
                raise RuntimeError("Dia generation produced no audio output.")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(produced[0].read_bytes())
        return str(output_path)

    def synthesize(self, text: str, output_path: str, voice: str | None = None) -> str:
        """Synthesize a single utterance as speaker S1."""
        safe_text = scrub_speech_text(text)
        return self._run_generation(f"[S1] {safe_text}", Path(output_path))

    def synthesize_dialogue(
        self,
        turns: Sequence[dict[str, str]],
        output_dir: str,
        *,
        host_speaker: str = "HOST",
    ) -> list[str]:
        """Synthesize a full two-speaker script in prosody-preserving chunks.

        Returns the list of chunk WAV paths in playback order.
        """
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        chunks: list[str] = []
        current: list[str] = []
        current_size = 0
        for turn in turns:
            text = scrub_speech_text(str(turn.get("text", "")).strip())
            if not text:
                continue
            tag = "[S1]" if str(turn.get("speaker", host_speaker)).strip() == host_speaker else "[S2]"
            fragment = f"{tag} {text}"
            if current and current_size + len(fragment) > MAX_CHUNK_CHARS:
                chunks.append(" ".join(current))
                current = []
                current_size = 0
            current.append(fragment)
            current_size += len(fragment)
        if current:
            chunks.append(" ".join(current))

        outputs: list[str] = []
        for chunk_index, chunk in enumerate(chunks, start=1):
            chunk_path = target_dir / f"dialogue_{chunk_index:03d}.wav"
            outputs.append(self._run_generation(chunk, chunk_path))
        return outputs
