"""Dia conversational TTS backend via mlx-audio.

Dia (Nari Labs, 1.6B) synthesizes a whole two-speaker exchange in one pass
using [S1]/[S2] speaker tags, so prosody carries across turn boundaries and
nonverbal cues like (laughs) are supported. Requires the optional mlx-audio
dependency: uv add mlx-audio
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Sequence

import soundfile as sf

from podcast.conversation.evidence import scrub_speech_text

from .base import TTSBackend

DEFAULT_MODEL = "mlx-community/Dia-1.6B-4bit"
# Dia compresses its speaking rate to fit the text it is given per generation:
# measured 313 wpm at ~700 chars, 259 at ~340, 213 at ~200. Small chunks help
# but plateau above natural pace (~170 wpm), so assembly additionally applies
# a calibrated time-stretch for Dia-rendered episodes.
MAX_CHUNK_CHARS = 220


class DiaTTS(TTSBackend):
    """Whole-conversation synthesis with cross-turn prosody."""

    def __init__(self, model_id: str = DEFAULT_MODEL, voice: str = "dialogue") -> None:
        try:
            from mlx_audio.tts.generate import generate_audio
            from mlx_audio.tts.utils import load_model
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "mlx-audio is required for the Dia backend. Install it with: uv add mlx-audio"
            ) from exc
        self._generate_audio = generate_audio
        self.model_id = model_id
        self.voice = voice
        # Load once and reuse across chunks; generate_audio(model=<str>) would
        # otherwise reload the full model for every chunk of the episode.
        self.model = load_model(model_id)

    def _run_generation(self, text: str, output_path: Path, *, attempts: int = 3) -> str:
        # Dia emits ~86 audio tokens per second; ~15 chars of text per spoken
        # second means ~6 tokens per char. Budget generously or output truncates.
        max_tokens = max(1500, min(8000, len(text) * 8))
        # Sampling occasionally emits EOS on the first step, yielding no (or
        # near-empty) audio; a retry with fresh sampling almost always recovers.
        min_seconds = max(1.0, len(text) / 60.0)

        for attempt in range(1, attempts + 1):
            with tempfile.TemporaryDirectory() as workdir:
                prefix = str(Path(workdir) / "segment")
                try:
                    self._generate_audio(
                        text=text,
                        model=self.model,
                        max_tokens=max_tokens,
                        file_prefix=prefix,
                        audio_format="wav",
                        join_audio=True,
                        verbose=False,
                    )
                except Exception:
                    continue
                produced = sorted(Path(workdir).glob("segment*.wav"))
                if not produced:
                    continue
                try:
                    duration = sf.info(str(produced[0])).duration
                except Exception:
                    continue
                if duration < min_seconds:
                    continue
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(produced[0].read_bytes())
                return str(output_path)

        raise RuntimeError(
            f"Dia produced no usable audio after {attempts} attempts for chunk starting "
            f"{text[:80]!r}; the underlying mlx-audio errors were printed above."
        )

    def synthesize(self, text: str, output_path: str, voice: str | None = None) -> str:
        """Synthesize a single utterance as speaker S1."""
        safe_text = scrub_speech_text(text)
        return self._run_generation(f"[S1] {safe_text}", Path(output_path))

    @staticmethod
    def _split_long_text(text: str, limit: int) -> list[str]:
        """Split one turn's text into sentence-aligned pieces no longer than limit."""
        sentences = re.split(r"(?<=[.!?…])\s+", text)
        parts: list[str] = []
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) + 1 > limit:
                parts.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip() if current else sentence
        if current:
            parts.append(current)

        pieces: list[str] = []
        for part in parts:
            while len(part) > limit:
                pieces.append(part[:limit])
                part = part[limit:]
            if part.strip():
                pieces.append(part.strip())
        return pieces

    @staticmethod
    def build_chunks(turns: Sequence[dict[str, str]], *, host_speaker: str = "HOST") -> list[str]:
        """Split a script into [S1]/[S2]-tagged chunks safe for Dia's constraints.

        Two hard constraints drive this: Dia silently produces NO audio for text
        that opens with [S2], and its text encoder degrades into instant EOS past
        roughly 1k characters. So speaker tags are assigned PER CHUNK (the chunk's
        first speaker is always [S1]) and chunks never exceed MAX_CHUNK_CHARS,
        with oversized single turns split at sentence boundaries.
        """
        fragments: list[tuple[str, str]] = []
        for turn in turns:
            text = scrub_speech_text(str(turn.get("text", "")).strip())
            if not text:
                continue
            speaker = str(turn.get("speaker", host_speaker)).strip() or host_speaker
            for piece in DiaTTS._split_long_text(text, MAX_CHUNK_CHARS - 5):
                fragments.append((speaker, piece))

        chunks: list[str] = []
        current: list[str] = []
        current_size = 0
        tag_by_speaker: dict[str, str] = {}
        for speaker, text in fragments:
            fragment_size = len(text) + 5
            if current and current_size + fragment_size > MAX_CHUNK_CHARS:
                chunks.append(" ".join(current))
                current = []
                current_size = 0
                tag_by_speaker = {}
            tag = tag_by_speaker.setdefault(speaker, "[S1]" if not tag_by_speaker else "[S2]")
            current.append(f"{tag} {text}")
            current_size += fragment_size
        if current:
            chunks.append(" ".join(current))
        return chunks

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

        chunks = self.build_chunks(turns, host_speaker=host_speaker)

        outputs: list[str] = []
        for chunk_index, chunk in enumerate(chunks, start=1):
            chunk_path = target_dir / f"dialogue_{chunk_index:03d}.wav"
            text_path = chunk_path.with_suffix(".txt")
            # Resume support: reuse a finished chunk only if it was synthesized
            # from EXACTLY this text (guards against stale chunk boundaries).
            if (
                chunk_path.exists()
                and chunk_path.stat().st_size > 0
                and text_path.exists()
                and text_path.read_text(encoding="utf-8") == chunk
            ):
                outputs.append(str(chunk_path))
                continue
            outputs.append(self._run_generation(chunk, chunk_path))
            text_path.write_text(chunk, encoding="utf-8")
        return outputs
