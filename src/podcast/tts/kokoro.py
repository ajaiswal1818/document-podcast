"""Kokoro-based TTS wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

try:  # pragma: no cover - optional MLX dependency path
    from kokoro_mlx import KokoroTTS as KokoroRuntime
except Exception:  # pragma: no cover
    KokoroRuntime = None


def _normalize_clip(data: np.ndarray, target_rms: float = 0.06) -> np.ndarray:
    """Bring a clip to a consistent perceived loudness without clipping."""
    rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
    if rms > 1e-6:
        data = data * (target_rms / rms)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 0.95:
        data = data * (0.95 / peak)
    return data.astype(np.float32, copy=False)


def assemble_audio_files(
    audio_files: Sequence[str],
    output_path: str,
    *,
    gap_seconds: float = 0.35,
    normalize: bool = True,
) -> str:
    """Concatenate WAV clips with natural inter-turn pauses and consistent loudness."""
    if not audio_files:
        raise ValueError("No audio files provided to assemble.")

    arrays: list[np.ndarray] = []
    sample_rate: int | None = None
    channels: int | None = None

    for file_path in audio_files:
        data, sr = sf.read(file_path, dtype="float32", always_2d=False)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"Audio sample rates do not match: {sr} != {sample_rate}")
        if channels is None:
            channels = data.shape[1]
        elif data.shape[1] != channels:
            raise ValueError("Audio files have inconsistent channel counts.")
        if normalize:
            data = _normalize_clip(data)
        arrays.append(data.astype(np.float32, copy=False))

    if sample_rate is None or not arrays:
        raise ValueError("No valid audio data could be assembled.")

    gap = np.zeros((int(sample_rate * max(0.0, gap_seconds)), channels or 1), dtype=np.float32)
    segments: list[np.ndarray] = []
    for index, clip in enumerate(arrays):
        if index > 0 and gap.size:
            segments.append(gap)
        segments.append(clip)

    combined = np.concatenate(segments, axis=0)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), combined, sample_rate)
    return str(target)


class KokoroTTS:
    """Small wrapper around the MLX Kokoro implementation."""

    def __init__(self, voice: str = "af_heart") -> None:
        self.voice = voice
        self.model: Any | None = None

        if KokoroRuntime is not None:
            try:
                self.model = KokoroRuntime.from_pretrained()
            except Exception as exc:  # pragma: no cover - download may fail at runtime
                raise RuntimeError(f"Unable to initialize Kokoro TTS model: {exc}") from exc

    def synthesize(self, text: str, output_path: str, voice: str | None = None) -> str:
        """Generate audio for a text string and save it to a WAV file."""
        chosen_voice = voice or self.voice

        if self.model is None:
            raise RuntimeError("Kokoro TTS model is not available.")

        try:
            if hasattr(self.model, "generate"):
                result = self.model.generate(text=text, voice=chosen_voice)
                audio = getattr(result, "audio", None)
                sample_rate = getattr(result, "sample_rate", 24000)
                if audio is None:
                    raise RuntimeError("Kokoro generate() did not return audio data.")
                audio_array = np.asarray(audio, dtype=np.float32)
                target = Path(output_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(target), audio_array, int(sample_rate))
                return str(target)

            if hasattr(self.model, "speak"):
                self.model.speak(text, voice=chosen_voice)
                return output_path

            raise RuntimeError("Kokoro model does not expose a supported generation method.")
        except Exception as exc:
            raise RuntimeError(f"Kokoro synthesis failed: {exc}") from exc
