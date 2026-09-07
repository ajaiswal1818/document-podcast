"""Final audio assembly: joining turn clips with humanized pauses and consistent loudness."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf


def _normalize_clip(data: np.ndarray, target_rms: float = 0.06) -> np.ndarray:
    """Bring a clip to a consistent perceived loudness without clipping."""
    rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
    if rms > 1e-6:
        data = data * (target_rms / rms)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 0.95:
        data = data * (0.95 / peak)
    return data.astype(np.float32, copy=False)


def _apply_edge_fades(data: np.ndarray, sample_rate: int, fade_seconds: float = 0.008) -> np.ndarray:
    """Short linear fades at clip edges so zero-gap joins do not click."""
    fade_samples = min(int(sample_rate * fade_seconds), data.shape[0] // 2)
    if fade_samples <= 0:
        return data
    ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32).reshape(-1, 1)
    data = data.copy()
    data[:fade_samples] *= ramp
    data[-fade_samples:] *= ramp[::-1]
    return data


def _time_stretch(data: np.ndarray, rate: float) -> np.ndarray:
    """Slow down (rate < 1) or speed up speech without changing pitch."""
    import librosa

    channels = [
        librosa.effects.time_stretch(np.ascontiguousarray(data[:, ch]), rate=rate)
        for ch in range(data.shape[1])
    ]
    length = min(len(ch) for ch in channels)
    return np.stack([ch[:length] for ch in channels], axis=1).astype(np.float32)


def assemble_audio_files(
    audio_files: Sequence[str],
    output_path: str,
    *,
    gap_seconds: float = 0.35,
    gaps: Sequence[float] | None = None,
    normalize: bool = True,
    stretch_rate: float | None = None,
) -> str:
    """Concatenate WAV clips with natural inter-turn pauses and consistent loudness.

    `gaps` gives an explicit pause duration before each clip after the first
    (len(audio_files) - 1 values); otherwise the uniform `gap_seconds` is used.
    `stretch_rate` < 1.0 slows the speech (pitch-preserving) by 1/rate; used to
    bring fast TTS models down to a natural speaking pace.
    """
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
        if stretch_rate is not None and 0 < stretch_rate < 1.0:
            data = _time_stretch(data, stretch_rate)
        if normalize:
            data = _normalize_clip(data)
        data = _apply_edge_fades(data, sr)
        arrays.append(data.astype(np.float32, copy=False))

    if sample_rate is None or not arrays:
        raise ValueError("No valid audio data could be assembled.")

    segments: list[np.ndarray] = []
    for index, clip in enumerate(arrays):
        if index > 0:
            pause = gaps[index - 1] if gaps is not None and index - 1 < len(gaps) else gap_seconds
            pause_samples = int(sample_rate * max(0.0, float(pause)))
            if pause_samples:
                segments.append(np.zeros((pause_samples, channels or 1), dtype=np.float32))
        segments.append(clip)

    combined = np.concatenate(segments, axis=0)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), combined, sample_rate)
    return str(target)
