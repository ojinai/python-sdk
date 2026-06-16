"""Resampler protocol + bundled default for the server-feed path.

Only the **copy of TTS audio sent to the inference server** is resampled (to
16 kHz, for lip-sync inference); the audio the avatar plays is the untouched
original. The async ``resample`` signature matches pipecat's resampler so
``OjinVideoService`` can inject pipecat's implementation directly.

The default prefers ``soxr`` (high quality, GIL-releasing) and falls back to a
pure-numpy linear interpolator when ``soxr`` is not installed. Both assume mono
int16 little-endian PCM — the shape of the server-feed audio.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Resampler(Protocol):
    """Async resampler: int16 PCM at ``in_rate`` → int16 PCM at ``out_rate``."""

    async def resample(self, pcm: bytes, in_rate: int, out_rate: int) -> bytes:
        """Return ``pcm`` resampled from ``in_rate`` to ``out_rate`` Hz."""
        ...


class SoxrResampler:
    """High-quality resampler backed by ``soxr`` (lazy-imported)."""

    async def resample(self, pcm: bytes, in_rate: int, out_rate: int) -> bytes:
        """Resample mono int16 PCM with soxr; identity when rates match."""
        if not pcm or in_rate == out_rate:
            return pcm
        soxr = importlib.import_module("soxr")  # lazy: avoided if you inject your own
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.int16)
        out = soxr.resample(samples, in_rate, out_rate)
        return out.astype("<i2").tobytes()


class NumpyLinearResampler:
    """Pure-numpy linear-interpolation fallback (no extra dependency)."""

    async def resample(self, pcm: bytes, in_rate: int, out_rate: int) -> bytes:
        """Resample mono int16 PCM by linear interpolation; identity on match."""
        if not pcm or in_rate == out_rate:
            return pcm
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        n_out = round(len(samples) * out_rate / in_rate)
        if n_out <= 0:
            return b""
        x_old = np.arange(len(samples))
        x_new = np.linspace(0, len(samples) - 1, n_out)
        out = np.interp(x_new, x_old, samples)
        return np.clip(np.round(out), -32768, 32767).astype("<i2").tobytes()


def default_resampler() -> Resampler:
    """Return the best available bundled resampler (soxr, else numpy-linear)."""
    if importlib.util.find_spec("soxr") is None:
        return NumpyLinearResampler()
    return SoxrResampler()
