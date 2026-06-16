"""Pure int16 PCM audio helpers for the STV client.

These are the amplitude-domain primitives used by the synchronizer for barge-in
fades and swap-time alignment. They are pure (no I/O, no shared state) and depend
only on numpy, so they are trivially unit-testable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

_BYTES_PER_SAMPLE = 2  # int16


def rms_int16(audio: bytes) -> Optional[float]:
    """Return the RMS amplitude of int16 little-endian PCM, or None.

    Args:
        audio: int16 little-endian PCM bytes.

    Returns:
        The root-mean-square amplitude as a float, or ``None`` when the input is
        empty or shorter than one sample (odd/zero length).

    """
    if not audio or len(audio) < _BYTES_PER_SAMPLE:
        return None
    samples = np.frombuffer(audio[: len(audio) - (len(audio) % 2)], dtype="<i2")
    if samples.size == 0:
        return None
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def fade_chunk(chunk: bytes, samples_emitted: int, fade_total_samples: int) -> bytes:
    """Linearly ramp int16 PCM volume toward silence across a fade window.

    Scales ``chunk`` by a per-sample gain that decreases linearly from
    ``1 - samples_emitted / fade_total_samples`` at the first sample, reaching 0
    once ``fade_total_samples`` samples have been emitted. The gain is computed in
    flat int16-sample space (channels interleaved), so it is channel-agnostic and
    click-free across chunk boundaries as long as the caller threads
    ``samples_emitted`` forward. Pure: no I/O, no shared state.

    Args:
        chunk: int16 little-endian PCM bytes for this tick.
        samples_emitted: flat int16 samples already emitted since the fade began.
        fade_total_samples: total flat int16 samples over which to ramp to silence
            (``fade_s * sample_rate * num_channels``); must be > 0.

    Returns:
        Scaled int16 little-endian PCM bytes (same sample count as ``chunk``).

    """
    n = len(chunk) // 2
    samples = np.frombuffer(chunk[: n * 2], dtype="<i2").astype(np.float32)
    idx = np.arange(samples_emitted, samples_emitted + n, dtype=np.float32)
    gain = np.clip(1.0 - idx / float(fade_total_samples), 0.0, 1.0)
    return (samples * gain).astype("<i2").tobytes()
