"""Unit tests for ojin.stv.audio_utils (pure audio math)."""

import numpy as np

from ojin.stv.audio_utils import fade_chunk, rms_int16


def test_rms_none_on_empty_or_odd() -> None:
    """RMS is undefined (None) for empty or sub-sample input."""
    assert rms_int16(b"") is None
    assert rms_int16(b"\x01") is None


def test_rms_known_signal() -> None:
    """RMS of a constant signal equals that constant."""
    pcm = np.full(100, 1000, dtype="<i2").tobytes()
    assert abs(rms_int16(pcm) - 1000.0) < 1e-6


def test_fade_chunk_ramps_to_zero_and_clips() -> None:
    """Gain starts ~1.0, reaches ~0 by the window end, and stays clipped past it."""
    n = 100
    pcm = np.full(n, 10000, dtype="<i2").tobytes()
    out = np.frombuffer(fade_chunk(pcm, 0, n), dtype="<i2")
    assert out[0] == 10000  # gain ~1.0 at the first sample
    assert abs(int(out[-1])) <= 200  # near zero at the end of the window
    # fully past the window → all zero
    out2 = np.frombuffer(fade_chunk(pcm, n, n), dtype="<i2")
    assert not out2.any()


def test_fade_chunk_continuous_across_chunks() -> None:
    """Threading samples_emitted keeps the ramp monotonic across chunk boundaries."""
    n = 50
    pcm = np.full(n, 10000, dtype="<i2").tobytes()
    a = np.frombuffer(fade_chunk(pcm, 0, 100), dtype="<i2")
    b = np.frombuffer(fade_chunk(pcm, 50, 100), dtype="<i2")
    assert a[-1] >= b[0]  # monotonic non-increasing across the boundary
