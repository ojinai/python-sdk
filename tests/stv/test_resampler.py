"""Unit tests for ojin.stv.resampler."""

import asyncio

import numpy as np

from ojin.stv.resampler import NumpyLinearResampler, default_resampler


def test_default_resample_changes_length_to_target_rate() -> None:
    """Resampling 0.1 s @ 24 kHz to 16 kHz yields ~0.1 s @ 16 kHz."""
    r = default_resampler()
    pcm = np.zeros(2400, dtype="<i2").tobytes()  # 2400 samples @ 24 kHz = 0.1 s
    out = asyncio.run(r.resample(pcm, 24000, 16000))
    n = len(out) // 2
    assert abs(n - 1600) <= 8  # ~0.1 s @ 16 kHz


def test_resample_identity_when_equal_rates() -> None:
    """Equal in/out rates return the input untouched."""
    r = default_resampler()
    pcm = np.arange(100, dtype="<i2").tobytes()
    assert asyncio.run(r.resample(pcm, 16000, 16000)) == pcm


def test_resample_empty_is_empty() -> None:
    """Empty input resamples to empty output."""
    r = default_resampler()
    assert asyncio.run(r.resample(b"", 24000, 16000)) == b""


def test_numpy_linear_resampler_downsamples() -> None:
    """The pure-numpy fallback also downsamples to the target length."""
    r = NumpyLinearResampler()
    pcm = np.zeros(2400, dtype="<i2").tobytes()
    out = asyncio.run(r.resample(pcm, 24000, 16000))
    assert abs(len(out) // 2 - 1600) <= 2
