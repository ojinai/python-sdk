"""Unit tests for ojin.stv.resampler.

The server feed is resampled one TTS chunk at a time. The regression these tests
guard is that the default resampler reconstructs (almost) the same waveform as a
single full-signal pass — which a batch-per-chunk resampler does not, because it
re-runs the anti-alias filter from cold at every chunk boundary.
"""

import asyncio
import importlib.util

import numpy as np
import pytest

from ojin.stv.resampler import (
    NumpyLinearResampler,
    SoxrResampler,
    SoxrStreamResampler,
    default_resampler,
)

HAS_SOXR = importlib.util.find_spec("soxr") is not None
requires_soxr = pytest.mark.skipif(not HAS_SOXR, reason="soxr not installed")

SR_IN, SR_OUT = 24000, 16000


def _broadband_24k(seconds: float = 4.0) -> np.ndarray:
    """Deterministic full-band 24 kHz signal: log sweep + tones + noise.

    Has energy above the 8 kHz output Nyquist, so the 24→16 anti-alias filter
    must actually work — which is where per-chunk batch resampling shows its
    boundary artifacts.
    """
    n = int(seconds * SR_IN)
    t = np.arange(n) / SR_IN
    f0, f1 = 50.0, 11500.0
    k = (f1 / f0) ** (1.0 / seconds)
    sig = 0.5 * np.sin(2 * np.pi * f0 * (k**t - 1) / np.log(k))
    for f in (1000, 3000, 6000, 9000, 10500):
        sig += 0.08 * np.sin(2 * np.pi * f * t)
    sig += 0.05 * np.random.default_rng(1234).standard_normal(n)
    sig /= np.max(np.abs(sig)) + 1e-9
    return (sig * 28000).astype("<i2")


def _chunks(sig_i16: np.ndarray, chunk_ms: int = 20) -> list[bytes]:
    raw = sig_i16.tobytes()
    step = int(SR_IN * chunk_ms / 1000) * 2
    return [raw[i : i + step] for i in range(0, len(raw), step)]


def _run_chunked(resampler, chunks: list[bytes]) -> np.ndarray:
    out = b"".join(asyncio.run(resampler.resample(c, SR_IN, SR_OUT)) for c in chunks)
    return np.frombuffer(out, dtype="<i2").astype(np.float64)


def _best_lag_snr_db(cand: np.ndarray, gt: np.ndarray, max_lag: int = 600) -> float:
    """SNR (dB) of ``cand`` vs ``gt`` after aligning within ±``max_lag`` samples."""
    n = min(len(cand), len(gt))
    cand, gt = cand[:n], gt[:n]
    denom = np.sum(gt**2) + 1e-9
    best = -1e9
    for lag in range(0, max_lag + 1):  # stream output lags ground truth
        a, b = cand[lag:], gt[: n - lag]
        m = min(len(a), len(b))
        resid = a[:m] - b[:m]
        best = max(best, 10 * np.log10(denom / (np.sum(resid**2) + 1e-9)))
    return best


@requires_soxr
def test_default_resampler_is_streaming_when_soxr_available() -> None:
    """The bundled default is the boundary-artifact-free streaming resampler."""
    assert isinstance(default_resampler(), SoxrStreamResampler)


@requires_soxr
def test_stream_chunked_reconstructs_fullsignal_far_better_than_batch() -> None:
    """Chunked streaming resample ~ full-signal pass; batch-per-chunk is far worse."""
    soxr = importlib.import_module("soxr")
    src = _broadband_24k()
    ground_truth = soxr.resample(src, SR_IN, SR_OUT, quality="VHQ").astype(np.float64)
    chunks = _chunks(src, chunk_ms=20)

    stream = _run_chunked(SoxrStreamResampler(), chunks)
    batch = _run_chunked(SoxrResampler(), chunks)
    stream_snr = _best_lag_snr_db(stream, ground_truth)
    batch_snr = _best_lag_snr_db(batch, ground_truth)

    assert stream_snr > 60.0, f"stream SNR too low: {stream_snr:.1f} dB"
    assert stream_snr > batch_snr + 20.0, (
        f"stream {stream_snr:.1f} not >> batch {batch_snr:.1f}"
    )


@requires_soxr
def test_stream_identity_when_equal_rates() -> None:
    """Equal in/out rates return the input untouched (no stream allocated)."""
    pcm = np.arange(100, dtype="<i2").tobytes()
    assert asyncio.run(SoxrStreamResampler().resample(pcm, 16000, 16000)) == pcm


@requires_soxr
def test_stream_empty_is_empty() -> None:
    """Empty input resamples to empty output."""
    assert asyncio.run(SoxrStreamResampler().resample(b"", SR_IN, SR_OUT)) == b""


@requires_soxr
def test_default_resampler_streaming_length_within_filter_delay() -> None:
    """0.5 s @ 24 kHz fed in chunks yields ~0.5 s @ 16 kHz minus the filter delay."""
    src = np.zeros(int(0.5 * SR_IN), dtype="<i2")  # 0.5 s
    out = _run_chunked(default_resampler(), _chunks(src, chunk_ms=20))
    target = int(0.5 * SR_OUT)  # 8000
    # The stream holds back a small constant filter delay (~500 samples) it never
    # flushes; everything else must be there, and it must never overshoot.
    assert 0 <= target - len(out) <= 600


@requires_soxr
def test_batch_vhq_downsamples_to_target_length() -> None:
    """SoxrResampler (full-signal) hits the target length within rounding."""
    pcm = np.zeros(2400, dtype="<i2").tobytes()  # 0.1 s @ 24 kHz
    out = asyncio.run(SoxrResampler().resample(pcm, SR_IN, SR_OUT))
    assert abs(len(out) // 2 - 1600) <= 8  # ~0.1 s @ 16 kHz


def test_numpy_linear_resampler_downsamples() -> None:
    """The pure-numpy fallback also downsamples to the target length."""
    pcm = np.zeros(2400, dtype="<i2").tobytes()
    out = asyncio.run(NumpyLinearResampler().resample(pcm, SR_IN, SR_OUT))
    assert abs(len(out) // 2 - 1600) <= 2


def test_numpy_linear_identity_and_empty() -> None:
    """Equal rates pass through; empty stays empty."""
    pcm = np.arange(100, dtype="<i2").tobytes()
    assert asyncio.run(NumpyLinearResampler().resample(pcm, 16000, 16000)) == pcm
    assert asyncio.run(NumpyLinearResampler().resample(b"", SR_IN, SR_OUT)) == b""


class _FakeClock:
    """Fake ``time`` for the resampler module.

    Lets tests drive the inter-chunk gap that decides history clearing, without
    real sleeps.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def monotonic(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@requires_soxr
def test_inter_chunk_gaps_do_not_change_output(monkeypatch) -> None:
    """Gapped delivery must resample identically to back-to-back delivery.

    Output depends only on the audio fed, never on the wall-clock timing of the
    calls. Regression for the Hume/24 kHz lip-sync drift: the resampler used to reset
    its filter history whenever the wall-clock gap since the last chunk exceeded
    a threshold. On a real-time TTS feed every chunk tripped that reset, so each
    was resampled from cold and the soxr group-delay tail was dropped — the feed
    came out ~1.7% short and the mouth ran ahead of the played audio at ~16 ms/s.
    """
    import ojin.stv.resampler as resampler_mod

    src = _broadband_24k(seconds=6.0)
    raw = src.tobytes()
    step = SR_IN * 2  # 1 s of 24 kHz int16 per chunk
    chunks = [raw[i : i + step] for i in range(0, len(raw), step)]

    def run(gap_s: float) -> bytes:
        clock = _FakeClock()
        # Inert now (the resampler consults no clock); if a wall-clock reset is
        # ever reintroduced this patch makes the two runs diverge and fail.
        monkeypatch.setattr(resampler_mod, "time", clock, raising=False)
        resampler = SoxrStreamResampler()
        out = bytearray()
        for c in chunks:
            clock.advance(gap_s)  # simulate arrival gap between chunks
            out += asyncio.run(resampler.resample(c, SR_IN, SR_OUT))
        return bytes(out)

    realtime = run(1.0)  # >0.2 s between chunks (real-time streaming)
    back_to_back = run(0.01)  # bulk delivery, no meaningful gap

    assert realtime == back_to_back, (
        f"inter-chunk timing changed the resample ({len(realtime)} vs "
        f"{len(back_to_back)} bytes) — output must depend only on the audio"
    )


@requires_soxr
def test_flush_drains_filter_tail_to_exact_duration() -> None:
    """flush() drains the held filter tail so a turn's feed is duration-exact.

    No missing sub-frame tail is left behind to accumulate into drift.
    """
    resampler = SoxrStreamResampler()
    src = _broadband_24k(seconds=5.0)
    raw = src.tobytes()
    step = SR_IN * 2
    chunks = [raw[i : i + step] for i in range(0, len(raw), step)]

    out = bytearray()
    for c in chunks:
        out += asyncio.run(resampler.resample(c, SR_IN, SR_OUT))
    tail = resampler.flush()
    out += tail

    in_samples = len(raw) // 2
    out_samples = len(out) // 2
    ideal = round(in_samples * SR_OUT / SR_IN)
    assert tail, "flush() should emit the held filter tail"
    assert abs(out_samples - ideal) <= 4, (
        f"flushed stream {out_samples} samples vs exact {ideal}"
    )


@requires_soxr
def test_flush_is_empty_and_resets_stream() -> None:
    """flush() drains once after feeding, then resets so a repeat flush is empty.

    A fresh stream flushes to nothing; after resampling it yields the tail once,
    then the next chunk starts a new segment.
    """
    resampler = SoxrStreamResampler()
    assert resampler.flush() == b""  # nothing fed yet
    asyncio.run(
        resampler.resample(np.zeros(SR_IN, dtype="<i2").tobytes(), SR_IN, SR_OUT)
    )
    assert resampler.flush()  # tail available after feeding
    assert resampler.flush() == b""  # already drained + reset


def test_stateless_resamplers_flush_empty() -> None:
    """The batch and numpy resamplers hold no state, so flush() is always empty."""
    assert NumpyLinearResampler().flush() == b""
    if HAS_SOXR:
        assert SoxrResampler().flush() == b""
