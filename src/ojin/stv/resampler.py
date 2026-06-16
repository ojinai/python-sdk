"""Resampler protocol + bundled default for the server-feed path.

Only the **copy of TTS audio sent to the inference server** is resampled (to
16 kHz, for lip-sync inference); the audio the avatar plays is the untouched
original. The async ``resample`` signature matches pipecat's resampler so
``OjinVideoService`` can inject pipecat's implementation directly.

The server feed is a *stream* of small TTS chunks, resampled one chunk at a time
(see ``OjinSTVClient.send_tts_audio``). Resampling each chunk in isolation with a
batch resampler re-runs the anti-alias filter from a cold state at every chunk
boundary, which injects audible discontinuities - measured at ~33-44 dB SNR
against a single full-signal pass, regardless of ``quality`` (HQ vs VHQ barely
moves it). The fix is a **streaming** resampler that carries filter history
across chunks: it reconstructs the same waveform as a full-signal resample
(~75-84 dB SNR) at the cost of a small constant filter delay.

So the default is ``SoxrStreamResampler`` (a faithful port of pipecat's
``SOXRStreamAudioResampler``); ``SoxrResampler`` (stateless VHQ batch, matching
pipecat's ``SOXRAudioResampler``) is kept for callers that genuinely have the
whole signal at once. A pure-numpy linear interpolator is the no-``soxr``
fallback. All assume mono int16 little-endian PCM — the shape of the
server-feed audio.
"""

from __future__ import annotations

import importlib
import importlib.util
import time
from typing import Protocol, runtime_checkable

import numpy as np

# A streaming resampler keeps filter history between chunks. When the gap since
# the last chunk exceeds this, the history is stale (a new turn / silence) and is
# cleared so old audio cannot bleed into the next turn's first samples.
_CLEAR_HISTORY_AFTER_S = 0.2


@runtime_checkable
class Resampler(Protocol):
    """Async resampler: int16 PCM at ``in_rate`` → int16 PCM at ``out_rate``."""

    async def resample(self, pcm: bytes, in_rate: int, out_rate: int) -> bytes:
        """Return ``pcm`` resampled from ``in_rate`` to ``out_rate`` Hz."""
        ...


class SoxrStreamResampler:
    """Streaming resampler backed by ``soxr.ResampleStream`` (VHQ, lazy-imported).

    Faithful port of pipecat's ``SOXRStreamAudioResampler``. It keeps the SoX
    resampler's internal filter history across calls, so feeding a chunked stream
    produces the same waveform as resampling the whole signal at once — no clicks
    at chunk boundaries. The trade-off is a small constant filter delay (~500
    samples at 24→16 kHz VHQ); it is not flushed, so the last ~30 ms of each turn
    stays buffered until the next chunk (inaudible on the lip-sync feed).

    This is the right default for the server-feed: it is called once per client
    with a fixed sample-rate pair and a stream of TTS chunks. Mono int16 only.
    """

    def __init__(self) -> None:
        """Initialize an empty stream; the SoX stream is built on first use."""
        self._in_rate: int | None = None
        self._out_rate: int | None = None
        self._last_resample_time: float = 0.0
        self._stream = None  # soxr.ResampleStream | None (lazy)

    def _ensure_stream(self, in_rate: int, out_rate: int) -> None:
        """(Re)build the SoX stream, or clear stale history after a gap.

        Unlike pipecat's resampler — which raises if reused with a different
        sample-rate pair — this is a library default, so a rate change simply
        rebuilds the stream for the new rates rather than erroring.
        """
        rates_changed = self._in_rate != in_rate or self._out_rate != out_rate
        if self._stream is None or rates_changed:
            soxr = importlib.import_module("soxr")
            self._in_rate = in_rate
            self._out_rate = out_rate
            self._stream = soxr.ResampleStream(
                in_rate=in_rate,
                out_rate=out_rate,
                num_channels=1,
                quality="VHQ",
                dtype="int16",
            )
        elif time.monotonic() - self._last_resample_time > _CLEAR_HISTORY_AFTER_S:
            self._stream.clear()
        self._last_resample_time = time.monotonic()

    async def resample(self, pcm: bytes, in_rate: int, out_rate: int) -> bytes:
        """Resample one chunk of mono int16 PCM; identity when rates match."""
        if not pcm or in_rate == out_rate:
            return pcm
        self._ensure_stream(in_rate, out_rate)
        samples = np.frombuffer(pcm, dtype="<i2")
        out = self._stream.resample_chunk(samples)
        return out.astype("<i2").tobytes()


class SoxrResampler:
    """Stateless high-quality batch resampler backed by ``soxr`` (lazy-imported).

    Matches pipecat's ``SOXRAudioResampler`` (``quality="VHQ"``). Each call
    resamples independently, so on a chunked stream it re-runs the anti-alias
    filter per chunk and produces boundary artifacts — prefer
    :class:`SoxrStreamResampler` for the server feed; use this only when you have
    the whole signal at once.
    """

    async def resample(self, pcm: bytes, in_rate: int, out_rate: int) -> bytes:
        """Resample mono int16 PCM with soxr (VHQ); identity when rates match."""
        if not pcm or in_rate == out_rate:
            return pcm
        soxr = importlib.import_module("soxr")  # lazy: avoided if you inject your own
        samples = np.frombuffer(pcm, dtype="<i2")
        out = soxr.resample(samples, in_rate, out_rate, quality="VHQ")
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
    """Return the best available bundled resampler for the chunked server feed.

    Prefers the streaming soxr resampler (high quality, boundary-artifact-free,
    GIL-releasing); falls back to the pure-numpy linear interpolator when
    ``soxr`` is not installed.
    """
    if importlib.util.find_spec("soxr") is None:
        return NumpyLinearResampler()
    return SoxrStreamResampler()
