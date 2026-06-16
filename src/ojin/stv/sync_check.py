"""Offline audio/video sync verification for an STV session's output stream.

The avatar plays the buffered original TTS audio while the inference server returns
the matching video frames; if the playback loop pairs them correctly the played
audio amplitude and the on-screen lip motion rise and fall together. This module
turns that into a measurable check: collect a :class:`TickSample` per emitted tick
(audio RMS + video-motion RMS, joined by ``pts``), then :func:`summarize`
cross-correlates the two envelopes to report the lag between sound and motion and
whether any video frames went missing.

Pure and numpy-only (no I/O, no client/transport coupling), so it is unit-testable
and reusable by any consumer that wants to verify a session — the file demo writes
the samples to ``session.json`` and logs the resulting :class:`SyncReport`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np

# A tick at 25 fps is 40 ms; one frame of lag is the smallest resolvable offset, so
# treat up to one frame either way as "in sync".
_DEFAULT_LAG_TOLERANCE_FRAMES = 1
# Below this peak correlation the two envelopes are not really tracking each other
# (e.g. near-silent clip), so a "lag" reading is not meaningful — fail closed.
_DEFAULT_MIN_CORRELATION = 0.4
# Search ±1 s for the best alignment; real desync from drops/trims is well inside.
_DEFAULT_MAX_LAG_FRAMES = 25

_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float64)
# Pearson correlation needs at least two paired points to be defined.
_MIN_SERIES_LEN = 2
# Motion is computed on at most this many evenly-strided pixels. A full 1280x720
# float64 luma diff per frame (~2.7M px) is tens of ms and would stall a consumer
# running on the playback event loop; a few thousand pixels keeps the same envelope
# while costing well under a millisecond.
_MOTION_SAMPLE_PIXELS = 30_000


@dataclass
class TickSample:
    """One emitted playback tick, joined by ``pts`` across the audio/video output.

    ``audio_rms`` is the amplitude of the *played* audio (original TTS) this tick;
    ``video_volume`` is the amplitude of the audio the server bundled with the
    frame (the lip-sync target); ``video_motion`` is the luma frame-delta amplitude
    of the rendered pixels. ``video_present`` is ``False`` when the tick carried
    audio but no video frame — a frame dropped between client and consumer.
    """

    pts: int
    frame_type: int
    audio_rms: float
    video_volume: float
    video_motion: float
    video_present: bool


@dataclass
class SyncReport:
    """The verdict of cross-correlating the played-audio and video envelopes."""

    ticks: int
    fps: int
    dropped_video_ticks: int
    lag_frames: int  # +ve => video trails (lags) audio; -ve => audio trails video
    lag_ms: float
    motion_correlation: float  # peak normalized corr of audio_rms vs video_motion
    volume_correlation: float  # peak normalized corr of audio_rms vs video_volume
    volume_lag_frames: int  # lag of audio_rms vs the server's bundled-audio envelope
    aligned: bool


def luma_motion_rms(
    rgb: Optional[bytes], prev_rgb: Optional[bytes], width: int, height: int
) -> float:
    """Return the RMS of the per-pixel luma change between two RGB frames.

    Rendered lip motion shows up as luma change between consecutive frames, so this
    is a pixel-domain proxy for "the avatar is moving/speaking now". Returns ``0.0``
    when either frame is missing (no decoded pixels / start of stream).

    Args:
        rgb: current frame's packed RGB bytes (``width*height*3``), or ``None``.
        prev_rgb: previous frame's packed RGB bytes, or ``None``.
        width: frame width in pixels.
        height: frame height in pixels.

    Returns:
        The root-mean-square of the luma difference image, as a float.

    """
    if rgb is None or prev_rgb is None or len(rgb) != len(prev_rgb):
        return 0.0
    cur = np.frombuffer(rgb, dtype=np.uint8)
    prev = np.frombuffer(prev_rgb, dtype=np.uint8)
    if cur.size != width * height * 3 or cur.size == 0:
        return 0.0
    # Evenly subsample whole RGB triples to bound the per-frame cost (see constant).
    pixels = width * height
    stride = max(1, pixels // _MOTION_SAMPLE_PIXELS)
    cur_px = cur.reshape(-1, 3)[::stride].astype(np.float64)
    prev_px = prev.reshape(-1, 3)[::stride].astype(np.float64)
    diff = cur_px @ _LUMA_WEIGHTS - prev_px @ _LUMA_WEIGHTS
    return float(np.sqrt(np.mean(diff * diff)))


def cross_correlation_lag(
    audio: List[float], video: List[float], max_lag: int
) -> tuple[int, float]:
    """Find the integer lag (in ticks) that best aligns ``video`` onto ``audio``.

    For each candidate shift ``d`` in ``[-max_lag, max_lag]`` the Pearson
    correlation of ``audio[t]`` against ``video[t + d]`` is computed over the
    overlapping region, and the ``d`` with the highest correlation is returned.

    A positive lag means the video envelope must be advanced to meet the audio —
    i.e. the video *trails* (lags) the audio by that many ticks. A negative lag
    means the audio trails the video.

    Args:
        audio: the played-audio amplitude envelope, one value per tick.
        video: the video (motion or bundled-audio) envelope, one value per tick.
        max_lag: maximum shift to search, in ticks.

    Returns:
        ``(lag, correlation)`` — the best-aligning lag and its correlation in
        ``[-1, 1]``; ``(0, 0.0)`` when there is not enough varying signal.

    """
    a = np.asarray(audio, dtype=np.float64)
    v = np.asarray(video, dtype=np.float64)
    n = min(a.size, v.size)
    if n < _MIN_SERIES_LEN:
        return 0, 0.0
    a, v = a[:n], v[:n]

    # Search lags in order of increasing magnitude so a tie (e.g. a periodic
    # envelope that correlates equally at several shifts) resolves to the smallest
    # offset rather than an arbitrary far one.
    best_lag, best_corr = 0, -2.0
    for d in sorted(range(-max_lag, max_lag + 1), key=lambda k: (abs(k), k)):
        overlap = n - abs(d)
        if overlap < _MIN_SERIES_LEN:
            continue
        if d >= 0:
            x, y = a[:overlap], v[d : d + overlap]
        else:
            x, y = a[-d : -d + overlap], v[:overlap]
        corr = _pearson(x, y)
        if corr > best_corr:
            best_lag, best_corr = d, corr
    if best_corr < -1.0:
        return 0, 0.0
    return best_lag, best_corr


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of two equal-length series; 0.0 if either is constant."""
    xc = x - x.mean()
    yc = y - y.mean()
    denom = float(np.sqrt(np.sum(xc * xc) * np.sum(yc * yc)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(xc * yc) / denom)


def summarize(
    samples: List[TickSample],
    fps: int,
    *,
    max_lag_frames: int = _DEFAULT_MAX_LAG_FRAMES,
    lag_tolerance_frames: int = _DEFAULT_LAG_TOLERANCE_FRAMES,
    min_correlation: float = _DEFAULT_MIN_CORRELATION,
) -> SyncReport:
    """Cross-correlate the played-audio and video envelopes into a verdict.

    Args:
        samples: per-tick samples, in emission order.
        fps: playback frame rate (to convert a tick lag to milliseconds).
        max_lag_frames: maximum lag searched, in ticks.
        lag_tolerance_frames: lag (either sign) still considered aligned.
        min_correlation: minimum motion correlation to trust the lag reading.

    Returns:
        A :class:`SyncReport`. ``aligned`` is True only when no video ticks were
        dropped, the audio/motion correlation is at least ``min_correlation``, and
        the detected lag is within ``lag_tolerance_frames``.

    """
    audio = [float(s.audio_rms) for s in samples]
    motion = [float(s.video_motion) for s in samples]
    volume = [float(s.video_volume) for s in samples]
    dropped = sum(1 for s in samples if not s.video_present)

    lag_frames, motion_corr = cross_correlation_lag(audio, motion, max_lag_frames)
    volume_lag, volume_corr = cross_correlation_lag(audio, volume, max_lag_frames)

    # The played-audio vs server-bundled-audio (volume) channel is the exact A/V
    # measure: both are audio amplitudes, and the bundled audio is generated with
    # the video, so a lag here is a true audio↔video offset. Pixel motion is a
    # weak, lagged proxy (the mouth shape does not track amplitude one-to-one), so
    # it is reported for context but must not gate the verdict.
    aligned = (
        dropped == 0
        and volume_corr >= min_correlation
        and abs(volume_lag) <= lag_tolerance_frames
    )
    return SyncReport(
        ticks=len(samples),
        fps=fps,
        dropped_video_ticks=dropped,
        lag_frames=lag_frames,
        lag_ms=lag_frames / fps * 1000.0,
        motion_correlation=motion_corr,
        volume_correlation=volume_corr,
        volume_lag_frames=volume_lag,
        aligned=aligned,
    )


def report_to_dict(report: SyncReport) -> dict:
    """Return a JSON-serializable dict for a :class:`SyncReport`."""
    return asdict(report)
