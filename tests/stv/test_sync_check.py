"""Unit tests for the offline A/V sync verifier (ojin.stv.sync_check)."""

from __future__ import annotations

import numpy as np

from ojin.stv.sync_check import (
    TickSample,
    cross_correlation_lag,
    luma_motion_rms,
    report_to_dict,
    summarize,
)


def _rgb(width: int, height: int, value: int) -> bytes:
    """Build a flat RGB buffer of a single grey ``value``."""
    return bytes([value]) * (width * height * 3)


# ----------------------------------------------------------------------
# luma_motion_rms
# ----------------------------------------------------------------------


def test_luma_motion_rms_identical_frames_is_zero() -> None:
    """Two identical frames have zero motion energy."""
    frame = _rgb(4, 4, 120)
    assert luma_motion_rms(frame, frame, 4, 4) == 0.0


def test_luma_motion_rms_detects_change() -> None:
    """A black->white frame transition produces large motion energy."""
    prev = _rgb(4, 4, 0)
    cur = _rgb(4, 4, 255)
    assert luma_motion_rms(cur, prev, 4, 4) > 200.0


def test_luma_motion_rms_none_inputs_are_zero() -> None:
    """A missing current or previous frame yields zero motion (no signal)."""
    frame = _rgb(4, 4, 120)
    assert luma_motion_rms(None, frame, 4, 4) == 0.0
    assert luma_motion_rms(frame, None, 4, 4) == 0.0


# ----------------------------------------------------------------------
# cross_correlation_lag
# ----------------------------------------------------------------------


def test_cross_correlation_lag_zero_when_aligned() -> None:
    """Identical envelopes correlate best at lag 0."""
    a = [0.0, 0.0, 5.0, 9.0, 3.0, 0.0, 0.0, 1.0, 4.0, 0.0]
    lag, corr = cross_correlation_lag(a, list(a), max_lag=4)
    assert lag == 0
    assert corr > 0.99


def test_cross_correlation_lag_detects_video_delay() -> None:
    """When video trails audio by 2 ticks, the detected lag is +2."""
    audio = [0.0, 0.0, 5.0, 9.0, 3.0, 0.0, 0.0, 1.0, 4.0, 0.0]
    video = [0.0, 0.0, *audio[:-2]]  # video[t] == audio[t-2]
    lag, corr = cross_correlation_lag(audio, video, max_lag=4)
    assert lag == 2
    assert corr > 0.99


def test_cross_correlation_lag_detects_audio_delay() -> None:
    """When audio trails video by 1 tick, the detected lag is -1."""
    video = [0.0, 0.0, 5.0, 9.0, 3.0, 0.0, 0.0, 1.0, 4.0, 0.0]
    audio = [0.0, *video[:-1]]  # audio[t] == video[t-1]
    lag, _corr = cross_correlation_lag(audio, video, max_lag=4)
    assert lag == -1


# ----------------------------------------------------------------------
# summarize
# ----------------------------------------------------------------------


def _samples_from(audio: list[float], video: list[float]) -> list[TickSample]:
    """Build tick samples pairing an audio and a video-motion envelope."""
    return [
        TickSample(
            pts=i * 40_000_000,
            frame_type=1,
            audio_rms=a,
            video_volume=v,
            video_motion=v,
            video_present=True,
        )
        for i, (a, v) in enumerate(zip(audio, video, strict=False))
    ]


def test_summarize_aligned_session_is_aligned() -> None:
    """A perfectly tick-aligned session reports lag 0 and aligned=True."""
    env = [0.0, 0.0, 5.0, 9.0, 3.0, 0.0, 0.0, 1.0, 4.0, 0.0] * 3
    report = summarize(_samples_from(env, env), fps=25)
    assert report.lag_frames == 0
    assert report.dropped_video_ticks == 0
    assert report.aligned is True


def test_summarize_detects_offset() -> None:
    """A session whose video trails audio is reported as not aligned."""
    audio = [0.0, 0.0, 5.0, 9.0, 3.0, 0.0, 0.0, 1.0, 4.0, 0.0] * 3
    video = [0.0, 0.0, 0.0, *audio[:-3]]  # video lags audio by 3 ticks
    report = summarize(_samples_from(audio, video), fps=25)
    assert report.lag_frames == 3
    assert abs(report.lag_ms - 120.0) < 1e-6
    assert report.aligned is False


def test_summarize_trusts_volume_channel_over_noisy_motion() -> None:
    """Perfect audio↔bundled-audio alignment is aligned even if pixel motion is noisy.

    The played-audio vs server-bundled-audio (``video_volume``) channel is the exact
    A/V-sync measure; the pixel-``video_motion`` channel is a weak, lagged proxy and
    must not by itself flip the verdict to misaligned.
    """
    audio = [0.0, 0.0, 5.0, 9.0, 3.0, 0.0, 0.0, 1.0, 4.0, 0.0] * 3
    samples = [
        TickSample(
            pts=i * 40_000_000,
            frame_type=1,
            audio_rms=a,
            video_volume=a,  # bundled audio tracks played audio exactly (lag 0)
            video_motion=7.0,  # constant -> zero correlation with audio
            video_present=True,
        )
        for i, a in enumerate(audio)
    ]
    report = summarize(samples, fps=25)
    assert report.volume_lag_frames == 0
    assert report.volume_correlation > 0.99
    assert report.aligned is True


def test_summarize_counts_dropped_video_ticks() -> None:
    """Ticks that carried audio but no video frame are counted as drops."""
    env = [0.0, 5.0, 9.0, 3.0, 1.0, 4.0, 2.0, 0.0]
    samples = _samples_from(env, env)
    samples[3].video_present = False  # a dropped video frame
    samples[3].video_motion = 0.0
    report = summarize(samples, fps=25)
    assert report.dropped_video_ticks == 1
    assert report.aligned is False


def test_summarize_uses_numpy_float_safely() -> None:
    """RMS values arriving as numpy floats are handled without error."""
    env = [np.float64(x) for x in (0.0, 5.0, 9.0, 3.0, 0.0, 1.0, 4.0, 0.0)]
    report = summarize(_samples_from(env, env), fps=25)
    assert report.aligned is True


# ----------------------------------------------------------------------
# to_perfetto_trace (Chrome Trace Event Format, loadable in ui.perfetto.dev)
# ----------------------------------------------------------------------


def test_report_to_dict_round_trips() -> None:
    """report_to_dict exposes the verdict fields for embedding in the trace."""
    env = [0.0, 5.0, 9.0, 3.0, 0.0]
    report = summarize(_samples_from(env, env), fps=25)
    d = report_to_dict(report)
    assert d["aligned"] == report.aligned
    assert d["volume_lag_frames"] == report.volume_lag_frames
