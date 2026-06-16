"""Tests for FPS and latency profiling helpers."""

from __future__ import annotations

import itertools
import time

import pytest

from ojin.profiling_utils import FPSTracker, LatencyTracker


@pytest.fixture(autouse=True)
def reset_latency_tracker() -> None:
    """Clear shared latency tracker state around each test."""
    LatencyTracker.reset()


def test_fps_history_getters_return_copies() -> None:
    """Callers must not be able to mutate tracker history through getters."""
    tracker = FPSTracker("test")
    tracker.fps_history = [24.0]
    tracker.partial_fps_history = [12.0]

    tracker.get_fps_history().append(1.0)
    tracker.get_partial_fps_history().append(1.0)

    assert tracker.fps_history == [24.0]
    assert tracker.partial_fps_history == [12.0]


def test_partial_reset_resets_partial_timing_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial FPS reset should not include idle gap in the next partial window."""
    tracker = FPSTracker("test")
    tracker.last_partial_time = 10.0
    monkeypatch.setattr(time, "perf_counter", lambda: 20.0)

    tracker.reset_partial_average()

    assert tracker.last_partial_time == pytest.approx(20.0)


def test_log_resets_partial_timing_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Logging a partial window should start the next partial timing window now."""
    tracker = FPSTracker("test")
    tracker.last_partial_time = 10.0
    monkeypatch.setattr(time, "perf_counter", lambda: 20.0)

    tracker.log()

    assert tracker.last_partial_time == pytest.approx(20.0)


def test_latency_tracker_prunes_completed_measures_but_keeps_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completed latency records should not accumulate in the active measure list."""
    times = itertools.count(start=100, step=2)
    monkeypatch.setattr(time, "perf_counter", lambda: next(times))

    LatencyTracker.start_latency_measure("sample")
    LatencyTracker.stop_latency_measure("sample")

    assert LatencyTracker._measures["sample"] == []
    assert LatencyTracker.average("sample") == pytest.approx(2.0)
    assert LatencyTracker.max("sample") == pytest.approx(2.0)
    assert LatencyTracker.min("sample") == pytest.approx(2.0)


def test_latency_tracker_stop_without_running_measure_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Stopping a missing or already-pruned measure should be a no-op."""
    LatencyTracker.stop_latency_measure("missing")
    LatencyTracker._measures["empty"] = []
    LatencyTracker.stop_latency_measure("empty")

    assert "Latency measure missing was not running" in caplog.text
    assert "Latency measure empty was not running" in caplog.text
