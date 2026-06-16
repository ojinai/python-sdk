"""Concrete Perfetto/Chrome-Trace session recorder for the STV client.

This is the framework-agnostic counterpart to pipecat ``OjinVideoService``'s
``OjinSessionTrace``: a recording :class:`~ojin.stv.tracing.Tracer` that
accumulates one Chrome Trace Event Format document per session. Drop the file onto
https://ui.perfetto.dev to see every audio/video event on one timeline — the
received and played streams split by frame type, audio playback, buffer swaps,
barge-ins, per-turn response latency, and the per-tick counters (buffer depth,
pending frames, playback fps, played/bundled-audio RMS, loop lag, tick cost).

The client feeds it through the :class:`~ojin.stv.tracing.Tracer` protocol
(``instant``/``span``/``counter``/``mark``/``now_us``/``record_response_latency``),
so injecting this instead of :class:`~ojin.stv.tracing.NullTracer` turns the trace
on. Every producer (receive loop, playback loop) runs on the one asyncio loop, so
each record is a single ``deque.append`` — no locking, ~free when not dumped.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Callable, Dict, Optional

# Perfetto thread (lane) ids. Stable integers; names attached via ``M`` metadata
# events at build time. Mirrors OjinVideoService so the two traces diff cleanly.
LANES: Dict[str, int] = {
    "lifecycle": 1,
    "tts_input": 2,
    "to_server": 3,
    "recv:speech": 4,
    "recv:new_turn": 5,
    "recv:idle": 6,
    "recv:fade": 7,
    "play:speech": 8,
    "play:new_turn": 9,
    "play:idle": 10,
    "play:fade": 11,
    "play:repeat": 12,
    "play_audio": 13,
    "buffers": 14,
    "interruption": 15,
    "speaking": 16,
    "lipsync": 17,
    "response": 18,
    "latency": 19,
}

_MIN_SPAN_US = 1.0


class OjinSessionTrace:
    """Accumulates Chrome-Trace events for one STV session (Perfetto-loadable)."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        session_id: str = "",
        config_id: str = "",
        pid: int = 1,
        clock: Callable[[], float] = time.perf_counter,
        max_events: int = 500_000,
    ) -> None:
        """Create an empty session trace bound to a monotonic ``clock``."""
        self.session_id = session_id
        self.config_id = config_id
        self._pid = pid
        self._clock = clock
        self._t0 = clock()
        self._events: deque[dict] = deque(maxlen=max_events)
        self._evicted = 0
        self._counts: Dict[str, int] = {}
        self._response_latencies: Dict[str, deque[float]] = {
            "recv": deque(maxlen=10_000),
            "played": deque(maxlen=10_000),
        }
        self._other: Dict[str, object] = {}

    # -- time -----------------------------------------------------------

    def now_us(self) -> float:
        """Microseconds since session start (Perfetto ts domain)."""
        return (self._clock() - self._t0) * 1e6

    def mark(self) -> float:
        """Capture a start timestamp (µs) for a later :meth:`span`."""
        return self.now_us()

    # -- recording (all O(1); single event loop, no locking) ------------

    def _append(self, ev: dict) -> None:
        if len(self._events) == self._events.maxlen:
            self._evicted += 1
        self._events.append(ev)

    def _bump(self, name: str) -> None:
        self._counts[name] = self._counts.get(name, 0) + 1

    def instant(
        self, lane: str, name: str, *, cat: str = "", args: Optional[dict] = None
    ) -> None:
        """Record an instant marker event (``ph='i'``) on ``lane``."""
        self._append(
            {
                "name": name,
                "cat": cat or lane,
                "ph": "i",
                "ts": self.now_us(),
                "pid": self._pid,
                "tid": LANES.get(lane, 0),
                "s": "t",
                "args": args or {},
            }
        )
        self._bump(name)

    def span(
        self,
        lane: str,
        name: str,
        start_us: float,
        *,
        cat: str = "",
        args: Optional[dict] = None,
    ) -> None:
        """Record a completed duration event (``ph='X'``) from ``start_us`` to now."""
        dur = self.now_us() - start_us
        self._append(
            {
                "name": name,
                "cat": cat or lane,
                "ph": "X",
                "ts": start_us,
                "dur": max(dur, _MIN_SPAN_US),
                "pid": self._pid,
                "tid": LANES.get(lane, 0),
                "args": args or {},
            }
        )
        self._bump(name)

    def counter(self, name: str, value: float, *, extra: Optional[dict] = None) -> None:
        """Record a counter (line-plot) event (``ph='C'``) for ``name``."""
        series = {name: value}
        if extra:
            series.update(extra)
        self._append(
            {
                "name": name,
                "ph": "C",
                "ts": self.now_us(),
                "pid": self._pid,
                "args": series,
            }
        )

    def record_response_latency(
        self, kind: str, start_us: float, *, args: Optional[dict] = None
    ) -> float:
        """Record a first-TTS → first-speech-video span; return its ms value.

        ``kind`` selects the endpoint: ``"recv"`` (first speech video frame from
        the server — the inference round-trip) or ``"played"`` (that frame emitted
        downstream). Drawn as a span on the ``response`` lane and aggregated into
        ``otherData.response_latency_ms``.
        """
        latency_ms = (self.now_us() - start_us) / 1000.0
        self.span("response", f"first_tts→first_video_{kind}", start_us, args=args)
        if kind in self._response_latencies:
            self._response_latencies[kind].append(latency_ms)
        return round(latency_ms, 1)

    def set_other_data(self, key: str, value: object) -> None:
        """Attach an extra ``otherData`` entry (e.g. the A/V-sync verdict)."""
        self._other[key] = value

    # -- build + dump ---------------------------------------------------

    @staticmethod
    def _summarise(vals: list) -> Dict[str, float]:
        if not vals:
            return {"count": 0}
        ordered = sorted(vals)
        return {
            "count": len(vals),
            "min_ms": round(ordered[0], 1),
            "max_ms": round(ordered[-1], 1),
            "mean_ms": round(sum(vals) / len(vals), 1),
            "p50_ms": round(ordered[len(ordered) // 2], 1),
            "last_ms": round(vals[-1], 1),
        }

    def build(self) -> dict:
        """Return the Chrome Trace document (events + ``otherData`` summary)."""
        meta: list[dict] = [
            {
                "name": "process_name",
                "ph": "M",
                "pid": self._pid,
                "args": {"name": "ojin_stv"},
            }
        ]
        meta.extend(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": self._pid,
                "tid": tid,
                "args": {"name": lane},
            }
            for lane, tid in LANES.items()
        )
        other = {
            "schema_version": self.SCHEMA_VERSION,
            "producer": "ojin_stv_client",
            "session_id": self.session_id,
            "config_id": self.config_id,
            "duration_s": round(self._clock() - self._t0, 3),
            "event_count": len(self._events),
            "events_evicted_overflow": self._evicted,
            "event_counts": dict(self._counts),
            "response_latency_ms": {
                kind: self._summarise(list(series))
                for kind, series in self._response_latencies.items()
            },
        }
        other.update(self._other)
        return {"traceEvents": meta + list(self._events), "otherData": other}

    def dump(self, path: str) -> str:
        """Write the built trace document to ``path`` (creating parent dirs)."""
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.build(), fh)
        return path
