"""Tracer protocol + no-op default for the STV client.

The client records per-tick audio/video events, swap/interrupt markers, and
per-turn response latencies through an injected tracer. Defining the protocol here
(rather than importing pipecat's ``OjinSessionTrace``) keeps ``ojin.stv``
framework-agnostic: any object with these methods works, and ``OjinSessionTrace``
already satisfies the shape by duck typing. The lane-name strings the client
passes (``"recv:speech"``, ``"play:idle"``, …) match the lanes ``OjinSessionTrace``
knows, so no coupling beyond the method names is needed.
"""

from __future__ import annotations

from typing import Dict, Optional, Protocol, runtime_checkable

# frame_type wire marker (0/1/2/3) → received-stream / played-stream lane name.
_RECV_LANE_FOR_FRAME_TYPE: Dict[int, str] = {
    0: "recv:idle",
    1: "recv:speech",
    2: "recv:fade",
    3: "recv:new_turn",
}
_PLAY_LANE_FOR_FRAME_TYPE: Dict[int, str] = {
    0: "play:idle",
    1: "play:speech",
    2: "play:fade",
    3: "play:new_turn",
}


def recv_lane_for_frame_type(frame_type: int) -> str:
    """Return the received-stream lane name for a wire ``frame_type`` marker."""
    return _RECV_LANE_FOR_FRAME_TYPE.get(frame_type, "recv:speech")


def play_lane_for_frame_type(frame_type: int) -> str:
    """Return the played-stream lane name for a wire ``frame_type`` marker."""
    return _PLAY_LANE_FOR_FRAME_TYPE.get(frame_type, "play:speech")


@runtime_checkable
class Tracer(Protocol):
    """Minimal recording surface the STV client uses (a subset of OjinSessionTrace)."""

    session_id: str

    def instant(
        self, lane: str, name: str, *, cat: str = "", args: Optional[dict] = None
    ) -> None:
        """Record an instant marker event on ``lane``."""
        ...

    def span(
        self,
        lane: str,
        name: str,
        start_us: float,
        *,
        cat: str = "",
        args: Optional[dict] = None,
    ) -> None:
        """Record a completed duration event from ``start_us`` to now."""
        ...

    def counter(self, name: str, value: float, *, extra: Optional[dict] = None) -> None:
        """Record a counter (line-plot) sample for ``name``."""
        ...

    def mark(self) -> float:
        """Capture a start timestamp (µs) for a later :meth:`span`."""
        ...

    def now_us(self) -> float:
        """Microseconds since session start."""
        ...

    def record_response_latency(
        self, kind: str, start_us: float, *, args: Optional[dict] = None
    ) -> float:
        """Record a first-TTS → first-video span and return its ms value."""
        ...


class NullTracer:
    """A tracer that records nothing — the default when no tracer is injected.

    Uses catch-all signatures so it satisfies the :class:`Tracer` surface for any
    call site without redeclaring each parameter; the timing/latency methods return
    0.0 so callers that store the value stay well-defined.
    """

    session_id: str = ""

    def instant(self, *args: object, **kwargs: object) -> None:
        """No-op."""

    def span(self, *args: object, **kwargs: object) -> None:
        """No-op."""

    def counter(self, *args: object, **kwargs: object) -> None:
        """No-op."""

    def mark(self) -> float:
        """Return 0.0 (no clock)."""
        return 0.0

    def now_us(self) -> float:
        """Return 0.0 (no clock)."""
        return 0.0

    def record_response_latency(self, *args: object, **kwargs: object) -> float:  # noqa: ARG002
        """Return 0.0 (nothing recorded)."""
        return 0.0
