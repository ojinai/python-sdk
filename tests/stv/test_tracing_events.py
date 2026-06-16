"""Unit tests for ojin.stv.tracing and ojin.stv.events."""

import asyncio

from ojin.stv.events import EventEmitter, STVEvent
from ojin.stv.tracing import (
    NullTracer,
    play_lane_for_frame_type,
    recv_lane_for_frame_type,
)


def test_null_tracer_is_a_safe_noop() -> None:
    """NullTracer satisfies the Tracer surface with harmless no-ops."""
    t = NullTracer()
    t.instant("x", "y")
    t.span("x", "y", 0.0)
    t.counter("c", 1)
    assert isinstance(t.mark(), float)
    assert isinstance(t.now_us(), float)
    assert t.record_response_latency("recv", 0.0) == 0.0
    assert isinstance(t.session_id, str)


def test_lane_helpers_map_frame_types() -> None:
    """Lane helpers map wire frame_type markers to recv/play lane names."""
    assert recv_lane_for_frame_type(3) == "recv:new_turn"
    assert recv_lane_for_frame_type(0) == "recv:idle"
    assert play_lane_for_frame_type(1) == "play:speech"
    assert play_lane_for_frame_type(2) == "play:fade"
    assert play_lane_for_frame_type(99) == "play:speech"  # unknown → speech


def test_emitter_calls_sync_and_async_handlers() -> None:
    """Both plain and coroutine handlers receive emitted events."""
    em = EventEmitter()
    seen = []
    em.add_listener(STVEvent.ERROR, lambda **k: seen.append(("sync", k)))

    async def ah(**k):
        seen.append(("async", k))

    em.add_listener(STVEvent.ERROR, ah)
    asyncio.run(em.emit(STVEvent.ERROR, message="boom", fatal=True))
    assert ("sync", {"message": "boom", "fatal": True}) in seen
    assert ("async", {"message": "boom", "fatal": True}) in seen


def test_emitter_decorator_registers() -> None:
    """The on() decorator registers a handler for the event."""
    em = EventEmitter()
    hits = []

    @em.on(STVEvent.SESSION_READY)
    def _ready(**k):
        hits.append(k)

    asyncio.run(em.emit(STVEvent.SESSION_READY, session_data={"a": 1}))
    assert hits == [{"session_data": {"a": 1}}]


def test_emitter_isolates_handler_errors() -> None:
    """A throwing handler does not stop the others."""
    em = EventEmitter()
    ok = []

    def bad(**kwargs):
        raise RuntimeError("x")

    def good(**kwargs):
        ok.append(1)

    em.add_listener(STVEvent.CLOSED, bad)
    em.add_listener(STVEvent.CLOSED, good)
    asyncio.run(em.emit(STVEvent.CLOSED))
    assert ok == [1]


def test_emitter_remove_listener() -> None:
    """A removed listener stops receiving events."""
    em = EventEmitter()
    hits = []

    def cb(**kwargs):
        hits.append(1)

    em.add_listener(STVEvent.CLOSED, cb)
    em.remove_listener(STVEvent.CLOSED, cb)
    asyncio.run(em.emit(STVEvent.CLOSED))
    assert hits == []
