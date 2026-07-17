"""In-memory fakes for deterministic OjinSTVClient unit tests."""

from __future__ import annotations

import asyncio
import inspect
from typing import Callable, List, Optional

from pydantic import BaseModel

from ojin.ojin_client_messages import (
    IOjinClient,
    OjinSessionReadyMessage,
    OjinWebRTCStatusMessage,
)
from ojin.stv.events import STVEvent
from ojin.stv.frames import STVAudioFrame, STVVideoFrame


class FakeOjinClient(IOjinClient):
    """An IOjinClient backed by in-memory queues (no sockets).

    ``connect`` enqueues a SessionReady so the client's receive loop drives
    startup; tests feed further server messages with :meth:`push`. Sent messages
    are recorded on ``sent`` for assertions.
    """

    def __init__(
        self,
        session_parameters: Optional[dict] = None,
        raise_on_connect: bool = False,
    ) -> None:
        """Create a disconnected fake client (optionally one that fails to connect)."""
        self.sent: List[BaseModel] = []
        self._queue: asyncio.Queue[BaseModel] = asyncio.Queue()
        self._running = False
        self.started_interaction = False
        self.closed = False
        self._session_parameters = session_parameters or {"persona": "test"}
        self._raise_on_connect = raise_on_connect
        self.queue_depths: dict[str, int] = {}
        self.webrtc_status_callback: Optional[
            Callable[[OjinWebRTCStatusMessage], object]
        ] = None

    async def connect(self) -> None:
        """Mark connected and enqueue a SessionReady message (or fail)."""
        if self._raise_on_connect:
            raise ConnectionError("fake connect failure")
        self._running = True
        await self._queue.put(
            OjinSessionReadyMessage(parameters=self._session_parameters)
        )

    async def send_message(self, message: BaseModel) -> None:
        """Record an outgoing message."""
        self.sent.append(message)

    async def start_interaction(self) -> None:
        """Record that an interaction was started."""
        self.started_interaction = True

    async def receive_message(self) -> Optional[BaseModel]:
        """Return the next queued server message (awaits if empty)."""
        return await self._queue.get()

    async def close(self) -> None:
        """Mark closed."""
        self._running = False
        self.closed = True

    async def push(self, message: BaseModel) -> None:
        """Test helper: enqueue a server message for the receive loop."""
        await self._queue.put(message)

    def set_webrtc_status_callback(
        self, callback: Callable[[OjinWebRTCStatusMessage], object]
    ) -> None:
        """Register the webrtcStatus handler, mirroring the real OjinClient."""
        self.webrtc_status_callback = callback

    async def push_webrtc_status(self, payload: dict) -> None:
        """Test helper: parse a status payload and invoke the callback inline."""
        callback = self.webrtc_status_callback
        assert callback is not None, "no webrtcStatus callback registered"
        result = callback(OjinWebRTCStatusMessage(**payload))
        if inspect.isawaitable(result):
            await result

    def debug_queue_depths(self) -> dict[str, int]:
        """Return canned receive-pipeline depths for trace-forwarding assertions."""
        return dict(self.queue_depths)


class RecordingTracer:
    """A Tracer that records calls so tests can assert on instrumentation."""

    def __init__(self) -> None:
        """Create an empty recording tracer."""
        self.session_id = "test-session"
        self.instants: List[tuple] = []
        self.spans: List[tuple] = []
        self.counters: List[tuple] = []
        self.other: dict = {}
        self._t = 0.0

    def instant(self, lane, name, *, cat="", args=None) -> None:
        """Record an instant marker."""
        self.instants.append((lane, name, args or {}))

    def span(self, lane, name, start_us, *, cat="", args=None) -> None:
        """Record a span."""
        self.spans.append((lane, name, start_us, args or {}))

    def counter(self, name, value, *, extra=None) -> None:
        """Record a counter sample."""
        self.counters.append((name, value))

    def mark(self) -> float:
        """Return a monotonically increasing fake timestamp."""
        self._t += 1.0
        return self._t

    def now_us(self) -> float:
        """Return a monotonically increasing fake timestamp."""
        self._t += 1.0
        return self._t

    def record_response_latency(self, kind, start_us, *, args=None) -> float:
        """Record a response-latency span and return a fake ms value."""
        self.spans.append(("response", f"latency_{kind}", start_us, args or {}))
        return 0.0

    def set_other_data(self, key, value) -> None:
        """Record an otherData entry (mirrors OjinSessionTrace)."""
        self.other[key] = value


class ListOutput:
    """An STVOutput that appends frames/events to lists for assertions."""

    def __init__(self) -> None:
        """Create an output sink with empty capture lists."""
        self.audio: List[STVAudioFrame] = []
        self.video: List[STVVideoFrame] = []
        self.events: List[tuple] = []

    async def write_audio(self, frame: STVAudioFrame) -> None:
        """Capture an emitted audio frame."""
        self.audio.append(frame)

    async def write_video(self, frame: STVVideoFrame) -> None:
        """Capture an emitted video frame."""
        self.video.append(frame)

    def on_event(self, event: STVEvent, **kwargs) -> None:
        """Capture a lifecycle event."""
        self.events.append((event, kwargs))
