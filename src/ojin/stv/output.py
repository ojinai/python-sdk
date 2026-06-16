"""Output sink protocol + a queue-backed async-iterator adapter.

The playback loop emits one audio frame every tick (real or silence) plus a video
frame whenever one is available, by calling an :class:`STVOutput`. Consumers either
implement the protocol directly (e.g. the pipecat adapter pushing frames
downstream) or use :class:`QueueOutput` and drain it with ``async for``.

Backpressure: audio is the clock and is never dropped; video is bounded
(``max_video``) and the oldest video frame is dropped on overflow, logged so the
truncation is never silent.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import AsyncIterator, Protocol, Tuple, Union, runtime_checkable

from ojin.stv.events import STVEvent
from ojin.stv.frames import STVAudioFrame, STVVideoFrame

logger = logging.getLogger(__name__)

STVFrame = Union[STVAudioFrame, STVVideoFrame]


@runtime_checkable
class STVOutput(Protocol):
    """Where the client emits synced audio/video each tick."""

    async def write_audio(self, frame: STVAudioFrame) -> None:
        """Emit one tick of played audio (never dropped)."""
        ...

    async def write_video(self, frame: STVVideoFrame) -> None:
        """Emit one avatar video frame."""
        ...

    def on_event(self, event: STVEvent, **kwargs) -> None:
        """Receive a lifecycle event (optional; default no-op)."""
        ...


class QueueOutput:
    """Buffers emitted frames for consumption via :meth:`stream` (``async for``).

    A single ordered buffer holds both audio and video. Audio is retained; video is
    capped at ``max_video`` with drop-oldest on overflow (logged). Producer and
    consumer run on the same event loop; an ``asyncio.Condition`` coordinates them.
    """

    def __init__(self, max_video: int = 60) -> None:
        """Create an empty output buffer bounding queued video to ``max_video``."""
        self._max_video = max_video
        self._items: deque[Tuple[str, STVFrame]] = deque()
        self._video_count = 0
        self._cond = asyncio.Condition()
        self._closed = False

    async def write_audio(self, frame: STVAudioFrame) -> None:
        """Enqueue an audio frame (never dropped)."""
        async with self._cond:
            self._items.append(("a", frame))
            self._cond.notify()

    async def write_video(self, frame: STVVideoFrame) -> None:
        """Enqueue a video frame, dropping the oldest video on overflow."""
        async with self._cond:
            self._items.append(("v", frame))
            self._video_count += 1
            if self._video_count > self._max_video:
                self._drop_oldest_video()
            self._cond.notify()

    def on_event(self, event: STVEvent, **kwargs) -> None:
        """Ignore lifecycle events (consumers can subclass to observe them)."""

    def _drop_oldest_video(self) -> None:
        for i, (kind, _) in enumerate(self._items):
            if kind == "v":
                del self._items[i]
                self._video_count -= 1
                logger.debug("QueueOutput: dropped oldest video frame (consumer slow)")
                return

    async def stream(self) -> AsyncIterator[STVFrame]:
        """Yield buffered frames in order, waiting when empty until closed."""
        while True:
            async with self._cond:
                while not self._items and not self._closed:
                    await self._cond.wait()
                if not self._items and self._closed:
                    return
                kind, frame = self._items.popleft()
                if kind == "v":
                    self._video_count -= 1
            yield frame

    async def aclose(self) -> None:
        """Signal end-of-stream so a waiting :meth:`stream` returns."""
        async with self._cond:
            self._closed = True
            self._cond.notify_all()
