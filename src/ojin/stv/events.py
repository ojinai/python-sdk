"""Event types + a small async-aware emitter for the STV client (anam-style).

Consumers register handlers with :meth:`EventEmitter.on` (decorator) or
:meth:`add_listener`. Handlers may be sync or async; :meth:`emit` awaits coroutine
results and isolates handler exceptions so one failing handler never breaks the
playback/receive loop that fired the event.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from enum import Enum
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)

EventCallback = Callable[..., object]


class STVEvent(Enum):
    """Lifecycle/state events emitted by :class:`OjinSTVClient`."""

    SESSION_READY = "session_ready"  # args: session_data
    BOT_STARTED_SPEAKING = "bot_started_speaking"
    BOT_STOPPED_SPEAKING = "bot_stopped_speaking"
    INTERRUPTED = "interrupted"
    ERROR = "error"  # args: message, fatal
    CLOSED = "closed"


class EventEmitter:
    """Registers and dispatches :class:`STVEvent` handlers (sync or async)."""

    def __init__(self) -> None:
        """Create an emitter with no listeners."""
        super().__init__()
        self._listeners: Dict[STVEvent, List[EventCallback]] = {}

    def on(self, event: STVEvent) -> Callable[[EventCallback], EventCallback]:
        """Register a handler via decorator (returns the handler unchanged)."""

        def _register(cb: EventCallback) -> EventCallback:
            self.add_listener(event, cb)
            return cb

        return _register

    def add_listener(self, event: STVEvent, cb: EventCallback) -> None:
        """Register ``cb`` to be called when ``event`` is emitted."""
        self._listeners.setdefault(event, []).append(cb)

    def remove_listener(self, event: STVEvent, cb: EventCallback) -> None:
        """Remove a previously registered handler (no-op if absent)."""
        handlers = self._listeners.get(event)
        if handlers and cb in handlers:
            handlers.remove(cb)

    async def emit(self, event: STVEvent, **kwargs: object) -> None:
        """Call every handler for ``event``; await coroutines; isolate failures."""
        for cb in list(self._listeners.get(event, [])):
            await self._dispatch(event, cb, kwargs)

    @staticmethod
    async def _dispatch(event: STVEvent, cb: EventCallback, kwargs: dict) -> None:
        """Invoke one handler, awaiting coroutines and isolating its exceptions."""
        try:
            result = cb(**kwargs)
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("STV event handler for %s failed", event)
