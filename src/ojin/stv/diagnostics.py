"""Optional event-loop stall diagnostics for the STV playback loop.

Ported from pipecat's ``OjinVideoService``: a watchdog thread dumps every thread's
stack (via ``faulthandler``, which walks frames without the GIL — so it captures the
main thread even while it is blocked in a C extension) when a playback tick fails
to advance within a threshold, and a loop exception handler names the connection
behind otherwise-anonymous asyncio socket errors.

All gated by the threshold knobs and **off by default** (``0`` disables a tier);
enable a tier by passing a positive threshold via :class:`~ojin.stv.config.STVConfig`.
Diagnostics must never break a session, so every operation is defensive.
"""

from __future__ import annotations

import asyncio
import contextlib
import faulthandler
import logging
import sys
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class LoopDiagnostics:
    """Watchdog + loop exception handler for the audio-clocked playback loop."""

    def __init__(
        self,
        *,
        watchdog_ms: float = 0.0,
        tick_warn_ms: float = 80.0,
        stall_probe_ms: float = 0.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Configure the diagnostics; construction starts nothing."""
        super().__init__()
        self._watchdog_ms = watchdog_ms
        self._tick_warn_ms = tick_warn_ms
        self._stall_probe_ms = stall_probe_ms
        self._clock = clock
        self._last_tick_perf: float = 0.0
        self._thread: Optional[threading.Thread] = None
        self._stop: Optional[threading.Event] = None
        self._prev_exc_handler = None
        self._started = False

    @property
    def tick_warn_ms(self) -> float:
        """Threshold (ms) above which the client logs a slow-tick warning."""
        return self._tick_warn_ms

    def note_tick(self) -> None:
        """Record that a playback tick just advanced (feeds the watchdog)."""
        self._last_tick_perf = self._clock()

    def start(self) -> None:
        """Install the loop exception handler and start the watchdog (idempotent)."""
        if self._started:
            return
        self._started = True
        try:
            loop = asyncio.get_running_loop()
            self._prev_exc_handler = loop.get_exception_handler()
            loop.set_exception_handler(self._loop_exception_handler)
        except Exception as exc:  # pragma: no cover - diagnostic only
            logger.warning("could not install loop exception handler: %s", exc)
        if (self._watchdog_ms > 0 or self._stall_probe_ms > 0) and self._thread is None:
            try:
                stop = threading.Event()
                thread = threading.Thread(
                    target=self._watchdog,
                    args=(self._watchdog_ms / 1000.0, stop),
                    name="ojin-stv-loop-watchdog",
                    daemon=True,
                )
                self._stop = stop
                self._thread = thread
                thread.start()
            except Exception as exc:  # pragma: no cover - diagnostic only
                logger.warning("could not start loop stall watchdog: %s", exc)

    def stop(self) -> None:
        """Restore the previous loop exception handler and stop the watchdog."""
        self._started = False
        try:
            loop = asyncio.get_running_loop()
            loop.set_exception_handler(self._prev_exc_handler)
        except Exception:  # pragma: no cover - diagnostic only
            pass
        stop, thread = self._stop, self._thread
        self._stop = self._thread = None
        if stop is not None:
            stop.set()
        if thread is not None:
            thread.join(timeout=1.0)

    def _watchdog(self, threshold_s: float, stop: threading.Event) -> None:
        """Dump all thread stacks when a playback tick stalls (background thread).

        Two latched tiers: a low ``stall_probe_ms`` probe for small stalls and the
        bigger ``watchdog_ms`` for hard freezes. The dump happens while the loop is
        still blocked (faulthandler walks frames without the GIL), so the stack
        shows what is actually running.
        """
        probe_s = self._stall_probe_ms / 1000.0 if self._stall_probe_ms > 0 else 0.0
        if probe_s > 0:
            check_s = max(0.005, min(probe_s / 2.0, 0.02))
        else:
            check_s = max(0.01, min(threshold_s / 2.0, 0.05))
        dumped_full = False
        dumped_probe = False
        while not stop.wait(check_s):
            try:
                last = self._last_tick_perf
                if last <= 0.0:
                    dumped_full = dumped_probe = False
                    continue
                stalled_s = self._clock() - last
                stalled_ms = stalled_s * 1000.0
                if probe_s > 0 and stalled_s >= probe_s and not dumped_probe:
                    dumped_probe = True
                    print(
                        f"\n[ojin-stv-stall-probe] playback loop stalled "
                        f"{stalled_ms:.0f}ms (probe {self._stall_probe_ms:.0f}ms) "
                        f"— dumping all thread stacks:",
                        file=sys.stderr,
                        flush=True,
                    )
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                if threshold_s > 0 and stalled_s >= threshold_s and not dumped_full:
                    dumped_full = True
                    print(
                        f"\n[ojin-stv-loop-watchdog] playback loop stalled "
                        f"{stalled_ms:.0f}ms (threshold {threshold_s * 1000:.0f}ms) "
                        f"— dumping all thread stacks:",
                        file=sys.stderr,
                        flush=True,
                    )
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                if threshold_s <= 0 or stalled_s < threshold_s:
                    dumped_full = False
                if probe_s > 0 and stalled_s < probe_s:
                    dumped_probe = False
            except Exception:  # pragma: no cover - diagnostic only
                pass

    def _loop_exception_handler(
        self, loop: asyncio.AbstractEventLoop, context: dict
    ) -> None:
        """Name the connection behind anonymous asyncio socket errors, then delegate."""
        try:
            transport = context.get("transport")
            info: dict = {}
            get_extra = getattr(transport, "get_extra_info", None)
            if callable(get_extra):
                for key in ("peername", "sockname", "server_hostname"):
                    with contextlib.suppress(Exception):
                        info[key] = get_extra(key)
                with contextlib.suppress(Exception):
                    sock = get_extra("socket")
                    fileno = getattr(sock, "fileno", None)
                    info["fd"] = fileno() if callable(fileno) else None
            exc = context.get("exception")
            logger.warning(
                "[ojin-stv-loop-exc] msg=%r exc=%s:%s transport_info=%s",
                context.get("message"),
                type(exc).__name__ if exc else None,
                exc,
                info,
            )
        except Exception:  # pragma: no cover - diagnostic only
            pass
        finally:
            try:
                if self._prev_exc_handler is not None:
                    self._prev_exc_handler(loop, context)
                else:
                    loop.default_exception_handler(context)
            except Exception:  # pragma: no cover - diagnostic only
                pass
