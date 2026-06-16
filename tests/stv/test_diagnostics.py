"""Unit tests for ojin.stv.diagnostics (optional loop-stall watchdog)."""

import asyncio

from ojin.stv.diagnostics import LoopDiagnostics


def test_diagnostics_disabled_is_idempotent_noop() -> None:
    """When disabled, start/stop/note_tick never raise and start no thread."""

    async def run() -> None:
        d = LoopDiagnostics(watchdog_ms=0, tick_warn_ms=0, stall_probe_ms=0)
        d.start()
        d.start()  # idempotent
        d.note_tick()
        d.stop()
        d.stop()  # idempotent
        assert d._thread is None

    asyncio.run(run())


def test_diagnostics_enabled_starts_and_stops_thread() -> None:
    """With the watchdog enabled, start spins a daemon thread that stop joins."""

    async def run() -> None:
        d = LoopDiagnostics(watchdog_ms=5000, tick_warn_ms=0, stall_probe_ms=0)
        d.start()
        d.note_tick()
        assert d._thread is not None and d._thread.is_alive()
        d.stop()
        assert d._thread is None

    asyncio.run(run())
