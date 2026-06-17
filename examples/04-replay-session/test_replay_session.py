"""Characterization test: the recorded barge-in session reproduces the cancel lag.

This locks the *observed bug* from session ``8b00aa4e27e2`` so it can't be lost: a
Cancel was issued the moment the user barged in, yet the first server FADE frame
did not reach the client until ~9s later. The assertions read the two bundled
Perfetto fixtures under ``traces/`` (no live server needed).

This is a regression FIXTURE of broken behaviour, not the fix's own test. The fix
lives in the inference server (give Cancel an out-of-band/priority path so it does
not queue behind buffered audio, and hold new speech until fade+idle is emitted),
and is validated by inference-server unit tests plus a fresh trace where this lag
collapses to ~1.4s. When that lands, flip ``BUG_LAG_THRESHOLD_MS`` expectations
here against the new traces.
"""

from __future__ import annotations

import pathlib

from replay_session import BUG_LAG_THRESHOLD_MS, analyze, load_trace

TRACES = pathlib.Path(__file__).parent / "traces"


def _analyze():
    return analyze(
        load_trace(TRACES / "client_session.json"),
        load_trace(TRACES / "server_session.json"),
    )


def test_client_fade_arrives_long_after_cancel() -> None:
    """Client-only signal: the FADE frame lags the cancel by far more than the budget."""
    r = _analyze()
    assert r["client_fade_lag_ms"] is not None
    # observed ~9092ms; the healthy path returns a fade within ~1.4s.
    assert r["client_fade_lag_ms"] > BUG_LAG_THRESHOLD_MS


def test_server_registers_interrupt_long_after_cancel() -> None:
    """Cross-clock signal: the server records interrupt_requested seconds after the cancel."""
    r = _analyze()
    assert r["cancel_to_server_interrupt_ms"] is not None
    # FIFO head-of-line blocking: the cancel sat behind ~8s of buffered audio.
    assert r["cancel_to_server_interrupt_ms"] > BUG_LAG_THRESHOLD_MS


def test_server_fade_itself_is_fast_once_dispatched() -> None:
    """The server fade machinery is healthy: the latency is all *before* interrupt() runs."""
    r = _analyze()
    assert r["server_fade_latency_ms"] is not None
    # requested→fade_emitted was ~346ms — the bug is delivery, not the fade.
    assert r["server_fade_latency_ms"] < 1000.0


def test_bug_signature_reproduced() -> None:
    """The overall verdict flags the reproduction."""
    assert _analyze()["bug_reproduced"] is True
