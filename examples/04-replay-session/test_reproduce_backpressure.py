"""The barge-in Cancel is delayed by faster-than-realtime audio — and pacing fixes it.

Drives the real ``OjinClient`` send path against a realtime-draining sink (see
``reproduce_backpressure.py``). Confirms P1 is delivery-side: flooding audio strands
the Cancel behind the backlog on the wire; metering audio to ~realtime delivers it
promptly. Sync wrappers (``asyncio.run``) so no pytest-asyncio plugin is needed.
"""

from __future__ import annotations

import asyncio

from reproduce_backpressure import run_experiment

_N_CHUNKS = 16
_CONSUME_S = 0.06


def test_flood_strands_the_cancel_behind_buffered_audio() -> None:
    """Pushing audio faster than realtime delays the Cancel by ~the backlog drain time."""
    r = asyncio.run(
        run_experiment(n_chunks=_N_CHUNKS, consume_s=_CONSUME_S, pace_audio=False)
    )
    assert r["cancel_latency_s"] is not None
    # The sink drains ~1 chunk per _CONSUME_S; a flooded Cancel waits behind most of them.
    assert r["cancel_latency_s"] > 0.4


def test_pacing_audio_to_realtime_delivers_the_cancel_promptly() -> None:
    """Metering audio to ~realtime keeps the wire empty, so the Cancel arrives at once."""
    r = asyncio.run(
        run_experiment(n_chunks=_N_CHUNKS, consume_s=_CONSUME_S, pace_audio=True)
    )
    assert r["cancel_latency_s"] is not None
    assert r["cancel_latency_s"] < 0.25


def test_pacing_is_dramatically_faster_than_flooding() -> None:
    """The contrast localises the latency to delivery, not server processing."""
    flood = asyncio.run(
        run_experiment(n_chunks=_N_CHUNKS, consume_s=_CONSUME_S, pace_audio=False)
    )
    paced = asyncio.run(
        run_experiment(n_chunks=_N_CHUNKS, consume_s=_CONSUME_S, pace_audio=True)
    )
    assert flood["cancel_latency_s"] > 4 * paced["cancel_latency_s"]
