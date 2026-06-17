"""Reproduce the barge-in cancel latency: a Cancel stuck behind faster-than-realtime audio.

This confirms P1 (the ~8.4s cancel latency from session ``8b00aa4e27e2``) is a
**delivery-path** problem — the Cancel sits behind buffered audio on the wire /
client write buffer — and is **not** the inference server's message queue (which
is unbounded and non-blocking).

It drives the REAL ``ojin.ojin_client.OjinClient`` send path:

* audio goes through ``send_message`` → ``_pending_client_messages_queue`` →
  the ``_process_client_messages`` worker → ``ws.send`` (ojin_client.py:374-432),
* the Cancel goes through ``send_message`` → a **direct** ``ws.send`` that bypasses
  the app-level queue (ojin_client.py:362-372),

against a local websocket server that consumes audio at ~realtime with a small
receive buffer — the same realtime-bound condition production imposes (server
trackers showed audio_feeder rtf≈1.0, frame_pacer rtf≈0.99).

The contrast is the proof:

* **flood** (audio pushed as fast as TTS delivers it, ~5× realtime): the audio
  banks in the client's write buffer, and the Cancel — though it skips the app
  queue — lands behind that buffered audio on the one shared websocket, so the
  server does not *receive* it until the backlog drains.
* **paced** (audio metered to ~realtime): no backlog, so the Cancel arrives within
  ~one chunk. This is the shape of the client-side fix.

The wall-clock per chunk is scaled down from 200ms purely to keep the test fast;
the backpressure mechanism is time-scale invariant.

Run::

    python reproduce_backpressure.py
    pytest test_reproduce_backpressure.py
"""

from __future__ import annotations

import asyncio
import time

from websockets.asyncio.server import serve

from ojin.ojin_client import OjinClient
from ojin.ojin_client_messages import (
    OjinAudioInputMessage,
    OjinCancelInteractionMessage,
)

# 200ms of 16kHz mono int16 audio — the chunk size the client actually sends.
_CHUNK_BYTES = 6400
# Non-silent payload (the server drops all-zero audio; this must look like speech).
_AUDIO_CHUNK = bytes([1, 2]) * (_CHUNK_BYTES // 2)


async def _realtime_sink(ws, *, consume_s: float, recv: dict) -> None:
    """Read messages, consuming each audio chunk at ~realtime; timestamp the Cancel.

    A small receive buffer plus a per-chunk sleep makes the sink drain slower than a
    flooding sender, so backpressure propagates to the sender exactly as the
    realtime-bound inference pipeline does.
    """
    async for message in ws:
        now = time.perf_counter()
        if isinstance(message, str):  # Cancel is JSON text; audio is binary
            recv["cancel_ts"] = now
            return
        recv["audio_ts"].append(now)
        await asyncio.sleep(consume_s)  # consume this audio chunk at ~realtime


async def _wait_until_forwarded(client: OjinClient, timeout: float = 2.0) -> None:
    """Wait until the worker has forwarded every app-queued message onto the wire.

    Models the prod condition: TTS streams over the whole turn, so by barge-in time
    ``_process_client_messages`` has already moved the audio out of the app queue and
    onto the wire (into the OS/proxy/server buffers). Only *then* is a Cancel able to
    land behind buffered audio.
    """
    deadline = time.perf_counter() + timeout
    queue = client._pending_client_messages_queue
    while queue.qsize() > 0 and time.perf_counter() < deadline:
        await asyncio.sleep(0.005)
    await asyncio.sleep(0.02)  # let the final ws.send flush to the socket


async def run_experiment(*, n_chunks: int, consume_s: float, pace_audio: bool) -> dict:
    """Stream ``n_chunks`` then a Cancel through a real OjinClient; measure cancel latency.

    ``pace_audio=False`` forwards the whole turn to the wire ahead of the Cancel
    (reproduces the bug); ``pace_audio=True`` meters audio to ``consume_s`` so the wire
    never backs up (the fix shape). Returns the cancel-issue→cancel-received latency.

    The sink's receive buffer is sized to hold the turn — it stands in for the
    cumulative client-OS + proxy + server buffering along the real path, which is
    where the audio banks ahead of the Cancel.
    """
    recv: dict = {"audio_ts": [], "cancel_ts": None}
    server = await serve(
        lambda ws: _realtime_sink(ws, consume_s=consume_s, recv=recv),
        "127.0.0.1",
        0,
        max_queue=n_chunks + 16,
    )
    port = server.sockets[0].getsockname()[1]
    uri = f"ws://127.0.0.1:{port}"

    client = OjinClient(ws_url=uri, api_key="k", config_id="c")
    import websockets

    client._ws = await websockets.connect(uri, max_queue=n_chunks + 16)
    client._running = True
    client._inference_server_ready = True
    worker = asyncio.create_task(client._process_client_messages())

    try:
        for _ in range(n_chunks):
            await client.send_message(OjinAudioInputMessage(audio_int16_bytes=_AUDIO_CHUNK))
            if pace_audio:
                await asyncio.sleep(consume_s)

        if not pace_audio:
            # The worker forwards the whole turn onto the wire before the barge-in,
            # exactly as it does in prod once TTS has streamed the turn.
            await _wait_until_forwarded(client)

        issue_ts = time.perf_counter()
        await client.send_message(OjinCancelInteractionMessage())

        deadline = time.perf_counter() + n_chunks * consume_s + 5.0
        while recv["cancel_ts"] is None and time.perf_counter() < deadline:
            await asyncio.sleep(0.01)
    finally:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        await client._ws.close()
        server.close()
        await server.wait_closed()

    cancel_latency_s = (
        None if recv["cancel_ts"] is None else recv["cancel_ts"] - issue_ts
    )
    return {
        "pace_audio": pace_audio,
        "n_chunks": n_chunks,
        "audio_received_before_cancel": len(recv["audio_ts"]),
        "cancel_latency_s": cancel_latency_s,
    }


async def main() -> int:
    """Run flood vs paced and print the contrast that localises the latency."""
    n_chunks, consume_s = 20, 0.08
    flood = await run_experiment(n_chunks=n_chunks, consume_s=consume_s, pace_audio=False)
    paced = await run_experiment(n_chunks=n_chunks, consume_s=consume_s, pace_audio=True)

    def fmt(r: dict) -> str:
        lat = r["cancel_latency_s"]
        return (
            f"  cancel latency = {lat * 1000:7.0f}ms"
            f"   (audio chunks the sink drained before seeing the cancel:"
            f" {r['audio_received_before_cancel']}/{r['n_chunks']})"
        )

    print(f"Backpressure repro — {n_chunks} chunks, sink drains 1 chunk / {consume_s*1000:.0f}ms")
    print("─" * 70)
    print(f"FLOOD (audio pushed ~as fast as TTS, like prod):\n{fmt(flood)}")
    print(f"PACED (audio metered to ~realtime, the fix shape):\n{fmt(paced)}")
    print("─" * 70)
    fl, pl = flood["cancel_latency_s"], paced["cancel_latency_s"]
    reproduced = fl is not None and pl is not None and fl > 0.4 and fl > 4 * pl
    print(
        f"DELIVERY-SIDE LATENCY REPRODUCED: {reproduced}  "
        f"(flood {fl*1000:.0f}ms vs paced {pl*1000:.0f}ms — pacing collapses it)"
    )
    return 0 if reproduced else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
