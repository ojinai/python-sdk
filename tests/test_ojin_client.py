"""Unit tests for OjinClient server-message handling robustness."""

import asyncio
import contextlib
import json

from ojin.entities.interaction_messages import ErrorResponseMessage
from ojin.ojin_client import OjinClient
from ojin.ojin_client_messages import (
    OjinAudioInputMessage,
    OjinCancelInteractionMessage,
    OjinSessionReadyMessage,
)


def _client() -> OjinClient:
    """Return an OjinClient that is constructed but never connected (queue only)."""
    return OjinClient(ws_url="ws://test/realtime", api_key="k", config_id="c")


class _FakeWS:
    """Records everything sent over the socket; no real I/O."""

    def __init__(self) -> None:
        self.sent: list = []

    async def send(self, data) -> None:
        self.sent.append(data)


async def _drain_send_loop(client, expected_sends, real_sleep, max_iters=1000):
    """Run _process_client_messages until it has sent ``expected_sends`` frames.

    ``real_sleep`` is the un-patched ``asyncio.sleep`` (so the driver can yield even
    when the module's sleep is monkeypatched to a no-op recorder).
    """
    task = asyncio.create_task(client._process_client_messages())
    for _ in range(max_iters):
        if (
            len(client._ws.sent) >= expected_sends
            and client._pending_client_messages_queue.empty()
        ):
            break
        await real_sleep(0)
    client._running = False
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def _audio_msg(pcm: bytes) -> OjinAudioInputMessage:
    return OjinAudioInputMessage(audio_int16_bytes=pcm)


async def test_cancel_drops_pending_client_messages() -> None:
    """A cancel clears audio still queued to send, so it can't outlive the barge-in."""
    client = _client()
    client._running = True
    client._inference_server_ready = True
    client._ws = _FakeWS()  # type: ignore[assignment]

    await client.send_message(_audio_msg(b"\x01\x02" * 100))
    await client.send_message(_audio_msg(b"\x03\x04" * 100))
    assert client._pending_client_messages_queue.qsize() == 2

    await client.send_message(OjinCancelInteractionMessage())

    assert client._pending_client_messages_queue.qsize() == 0  # drained on cancel
    assert len(client._ws.sent) == 1  # only the cancel frame reached the wire


async def test_large_audio_split_into_max_chunks_and_paced(monkeypatch) -> None:
    """A payload larger than the cap is split, and split chunks are gapped."""
    real_sleep = asyncio.sleep
    sleeps: list = []

    async def fake_sleep(delay):
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    client = OjinClient(
        ws_url="ws://t",
        api_key="k",
        config_id="c",
        max_input_chunk_bytes=4,
        send_chunk_gap_s=0.2,
    )
    client._running = True
    client._ws = _FakeWS()  # type: ignore[assignment]
    await client._pending_client_messages_queue.put(_audio_msg(b"\x01\x02\x03\x04" * 3))

    await _drain_send_loop(client, expected_sends=3, real_sleep=real_sleep)

    assert len(client._ws.sent) == 3  # 12 bytes / 4-byte cap = 3 chunks
    # Time-based pacing: no wait before the first send, one ~gap-length sleep
    # before each later chunk (wall clock barely advances between them in-test).
    assert len(sleeps) == 2
    assert all(0.15 <= s <= 0.2 for s in sleeps)


async def test_single_message_sends_without_gap(monkeypatch) -> None:
    """One message with no backlog is sent at once — pacing never delays realtime."""
    real_sleep = asyncio.sleep
    sleeps: list = []

    async def fake_sleep(delay):
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    client = OjinClient(
        ws_url="ws://t", api_key="k", config_id="c", send_chunk_gap_s=0.2
    )
    client._running = True
    client._ws = _FakeWS()  # type: ignore[assignment]
    await client._pending_client_messages_queue.put(_audio_msg(b"\x01\x02" * 10))

    await _drain_send_loop(client, expected_sends=1, real_sleep=real_sleep)

    assert len(client._ws.sent) == 1
    assert sleeps == []  # queue drained empty → no gap inserted


async def test_backlog_of_messages_is_paced(monkeypatch) -> None:
    """A queued burst of messages is spaced by the gap, one per message boundary."""
    real_sleep = asyncio.sleep
    sleeps: list = []

    async def fake_sleep(delay):
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    client = OjinClient(
        ws_url="ws://t", api_key="k", config_id="c", send_chunk_gap_s=0.2
    )
    client._running = True
    client._ws = _FakeWS()  # type: ignore[assignment]
    for _ in range(3):
        await client._pending_client_messages_queue.put(_audio_msg(b"\x01\x02" * 10))

    await _drain_send_loop(client, expected_sends=3, real_sleep=real_sleep)

    assert len(client._ws.sent) == 3
    # First message goes straight out; each later one waits out the remainder of
    # the gap since the previous send.
    assert len(sleeps) == 2
    assert all(0.15 <= s <= 0.2 for s in sleeps)


async def test_interleaved_producer_is_still_paced(monkeypatch) -> None:
    """A producer that enqueues one message per await is still gap-paced.

    Regression guard for the session-7 burst (2026-07-12): the deferred-TTS
    replay enqueues messages one at a time, interleaving with this consumer so
    the queue never holds more than one item. Queue-depth-based pacing saw "no
    backlog" and sent everything back-to-back; time-based pacing spaces the
    sends regardless of how the queue interleaves.
    """
    real_sleep = asyncio.sleep
    sleeps: list = []

    async def fake_sleep(delay):
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    client = OjinClient(
        ws_url="ws://t", api_key="k", config_id="c", send_chunk_gap_s=0.2
    )
    client._running = True
    client._ws = _FakeWS()  # type: ignore[assignment]

    task = asyncio.create_task(client._process_client_messages())
    try:
        for i in range(3):
            await client._pending_client_messages_queue.put(
                _audio_msg(b"\x01\x02" * 10)
            )
            # Wait for THIS message to hit the wire before enqueueing the next —
            # the queue is empty at every gap decision, mimicking the replay race.
            for _ in range(200):
                if len(client._ws.sent) > i:
                    break
                await real_sleep(0)
    finally:
        client._running = False
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert len(client._ws.sent) == 3
    assert len(sleeps) == 2  # 2nd and 3rd sends each waited out the gap
    assert all(0.15 <= s <= 0.2 for s in sleeps)


async def test_non_json_text_surfaces_clean_error() -> None:
    """A plain-text server error (e.g. 'Model not found') becomes an error message.

    Previously json.loads() raised on the text and killed the receive loop with no
    error surfaced to consumers; now it is queued as an ErrorResponseMessage.
    """
    client = _client()
    await client._handle_server_message("Model not found")  # must not raise
    msg = client._available_response_messages_queue.get_nowait()
    assert isinstance(msg, ErrorResponseMessage)
    assert msg.payload.message == "Model not found"


async def test_empty_text_surfaces_clean_error() -> None:
    """An empty text frame is surfaced as an error rather than crashing."""
    client = _client()
    await client._handle_server_message("")  # must not raise
    msg = client._available_response_messages_queue.get_nowait()
    assert isinstance(msg, ErrorResponseMessage)


async def test_session_ready_json_still_parsed() -> None:
    """Valid sessionReady JSON is still parsed into OjinSessionReadyMessage."""
    client = _client()
    payload = {"type": "sessionReady", "payload": {"parameters": {"persona": "x"}}}
    await client._handle_server_message(json.dumps(payload))
    msg = client._available_response_messages_queue.get_nowait()
    assert isinstance(msg, OjinSessionReadyMessage)
    assert msg.parameters == {"persona": "x"}


def test_debug_queue_depths_disconnected_reports_only_parsed_queue() -> None:
    """With no websocket, only the always-readable parsed-queue depth is reported."""
    client = _client()
    assert client.debug_queue_depths() == {"server_msgs": 0}
    client._available_response_messages_queue.put_nowait(
        OjinSessionReadyMessage(parameters={})
    )
    assert client.debug_queue_depths()["server_msgs"] == 1


def test_debug_queue_depths_reads_len_based_frame_queue() -> None:
    """The websockets asyncio queue exposes len() (not qsize) — read it via len()."""

    class _LenFrames:
        """Mimics websockets' custom SimpleQueue: supports len(), not qsize()."""

        def __len__(self) -> int:
            return 4

    class _Assembler:
        frames = _LenFrames()
        paused = True

    class _WS:
        recv_messages = _Assembler()
        transport = None  # skip the Unix-only FIONREAD probe in this test

    client = _client()
    client._ws = _WS()  # type: ignore[assignment]
    depths = client.debug_queue_depths()
    assert depths["ws_frames"] == 4
    assert depths["ws_paused"] == 1
    assert depths["server_msgs"] == 0


def test_debug_queue_depths_reads_qsize_based_frame_queue() -> None:
    """A stdlib queue.SimpleQueue (qsize, no len) is read via the qsize fallback."""

    class _QSizeFrames:
        def qsize(self) -> int:
            return 7

    class _Assembler:
        frames = _QSizeFrames()
        paused = False

    class _WS:
        recv_messages = _Assembler()
        transport = None

    client = _client()
    client._ws = _WS()  # type: ignore[assignment]
    depths = client.debug_queue_depths()
    assert depths["ws_frames"] == 7
    assert depths["ws_paused"] == 0


def test_debug_queue_depths_never_raises_on_broken_internals() -> None:
    """A websockets-internals shape change degrades to a missing key, not a crash."""

    class _BadWS:
        """A websocket whose internals don't match what the probe expects."""

    client = _client()
    client._ws = _BadWS()  # type: ignore[assignment]
    # Must not raise; ws gauges are simply absent, server_msgs still present.
    assert client.debug_queue_depths() == {"server_msgs": 0}
