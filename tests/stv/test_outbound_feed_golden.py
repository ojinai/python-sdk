"""Golden-sequence characterization of OjinSTVClient's outbound server feed.

Pins the exact ordered wire messages (types + payload bytes) a scripted session
produces — seed at ready, initial/min batching, lead-gated release, interrupt +
batcher reset, deferred-turn replay, and close flush — plus the idle-flush tick
path on a fake clock. The outbound-feed extraction must keep every assertion
here green, byte for byte.
"""

import asyncio

from ojin.ojin_client_messages import (
    FrameType,
    OjinAudioInputMessage,
    OjinCancelInteractionMessage,
    OjinInteractionResponseMessage,
)
from ojin.stv.config import STVConfig
from ojin.stv.ojin_stv_client import OjinSTVClient
from ojin.stv.send_batcher import SendBatcher
from ojin.stv.synchronizer import AudioBuffer
from tests.stv.fakes import FakeOjinClient, ListOutput, RecordingTracer

_SEED = b"\x00" * 1280  # one 40 ms silence frame @ 16 kHz int16


def _chunk(index: int) -> bytes:
    """Return a distinct non-silent 40 ms @ 16 kHz mono int16 chunk."""
    return bytes([index % 251, (index * 7) % 251 or 1]) * 640


def _make_client(**config_overrides) -> tuple[OjinSTVClient, FakeOjinClient]:
    """Build a client on in-memory fakes with the golden-run configuration."""
    fake_client = FakeOjinClient()
    client = OjinSTVClient(
        client=fake_client,
        output=ListOutput(),
        tracer=RecordingTracer(),
        config=STVConfig(
            loop_stall_watchdog_ms=0,
            stall_probe_ms=0,
            server_feed_initial_chunk_ms=1000,
            server_feed_min_chunk_ms=400,
            server_feed_flush_idle_ms=10000,
            server_feed_max_lead_ms=1200,
            **config_overrides,
        ),
    )
    return client, fake_client


def _sent_sequence(fake_client: FakeOjinClient) -> list[tuple[str, bytes | None]]:
    """Map recorded outbound messages to comparable (kind, payload) tuples."""
    sequence: list[tuple[str, bytes | None]] = []
    for message in fake_client.sent:
        if isinstance(message, OjinAudioInputMessage):
            sequence.append(("audio", message.audio_int16_bytes))
        elif isinstance(message, OjinCancelInteractionMessage):
            sequence.append(("cancel", None))
        else:
            sequence.append((type(message).__name__, None))
    return sequence


def _interaction_response(frame_type: FrameType) -> OjinInteractionResponseMessage:
    """Build a minimal server interaction-response message of the given type."""
    return OjinInteractionResponseMessage(
        interaction_id="i1",
        video_frame_bytes=b"",
        audio_frame_bytes=b"\x00\x00" * 320,
        is_final_response=False,
        index=0,
        frame_type=frame_type,
    )


def _live_buffer() -> AudioBuffer:
    """Build a fresh, non-interrupted buffer with audio (interruptible)."""
    buffer = AudioBuffer(sample_rate=16000)
    buffer.bytes_.extend(b"\x01\x02" * 100)
    return buffer


async def test_golden_outbound_sequence() -> None:
    """A scripted run produces the exact ordered outbound types and bytes.

    Covers: seed at ready, 32000-byte initial batch, 12800-byte min batches,
    lead-gated release, interrupt (cancel + batcher reset + pending discard),
    deferred-turn replay after the ack frame, and the close-time tail flush.
    """
    chunks = {i: _chunk(i) for i in range(1, 79)}
    client, fake_client = _make_client()

    await client.start()
    await asyncio.sleep(0.05)  # let the receive loop process sessionReady

    # Turn 1: 25 chunks reach the 1000 ms initial threshold -> one 32000 B batch.
    await client.start_turn()
    for i in range(1, 26):
        await client.send_tts_audio(chunks[i], 16000, 1)
    # 10 more reach the 400 ms min threshold; lead 1000 < 1200 -> sent directly.
    for i in range(26, 36):
        await client.send_tts_audio(chunks[i], 16000, 1)
    # 10 more; lead 1400 >= 1200 -> the batch is gated client-side.
    for i in range(36, 46):
        await client.send_tts_audio(chunks[i], 16000, 1)
    assert len(client._feed_pending) == 1

    # Playback advances 600 ms -> the feeder releases the gated batch.
    client._played_real_ms = 600.0
    client._feed_wake.set()
    await asyncio.sleep(0.05)
    assert not client._feed_pending

    # 5 sub-threshold chunks sit in the batcher, then a barge-in discards them.
    for i in range(46, 51):
        await client.send_tts_audio(chunks[i], 16000, 1)
    assert client._batcher.pending_bytes == 6400
    client._synchronizer.current_buffer = _live_buffer()
    await client.interrupt()
    assert client._batcher.pending_bytes == 0
    assert client._server_fed_ms == client._played_real_ms

    # A new turn during the barge-in window is deferred, then replayed whole
    # once the server's idle frame acknowledges the cancel.
    await client.start_turn()
    for i in range(51, 76):
        await client.send_tts_audio(chunks[i], 16000, 1)
    assert len(client._interrupt_deferred) == 26  # turn + 25 audio ops
    await client._handle_message(_interaction_response(FrameType.IDLE))
    assert client._interrupt_deferred == []

    # A sub-threshold tail is flushed by close().
    for i in range(76, 79):
        await client.send_tts_audio(chunks[i], 16000, 1)
    await client.close()

    expected = [
        ("audio", _SEED),
        ("audio", b"".join(chunks[i] for i in range(1, 26))),
        ("audio", b"".join(chunks[i] for i in range(26, 36))),
        ("audio", b"".join(chunks[i] for i in range(36, 46))),
        ("cancel", None),
        ("audio", b"".join(chunks[i] for i in range(51, 76))),
        ("audio", b"".join(chunks[i] for i in range(76, 79))),
    ]
    assert _sent_sequence(fake_client) == expected


async def test_golden_idle_flush_tick_releases_partial_batch() -> None:
    """The idle-flush tick releases a sub-threshold batch on the batcher clock.

    Drives _batch_flush_tick directly with a fake-clock batcher: the first tick
    only restarts the debounce (arrival event set), the second finds the tail
    not yet idle-due, and after the fake clock advances past the idle window
    the third tick flushes the exact buffered bytes.
    """
    client, fake_client = _make_client()
    await client.start()
    await asyncio.sleep(0.05)
    fake_client.sent.clear()

    fake_now = [0.0]
    client._batcher = SendBatcher(
        initial_chunk_bytes=32000,
        min_chunk_bytes=12800,
        flush_idle_s=0.2,
        clock=lambda: fake_now[0],
    )

    await client.start_turn()
    chunks = [_chunk(i) for i in range(1, 6)]
    for chunk in chunks:
        await client.send_tts_audio(chunk, 16000, 1)
    assert _sent_sequence(fake_client) == []  # 200 ms < initial threshold

    await client._batch_flush_tick(0.01)  # arrival event set -> debounce restart
    assert _sent_sequence(fake_client) == []
    await client._batch_flush_tick(0.01)  # timeout, but the tail is not idle yet
    assert _sent_sequence(fake_client) == []

    fake_now[0] += 0.5
    await client._batch_flush_tick(0.01)  # idle window elapsed -> tail flush
    assert _sent_sequence(fake_client) == [("audio", b"".join(chunks))]

    await client.close()
    assert _sent_sequence(fake_client) == [("audio", b"".join(chunks))]
