"""Fixed audio packets preserve PCM across batching, pacing, and cancellation."""

import asyncio
from types import SimpleNamespace

import pytest

import ojin.ojin_client as transport_module
from ojin.entities.interaction_messages import InteractionInputMessage
from ojin.ojin_client import OjinClient
from ojin.ojin_client_messages import (
    FrameType,
    OjinAudioInputMessage,
    OjinSessionReadyMessage,
)
from ojin.stv import OjinSTVClient, STVConfig, WebRTCSettings
from ojin.stv.synchronizer import AudioBuffer
from tests.stv.fakes import FakeOjinClient, RecordingTracer
from tests.stv.test_webrtc_client import _frame
from tests.stv.test_webrtc_negotiation import READY_CONNECTED_PARAMETERS


def _config() -> STVConfig:
    return STVConfig(
        server_feed_initial_chunk_ms=500,
        server_feed_min_chunk_ms=200,
        server_feed_fixed_chunk_size=True,
        server_feed_send_gap_ms=100,
        server_feed_flush_idle_ms=10000,
        server_feed_max_lead_ms=0,
    )


def _client(transport, tracer, direct: bool, rate: int):
    return OjinSTVClient(
        client=transport,
        tracer=tracer,
        config=_config(),
        webrtc=(
            WebRTCSettings(
                room_url="https://ojin.daily.co/fixed-feed",
                token="test-token",
                audio_sample_rate=rate,
            )
            if direct
            else None
        ),
    )


def _audio(transport: FakeOjinClient) -> list[bytes]:
    return [
        message.audio_int16_bytes
        for message in transport.sent
        if isinstance(message, OjinAudioInputMessage) and any(message.audio_int16_bytes)
    ]


async def _wait_for_writes(sent, count: int) -> None:
    for _ in range(100):
        if len(sent) == count:
            return
        await asyncio.sleep(0)
    assert len(sent) == count


@pytest.mark.parametrize("direct,rate", [(False, 16000), (True, 24000), (True, 44100)])
async def test_oversized_input_preserves_exact_packets_tail_and_cancel_rearm(
    direct: bool, rate: int
) -> None:
    """TTS packet boundaries do not affect output bytes or the next turn's lead."""
    transport = FakeOjinClient()
    client = _client(transport, RecordingTracer(), direct, rate)
    engine = client._webrtc or client
    ready = OjinSessionReadyMessage(parameters=READY_CONNECTED_PARAMETERS)
    pcm = b"\x01\x02" * (rate * 23 // 20)
    initial_bytes, steady_bytes = rate, rate * 2 // 5
    try:
        await engine._handle_message(ready)
        await client.start_turn()
        await client.send_tts_audio(pcm[: rate // 5], rate, 1)
        assert not _audio(transport)
        await client.send_tts_audio(pcm[rate // 5 :], rate, 1)
        packets = _audio(transport)
        assert [len(packet) for packet in packets] == [
            initial_bytes,
            steady_bytes,
            steady_bytes,
            steady_bytes,
        ]
        assert b"".join(packets) == pcm[: rate * 22 // 10]

        await client.start_turn()
        assert b"".join(_audio(transport)) == pcm
        assert len(_audio(transport)[-1]) == rate // 10
        await client.send_tts_audio(b"\x03\x04" * (rate * 2 // 5), rate, 1)
        assert len(_audio(transport)) == 5
        if not direct:
            engine._synchronizer.current_buffer = AudioBuffer(sample_rate=rate)
            engine._synchronizer.current_buffer.bytes_.extend(pcm)
        assert await client.interrupt()
        await client.start_turn()
        fresh = b"\x05\x06" * (rate // 2)
        await client.send_tts_audio(fresh, rate, 1)
        assert len(_audio(transport)) == 5
        await engine._handle_message(_frame(FrameType.IDLE))
        assert _audio(transport)[-1] == fresh
        assert len(_audio(transport)) == 6
    finally:
        await client.close()


async def test_exact_packets_are_paced_on_wire_and_traced_after_write(monkeypatch):
    """A 950 ms burst becomes 500/200/200 ms writes and a real 50 ms idle tail."""
    rate = 24000
    now = [100.0]
    tracer = RecordingTracer()
    transport = OjinClient(
        ws_url="ws://test", api_key="test", config_id="test", send_chunk_gap_s=0.1
    )
    transport._running = True
    transport._inference_server_ready = True
    sent = []

    class RecordingSocket:
        async def send(self, data):
            sent.append((now[0], data))
            now[0] += 0.002

        async def close(self):
            pass

    transport._ws = RecordingSocket()
    client = _client(transport, tracer, True, rate)
    engine = client._webrtc
    await engine._handle_message(
        OjinSessionReadyMessage(parameters=READY_CONNECTED_PARAMETERS)
    )
    real_sleep = asyncio.sleep

    async def paced_sleep(delay):
        now[0] += delay
        await real_sleep(0)

    monkeypatch.setattr(
        transport_module,
        "asyncio",
        SimpleNamespace(sleep=paced_sleep, CancelledError=asyncio.CancelledError),
    )
    monkeypatch.setattr(
        transport_module, "time", SimpleNamespace(monotonic=lambda: now[0])
    )
    pcm = b"\x01\x02" * (rate * 19 // 20)
    try:
        await client.start_turn()
        await client.send_tts_audio(pcm, rate, 1)
        assert sent == []
        queued = [args for _, name, args in tracer.instants if name == "audio_queued"]
        assert [args["duration_ms"] for args in queued] == [500, 200, 200]
        assert not any(name == "audio_sent" for _, name, _ in tracer.instants)
        transport._process_messages_task = asyncio.create_task(
            transport._process_client_messages()
        )
        await _wait_for_writes(sent, 3)
        assert [timestamp for timestamp, _ in sent] == pytest.approx(
            [100.0, 100.102, 100.204]
        )
        assert [
            len(InteractionInputMessage.from_bytes(data).payload.payload)
            for _, data in sent
        ] == [24000, 9600, 9600]
        wire = [args for _, name, args in tracer.instants if name == "audio_sent"]
        assert [args["duration_ms"] for args in wire] == [500, 200, 200]
        assert [args["queue_wait_ms"] for args in wire] == [0, 102, 204]
        assert [args["send_ms"] for args in wire] == [2, 2, 2]

        engine._batch_added.clear()
        engine._batcher._clock = lambda: engine._batcher._last_add_ts + 11
        now[0] += 1
        await engine._batch_flush_tick(0.0001)
        await _wait_for_writes(sent, 4)
        audio = [
            InteractionInputMessage.from_bytes(data).payload.payload for _, data in sent
        ]
        assert len(audio) == 4
        assert len(audio[-1]) == 2400
        assert b"".join(audio) == pcm
        await engine._batch_flush_tick(0.0001)
        await real_sleep(0)
        assert len(sent) == 4
    finally:
        await client.close()


async def test_short_utterance_idle_flushes_without_padding() -> None:
    """An utterance shorter than the startup lead still reaches the server."""
    transport = FakeOjinClient()
    client = _client(transport, RecordingTracer(), True, 24000)
    engine = client._webrtc
    pcm = b"\x01\x02" * 3600
    try:
        await engine._handle_message(
            OjinSessionReadyMessage(parameters=READY_CONNECTED_PARAMETERS)
        )
        await client.start_turn()
        await client.send_tts_audio(pcm, 24000, 1)
        assert not _audio(transport)
        engine._batch_added.clear()
        engine._batcher._clock = lambda: engine._batcher._last_add_ts + 11
        await engine._batch_flush_tick(0.0001)
        assert _audio(transport) == [pcm]
        await engine._batch_flush_tick(0.0001)
        assert _audio(transport) == [pcm]
    finally:
        await client.close()


@pytest.mark.parametrize("fixed", [False, True])
async def test_fractional_sample_thresholds_preserve_complete_pcm_samples(
    fixed: bool,
) -> None:
    """Fixed packets round up to whole samples; default mode keeps the burst."""
    rate = 44100
    transport = FakeOjinClient()
    config = _config()
    config.server_feed_initial_chunk_ms = 329
    config.server_feed_min_chunk_ms = 201
    config.server_feed_fixed_chunk_size = fixed
    client = OjinSTVClient(
        client=transport,
        tracer=RecordingTracer(),
        config=config,
        webrtc=WebRTCSettings(
            room_url="https://ojin.daily.co/fixed-feed",
            token="test-token",
            audio_sample_rate=rate,
        ),
    )
    engine = client._webrtc
    pcm = b"\x01\x02" * rate
    try:
        await engine._handle_message(
            OjinSessionReadyMessage(parameters=READY_CONNECTED_PARAMETERS)
        )
        await client.start_turn()
        await client.send_tts_audio(pcm, rate, 1)
        assert [len(packet) for packet in _audio(transport)] == (
            [29018, 17730, 17730, 17730] if fixed else [len(pcm)]
        )
        await client.start_turn()
        packets = _audio(transport)
        assert all(len(packet) % 2 == 0 for packet in packets)
        assert b"".join(packets) == pcm
    finally:
        await client.close()
