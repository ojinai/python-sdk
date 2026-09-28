"""Server feed hints apply before audio replay without changing explicit settings."""

import pytest

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


def _client(direct: bool, initial: int | None = None, rate: int = 16000):
    transport = FakeOjinClient()
    trace = RecordingTracer()
    config = STVConfig(
        server_feed_flush_idle_ms=10000,
        server_feed_max_lead_ms=0,
    )
    if initial is not None:
        config.server_feed_initial_chunk_ms = initial
    client = OjinSTVClient(
        client=transport,
        tracer=trace,
        config=config,
        webrtc=(
            WebRTCSettings(
                room_url="https://ojin.daily.co/feed-hint",
                token="feed-hint-test",
                audio_sample_rate=rate,
            )
            if direct
            else None
        ),
    )
    return client, client._webrtc or client, transport, trace


def _ready(hint: object = None) -> OjinSessionReadyMessage:
    parameters = dict(READY_CONNECTED_PARAMETERS)
    if hint is not None:
        parameters["server_feed_initial_chunk_ms"] = hint
    return OjinSessionReadyMessage(parameters=parameters)


def _audio(transport: FakeOjinClient) -> list[bytes]:
    return [
        message.audio_int16_bytes
        for message in transport.sent
        if isinstance(message, OjinAudioInputMessage) and any(message.audio_int16_bytes)
    ]


@pytest.mark.parametrize("direct,rate", [(False, 16000), (True, 16000), (True, 44100)])
async def test_hint_releases_preinit_audio_and_rearms_after_cancel(
    direct: bool, rate: int
) -> None:
    """A 500 ms first batch reaches the server both before and after barge-in."""
    client, engine, transport, _trace = _client(direct, rate=rate)
    pcm = b"\x01\x02" * (rate // 2)
    try:
        await client.start_turn()
        await client.send_tts_audio(pcm, rate, 1)
        assert not _audio(transport)
        await engine._handle_message(_ready(500))
        assert _audio(transport) == [pcm]
        if not direct:
            engine._synchronizer.current_buffer = AudioBuffer(sample_rate=rate)
            engine._synchronizer.current_buffer.bytes_.extend(pcm)
        assert await client.interrupt()
        await client.start_turn()
        split_at = rate * 2 * 400 // 1000
        await client.send_tts_audio(pcm[:split_at], rate, 1)
        await engine._handle_message(_frame(FrameType.IDLE))
        assert len(_audio(transport)) == 1
        await client.send_tts_audio(pcm[split_at:], rate, 1)
        assert _audio(transport) == [pcm, pcm]
    finally:
        await client.close()


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("hint", [None, False, True, 0, -1, 500.0, "500", 8001, {}, []])
async def test_missing_or_invalid_hint_keeps_legacy_threshold(
    direct: bool, hint: object
) -> None:
    """Untrusted hint shapes cannot silently change the legacy initial batch."""
    client, engine, transport, _trace = _client(direct)
    pcm = b"\x01\x02" * 8000
    try:
        await engine._handle_message(_ready(hint))
        await client.start_turn()
        await client.send_tts_audio(pcm, 16000, 1)
        assert not _audio(transport)
        await client.send_tts_audio(pcm, 16000, 1)
        assert _audio(transport) == [pcm * 2]
    finally:
        await client.close()


@pytest.mark.parametrize("direct", [False, True])
async def test_explicit_threshold_overrides_server_hint(direct: bool) -> None:
    """A caller's configured startup lead wins over the server recommendation."""
    client, engine, transport, _trace = _client(direct, initial=1200)
    try:
        await engine._handle_message(_ready(500))
        await client.start_turn()
        await client.send_tts_audio(b"\x01\x02" * 16000, 16000, 1)
        assert not _audio(transport)
        await client.send_tts_audio(b"\x01\x02" * 3200, 16000, 1)
        assert _audio(transport) == [b"\x01\x02" * 19200]
    finally:
        await client.close()


@pytest.mark.parametrize("direct", [False, True])
async def test_portrait_threshold_sends_whole_burst_then_rearms(direct: bool) -> None:
    """An explicit 200 ms lead overrides older hints and preserves TTS bursts."""
    client, engine, transport, _trace = _client(direct, initial=200)
    pcm = b"\x01\x02" * 16000
    try:
        await engine._handle_message(_ready(500))
        await client.start_turn()
        await client.send_tts_audio(pcm[:3200], 16000, 1)
        assert not _audio(transport)
        await client.send_tts_audio(pcm[3200:], 16000, 1)
        assert _audio(transport) == [pcm]

        await client.start_turn()
        initial = pcm[: 200 * 32]
        await client.send_tts_audio(initial[:-2], 16000, 1)
        assert _audio(transport) == [pcm]
        await client.send_tts_audio(initial[-2:], 16000, 1)
        assert _audio(transport) == [pcm, initial]
    finally:
        await client.close()


@pytest.mark.parametrize("direct", [False, True])
async def test_later_handshake_restores_fallback_without_losing_pending(
    direct: bool,
) -> None:
    """Another ready event updates the lead while retaining unsent PCM."""
    client, engine, transport, _trace = _client(direct)
    pcm = b"\x01\x02" * 6400
    try:
        await engine._handle_message(_ready(500))
        await client.start_turn()
        await client.send_tts_audio(pcm, 16000, 1)
        assert not _audio(transport)
        await engine._handle_message(_ready())
        await client.send_tts_audio(pcm, 16000, 1)
        assert not _audio(transport)
        await client.send_tts_audio(pcm[:6400], 16000, 1)
        assert _audio(transport) == [pcm * 2 + pcm[:6400]]
        await client.send_tts_audio(pcm[:6400], 16000, 1)
        await engine._handle_message(_ready(500))
        await client.send_tts_audio(pcm[:6400], 16000, 1)
        assert _audio(transport)[1] == pcm
    finally:
        await client.close()
