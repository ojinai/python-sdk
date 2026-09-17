"""OjinSTVClient(webrtc=...): the direct-WebRTC path behind the unchanged client API.

The same calls a WebSocket program makes (start / start_turn / send_tts_audio /
say / interrupt / close, event listeners, output_stream) must work unchanged
when ``webrtc`` settings are passed; the media goes to the room, and any WebRTC
failure is reported as a fatal ERROR that closes the session.
"""

import asyncio

import pytest

from ojin.ojin_client_messages import (
    FrameType,
    OjinAudioInputMessage,
    OjinCancelInteractionMessage,
)
from ojin.stv.config import STVConfig, WebRTCProvider, WebRTCSettings
from ojin.stv.events import STVEvent
from ojin.stv.ojin_stv_client import OjinSTVClient
from tests.stv.fakes import FakeOjinClient, ListOutput, RecordingTracer
from tests.stv.test_webrtc_client import STATUS_FAILED_REJOIN, _frame
from tests.stv.test_webrtc_negotiation import (
    READY_CONNECTED_PARAMETERS,
    READY_FAILED_PARAMETERS,
    READY_NOT_SUPPORTED_PARAMETERS,
    ROOM_URL,
    TOKEN,
)

RATE = 24000
CHUNK = b"\x01\x02" * (RATE // 25)  # one 40 ms mono int16 chunk at 24 kHz


class _NeverReadyClient(FakeOjinClient):
    """A transport that connects but never delivers sessionReady."""

    async def connect(self) -> None:
        """Mark connected without enqueueing a SessionReady."""
        self._running = True


def make_client(
    session_parameters: dict = READY_CONNECTED_PARAMETERS,
    *,
    output: object = None,
    fake: FakeOjinClient | None = None,
    join_timeout_s: float = 10.0,
):
    """Build a WebRTC-mode OjinSTVClient on in-memory fakes."""
    fake_client = fake or FakeOjinClient(session_parameters=session_parameters)
    client = OjinSTVClient(
        webrtc=WebRTCSettings(
            room_url=ROOM_URL,
            token=TOKEN,
            provider="livekit",
            audio_sample_rate=RATE,
            webrtc_join_timeout_s=join_timeout_s,
        ),
        client=fake_client,
        output=output,  # type: ignore[arg-type]
        tracer=RecordingTracer(),
        config=STVConfig(
            loop_stall_watchdog_ms=0,
            stall_probe_ms=0,
            server_feed_batching_enabled=False,
        ),
    )
    return client, fake_client


def _record(client: OjinSTVClient, event: STVEvent) -> list[dict]:
    """Collect every emission of ``event`` as its kwargs dict."""
    calls: list[dict] = []
    client.add_listener(event, lambda **kwargs: calls.append(kwargs))
    return calls


def _audio(fake_client: FakeOjinClient) -> list[bytes]:
    """Return the recorded outbound audio payloads."""
    return [
        m.audio_int16_bytes
        for m in fake_client.sent
        if isinstance(m, OjinAudioInputMessage)
    ]


async def _drain(stream) -> list:
    """Consume an output stream to the end."""
    return [frame async for frame in stream]


def test_webrtc_settings_ride_the_connect_request() -> None:
    """Passing ``webrtc`` declares the settings on the transport before connect."""
    _client, fake_client = make_client()
    settings = fake_client.webrtc_connect_settings
    assert isinstance(settings, WebRTCSettings)
    assert settings.provider is WebRTCProvider.LIVEKIT
    assert settings.to_connect_query_params()["webrtc_audio_sample_rate"] == "24000"


def test_websocket_mode_declares_no_webrtc_settings() -> None:
    """Without ``webrtc`` the client stays on the plain WebSocket path."""
    fake_client = FakeOjinClient()
    client = OjinSTVClient(client=fake_client, output=ListOutput())
    assert client._webrtc is None
    assert fake_client.webrtc_connect_settings is None


async def test_connected_session_uses_the_same_input_api() -> None:
    """The start / start_turn / send_tts_audio calls work unchanged in WebRTC mode."""
    client, fake_client = make_client()
    ready = _record(client, STVEvent.SESSION_READY)
    connected = _record(client, STVEvent.WEBRTC_CONNECTED)
    errors = _record(client, STVEvent.ERROR)

    await client.start_turn()  # before the session is ready: held, then replayed
    await client.send_tts_audio(CHUNK, RATE, 1)
    await client.start()
    await asyncio.sleep(0.05)

    assert len(ready) == 1
    assert connected == [{"participant_id": "prt-1234"}]
    assert errors == []
    assert client.is_connected is True
    assert client.session_data == READY_CONNECTED_PARAMETERS
    # No WebSocket-mode silence seed: only the TTS audio, at the declared rate.
    assert _audio(fake_client) == [CHUNK]
    await client.close()
    assert client.is_connected is False


async def test_say_and_interrupt_work_in_webrtc_mode() -> None:
    """The one-shot say() helper and barge-in go through the WebRTC engine."""
    client, fake_client = make_client()
    interrupted = _record(client, STVEvent.INTERRUPTED)
    await client.start()
    await asyncio.sleep(0.05)

    await client.say(CHUNK, RATE, 1)
    assert _audio(fake_client) == [CHUNK]

    assert await client.interrupt() is True
    assert any(isinstance(m, OjinCancelInteractionMessage) for m in fake_client.sent)
    assert interrupted == [{}]
    await client.close()


async def test_no_local_playback_pipeline_is_started() -> None:
    """The decode worker, playback loop and WebSocket receive loop never run."""
    client, _fake_client = make_client()
    await client.start()
    await asyncio.sleep(0.05)

    assert client._decode_thread is None
    assert client._playback_task is None
    assert client._receive_task is None
    await client.close()


async def test_output_stream_ends_empty_and_events_flow() -> None:
    """Metadata frames drive events; output_stream() yields nothing and ends."""
    client, fake_client = make_client()
    first = _record(client, STVEvent.FIRST_FRAME)
    started = _record(client, STVEvent.BOT_STARTED_SPEAKING)
    stopped = _record(client, STVEvent.BOT_STOPPED_SPEAKING)
    consumer = asyncio.create_task(_drain(client.output_stream()))

    await client.start()
    await asyncio.sleep(0.05)
    for frame_type in (
        FrameType.IDLE,
        FrameType.START_OF_SPEECH,
        FrameType.SPEECH,
        FrameType.IDLE,
    ):
        await fake_client.push(_frame(frame_type))
    await asyncio.sleep(0.05)

    assert first == [{"frame_type": int(FrameType.IDLE)}]
    assert len(started) == 1
    assert len(stopped) == 1
    await client.close()
    assert await asyncio.wait_for(consumer, timeout=1.0) == []


async def test_custom_output_receives_no_frames() -> None:
    """An injected STVOutput sink is left untouched: the media is in the room."""
    output = ListOutput()
    client, fake_client = make_client(output=output)
    await client.start()
    await asyncio.sleep(0.05)
    await fake_client.push(_frame(FrameType.SPEECH))
    await asyncio.sleep(0.05)
    await client.close()

    assert output.audio == []
    assert output.video == []


@pytest.mark.parametrize(
    ("parameters", "code"),
    [
        (READY_FAILED_PARAMETERS, "WEBRTC_AUTH_FAILED"),
        (READY_NOT_SUPPORTED_PARAMETERS, "WEBRTC_NOT_SUPPORTED"),
    ],
)
async def test_webrtc_failure_is_reported_and_closes(
    parameters: dict, code: str
) -> None:
    """A failed or unsupported WebRTC session is a fatal ERROR, then CLOSED."""
    client, fake_client = make_client(parameters)
    errors = _record(client, STVEvent.ERROR)
    closed = _record(client, STVEvent.CLOSED)
    consumer = asyncio.create_task(_drain(client.output_stream()))

    await client.say(CHUNK, RATE, 1)
    await client.start()
    await asyncio.sleep(0.05)

    assert [(e["code"], e["fatal"]) for e in errors] == [(code, True)]
    assert closed == [{}]
    assert client.is_connected is False
    assert fake_client.closed is True
    assert _audio(fake_client) == []  # nothing ever reached the server
    # A consumer blocked on output_stream() is released by the failure close.
    assert await asyncio.wait_for(consumer, timeout=1.0) == []
    await client.close()  # idempotent
    assert closed == [{}]


async def test_join_timeout_is_reported_and_closes() -> None:
    """No sessionReady within the join timeout → fatal WEBRTC_JOIN_TIMEOUT."""
    client, _fake_client = make_client(fake=_NeverReadyClient(), join_timeout_s=0.05)
    errors = _record(client, STVEvent.ERROR)
    closed = _record(client, STVEvent.CLOSED)

    await client.start()
    await asyncio.sleep(0.2)

    assert [e["code"] for e in errors] == ["WEBRTC_JOIN_TIMEOUT"]
    assert closed == [{}]


async def test_mid_session_room_loss_is_reported_and_closes() -> None:
    """A post-connect webrtcStatus failed(REJOIN_FAILED) ends the session."""
    client, fake_client = make_client()
    errors = _record(client, STVEvent.ERROR)
    closed = _record(client, STVEvent.CLOSED)
    await client.start()
    await asyncio.sleep(0.05)

    await fake_client.push_webrtc_status(STATUS_FAILED_REJOIN)
    await asyncio.sleep(0.05)

    assert [e["code"] for e in errors] == ["WEBRTC_ROOM_LOST"]
    assert "REJOIN_FAILED" in errors[0]["message"]
    assert closed == [{}]
    assert client.is_connected is False


async def test_context_manager_works_in_webrtc_mode() -> None:
    """``async with`` starts and closes the WebRTC session like a WebSocket one."""
    client, fake_client = make_client()
    closed = _record(client, STVEvent.CLOSED)
    async with client:
        await asyncio.sleep(0.05)
        assert client.is_connected is True
    assert closed == [{}]
    assert fake_client.closed is True
