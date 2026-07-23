"""OjinSTVWebRTCClient metadata-frame handling: clock, events, ack, watchdog."""

import asyncio
import logging

from ojin.ojin_client_messages import (
    FrameType,
    OjinAudioInputMessage,
    OjinCancelInteractionMessage,
    OjinInteractionResponseMessage,
    OjinSessionReadyMessage,
)
from ojin.stv.config import STVConfig, WebRTCSettings
from ojin.stv.events import STVEvent
from ojin.stv.ojin_stv_client import OjinSTVClient
from ojin.stv.ojin_stv_webrtc_client import OjinSTVWebRTCClient
from tests.stv.fakes import FakeOjinClient, ListOutput, RecordingTracer
from tests.stv.test_webrtc_negotiation import READY_CONNECTED_PARAMETERS

# Canonical v2 post-connect webrtcStatus payloads (design §1.3): a mid-session
# room drop surfaces as ``disconnected`` then ``failed(REJOIN_FAILED)``.
STATUS_DISCONNECTED = {
    "status": "disconnected",
    "provider": "daily",
    "timestamp": 1789000000000,
}
STATUS_FAILED_REJOIN = {
    "status": "failed",
    "provider": "daily",
    "error": {"code": "REJOIN_FAILED", "message": "publisher dropped"},
    "timestamp": 1789000000000,
}


def make_client(clock=None, audio_sample_rate=16000, **config_overrides):
    """Build a webrtc client on fakes (no receive loop; handlers driven direct)."""
    fake_client = FakeOjinClient(session_parameters=READY_CONNECTED_PARAMETERS)
    tracer = RecordingTracer()
    kwargs = {"clock": clock} if clock is not None else {}
    client = OjinSTVWebRTCClient(
        webrtc_settings=WebRTCSettings(
            room_url="https://ojin.daily.co/room-abc",
            token="tok-secret-123",
            audio_sample_rate=audio_sample_rate,
        ),
        client=fake_client,
        tracer=tracer,
        config=STVConfig(
            loop_stall_watchdog_ms=0, stall_probe_ms=0, **config_overrides
        ),
        **kwargs,
    )
    return client, fake_client, tracer


async def _connect(client: OjinSTVWebRTCClient, fake_client: FakeOjinClient) -> None:
    """Drive the client to the CONNECTED state deterministically.

    Protocol v2: a single sessionReady carrying the connected result resolves
    the negotiation — there is no webrtcStatus ack.
    """
    await client._handle_message(
        OjinSessionReadyMessage(parameters=READY_CONNECTED_PARAMETERS)
    )


def _frame(
    frame_type: FrameType, index: int = 0, video: bytes = b"", audio: bytes = b""
) -> OjinInteractionResponseMessage:
    """Build a server frame (metadata by default; pass payloads for full frames)."""
    return OjinInteractionResponseMessage(
        interaction_id="i1",
        video_frame_bytes=video,
        audio_frame_bytes=audio,
        is_final_response=False,
        index=index,
        frame_type=frame_type,
    )


def _record(client: OjinSTVWebRTCClient, event: STVEvent) -> list[dict]:
    """Collect every emission of ``event`` as its kwargs dict."""
    calls: list[dict] = []
    client.add_listener(event, lambda **kwargs: calls.append(kwargs))
    return calls


def _audio_messages(fake_client: FakeOjinClient) -> list[OjinAudioInputMessage]:
    """Return the recorded outbound audio messages."""
    return [m for m in fake_client.sent if isinstance(m, OjinAudioInputMessage)]


async def test_clock_advances_40ms_per_speech_frame() -> None:
    """SPEECH and START_OF_SPEECH metadata frames advance the clock 40 ms each."""
    client, fake_client, _tracer = make_client()
    await _connect(client, fake_client)

    await client._handle_message(_frame(FrameType.START_OF_SPEECH))
    assert client._played_real_ms == 40.0
    await client._handle_message(_frame(FrameType.SPEECH))
    await client._handle_message(_frame(FrameType.SPEECH))
    assert client._played_real_ms == 120.0
    await client.close()


async def test_idle_and_fadeout_never_advance_clock() -> None:
    """IDLE and FADE_OUT frames flow at 25/s and must not move the lead clock."""
    client, fake_client, _tracer = make_client()
    await _connect(client, fake_client)

    for _ in range(5):
        await client._handle_message(_frame(FrameType.IDLE))
    await client._handle_message(_frame(FrameType.FADE_OUT))
    assert client._played_real_ms == 0.0
    await client.close()


async def test_lead_gated_sends_release_on_speech_frames() -> None:
    """The lead cap gates sends; SPEECH metadata frames release the backlog."""
    client, fake_client, _tracer = make_client(
        server_feed_batching_enabled=False, server_feed_max_lead_ms=100
    )
    await _connect(client, fake_client)
    fake_client.sent.clear()
    await client.start_turn()

    chunk = b"\x01\x02" * 1280  # 80 ms @ 16 kHz mono (identity resample)
    for _ in range(3):
        await client.send_tts_audio(chunk, 16000, 1)
    assert len(_audio_messages(fake_client)) == 2  # third is over the cap
    assert len(client._feed_pending) == 1

    for _ in range(3):  # 120 ms of published speech
        await client._handle_message(_frame(FrameType.SPEECH))
    await asyncio.sleep(0.05)  # let the feeder task release the backlog
    assert len(_audio_messages(fake_client)) == 3
    assert not client._feed_pending
    await client.close()


async def test_spurious_start_of_speech_after_speech_dropped_whole() -> None:
    """A boomerang START_OF_SPEECH right after SPEECH is dropped entirely."""
    client, fake_client, _tracer = make_client()
    started = _record(client, STVEvent.BOT_STARTED_SPEAKING)
    stopped = _record(client, STVEvent.BOT_STOPPED_SPEAKING)
    await _connect(client, fake_client)

    await client._handle_message(_frame(FrameType.SPEECH))
    assert client._played_real_ms == 40.0
    await client._handle_message(_frame(FrameType.START_OF_SPEECH))
    assert client._played_real_ms == 40.0  # no clock advance
    assert client._last_frame_type == int(FrameType.SPEECH)  # not recorded
    assert len(started) == 1 and stopped == []  # no extra edges
    await client.close()


async def test_first_frame_fires_exactly_once_any_frame_type() -> None:
    """The first metadata frame after connected emits FIRST_FRAME, one-shot."""
    client, fake_client, _tracer = make_client()
    first = _record(client, STVEvent.FIRST_FRAME)
    await _connect(client, fake_client)

    await client._handle_message(_frame(FrameType.IDLE))
    assert first == [{"frame_type": int(FrameType.IDLE)}]
    await client._handle_message(_frame(FrameType.SPEECH))
    await client._handle_message(_frame(FrameType.IDLE))
    assert len(first) == 1
    await client.close()


async def test_speaking_edges_from_frame_type_transitions() -> None:
    """IDLE/FADE_OUT→SPEECH starts speaking; SPEECH→FADE_OUT/IDLE stops it."""
    client, fake_client, _tracer = make_client()
    started = _record(client, STVEvent.BOT_STARTED_SPEAKING)
    stopped = _record(client, STVEvent.BOT_STOPPED_SPEAKING)
    await _connect(client, fake_client)

    await client._handle_message(_frame(FrameType.IDLE))
    await client._handle_message(_frame(FrameType.SPEECH))
    assert len(started) == 1 and len(stopped) == 0
    await client._handle_message(_frame(FrameType.FADE_OUT))
    assert len(started) == 1 and len(stopped) == 1
    await client._handle_message(_frame(FrameType.SPEECH))  # FADE_OUT→SPEECH
    assert len(started) == 2
    await client._handle_message(_frame(FrameType.IDLE))  # SPEECH→IDLE
    assert len(stopped) == 2
    await client.close()


async def test_start_of_speech_to_fadeout_emits_stopped() -> None:
    """A barge-in right after a post-cancel turn's first frame still stops."""
    client, fake_client, _tracer = make_client()
    stopped = _record(client, STVEvent.BOT_STOPPED_SPEAKING)
    await _connect(client, fake_client)

    await client._handle_message(_frame(FrameType.IDLE))
    await client._handle_message(_frame(FrameType.START_OF_SPEECH))
    await client._handle_message(_frame(FrameType.FADE_OUT))
    assert len(stopped) == 1
    await client.close()


async def test_frames_before_session_ready_parsed_and_discarded() -> None:
    """Frames before the sessionReady outcome produce no events, no clock.

    v2 has no pre-connected legacy-frame window — direct mode starts at the
    first frame — but a frame racing the outcome must still be harmless.
    """
    client, fake_client, _tracer = make_client()
    first = _record(client, STVEvent.FIRST_FRAME)
    started = _record(client, STVEvent.BOT_STARTED_SPEAKING)

    await client._handle_message(
        _frame(FrameType.SPEECH, video=b"\xff\xd8jpeg-ish", audio=b"\x01\x02" * 320)
    )
    await client._handle_message(_frame(FrameType.SPEECH))  # metadata crossing early
    assert first == [] and started == []
    assert client._played_real_ms == 0.0
    assert client._last_frame_type is None

    await _connect(client, fake_client)
    await client._handle_message(_frame(FrameType.IDLE))
    assert len(first) == 1  # the gate arms at the connected outcome
    await client.close()


async def test_full_payload_frame_after_connected_is_tolerated() -> None:
    """A legacy full-payload frame after connected is metadata-processed."""
    client, fake_client, _tracer = make_client()
    first = _record(client, STVEvent.FIRST_FRAME)
    started = _record(client, STVEvent.BOT_STARTED_SPEAKING)
    await _connect(client, fake_client)

    await client._handle_message(
        _frame(FrameType.SPEECH, video=b"\xff\xd8jpeg-ish", audio=b"\x01\x02" * 320)
    )
    assert first == [{"frame_type": int(FrameType.SPEECH)}]
    assert len(started) == 1
    assert client._played_real_ms == 40.0
    await client.close()


async def test_cancelled_turn_stragglers_are_dropped() -> None:
    """Audio for a cancelled turn (no new start_turn) is dropped, not sent."""
    client, fake_client, _tracer = make_client(server_feed_batching_enabled=False)
    await _connect(client, fake_client)
    await client.start_turn()
    await client.send_tts_audio(b"\x01\x02" * 640, 16000, 1)
    assert len(_audio_messages(fake_client)) == 1

    assert await client.interrupt() is True
    await client.send_tts_audio(b"\x03\x04" * 640, 16000, 1)  # straggler
    assert len(_audio_messages(fake_client)) == 1
    assert client._interrupt_deferred == []  # dropped, not deferred
    await client.close()


async def test_interrupt_cancels_resets_and_acks_on_idle() -> None:
    """Interrupt sends the cancel, resets the batcher, reconciles the lead."""
    client, fake_client, _tracer = make_client()
    await _connect(client, fake_client)
    await client.start_turn()
    frame = b"\x01\x02" * 640
    for _ in range(5):  # 200 ms — below the initial threshold, sits in batcher
        await client.send_tts_audio(frame, 16000, 1)
    assert client._batcher.pending_bytes == 6400
    client._played_real_ms = 100.0
    client._server_fed_ms = 500.0

    assert await client.interrupt() is True
    assert any(isinstance(m, OjinCancelInteractionMessage) for m in fake_client.sent)
    assert client._batcher.pending_bytes == 0
    assert client._server_fed_ms == client._played_real_ms == 100.0
    assert client._interruption_ongoing is True

    await client._handle_message(_frame(FrameType.IDLE))  # server ack
    assert client._interruption_ongoing is False
    await client.close()


async def test_new_turn_during_ack_window_is_deferred_then_replayed() -> None:
    """A turn opened while a cancel is settling is held, then shipped on the ack.

    The server discards audio sent mid-cancel, so the new turn must not be fed
    until the first idle/fade-out metadata frame closes the window; then the
    deferred turn and its audio replay in order.
    """
    client, fake_client, _tracer = make_client(server_feed_batching_enabled=False)
    await _connect(client, fake_client)
    await client.start_turn()
    chunk_a = b"\x01\x02" * 640
    await client.send_tts_audio(chunk_a, 16000, 1)
    assert [m.audio_int16_bytes for m in _audio_messages(fake_client)] == [chunk_a]

    assert await client.interrupt() is True
    assert client._interruption_ongoing is True

    chunk_b = b"\x03\x04" * 640
    await client.start_turn()  # new turn during the ack window
    await client.send_tts_audio(chunk_b, 16000, 1)
    assert client._deferring_input is True
    assert len(client._interrupt_deferred) == 2  # turn + audio, held
    assert [m.audio_int16_bytes for m in _audio_messages(fake_client)] == [chunk_a]

    await client._handle_message(_frame(FrameType.IDLE))  # server ack
    assert client._interruption_ongoing is False
    assert client._deferring_input is False
    assert client._interrupt_deferred == []
    assert [m.audio_int16_bytes for m in _audio_messages(fake_client)] == [
        chunk_a,
        chunk_b,
    ]
    await client.close()


async def test_metadata_watchdog_logs_but_never_fails(caplog) -> None:
    """A 5 s metadata gap logs a warning; the session stays healthy."""
    fake_now = [100.0]
    client, fake_client, _tracer = make_client(clock=lambda: fake_now[0])
    errors = _record(client, STVEvent.ERROR)

    assert client._check_metadata_watchdog() is False  # not connected yet
    await _connect(client, fake_client)
    assert client._check_metadata_watchdog() is False  # gap below threshold

    fake_now[0] += 6.0
    with caplog.at_level(logging.WARNING):
        assert client._check_metadata_watchdog() is True
    assert any("metadata frame" in record.message for record in caplog.records)
    assert client._check_metadata_watchdog() is False  # re-armed, no log spam
    assert errors == []
    await client.close()


async def test_failed_after_connected_is_fatal() -> None:
    """A post-connected failed status (REJOIN_FAILED) is fatal like any other."""
    client, fake_client, _tracer = make_client()
    errors = _record(client, STVEvent.ERROR)
    await _connect(client, fake_client)
    await client._handle_message(_frame(FrameType.IDLE))

    await fake_client.push_webrtc_status(STATUS_FAILED_REJOIN)
    assert len(errors) == 1
    assert errors[0]["code"] == "WEBRTC_JOIN_FAILED"
    assert errors[0]["fatal"] is True
    assert "REJOIN_FAILED" in errors[0]["message"]
    await client.close()


async def test_disconnected_alone_is_telemetry_only() -> None:
    """A disconnected status never fails the session; the failed after it does."""
    client, fake_client, _tracer = make_client()
    errors = _record(client, STVEvent.ERROR)
    await _connect(client, fake_client)

    await fake_client.push_webrtc_status(STATUS_DISCONNECTED)
    assert errors == []
    await client._handle_message(_frame(FrameType.IDLE))  # still processing frames
    assert client._played_real_ms == 0.0

    await fake_client.push_webrtc_status(STATUS_FAILED_REJOIN)
    assert len(errors) == 1 and errors[0]["fatal"] is True
    await client.close()


def test_webrtc_settings_repr_hides_token_and_pins_defaults() -> None:
    """The token never appears in repr/str; defaults match the wire contract."""
    settings = WebRTCSettings(
        room_url="https://ojin.daily.co/room-abc", token="super-secret-token"
    )
    assert "super-secret-token" not in repr(settings)
    assert "super-secret-token" not in str(settings)
    assert settings.provider == "daily"
    assert settings.audio_sample_rate == 16000
    assert settings.version == 2  # protocol v2 (DR-006)
    # Client-local; in v2 it governs the whole sessionReady wait and must not
    # appear in the connect declaration.
    assert settings.webrtc_join_timeout_s == 10.0
    assert "webrtc_join_timeout_s" not in settings.to_connect_query_params()
    assert settings.to_connect_query_params()["webrtc_version"] == "2"
    # The token travels only in the header — never in a query param.
    assert "super-secret-token" not in repr(settings.to_connect_query_params())
    assert settings.to_connect_headers() == {
        "X-Ojin-Webrtc-Token": "super-secret-token"
    }


def test_native_rate_feed_uses_48_bytes_per_ms() -> None:
    """A 24 kHz declared rate drives a 48 B/ms outbound clock, not 32 (D3/D5)."""
    client, _fake_client, _tracer = make_client(audio_sample_rate=24000)
    assert client._bytes_per_ms == 48.0


def test_legacy_client_uses_32_bytes_per_ms() -> None:
    """The legacy client defaults to the 16 kHz 32 B/ms clock (byte-identical)."""
    legacy = OjinSTVClient(client=FakeOjinClient(), output=ListOutput())
    assert legacy._bytes_per_ms == 32.0


async def test_24khz_tts_reaches_ws_unresampled() -> None:
    """At a 24 kHz declared rate, 24 kHz TTS is fed to the server verbatim.

    The declared rate equals the TTS native rate (the intended configuration),
    so the resample is a no-op and the exact input bytes reach the wire — no
    down-resample to 16 kHz.
    """
    client, fake_client, _tracer = make_client(
        audio_sample_rate=24000, server_feed_batching_enabled=False
    )
    await _connect(client, fake_client)
    fake_client.sent.clear()
    await client.start_turn()

    chunk_24k = b"\x01\x02" * 2400  # 100 ms @ 24 kHz mono int16
    await client.send_tts_audio(chunk_24k, 24000, 1)

    audio = _audio_messages(fake_client)
    assert len(audio) == 1
    assert audio[0].audio_int16_bytes == chunk_24k  # identity resample, verbatim
    await client.close()
