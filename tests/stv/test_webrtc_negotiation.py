"""OjinSTVWebRTCClient negotiation: capability gate, request wire shape, timers.

All wire inputs (the capability-advertising sessionReady parameters and the
webrtcStatus payloads) are the canonical pinned literals from the direct-webrtc
design, shared verbatim with the server-side suites so emitter and parser drift
are both unrepresentable without a failing test.
"""

import asyncio
import json

from ojin.entities.session_messages import SessionUpdateMessage
from ojin.ojin_client_messages import (
    FrameType,
    OjinAudioInputMessage,
    OjinCancelInteractionMessage,
    OjinInteractionResponseMessage,
)
from ojin.stv.config import STVConfig, WebRTCSettings
from ojin.stv.events import STVEvent
from ojin.stv.ojin_stv_webrtc_client import OjinSTVWebRTCClient
from tests.stv.fakes import FakeOjinClient, RecordingTracer

ROOM_URL = "https://ojin.daily.co/room-abc"
TOKEN = "tok-secret-123"

# Canonical capability advert content (design §1.1), as sessionReady parameters.
CAPABILITY_PARAMETERS = {"webrtc": {"version": 1, "providers": ["daily"]}}

# Canonical webrtcStatus payloads (design §1.3).
STATUS_CONNECTING = {
    "status": "connecting",
    "provider": "daily",
    "timestamp": 1789000000000,
}
STATUS_CONNECTED = {
    "status": "connected",
    "provider": "daily",
    "participant_id": "prt-1234",
    "timestamp": 1789000000000,
}
STATUS_FAILED = {
    "status": "failed",
    "provider": "daily",
    "error": {"code": "NETWORK", "message": "join failed"},
    "timestamp": 1789000000000,
}


def make_client(session_parameters=None, tracer=None, **config_overrides):
    """Build a webrtc client wired to in-memory fakes."""
    fake_client = FakeOjinClient(
        session_parameters=(
            session_parameters
            if session_parameters is not None
            else CAPABILITY_PARAMETERS
        )
    )
    recording_tracer = tracer or RecordingTracer()
    client = OjinSTVWebRTCClient(
        webrtc_settings=WebRTCSettings(room_url=ROOM_URL, token=TOKEN),
        client=fake_client,
        tracer=recording_tracer,
        config=STVConfig(
            loop_stall_watchdog_ms=0, stall_probe_ms=0, **config_overrides
        ),
    )
    return client, fake_client, recording_tracer


def _record(client: OjinSTVWebRTCClient, event: STVEvent) -> list[dict]:
    """Collect every emission of ``event`` as its kwargs dict."""
    calls: list[dict] = []
    client.add_listener(event, lambda **kwargs: calls.append(kwargs))
    return calls


def _audio_messages(fake_client: FakeOjinClient) -> list[OjinAudioInputMessage]:
    """Return the recorded outbound audio messages."""
    return [m for m in fake_client.sent if isinstance(m, OjinAudioInputMessage)]


async def test_sends_pinned_session_update_on_capability_ready() -> None:
    """A capability-advertising sessionReady triggers the exact pinned request."""
    client, fake_client, _tracer = make_client()
    await client.start()
    await asyncio.sleep(0.05)

    updates = [m for m in fake_client.sent if isinstance(m, SessionUpdateMessage)]
    assert len(updates) == 1
    wire = json.loads(updates[0].model_dump_json())
    timestamp = wire["payload"]["timestamp"]
    assert isinstance(timestamp, int) and timestamp > 0
    assert wire == {
        "type": "sessionUpdate",
        "payload": {
            "parameters": {
                "webrtc": {
                    "version": 1,
                    "provider": "daily",
                    "room_url": ROOM_URL,
                    "token": TOKEN,
                    "audio_sample_rate": 16000,
                }
            },
            "timestamp": timestamp,
        },
    }
    assert _audio_messages(fake_client) == []  # never a seed in webrtc mode
    await client.close()


async def test_no_capability_is_fatal_unsupported_and_never_requests() -> None:
    """Without the advert the client raises WEBRTC_UNSUPPORTED and stays silent."""
    client, fake_client, _tracer = make_client(session_parameters={"persona": "x"})
    errors = _record(client, STVEvent.ERROR)
    await client.start_turn()
    await client.send_tts_audio(b"\x01\x02" * 640, 16000, 1)

    await client.start()
    await asyncio.sleep(0.05)

    assert errors and errors[0]["code"] == "WEBRTC_UNSUPPORTED"
    assert errors[0]["fatal"] is True
    assert not any(isinstance(m, SessionUpdateMessage) for m in fake_client.sent)
    assert _audio_messages(fake_client) == []
    assert client._preinit_inputs == []  # held input discarded
    await client.close()


async def test_connecting_does_not_stop_the_join_timer() -> None:
    """Only a terminal status stops the timer; timeout is fatal JOIN_FAILED."""
    client, fake_client, _tracer = make_client()
    errors = _record(client, STVEvent.ERROR)
    await client.start()
    await asyncio.sleep(0.05)

    await fake_client.push_webrtc_status(STATUS_CONNECTING)
    timer = client._join_timer_task
    assert timer is not None and not timer.done()

    await client._handle_join_timeout()  # deadline reached with no terminal status
    assert errors and errors[0]["code"] == "WEBRTC_JOIN_FAILED"
    assert errors[0]["fatal"] is True
    await client.close()


async def test_connected_stops_timer_and_emits_webrtc_connected() -> None:
    """A terminal connected status cancels the timer and emits the event."""
    client, fake_client, _tracer = make_client()
    connected = _record(client, STVEvent.WEBRTC_CONNECTED)
    errors = _record(client, STVEvent.ERROR)
    await client.start()
    await asyncio.sleep(0.05)

    await fake_client.push_webrtc_status(STATUS_CONNECTED)
    assert connected == [{"participant_id": "prt-1234"}]
    assert client._join_timer_task is None

    await client._handle_join_timeout()  # a late fire is a no-op after terminal
    assert errors == []
    await client.close()


async def test_failed_status_is_fatal() -> None:
    """A failed status maps to a fatal WEBRTC_JOIN_FAILED error."""
    client, fake_client, _tracer = make_client()
    errors = _record(client, STVEvent.ERROR)
    await client.start()
    await asyncio.sleep(0.05)

    await fake_client.push_webrtc_status(STATUS_FAILED)
    assert len(errors) == 1
    assert errors[0]["code"] == "WEBRTC_JOIN_FAILED"
    assert errors[0]["fatal"] is True
    assert "NETWORK" in errors[0]["message"]
    await client.close()


async def test_input_held_until_connected_then_flushed_without_seed() -> None:
    """Audio fed pre-ready and mid-negotiation stays off the wire until connected.

    The legacy flush-at-ready path must not leak the preinit buffer at ready;
    on connected the held audio flushes in order with no seed preceding it.
    """
    client, fake_client, _tracer = make_client(server_feed_batching_enabled=False)
    chunk_16k = b"\x01\x02" * 640
    await client.start_turn()
    await client.send_tts_audio(chunk_16k, 16000, 1)  # pre-ready

    await client.start()
    await asyncio.sleep(0.05)  # sessionReady processed, request sent
    await client.send_tts_audio(b"\x05\x06" * 2400, 24000, 1)  # request→connected
    assert _audio_messages(fake_client) == []

    await fake_client.push_webrtc_status(STATUS_CONNECTED)
    audio = _audio_messages(fake_client)
    assert len(audio) == 2  # both held payloads, in order — and nothing else
    assert audio[0].audio_int16_bytes == chunk_16k  # 16 kHz is identity
    resampled = audio[1].audio_int16_bytes
    assert 0 < len(resampled) < 4800  # 24 kHz payload came out resampled
    assert b"\x00" * 1280 not in [m.audio_int16_bytes for m in audio]
    await client.close()


async def test_failure_discards_held_audio() -> None:
    """On a failed negotiation the held input is dropped, never sent."""
    client, fake_client, _tracer = make_client(server_feed_batching_enabled=False)
    await client.start_turn()
    await client.send_tts_audio(b"\x01\x02" * 640, 16000, 1)
    await client.start()
    await asyncio.sleep(0.05)

    await fake_client.push_webrtc_status(STATUS_FAILED)
    assert client._preinit_inputs == []
    await fake_client.push_webrtc_status(STATUS_CONNECTED)  # ignored: terminal
    assert _audio_messages(fake_client) == []
    await client.close()


async def test_interrupt_in_requested_state_clears_buffer_without_cancel() -> None:
    """A barge-in during the join window never touches the wire.

    The buffered turn is cleared, INTERRUPTED is emitted locally, no cancel is
    sent, no ack-suppression window opens, and the greeting is not replayed at
    connected.
    """
    client, fake_client, _tracer = make_client(server_feed_batching_enabled=False)
    interrupted = _record(client, STVEvent.INTERRUPTED)
    await client.start()
    await asyncio.sleep(0.05)

    await client.start_turn()
    await client.send_tts_audio(b"\x01\x02" * 640, 16000, 1)
    assert len(client._preinit_inputs) == 2

    assert await client.interrupt() is False
    assert interrupted == [{}]
    assert client._preinit_inputs == []
    assert client._interruption_ongoing is False  # no ack-suppression window
    assert not any(
        isinstance(m, OjinCancelInteractionMessage) for m in fake_client.sent
    )

    await fake_client.push_webrtc_status(STATUS_CONNECTED)
    assert _audio_messages(fake_client) == []  # the cleared turn is not replayed
    await client.close()


async def test_trace_records_webrtc_lane_and_other_data() -> None:
    """The webrtc lane, negotiate span, recv latency and otherData are recorded."""
    client, fake_client, tracer = make_client(server_feed_batching_enabled=False)
    await client.start()
    await asyncio.sleep(0.05)
    await fake_client.push_webrtc_status(STATUS_CONNECTING)
    await fake_client.push_webrtc_status(STATUS_CONNECTED)

    await client.start_turn()
    await client.send_tts_audio(b"\x01\x02" * 640, 16000, 1)
    await client._handle_message(
        OjinInteractionResponseMessage(
            interaction_id="i1",
            video_frame_bytes=b"",
            audio_frame_bytes=b"",
            is_final_response=False,
            index=0,
            frame_type=FrameType.SPEECH,
        )
    )

    webrtc_instants = [
        (name, args) for (lane, name, args) in tracer.instants if lane == "webrtc"
    ]
    assert ("request_sent", {}) in webrtc_instants
    assert ("webrtc_status", {"status": "connecting"}) in webrtc_instants
    assert ("webrtc_status", {"status": "connected"}) in webrtc_instants
    assert any(name == "first_metadata_frame" for name, _args in webrtc_instants)
    assert any(
        lane == "webrtc" and name == "negotiate"
        for (lane, name, _start, _args) in tracer.spans
    )
    assert any(name == "latency_recv" for (_lane, name, _s, _a) in tracer.spans)

    assert tracer.other["producer"] == "ojin_stv_webrtc_client"
    assert "recv_latency_semantics" in tracer.other
    summary = tracer.other["webrtc"]
    assert set(summary) == {"provider", "join_ms", "participant_id"}
    assert summary["provider"] == "daily"
    assert summary["participant_id"] == "prt-1234"
    dumped = json.dumps(tracer.other)
    assert TOKEN not in dumped
    assert ROOM_URL not in dumped
    await client.close()
