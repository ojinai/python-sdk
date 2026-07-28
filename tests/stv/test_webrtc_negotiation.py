"""OjinSTVWebRTCClient protocol-v2 negotiation: setup embedding + ready outcome.

Protocol v2 (DR-006): the webrtc request rides ``sessionSetup.parameters`` —
the connection's first message — and the server reports the outcome inside
``sessionReady.parameters.webrtc``. All wire inputs here are the canonical
pinned literals from the direct-webrtc design, shared verbatim with the
server-side suites so emitter and parser drift are both unrepresentable
without a failing test.
"""

import asyncio
import json
import logging

from ojin.entities.session_messages import SessionUpdateMessage
from ojin.ojin_client_messages import (
    OjinAudioInputMessage,
    OjinCancelInteractionMessage,
    OjinSessionReadyMessage,
)
from ojin.stv.config import STVConfig, WebRTCSettings
from ojin.stv.events import STVEvent
from ojin.stv.ojin_stv_webrtc_client import OjinSTVWebRTCClient
from tests.stv.fakes import FakeOjinClient, RecordingTracer

ROOM_URL = "https://ojin.daily.co/room-abc"
TOKEN = "tok-secret-123"

# Canonical v2 connect declaration (design §1.1, 2026-07-24 amendment): the
# non-secret fields ride webrtc_* query params on the upgrade request; the
# token rides the X-Ojin-Webrtc-Token header, never the URL.
PINNED_QUERY_PARAMS = {
    "webrtc_version": "2",
    "webrtc_provider": "daily",
    "webrtc_room_url": ROOM_URL,
    "webrtc_audio_sample_rate": "16000",
}
PINNED_HEADERS = {"X-Ojin-Webrtc-Token": TOKEN}

# Canonical v2 result objects (design §1.2) as sessionReady parameters.
READY_CONNECTED_PARAMETERS = {
    "webrtc": {
        "version": 2,
        "status": "connected",
        "provider": "daily",
        "participant_id": "prt-1234",
    }
}
READY_FAILED_PARAMETERS = {
    "webrtc": {
        "version": 2,
        "status": "failed",
        "provider": "daily",
        "error": {"code": "AUTH", "message": "token rejected"},
    }
}
# A server without direct-webrtc support omits the key entirely (relay mode).
READY_RELAY_PARAMETERS = {"persona": "x"}


def make_client(session_parameters=None, tracer=None, **config_overrides):
    """Build a webrtc client wired to in-memory fakes."""
    fake_client = FakeOjinClient(
        session_parameters=(
            session_parameters
            if session_parameters is not None
            else READY_CONNECTED_PARAMETERS
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


def test_settings_declared_on_transport_with_pinned_connect_shapes() -> None:
    """Construction declares the settings on the transport for connect time.

    The request rides the WebSocket upgrade request — non-secret fields as the
    pinned ``webrtc_*`` query params, the token in the pinned header, never in
    the query params.
    """
    _client, fake_client, _tracer = make_client()
    settings = fake_client.webrtc_connect_settings
    assert isinstance(settings, WebRTCSettings)
    assert settings.to_connect_query_params() == PINNED_QUERY_PARAMS
    assert settings.to_connect_headers() == PINNED_HEADERS
    assert TOKEN not in json.dumps(settings.to_connect_query_params())


async def test_no_session_update_and_no_seed_ever_sent() -> None:
    """The v1 machinery is dead: no webrtc sessionUpdate, no audio seed."""
    client, fake_client, _tracer = make_client()
    await client.start()
    await asyncio.sleep(0.05)

    assert not any(isinstance(m, SessionUpdateMessage) for m in fake_client.sent)
    assert _audio_messages(fake_client) == []
    await client.close()
    assert not any(isinstance(m, SessionUpdateMessage) for m in fake_client.sent)


async def test_connected_result_opens_direct_mode_and_flushes_held_input() -> None:
    """A connected sessionReady opens the direct path from the first frame.

    Input fed before the outcome stays off the wire, then flushes in order
    with no seed preceding it.
    """
    client, fake_client, _tracer = make_client(server_feed_batching_enabled=False)
    connected = _record(client, STVEvent.WEBRTC_CONNECTED)
    errors = _record(client, STVEvent.ERROR)
    chunk_16k = b"\x01\x02" * 640
    await client.start_turn()
    await client.send_tts_audio(chunk_16k, 16000, 1)  # held: outcome pending
    assert _audio_messages(fake_client) == []

    await client.start()
    await asyncio.sleep(0.05)

    assert connected == [{"participant_id": "prt-1234"}]
    assert errors == []
    audio = _audio_messages(fake_client)
    assert [m.audio_int16_bytes for m in audio] == [chunk_16k]
    assert client._join_timer_task is None  # timer stopped by the outcome

    await client._handle_join_timeout()  # a late fire is a no-op after ready
    assert errors == []
    await client.close()


async def test_failed_result_is_fatal_and_discards_held_input() -> None:
    """status=failed maps to a fatal WEBRTC_JOIN_FAILED (fail-fast retained)."""
    client, fake_client, _tracer = make_client(
        session_parameters=READY_FAILED_PARAMETERS,
        server_feed_batching_enabled=False,
    )
    errors = _record(client, STVEvent.ERROR)
    connected = _record(client, STVEvent.WEBRTC_CONNECTED)
    await client.start_turn()
    await client.send_tts_audio(b"\x01\x02" * 640, 16000, 1)

    await client.start()
    await asyncio.sleep(0.05)

    assert len(errors) == 1
    assert errors[0]["code"] == "WEBRTC_JOIN_FAILED"
    assert errors[0]["fatal"] is True
    assert "AUTH" in errors[0]["message"]
    assert connected == []
    assert client._preinit_inputs == []  # held input discarded, never sent
    assert _audio_messages(fake_client) == []
    await client.close()


async def test_absent_key_falls_back_to_relay_nonfatal(caplog) -> None:
    """No webrtc result key → graceful relay fallback, NOT an error.

    This replaces v1's client-fatal WEBRTC_UNSUPPORTED: the session continues
    as a legacy relay session — the feed opens and held input flushes — with
    only log + trace telemetry surfacing the fallback.
    """
    client, fake_client, tracer = make_client(
        session_parameters=READY_RELAY_PARAMETERS,
        server_feed_batching_enabled=False,
    )
    errors = _record(client, STVEvent.ERROR)
    connected = _record(client, STVEvent.WEBRTC_CONNECTED)
    ready = _record(client, STVEvent.SESSION_READY)
    chunk_16k = b"\x01\x02" * 640
    await client.start_turn()
    await client.send_tts_audio(chunk_16k, 16000, 1)

    with caplog.at_level(logging.WARNING):
        await client.start()
        await asyncio.sleep(0.05)

    assert errors == []  # pinned: relay fallback is non-fatal
    assert connected == []  # the direct path never opened
    assert len(ready) == 1
    # The session continues: the held input flushed onto the relay session.
    assert [m.audio_int16_bytes for m in _audio_messages(fake_client)] == [chunk_16k]
    # Telemetry surfaces the fallback.
    assert any("relay mode" in record.message for record in caplog.records)
    assert ("relay_fallback", {"reason": "absent"}) in [
        (name, args) for (lane, name, args) in tracer.instants if lane == "webrtc"
    ]
    assert tracer.other["webrtc"] == {"mode": "relay", "reason": "absent"}
    await client.close()


async def test_unknown_result_status_falls_back_to_relay() -> None:
    """A webrtc result with an unrecognized status degrades to relay, not fatal."""
    client, _fake_client, _tracer = make_client(
        session_parameters={"webrtc": {"version": 2, "status": "connecting"}},
        server_feed_batching_enabled=False,
    )
    errors = _record(client, STVEvent.ERROR)
    await client.start()
    await asyncio.sleep(0.05)

    assert errors == []
    assert client._feed_gate_open() is True
    await client.close()


async def test_no_session_ready_within_timeout_is_fatal() -> None:
    """The join timer governs the whole sessionReady wait; expiry is fatal."""
    client, fake_client, _tracer = make_client(server_feed_batching_enabled=False)
    errors = _record(client, STVEvent.ERROR)
    await client.start_turn()
    await client.send_tts_audio(b"\x01\x02" * 640, 16000, 1)

    # Arm the timer as start() does, but resolve nothing (no sessionReady).
    client._request_sent_at = 0.0
    await client._handle_join_timeout()

    assert len(errors) == 1
    assert errors[0]["code"] == "WEBRTC_JOIN_FAILED"
    assert errors[0]["fatal"] is True
    assert "sessionReady" in errors[0]["message"]
    assert client._preinit_inputs == []  # held input discarded on timeout
    assert _audio_messages(fake_client) == []
    await client.close()


async def test_second_session_ready_never_renegotiates() -> None:
    """One negotiation per connection: a later sessionReady changes nothing."""
    client, _fake_client, tracer = make_client()
    await client.start()
    await asyncio.sleep(0.05)
    assert client._state.value == "connected"

    await client._handle_message(
        OjinSessionReadyMessage(parameters=READY_RELAY_PARAMETERS)
    )
    assert client._state.value == "connected"  # not demoted to relay
    assert not any(
        name == "relay_fallback"
        for (lane, name, _args) in tracer.instants
        if lane == "webrtc"
    )
    await client.close()


async def test_interrupt_while_outcome_pending_clears_buffer_without_cancel() -> None:
    """A barge-in before the sessionReady outcome never touches the wire."""
    client, fake_client, _tracer = make_client(server_feed_batching_enabled=False)
    interrupted = _record(client, STVEvent.INTERRUPTED)

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

    await client.start()  # connected outcome arrives afterwards
    await asyncio.sleep(0.05)
    assert _audio_messages(fake_client) == []  # the cleared turn is not replayed
    await client.close()


async def test_token_never_in_logs_or_traces_any_outcome(caplog) -> None:
    """The meeting token appears in no log record and no trace surface."""
    for parameters in (
        READY_CONNECTED_PARAMETERS,
        READY_FAILED_PARAMETERS,
        READY_RELAY_PARAMETERS,
    ):
        client, _fake_client, tracer = make_client(session_parameters=parameters)
        with caplog.at_level(logging.DEBUG):
            await client.start()
            await asyncio.sleep(0.05)
            await client.close()
        for record in caplog.records:
            assert TOKEN not in record.getMessage()
        dumped = json.dumps(
            {
                "other": tracer.other,
                "instants": [args for (_lane, _name, args) in tracer.instants],
                "spans": [args for (_lane, _name, _start, args) in tracer.spans],
            }
        )
        assert TOKEN not in dumped
        assert ROOM_URL not in dumped


async def test_trace_records_negotiate_span_and_webrtc_summary() -> None:
    """The webrtc lane records request_sent and a connected negotiate span."""
    client, _fake_client, tracer = make_client(server_feed_batching_enabled=False)
    await client.start()
    await asyncio.sleep(0.05)

    webrtc_instants = [
        (name, args) for (lane, name, args) in tracer.instants if lane == "webrtc"
    ]
    assert ("request_sent", {}) in webrtc_instants
    negotiate_spans = [
        args
        for (lane, name, _start, args) in tracer.spans
        if lane == "webrtc" and name == "negotiate"
    ]
    assert len(negotiate_spans) == 1
    assert negotiate_spans[0]["outcome"] == "connected"
    assert "join_ms" in negotiate_spans[0]

    summary = tracer.other["webrtc"]
    assert summary["mode"] == "direct"
    assert summary["provider"] == "daily"
    assert summary["participant_id"] == "prt-1234"
    assert set(summary) == {"mode", "provider", "join_ms", "participant_id"}
    await client.close()
