"""Separate queued audio, socket submission, and received speech timestamps."""

import asyncio
import contextlib

import pytest

from ojin.ojin_client import OjinClient
from ojin.ojin_client_messages import (
    FrameType,
    OjinInteractionResponseMessage,
    OjinSessionReadyMessage,
)
from ojin.stv import OjinSessionTrace, OjinSTVClient, STVConfig, WebRTCSettings
from tests.stv.test_webrtc_negotiation import READY_CONNECTED_PARAMETERS


@pytest.mark.parametrize("direct_webrtc", [False, True])
async def test_audio_trace_distinguishes_queue_socket_and_receipt(direct_webrtc):
    """A blocked socket must not look sent when the SDK merely queues audio."""
    now = [0.0]
    trace = OjinSessionTrace(clock=lambda: now[0])
    submitted, release, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class PausedSocket:
        async def send(self, _data):
            submitted.set()
            await release.wait()
            completed.set()

    transport = OjinClient("ws://localhost:8000/realtime", "timing-test", "portrait")
    transport._running = True
    transport._inference_server_ready = True
    transport._ws = PausedSocket()
    client = OjinSTVClient(
        client=transport,
        tracer=trace,
        config=STVConfig(
            server_feed_batching_enabled=False,
            server_feed_max_lead_ms=0,
            loop_stall_watchdog_ms=0,
            stall_probe_ms=0,
        ),
        webrtc=(
            WebRTCSettings(
                room_url="https://ojin.daily.co/timing-test", token="timing-test"
            )
            if direct_webrtc
            else None
        ),
    )
    engine = client._webrtc or client
    if direct_webrtc:
        await engine._handle_message(
            OjinSessionReadyMessage(parameters=READY_CONNECTED_PARAMETERS)
        )
    else:
        engine._initialized = True
    now[0] = 1.0
    await client.start_turn()
    await client.send_tts_audio(b"\x01\x02" * 3200, 16000, 1)
    before_send = trace.build()["traceEvents"]
    assert [
        event["name"] for event in before_send if event["name"].startswith("audio_")
    ] == ["audio_enqueued"]

    now[0] = 2.0
    sender = asyncio.create_task(transport._process_client_messages())
    try:
        await asyncio.wait_for(submitted.wait(), 1)
        assert not any(
            event["name"] == "audio_send_complete"
            for event in trace.build()["traceEvents"]
        )
        now[0] = 2.1
        release.set()
        await asyncio.wait_for(completed.wait(), 1)
        now[0] = 3.0
        await engine._handle_message(
            OjinInteractionResponseMessage(
                interaction_id="timing-test",
                video_frame_bytes=b"",
                audio_frame_bytes=b"",
                is_final_response=False,
                index=5,
                frame_type=FrameType.SPEECH,
            )
        )
    finally:
        transport._running = False
        sender.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sender

    result = trace.build()
    audio_events = [
        event for event in result["traceEvents"] if event["name"].startswith("audio_")
    ]
    assert [event["name"] for event in audio_events] == [
        "audio_enqueued",
        "audio_send_start",
        "audio_send_complete",
    ]
    assert [event["ts"] for event in audio_events] == pytest.approx(
        [1_000_000, 2_000_000, 2_100_000]
    )
    assert [event["args"] for event in audio_events] == [{"bytes": 6400}] * 3
    assert result["otherData"]["response_latency_ms"]["recv"]["last_ms"] == 2000.0
    if direct_webrtc:
        assert result["otherData"]["recv_latency_semantics"] == (
            "recv marks local receipt of speech metadata over WebSocket, "
            "not server publication or browser media playback"
        )
