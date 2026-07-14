"""Unit tests for OjinSTVClient (async lifecycle, receive, send, playback)."""

import asyncio

import cv2
import numpy as np

from ojin.entities.interaction_messages import ErrorResponse, ErrorResponseMessage
from ojin.ojin_client_messages import (
    FrameType,
    OjinAudioInputMessage,
    OjinCancelInteractionMessage,
    OjinInteractionResponseMessage,
)
from ojin.stv.config import STVConfig
from ojin.stv.events import STVEvent
from ojin.stv.ojin_stv_client import OjinSTVClient
from ojin.stv.synchronizer import AudioBuffer
from tests.stv.fakes import FakeOjinClient, ListOutput, RecordingTracer


def make_client(**config_overrides):
    """Build a client wired to in-memory fakes (config overrides optional)."""
    fc = FakeOjinClient()
    out = ListOutput()
    c = OjinSTVClient(
        client=fc,
        output=out,
        tracer=RecordingTracer(),
        config=STVConfig(
            loop_stall_watchdog_ms=0, stall_probe_ms=0, **config_overrides
        ),
    )
    return c, fc, out


# ----------------------------------------------------------------------
# Task 13: lifecycle + receive + send
# ----------------------------------------------------------------------


def test_start_emits_session_ready_and_seeds() -> None:
    """Start connects, fires SESSION_READY with session_data, and seeds silence."""

    async def run() -> None:
        c, fc, _out = make_client()
        ready = []

        def on_ready(**kwargs):
            ready.append(kwargs)

        c.add_listener(STVEvent.SESSION_READY, on_ready)
        await c.start()
        await asyncio.sleep(0.05)
        assert ready and ready[0]["session_data"] == {"persona": "test"}
        assert any(isinstance(m, OjinAudioInputMessage) for m in fc.sent)  # seed
        assert c.is_connected is True
        await c.close()

    asyncio.run(run())


def test_send_tts_audio_buffers_original_and_sends_resampled() -> None:
    """With batching off, send_tts_audio buffers the original and sends each chunk."""

    async def run() -> None:
        c, fc, _out = make_client(server_feed_batching_enabled=False)
        await c.start()
        await asyncio.sleep(0.02)
        fc.sent.clear()
        await c.start_turn()
        await c.send_tts_audio(b"\x01\x02" * 640, 16000, 1)  # 40 ms @ 16 kHz (identity)
        assert c._synchronizer.audio_buffers[-1].bytes_  # original buffered
        audio = [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        assert len(audio) == 1 and len(audio[0].audio_int16_bytes) == 1280
        await c.close()

    asyncio.run(run())


def test_batching_combines_to_initial_then_min() -> None:
    """Batching emits one ~1000 ms initial chunk, then ~400 ms min chunks."""

    async def run() -> None:
        c, fc, _out = make_client(
            server_feed_initial_chunk_ms=1000, server_feed_min_chunk_ms=400
        )
        await c.start()
        await asyncio.sleep(0.02)
        fc.sent.clear()
        await c.start_turn()
        frame = b"\x01\x02" * 640  # 40 ms @ 16 kHz = 1280 bytes, non-silent
        for _ in range(25):  # 25 * 40 ms = 1000 ms = initial threshold
            await c.send_tts_audio(frame, 16000, 1)
        audio = [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        assert len(audio) == 1
        assert len(audio[0].audio_int16_bytes) == 32000  # one 1000 ms batch
        for _ in range(10):  # 10 * 40 ms = 400 ms = min threshold
            await c.send_tts_audio(frame, 16000, 1)
        audio = [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        assert len(audio) == 2
        assert len(audio[1].audio_int16_bytes) == 12800  # one 400 ms batch
        await c.close()

    asyncio.run(run())


def test_send_discards_half_second_trailing_silence() -> None:
    """The 0.5 s all-zero TTS sentinel is discarded (not buffered or sent)."""

    async def run() -> None:
        c, fc, _out = make_client()
        await c.start()
        await asyncio.sleep(0.02)
        fc.sent.clear()
        await c.start_turn()
        silence = b"\x00" * (16000 * 1 * 2 // 2)  # 0.5 s @ 16 kHz
        await c.send_tts_audio(silence, 16000, 1)
        assert not fc.sent
        assert not c._synchronizer.audio_buffers[-1].bytes_
        await c.close()

    asyncio.run(run())


def test_error_message_emits_error_event() -> None:
    """A server error message surfaces as an ERROR event."""

    async def run() -> None:
        c, fc, _out = make_client()
        errors = []

        def on_error(**kwargs):
            errors.append(kwargs)

        c.add_listener(STVEvent.ERROR, on_error)
        await c.start()
        await asyncio.sleep(0.02)
        await fc.push(
            ErrorResponseMessage(
                payload=ErrorResponse(
                    interaction_id=None,
                    code="BOOM",
                    message="bad",
                    timestamp=0,
                    details=None,
                )
            )
        )
        await asyncio.sleep(0.05)
        assert errors and errors[0]["fatal"] is True
        assert errors[0]["message"] == "bad" and errors[0]["code"] == "BOOM"
        await c.close()

    asyncio.run(run())


# ----------------------------------------------------------------------
# Task 14: playback tick + interrupt
# ----------------------------------------------------------------------


def test_emit_tick_always_emits_one_audio_frame() -> None:
    """A single tick emits exactly one audio frame even with no buffer (silence)."""

    async def run() -> None:
        c, _fc, out = make_client()
        await c._emit_tick()  # driven directly; no real loop running
        assert len(out.audio) == 1
        assert out.audio[0].pcm == b"\x00" * (int(16000 / 25) * 2)  # silence chunk

    asyncio.run(run())


def test_interrupt_sends_cancel_only_when_interruptible() -> None:
    """Interrupt sends a cancel only when a live buffer is playing."""

    async def run() -> None:
        c, fc, _out = make_client()
        await c.start()
        await asyncio.sleep(0.02)
        fc.sent.clear()
        await c.interrupt()  # idle → no cancel
        assert not any(isinstance(m, OjinCancelInteractionMessage) for m in fc.sent)
        c._synchronizer.current_buffer = AudioBuffer()
        c._synchronizer.current_buffer.bytes_.extend(b"\x01\x02" * 100)
        await c.interrupt()
        assert any(isinstance(m, OjinCancelInteractionMessage) for m in fc.sent)
        await c.close()

    asyncio.run(run())


def _interaction_response(frame_type):
    """Build a minimal server interaction-response message of the given type."""
    return OjinInteractionResponseMessage(
        interaction_id="i1",
        video_frame_bytes=b"",
        audio_frame_bytes=b"\x00\x00" * 320,
        is_final_response=False,
        index=0,
        frame_type=frame_type,
    )


def _live_buffer():
    """Build a fresh, non-interrupted buffer with audio to fade (interruptible)."""
    buf = AudioBuffer(sample_rate=16000)
    buf.bytes_.extend(b"\x01\x02" * 100)
    return buf


def test_interrupt_suppressed_while_one_is_ongoing() -> None:
    """A second barge-in is dropped while an interruption is still in flight."""

    async def run() -> None:
        c, fc, _out = make_client()
        c._initialized = True  # a live turn only exists once the session is ready
        c._synchronizer.current_buffer = _live_buffer()
        await c.interrupt()
        assert c._interruption_ongoing is True
        cancels = [m for m in fc.sent if isinstance(m, OjinCancelInteractionMessage)]
        assert len(cancels) == 1

        # A new turn may swap in a fresh, interruptible buffer, but the prior cancel
        # is still unacknowledged — a second barge-in must not fire another cancel.
        c._synchronizer.current_buffer = _live_buffer()
        await c.interrupt()
        cancels = [m for m in fc.sent if isinstance(m, OjinCancelInteractionMessage)]
        assert len(cancels) == 1, "second interrupt should not send another cancel"

        # In-flight speech frames do not end the interruption window.
        c._on_interaction_response(_interaction_response(FrameType.SPEECH))
        assert c._interruption_ongoing is True

    asyncio.run(run())


def test_interrupt_window_closes_on_idle_or_fadeout_frame() -> None:
    """The first idle/fade-out frame after a cancel reopens interruptibility."""

    async def run() -> None:
        for end_frame in (FrameType.IDLE, FrameType.FADE_OUT):
            c, fc, _out = make_client()
            c._initialized = True  # a live turn only exists once the session is ready
            c._synchronizer.current_buffer = _live_buffer()
            await c.interrupt()
            assert c._interruption_ongoing is True

            c._on_interaction_response(_interaction_response(end_frame))
            assert c._interruption_ongoing is False

            # Window closed → a fresh barge-in sends a new cancel.
            c._synchronizer.current_buffer = _live_buffer()
            await c.interrupt()
            cancels = [
                m for m in fc.sent if isinstance(m, OjinCancelInteractionMessage)
            ]
            assert len(cancels) == 2

    asyncio.run(run())


def test_turn_opened_during_interruption_is_deferred_then_replayed() -> None:
    """A turn opened mid-barge-in is buffered, not fed, then replayed on window close.

    The server discards audio sent while it is still cancelling, so feeding the new
    turn now would desync playback. Deferred input is replayed in order once the
    idle/fade frame clears the window.
    """

    async def run() -> None:
        c, fc, _out = make_client(server_feed_batching_enabled=False)
        c._initialized = True  # a live turn only exists once the session is ready
        c._synchronizer.current_buffer = _live_buffer()
        await c.interrupt()
        assert c._interruption_ongoing is True
        fc.sent.clear()

        # New turn + its audio open while the cancel is still unacknowledged.
        await c.start_turn()
        assert c._deferring_input is True
        await c.send_tts_audio(b"\x01\x02" * 640, 16000, 1)
        await c.send_tts_audio(b"\x03\x04" * 640, 16000, 1)
        # Nothing opened for playback, nothing fed to the server — all buffered.
        assert len(c._interrupt_deferred) == 3  # turn + 2 audio ops
        assert not [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]

        # Server's idle frame closes the window → deferred input replays (in order).
        await c._handle_message(_interaction_response(FrameType.IDLE))
        assert c._interruption_ongoing is False
        assert c._deferring_input is False
        assert c._interrupt_deferred == []
        sent_audio = [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        assert len(sent_audio) == 2  # both audio ops fed, in order
        assert c._synchronizer.audio_buffers[-1].bytes_  # turn buffered for playback

    asyncio.run(run())


def test_trailing_audio_of_interrupted_turn_is_not_deferred() -> None:
    """Audio for the cancelled turn (no new start_turn) is dropped, not buffered.

    Deferral begins only at a NEW turn's start_turn during the window; the
    interrupted turn's own straggler audio must still be dropped (it has no buffer),
    never replayed after the barge-in.
    """

    async def run() -> None:
        c, fc, _out = make_client(server_feed_batching_enabled=False)
        c._initialized = True
        c._synchronizer.current_buffer = _live_buffer()
        await c.interrupt()
        fc.sent.clear()

        assert c._deferring_input is False  # no new turn opened
        await c.send_tts_audio(b"\x01\x02" * 640, 16000, 1)

        assert c._interrupt_deferred == []  # not buffered
        # Dropped (no buffer for the cancelled turn), not fed to the server.
        assert not [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]

    asyncio.run(run())


def test_lead_cap_gates_sends_and_releases_as_playback_advances() -> None:
    """Audio beyond the lead cap waits client-side and ships as playback catches up.

    Guards the session-7 finding (2026-07-12): shipping a long answer's whole
    audio up front built a server-side backlog that made barge-in cancels take
    seconds to acknowledge. With the cap, only ~cap ms may be in flight ahead of
    playback; the rest is released by the feeder as ticks consume audio.
    """

    async def run() -> None:
        c, fc, _out = make_client(
            server_feed_batching_enabled=False, server_feed_max_lead_ms=100
        )
        await c.start()
        await asyncio.sleep(0.05)
        fc.sent.clear()
        await c.start_turn()

        chunk = b"\x01\x02" * 1280  # 80 ms @ 16 kHz mono (identity resample)
        for _ in range(3):
            await c.send_tts_audio(chunk, 16000, 1)

        # First send: lead 0 < 100 → out. Second: lead 80 < 100 → out (one payload
        # may overshoot the cap). Third: lead 160 >= 100 → gated.
        sent = [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        assert len(sent) == 2
        assert len(c._feed_pending) == 1

        # Playback consumes 200 ms → the feeder releases the gated payload.
        c._played_real_ms += 200.0
        c._feed_wake.set()
        await asyncio.sleep(0.05)
        sent = [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        assert len(sent) == 3
        assert not c._feed_pending
        await c.close()

    asyncio.run(run())


def test_interrupt_discards_lead_gated_audio_and_levels_the_feed() -> None:
    """A barge-in drops gated payloads and resets the lead to the played position."""

    async def run() -> None:
        c, fc, _out = make_client(
            server_feed_batching_enabled=False, server_feed_max_lead_ms=100
        )
        c._initialized = True
        c._synchronizer.current_buffer = _live_buffer()
        c._server_fed_ms = 500.0
        c._played_real_ms = 120.0
        c._feed_pending.append(b"\x01\x02" * 100)

        await c.interrupt()

        assert not c._feed_pending  # cancelled turn's gated audio discarded
        assert c._server_lead_ms() == 0.0  # feed restarts level with playback
        # Nothing but the cancel went to the server.
        assert not [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]

    asyncio.run(run())


def test_close_flushes_lead_gated_pending() -> None:
    """close() best-effort flushes gated payloads before tearing down."""

    async def run() -> None:
        c, fc, _out = make_client(
            server_feed_batching_enabled=False, server_feed_max_lead_ms=100
        )
        await c.start()
        await asyncio.sleep(0.05)
        fc.sent.clear()
        c._feed_pending.append(b"\x01\x02" * 100)
        c._server_fed_ms = 1000.0  # over cap → would never send while live

        await c.close()

        sent = [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        assert len(sent) == 1  # flushed on close despite the cap

    asyncio.run(run())


def test_emit_tick_drains_lead_on_silence_ticks() -> None:
    """A silent tick still advances the server-feed drain so the gate can't latch.

    Regression: gating the drain on real-audio-present let the lead freeze once the
    cap engaged — the gate withholds audio, the server starves, no speech frames
    come back, local playback underruns, the drain stops, and the gate never
    reopens (a self-sustaining deadlock). Every tick advances one frame of drain
    and, with a payload gated, wakes the feeder to re-check the lead.
    """

    async def run() -> None:
        c, _fc, _out = make_client(server_feed_max_lead_ms=100)
        c._feed_pending.append(b"\x00\x00")  # a payload stuck behind a stale lead
        c._feed_wake.clear()
        before = c._played_real_ms

        await c._emit_tick()  # empty synchronizer -> silence tick (audio_chunk None)

        assert c._played_real_ms == before + 1000.0 / c._config.fps
        assert c._feed_wake.is_set()  # feeder re-checks the now-lower lead

    asyncio.run(run())


def test_start_turn_relevels_lead_and_releases_gated_backlog() -> None:
    """A new turn clears a stale gated backlog and re-levels the lead to playback.

    Regression: if the previous turn left the lead stuck above the cap (its drain
    and the server's real consumption diverged), every later turn's audio was gated
    behind it and the session went silent. start_turn reconciles fed == played and
    drops the stale backlog, mirroring interrupt().
    """

    async def run() -> None:
        c, fc, _out = make_client(
            server_feed_batching_enabled=False, server_feed_max_lead_ms=100
        )
        c._initialized = True
        c._server_fed_ms = 5000.0  # stuck far ahead of playback
        c._played_real_ms = 100.0
        c._feed_pending.append(b"\x01\x02" * 100)  # prior turn's stranded payload

        await c.start_turn()

        assert not c._feed_pending  # stale backlog dropped at the turn boundary
        assert c._server_lead_ms() == 0.0  # feed re-leveled to playback
        # Only the turn opened — no stale audio (and no cancel) went to the server.
        assert not [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]

    asyncio.run(run())


def test_emit_tick_fires_started_speaking_event() -> None:
    """When a buffer begins draining, the tick emits BOT_STARTED_SPEAKING."""

    async def run() -> None:
        c, _fc, out = make_client()
        started = []

        def on_started(**kwargs):
            started.append(1)

        c.add_listener(STVEvent.BOT_STARTED_SPEAKING, on_started)
        buf = AudioBuffer(sample_rate=16000)
        buf.bytes_.extend(b"\x01\x02" * 2000)
        c._synchronizer.current_buffer = buf
        await c._emit_tick()
        assert started == [1]
        assert out.audio[0].pcm != b"\x00" * len(out.audio[0].pcm)  # real audio

    asyncio.run(run())


def test_close_tears_down() -> None:
    """Close stops the client and closes the transport."""

    async def run() -> None:
        c, fc, _out = make_client()
        await c.start()
        await asyncio.sleep(0.02)
        await c.close()
        assert c.is_connected is False
        assert fc.closed is True

    asyncio.run(run())


def test_connect_failure_emits_fatal_error() -> None:
    """If the transport never connects, start emits a fatal ERROR and cleans up."""

    async def run() -> None:
        fc = FakeOjinClient(raise_on_connect=True)
        out = ListOutput()
        c = OjinSTVClient(
            client=fc,
            output=out,
            tracer=RecordingTracer(),
            config=STVConfig(
                client_connect_max_retries=2,
                client_reconnect_delay=0.0,
                loop_stall_watchdog_ms=0,
                stall_probe_ms=0,
            ),
        )
        errors = []

        def on_error(**kwargs):
            errors.append(kwargs)

        c.add_listener(STVEvent.ERROR, on_error)
        await c.start()
        assert errors and errors[0]["fatal"] is True
        assert c.is_connected is False

    asyncio.run(run())


def test_interaction_response_renders_decoded_video() -> None:
    """A server speech frame is decoded off-loop and emitted as an RGB video frame."""

    async def run() -> None:
        fc = FakeOjinClient()
        out = ListOutput()
        c = OjinSTVClient(
            client=fc,
            output=out,
            tracer=RecordingTracer(),
            config=STVConfig(
                initial_buffer_frames=0, loop_stall_watchdog_ms=0, stall_probe_ms=0
            ),
        )
        await c.start()
        await asyncio.sleep(0.02)
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        img[:, :, 2] = 255  # red in BGR
        jpeg = cv2.imencode(".jpg", img)[1].tobytes()
        await fc.push(
            OjinInteractionResponseMessage(
                interaction_id="i1",
                video_frame_bytes=jpeg,
                audio_frame_bytes=b"\x10\x00" * 320,
                is_final_response=False,
                index=0,
                frame_type=FrameType.SPEECH,
            )
        )
        await asyncio.sleep(0.2)  # decode worker + a few playback ticks
        assert out.video, "expected a decoded video frame to be emitted"
        # The frame is emitted at the server's native size (the 64x48 JPEG above),
        # not resized — image_size is gone.
        vf = out.video[0]
        assert vf.rgb is not None and len(vf.rgb) == 64 * 48 * 3
        assert (vf.width, vf.height) == (64, 48)
        await c.close()

    asyncio.run(run())


def test_emit_tick_records_rich_counters_and_lanes() -> None:
    """A speaking tick records the OjinVideoService-parity counters + play_audio."""

    async def run() -> None:
        fc = FakeOjinClient()
        tr = RecordingTracer()
        c = OjinSTVClient(
            client=fc,
            output=ListOutput(),
            tracer=tr,
            config=STVConfig(loop_stall_watchdog_ms=0, stall_probe_ms=0),
        )
        buf = AudioBuffer(sample_rate=16000)
        buf.bytes_.extend(b"\x01\x02" * 2000)
        c._synchronizer.current_buffer = buf
        await c._emit_tick()
        counters = {name for (name, _v) in tr.counters}
        assert {
            "current_buffer_ms",
            "queued_buffers",
            "pending_video_frames",
            "playback_fps",
            "output_audio_rms",
            "audio_underruns_total",
            "idle_backlog_skips_total",
            "loop_lag_ms",
            "tick_work_ms",
            "recv_decode_in",
            "recv_decode_out",
        } <= counters
        assert any(lane == "play_audio" for (lane, _n, _a) in tr.instants)

    asyncio.run(run())


def test_emit_tick_forwards_client_pipeline_depths() -> None:
    """Receive-pipeline depth gauges from the client are emitted with a recv_ prefix."""

    async def run() -> None:
        fc = FakeOjinClient()
        fc.queue_depths = {"ws_frames": 5, "ws_paused": 1, "server_msgs": 3}
        tr = RecordingTracer()
        c = OjinSTVClient(
            client=fc,
            output=ListOutput(),
            tracer=tr,
            config=STVConfig(loop_stall_watchdog_ms=0, stall_probe_ms=0),
        )
        buf = AudioBuffer(sample_rate=16000)
        buf.bytes_.extend(b"\x01\x02" * 2000)
        c._synchronizer.current_buffer = buf
        await c._emit_tick()
        depths = dict(tr.counters)
        assert depths["recv_ws_frames"] == 5
        assert depths["recv_ws_paused"] == 1
        assert depths["recv_server_msgs"] == 3
        # Owned-queue gauges are always present, independent of the client probe.
        assert "recv_decode_in" in depths
        assert "recv_decode_out" in depths

    asyncio.run(run())


def test_send_tts_audio_records_input_rms() -> None:
    """send_tts_audio records the tts_audio marker and input_audio_rms counter."""

    async def run() -> None:
        fc = FakeOjinClient()
        tr = RecordingTracer()
        c = OjinSTVClient(
            client=fc,
            output=ListOutput(),
            tracer=tr,
            config=STVConfig(loop_stall_watchdog_ms=0, stall_probe_ms=0),
        )
        await c.start()
        await asyncio.sleep(0.02)
        await c.start_turn()
        # 100 ms @ 24 kHz: enough to clear the streaming resampler's filter
        # warm-up (a single ~20 ms chunk is fully absorbed and emits nothing).
        await c.send_tts_audio(b"\x05\x06" * 2400, 24000, 1)
        assert any(name == "input_audio_rms" for (name, _v) in tr.counters)
        assert any(n == "tts_audio" for (_lane, n, _a) in tr.instants)
        await c.close()

    asyncio.run(run())


def test_null_tracer_disables_per_tick_bookkeeping() -> None:
    """With the default NullTracer, per-tick trace bookkeeping stays inert."""

    async def run() -> None:
        fc = FakeOjinClient()
        c = OjinSTVClient(
            client=fc,
            output=ListOutput(),
            config=STVConfig(loop_stall_watchdog_ms=0, stall_probe_ms=0),
        )  # default NullTracer
        assert c._tracing is False
        buf = AudioBuffer(sample_rate=16000)
        buf.bytes_.extend(b"\x01\x02" * 2000)
        c._synchronizer.current_buffer = buf
        await c._emit_tick()
        assert not c._tr_emit_times  # never appended → no unbounded growth

    asyncio.run(run())


def test_idle_timeout_flushes_subthreshold_tail() -> None:
    """A turn shorter than the initial threshold flushes its tail after idle."""

    async def run() -> None:
        c, fc, _out = make_client(
            server_feed_initial_chunk_ms=1000,
            server_feed_min_chunk_ms=400,
            server_feed_flush_idle_ms=60,
        )
        await c.start()
        await asyncio.sleep(0.02)
        fc.sent.clear()
        await c.start_turn()
        frame = b"\x01\x02" * 640  # 40 ms @ 16 kHz
        for _ in range(5):  # 200 ms < 1000 ms initial → no size send
            await c.send_tts_audio(frame, 16000, 1)
        assert not [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        await asyncio.sleep(0.15)  # > flush_idle (60 ms) → idle flush fires
        audio = [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        assert len(audio) == 1
        assert len(audio[0].audio_int16_bytes) == 6400  # 5 * 1280 = 200 ms tail
        await c.close()

    asyncio.run(run())


def test_start_turn_flushes_previous_tail_and_rearms_initial() -> None:
    """start_turn sends the prior turn's tail, then requires a fresh initial chunk."""

    async def run() -> None:
        c, fc, _out = make_client(
            server_feed_initial_chunk_ms=1000,
            server_feed_min_chunk_ms=400,
            server_feed_flush_idle_ms=10000,  # large: isolate from the idle flush
        )
        await c.start()
        await asyncio.sleep(0.02)
        fc.sent.clear()
        await c.start_turn()
        frame = b"\x01\x02" * 640
        for _ in range(5):  # 200 ms tail, below initial → not size-sent
            await c.send_tts_audio(frame, 16000, 1)
        assert not [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        await c.start_turn()  # flushes the 200 ms tail, re-arms initial
        audio = [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        assert len(audio) == 1 and len(audio[0].audio_int16_bytes) == 6400
        for _ in range(10):  # 400 ms — would hit min, but initial is armed → no send
            await c.send_tts_audio(frame, 16000, 1)
        audio = [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        assert len(audio) == 1  # still just the flushed tail
        await c.close()

    asyncio.run(run())


def test_start_turn_discards_resampler_tail() -> None:
    """A new turn resets the resampler but never sends the stale tail.

    The streaming resampler holds back a filter-delay tail from audio fed
    seconds ago. Sending it at the next ``start_turn`` lands it at the HEAD of
    the new turn's server feed, where the server renders it as near-zero speech
    frames the local buffer does not have — a constant video-late offset for
    the whole turn (measured on staging: every natural-turn entry was offset by
    exactly the tail's duration). The boundary must flush the filter state so
    the new turn starts clean, but the stale bytes are discarded — mirroring
    the interrupt path.
    """
    tail_marker = b"\xab\xcd" * 8

    class _TailResampler:
        def __init__(self) -> None:
            self.pending = False
            self.flushes = 0

        async def resample(self, pcm: bytes, in_rate: int, out_rate: int) -> bytes:
            self.pending = True  # a tail becomes available once audio is fed
            return pcm

        def flush(self) -> bytes:
            self.flushes += 1
            if self.pending:
                self.pending = False
                return tail_marker
            return b""

    async def run() -> None:
        fc = FakeOjinClient()
        rs = _TailResampler()
        c = OjinSTVClient(
            client=fc,
            output=ListOutput(),
            tracer=RecordingTracer(),
            resampler=rs,
            config=STVConfig(
                loop_stall_watchdog_ms=0,
                stall_probe_ms=0,
                server_feed_batching_enabled=False,
            ),
        )
        await c.start()
        await asyncio.sleep(0.02)
        await c.start_turn()
        await c.send_tts_audio(b"\x03\x04" * 480, 24000, 1)  # 20 ms @ 24 kHz
        fc.sent.clear()
        await c.start_turn()  # turn boundary → reset the filter, drop the tail
        sent = [
            m.audio_int16_bytes for m in fc.sent if isinstance(m, OjinAudioInputMessage)
        ]
        assert rs.flushes >= 1  # the filter state was still reset at the boundary
        assert tail_marker not in sent, f"stale tail leaked into the new turn: {sent}"
        assert not sent  # nothing else should be sent by an empty-batch boundary
        await c.close()

    asyncio.run(run())


def test_interrupt_discards_pending_batch() -> None:
    """Barge-in throws away un-sent audio of the cancelled turn."""

    async def run() -> None:
        c, fc, _out = make_client(
            server_feed_initial_chunk_ms=1000, server_feed_flush_idle_ms=10000
        )
        await c.start()
        await asyncio.sleep(0.02)
        c._synchronizer.current_buffer = AudioBuffer(sample_rate=16000)
        c._synchronizer.current_buffer.bytes_.extend(b"\x01\x02" * 100)  # interruptible
        fc.sent.clear()
        await c.start_turn()
        frame = b"\x01\x02" * 640
        for _ in range(5):  # 200 ms pending in the batcher
            await c.send_tts_audio(frame, 16000, 1)
        assert c._batcher.pending_bytes == 6400
        await c.interrupt()
        assert c._batcher.pending_bytes == 0  # discarded
        audio = [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        assert not audio  # the cancelled turn's audio was never sent
        await c.close()

    asyncio.run(run())


def test_close_flushes_final_tail() -> None:
    """Close best-effort flushes a buffered tail before the transport closes."""

    async def run() -> None:
        c, fc, _out = make_client(
            server_feed_initial_chunk_ms=1000, server_feed_flush_idle_ms=10000
        )
        await c.start()
        await asyncio.sleep(0.02)
        fc.sent.clear()
        await c.start_turn()
        frame = b"\x01\x02" * 640
        for _ in range(3):  # 120 ms tail, below initial and idle window won't fire
            await c.send_tts_audio(frame, 16000, 1)
        assert not [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        await c.close()
        audio = [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        assert len(audio) == 1 and len(audio[0].audio_int16_bytes) == 3840  # 3 * 1280

    asyncio.run(run())


def test_close_ends_output_stream() -> None:
    """With the default QueueOutput, close() terminates a live output_stream()."""

    async def run() -> None:
        fc = FakeOjinClient()
        c = OjinSTVClient(
            client=fc,
            tracer=RecordingTracer(),
            config=STVConfig(loop_stall_watchdog_ms=0, stall_probe_ms=0),
        )  # default output = QueueOutput
        await c.start()
        await asyncio.sleep(0.02)

        items = []

        async def consume() -> None:
            items.extend([frame async for frame in c.output_stream()])

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        await c.close()
        await asyncio.wait_for(task, 1.0)  # consumer returns when the stream ends
        assert items  # some frames were emitted before close

    asyncio.run(run())
