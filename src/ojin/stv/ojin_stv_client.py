"""High-level Ojin Speech-To-Video client (framework-agnostic).

``OjinSTVClient`` packs everything pipecat's ``OjinVideoService`` did — connection
lifecycle, TTS-audio playback buffering, the audio-as-clock playback loop,
post-interruption audio/video re-sync, off-loop JPEG decode, session tracing, and
loop diagnostics — behind an anam-inspired API, with no pipecat dependency.

The avatar plays the **original** TTS audio buffered here; only a separate copy is
resampled and sent to the inference server for lip-sync. Cross-cutting concerns
(transport, resampler, decoder, tracer, output) are injected behind protocols, so
the WebSocket transport can later be swapped for WebRTC without touching this class.

Typical use::

    async with OjinSTVClient(api_key=..., config_id=...) as client:
        await client.start_turn()
        await client.send_tts_audio(pcm, 24000, 1)
        async for frame in client.output_stream():
            render(frame)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import queue
import threading
import time
from collections import deque
from typing import AsyncIterator, Callable, Optional

from pydantic import BaseModel

from ojin.entities.interaction_messages import ErrorResponseMessage
from ojin.ojin_client import OjinClient
from ojin.ojin_client_messages import (
    FrameType,
    IOjinClient,
    OjinAudioInputMessage,
    OjinCancelInteractionMessage,
    OjinInteractionResponseMessage,
    OjinSessionReadyMessage,
)
from ojin.stv.audio_utils import rms_int16
from ojin.stv.config import STVConfig
from ojin.stv.diagnostics import LoopDiagnostics
from ojin.stv.events import EventEmitter, STVEvent
from ojin.stv.frames import STVAudioFrame, STVVideoFrame
from ojin.stv.output import QueueOutput, STVOutput
from ojin.stv.resampler import Resampler, default_resampler
from ojin.stv.send_batcher import SendBatcher
from ojin.stv.synchronizer import (
    BYTES_PER_FRAME,
    OJIN_PERSONA_SAMPLE_RATE,
    Synchronizer,
    TickResult,
    VideoFrame,
)
from ojin.stv.tracing import (
    NullTracer,
    Tracer,
    play_lane_for_frame_type,
    recv_lane_for_frame_type,
)
from ojin.stv.video_decode import OpenCVDecoder, VideoDecoder

logger = logging.getLogger(__name__)

_DEFAULT_WS_URL = "wss://models.ojin.ai/realtime"
_HALF_SECOND = 0.5
_HALF_SECOND_TOL = 0.01
_ONE_SEC_US = 1_000_000  # rolling window for the playback_fps counter
_BYTES_PER_MS_16K = OJIN_PERSONA_SAMPLE_RATE * 2 / 1000.0  # 32 B/ms mono int16


class OjinSTVClient:
    """Drives one Ojin STV session: feed TTS audio in, get synced A/V out."""

    def __init__(  # noqa: PLR0913, PLR0915 — injectable deps + flat field init
        self,
        *,
        api_key: str = "",
        config_id: str = "",
        ws_url: str = _DEFAULT_WS_URL,
        output: Optional[STVOutput] = None,
        resampler: Optional[Resampler] = None,
        decoder: Optional[VideoDecoder] = None,
        tracer: Optional[Tracer] = None,
        client: Optional[IOjinClient] = None,
        config: Optional[STVConfig] = None,
        buffer_preinit_tts_audio: bool = True,
    ) -> None:
        """Create a client; pass your own transport/resampler/decoder/tracer/output.

        Defaults make it work standalone: a WebSocket ``OjinClient``, the bundled
        resampler, an opencv decoder, a :class:`QueueOutput` (drained via
        :meth:`output_stream`), and a no-op tracer. Emitted video frames carry
        whatever resolution the server sends (see :class:`STVVideoFrame`).

        When ``buffer_preinit_tts_audio`` is ``True`` (default), input turns and
        TTS audio sent via :meth:`start_turn` / :meth:`send_tts_audio` before the
        session is ready (``SESSION_READY``) are queued in order and replayed once
        it is, instead of being dropped — so a caller can start speaking during the
        cold-start handshake (e.g. an opening line) without losing audio. Set it to
        ``False`` to restore the previous behaviour: input before initialization is
        dropped with a warning.
        """
        super().__init__()
        self._config = config or STVConfig()

        self._client: IOjinClient = client or OjinClient(
            ws_url=ws_url,
            api_key=api_key,
            config_id=config_id,
            mode=os.getenv("OJIN_MODE", ""),
            # Transport-level send pacing: cap a single send and gap a backlog so a
            # buffered burst (e.g. TTS replayed after a barge-in) can't flood.
            max_input_chunk_bytes=max(1, int(self._config.server_feed_max_chunk_bytes)),
            send_chunk_gap_s=self._config.server_feed_send_gap_ms / 1000.0,
        )
        self._resampler: Resampler = resampler or default_resampler()
        self._decoder: VideoDecoder = decoder or OpenCVDecoder()
        self._tracer: Tracer = tracer or NullTracer()
        self._output: STVOutput = output or QueueOutput()
        self._events = EventEmitter()
        self._synchronizer = Synchronizer(self._config)
        self._batcher = SendBatcher(
            initial_chunk_bytes=int(
                self._config.server_feed_initial_chunk_ms * _BYTES_PER_MS_16K
            ),
            min_chunk_bytes=int(
                self._config.server_feed_min_chunk_ms * _BYTES_PER_MS_16K
            ),
            flush_idle_s=self._config.server_feed_flush_idle_ms / 1000.0,
        )
        self._batch_added = asyncio.Event()
        self._batch_flush_task: Optional[asyncio.Task] = None

        # Server-feed lead cap (see STVConfig.server_feed_max_lead_ms). The gate
        # compares how much audio has been shipped to the server (_server_fed_ms)
        # against how much real audio playback has consumed (_played_real_ms);
        # payloads that would push the lead past the cap wait in _feed_pending and
        # are released by _server_feed_loop as ticks advance playback. All
        # server-bound payloads are 16 kHz mono int16, so ms = bytes / 32.
        self._server_feed_max_lead_ms = float(
            max(0, self._config.server_feed_max_lead_ms)
        )
        self._server_fed_ms = 0.0
        self._played_real_ms = 0.0
        self._feed_pending: "deque[bytes]" = deque()
        self._feed_wake = asyncio.Event()
        self._feed_task: Optional[asyncio.Task] = None

        self._initialized = False
        self._session_data: Optional[dict] = None
        self._waiting_for_first_tts = False

        # Pre-initialization input buffer: when enabled, start_turn / send_tts_audio
        # calls that arrive before SESSION_READY are recorded here (in order) and
        # replayed once the session is ready, instead of being dropped. Entries are
        # ("turn",) or ("audio", pcm, sample_rate, num_channels).
        self._buffer_preinit_tts_audio = buffer_preinit_tts_audio
        self._preinit_inputs: list[tuple] = []
        self._last_audio_shape = (OJIN_PERSONA_SAMPLE_RATE, 1)
        self._last_video_frame: Optional[VideoFrame] = None
        self._last_played_rgb: Optional[bytes] = None
        self._last_played_size: Optional[tuple[int, int]] = None  # native (w, h)
        # Barge-in in flight: set when a cancel is sent, cleared when the server's
        # first idle/fade-out frame acknowledges it. While set, further interrupts
        # are dropped so a re-fire (e.g. VAD flap, or a new turn that swapped in
        # before the cancel landed) can't stack a second cancel on the server.
        self._interruption_ongoing = False

        # Deferred input during a barge-in. A NEW turn that opens (start_turn) while a
        # cancel is still settling server-side must not be fed yet — the server would
        # discard it, desyncing playback. Instead we buffer that turn's start + audio
        # here (in order) and replay it once the server's idle/fade frame clears the
        # window (see _flush_interrupt_deferred). Deferral begins at the new turn's
        # start_turn (NOT at interrupt: the interrupted turn's own trailing audio must
        # still be dropped, not replayed). Entries mirror _preinit_inputs:
        # ("turn",) or ("audio", pcm, sample_rate, num_channels).
        self._interrupt_deferred: list[tuple] = []
        self._deferring_input = False

        # Per-tick trace bookkeeping (only touched when a real tracer is injected,
        # so a NullTracer session pays nothing). Mirrors OjinVideoService.
        self._tracing = not isinstance(self._tracer, NullTracer)
        self._tr_emit_times: "deque[float]" = deque()
        self._tr_underruns = 0
        self._tr_idle_skips = 0
        self._tr_catchup_drops = 0
        self._prev_tick_perf = 0.0
        # (cumulative ingress bytes, perf_counter) at the previous tick — the
        # delta between ticks drives the ingress-bandwidth counter lane.
        self._tr_prev_ingress: "tuple[int, float] | None" = None

        # Off-loop JPEG decode pipeline.
        self._decode_in: "queue.Queue[Optional[VideoFrame]]" = queue.Queue()
        self._decode_out: "queue.Queue[VideoFrame]" = queue.Queue()
        self._decode_thread: Optional[threading.Thread] = None

        # Tasks + diagnostics.
        self._receive_task: Optional[asyncio.Task] = None
        self._playback_task: Optional[asyncio.Task] = None
        self._diagnostics = LoopDiagnostics(
            watchdog_ms=self._config.loop_stall_watchdog_ms,
            tick_warn_ms=self._config.tick_warn_ms,
            stall_probe_ms=self._config.stall_probe_ms,
        )

        # Trace anchors (µs marks).
        self._tr_session_start = 0.0
        self._tr_connect_start = 0.0
        self._tr_speaking_start: Optional[float] = None
        self._tr_interrupt_start: Optional[float] = None
        self._tr_first_tts_audio_at: Optional[float] = None
        self._awaiting_first_recv_video = False
        self._awaiting_first_played_video = False

    # ------------------------------------------------------------------
    # Introspection + events
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Whether the session is live (SessionReady received, not closed)."""
        return self._initialized

    @property
    def session_data(self) -> Optional[dict]:
        """Server-provided session parameters, available after SESSION_READY."""
        return self._session_data

    def on(self, event: STVEvent) -> Callable:
        """Register an event handler via decorator (see :class:`STVEvent`)."""
        return self._events.on(event)

    def add_listener(self, event: STVEvent, cb: Callable) -> None:
        """Register an event handler programmatically."""
        self._events.add_listener(event, cb)

    def remove_listener(self, event: STVEvent, cb: Callable) -> None:
        """Remove a previously registered event handler."""
        self._events.remove_listener(event, cb)

    def output_stream(self) -> AsyncIterator:
        """Async-iterate emitted frames (requires the default QueueOutput)."""
        if not isinstance(self._output, QueueOutput):
            raise RuntimeError(
                "output_stream() requires the default QueueOutput; a custom "
                "output sink was injected — consume it directly instead."
            )
        return self._output.stream()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect_with_retry(self) -> bool:
        """Connect the transport, retrying up to the configured max attempts."""
        last_error: Optional[Exception] = None
        for attempt in range(self._config.client_connect_max_retries):
            try:
                await self._client.connect()
                logger.info("OjinSTVClient connected")
                return True
            except ConnectionError as exc:  # noqa: PERF203 — bounded retry loop
                last_error = exc
                logger.warning("Connect attempt %d failed: %s", attempt + 1, exc)
                if attempt < self._config.client_connect_max_retries - 1:
                    await asyncio.sleep(self._config.client_reconnect_delay)
        await self._events.emit(
            STVEvent.ERROR,
            message=f"Failed to connect after "
            f"{self._config.client_connect_max_retries} attempts: {last_error}",
            fatal=True,
        )
        return False

    async def start(self) -> None:
        """Connect, start diagnostics + decode worker, and run the receive loop."""
        self._tr_session_start = self._tracer.mark()
        self._tr_connect_start = self._tr_session_start
        self._diagnostics.start()
        self._start_decode_worker()
        if not await self.connect_with_retry():
            await self.close()
            return
        self._receive_task = asyncio.create_task(self._receive_loop())
        await self._client.start_interaction()

    async def __aenter__(self) -> "OjinSTVClient":
        """Context-manager entry: :meth:`start`."""
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Context-manager exit: :meth:`close`."""
        await self.close()

    async def close(self) -> None:
        """Tear down loops, decode worker, and transport; record the session span."""
        was_initialized = self._initialized
        self._initialized = False
        # Drop any input buffered before init; we never became ready to replay it.
        self._preinit_inputs.clear()
        self._diagnostics.stop()
        self._stop_decode_worker()
        if was_initialized:
            self._tracer.span("lifecycle", "session", self._tr_session_start)
        # Stop the batch-flush and lead-gate feeder tasks before the transport
        # closes.
        for task_attr in ("_batch_flush_task", "_feed_task"):
            task = getattr(self, task_attr)
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
                setattr(self, task_attr, None)
        if was_initialized:
            # Flush what is still owed to the server, in order: lead-gated
            # payloads first (older), then the batcher's sub-threshold tail.
            # Best-effort straight sends — the session is ending, the lead cap
            # no longer matters.
            while self._feed_pending:
                with contextlib.suppress(Exception):
                    await self._send_audio_now(self._feed_pending.popleft())
            if self._config.server_feed_batching_enabled:
                pending = self._batcher.drain()
                if pending is not None:
                    with contextlib.suppress(Exception):
                        await self._send_audio_now(pending)
        try:
            await self._client.close()
        except Exception as exc:  # never let teardown raise
            logger.warning("Error closing transport: %s", exc)
        for task in (self._receive_task, self._playback_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self._receive_task = None
        self._playback_task = None
        # End a live output_stream() (default QueueOutput) so consumers iterating
        # the output finish cleanly. Custom sinks without aclose are left alone.
        aclose = getattr(self._output, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(Exception):
                await aclose()
        await self._events.emit(STVEvent.CLOSED)

    # ------------------------------------------------------------------
    # Input — turn + audio + interrupt
    # ------------------------------------------------------------------

    async def start_turn(self) -> None:
        """Open a new audio buffer for the next utterance (≈ TTSStartedFrame)."""
        if not self._initialized and self._buffer_preinit_tts_audio:
            # Defer until the session is ready so this turn boundary is replayed in
            # order with its audio (see _flush_preinit_inputs). Opening a buffer now
            # would strand it: add_audio always targets the tail buffer, so audio
            # replayed later would collapse into the most recent pre-init turn.
            self._preinit_inputs.append(("turn",))
            self._tracer.instant("tts_input", "tts_started_buffered")
            return
        if self._deferring_input or self._interruption_ongoing:
            # A new turn opened while a barge-in is still settling server-side. Buffer
            # its start (and everything that follows) instead of feeding it now — the
            # server would discard audio sent mid-cancel, desyncing playback. Replayed
            # in order once the window clears (see _flush_interrupt_deferred).
            self._deferring_input = True
            self._interrupt_deferred.append(("turn",))
            self._tracer.instant("tts_input", "tts_started_deferred")
            return
        await self._start_turn_core()

    async def _start_turn_core(self) -> None:
        """Open the next turn's buffer and flush the prior turn's tail (no guards).

        The body of :meth:`start_turn` once buffering/deferral guards have passed;
        also the seam :meth:`_flush_interrupt_deferred` replays through, so it must
        never re-check those guards.
        """
        # Flush and DISCARD the previous turn's resampler tail (soxr holds back
        # ~30 ms of filter delay). By start_turn time the previous turn ended
        # seconds ago and its audio has already played; sending the tail now
        # would land it at the HEAD of the new turn's server feed, where the
        # server renders it as 1-2 near-zero speech frames the local buffer does
        # not have — measured as a constant 40-80 ms video-late offset for the
        # whole turn (staging session d65312d758f0: every natural-turn entry was
        # offset by exactly the flushed tail's duration; barge-in turns, whose
        # interrupt path already discards the tail, were clean). Dropping it
        # only shortens the rendered tail of the PREVIOUS turn by ~1 frame —
        # the server pads/fades a turn's end anyway. This mirrors interrupt().
        self._flush_resampler()
        buf = self._synchronizer.open_turn()
        self._waiting_for_first_tts = True
        self._tracer.instant(
            "tts_input", "tts_started", args={"buffer_id": buf.buffer_id}
        )
        if self._config.server_feed_batching_enabled:
            pending = self._batcher.drain()
            if pending:
                await self._send_audio_message(pending)
            self._batcher.rearm_initial()

    def _flush_resampler(self) -> bytes:
        """Drain the streaming resampler's held tail at a turn boundary.

        Guarded so an injected resampler without ``flush`` degrades gracefully to
        the old drop-the-tail behaviour rather than raising.
        """
        flush = getattr(self._resampler, "flush", None)
        if not callable(flush):
            return b""
        tail = flush()
        return tail if isinstance(tail, bytes) else b""

    def _server_lead_ms(self) -> float:
        """How far the server feed is ahead of local playback, in ms (>= 0)."""
        return max(0.0, self._server_fed_ms - self._played_real_ms)

    async def _send_audio_message(self, pcm: bytes) -> None:
        """Ship one server-bound audio payload, subject to the lead cap.

        Under the cap (or with it disabled) the payload goes straight out; over
        the cap it waits in ``_feed_pending`` for :meth:`_server_feed_loop` to
        release it as playback advances. A barge-in discards the pending queue
        (see :meth:`interrupt`). Wire chunking + time pacing live in
        ``OjinClient``; this layer only bounds how far AHEAD of playback the
        server is fed.
        """
        if not pcm:
            return
        if self._server_feed_max_lead_ms <= 0 or (
            not self._feed_pending
            and self._server_lead_ms() < self._server_feed_max_lead_ms
        ):
            await self._send_audio_now(pcm)
            return
        self._feed_pending.append(pcm)
        self._feed_wake.set()
        self._tracer.instant(
            "to_server",
            "audio_lead_gated",
            args={"bytes": len(pcm), "lead_ms": round(self._server_lead_ms())},
        )

    async def _send_audio_now(self, pcm: bytes) -> None:
        """Send one server-bound audio payload and record the to_server trace."""
        await self._client.send_message(OjinAudioInputMessage(audio_int16_bytes=pcm))
        self._server_fed_ms += len(pcm) / _BYTES_PER_MS_16K
        self._tracer.instant("to_server", "audio_sent", args={"bytes": len(pcm)})

    async def _server_feed_loop(self) -> None:
        """Release lead-gated payloads as playback advances (lead cap only).

        Wakes on new pending payloads and on playback ticks; sends the head of
        ``_feed_pending`` whenever the lead is back under the cap. One payload may
        overshoot the cap by its own duration — the cap bounds the backlog, it is
        not a hard quantum.
        """
        while self._initialized:
            if (
                self._feed_pending
                and self._server_lead_ms() < self._server_feed_max_lead_ms
            ):
                pcm = self._feed_pending.popleft()
                try:
                    await self._send_audio_now(pcm)
                except asyncio.CancelledError:
                    raise
                except Exception:  # one bad send must not kill the feeder
                    logger.exception("server feed loop send error — continuing")
                continue
            self._feed_wake.clear()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._feed_wake.wait(), 0.1)

    async def send_tts_audio(
        self, pcm: bytes, sample_rate: int, num_channels: int
    ) -> None:
        """Buffer original TTS audio for playback and send a resampled copy.

        Discards the 0.5 s all-zero trailing-silence sentinel some TTS engines emit.
        Buffers the **original** bytes (played verbatim) and sends a 16 kHz copy to
        the server for lip-sync inference.
        """
        if not self._initialized:
            if self._buffer_preinit_tts_audio:
                # Queue verbatim; replayed in order once the session is ready.
                self._preinit_inputs.append(("audio", pcm, sample_rate, num_channels))
                self._tracer.instant(
                    "tts_input", "tts_audio_buffered", args={"bytes": len(pcm)}
                )
            else:
                logger.warning("send_tts_audio before session ready — dropping")
            return

        if self._deferring_input:
            # A turn opened during a barge-in (see start_turn); hold its audio until
            # the window clears so it replays in order with its turn boundary.
            self._interrupt_deferred.append(("audio", pcm, sample_rate, num_channels))
            self._tracer.instant(
                "tts_input", "tts_audio_deferred", args={"bytes": len(pcm)}
            )
            return

        await self._send_tts_audio_core(pcm, sample_rate, num_channels)

    async def _send_tts_audio_core(
        self, pcm: bytes, sample_rate: int, num_channels: int
    ) -> None:
        """Buffer + resample + feed one TTS audio payload (no deferral guards).

        The body of :meth:`send_tts_audio` once the not-ready / deferral guards have
        passed; also the seam :meth:`_flush_interrupt_deferred` replays through.
        """
        duration = len(pcm) / (sample_rate * num_channels * 2)
        if abs(duration - _HALF_SECOND) < _HALF_SECOND_TOL and pcm == b"\x00" * len(
            pcm
        ):
            self._tracer.instant("tts_input", "tts_silence_discarded")
            return

        target = self._synchronizer.add_audio(pcm, sample_rate, num_channels)
        if target is None:
            logger.warning(
                "TTS audio with no target buffer — dropping %d bytes", len(pcm)
            )
            return
        self._last_audio_shape = (sample_rate, num_channels)

        resampled = await self._resampler.resample(
            pcm, sample_rate, OJIN_PERSONA_SAMPLE_RATE
        )

        if self._waiting_for_first_tts:
            self._waiting_for_first_tts = False
            self._tr_first_tts_audio_at = self._tracer.mark()
            self._awaiting_first_recv_video = True
            self._awaiting_first_played_video = True

        if self._tracing:
            self._tracer.instant(
                "tts_input",
                "tts_audio",
                args={"bytes": len(pcm), "duration_ms": round(duration * 1000, 1)},
            )
            input_rms = rms_int16(resampled)
            if input_rms is not None:
                self._tracer.counter("input_audio_rms", round(input_rms, 1))

        if self._config.server_feed_batching_enabled:
            to_send = self._batcher.add(resampled)
            self._batch_added.set()  # any arrival resets the idle-flush timer
            if to_send is not None:
                await self._send_audio_message(to_send)
        else:
            await self._send_audio_message(resampled)

    async def say(self, pcm: bytes, sample_rate: int, num_channels: int) -> None:
        """Open a turn and send one audio payload (one-shot helper)."""
        await self.start_turn()
        await self.send_tts_audio(pcm, sample_rate, num_channels)

    async def _flush_preinit_inputs(self) -> None:
        """Replay input buffered before the session was ready, in arrival order.

        Called once from the ``SESSION_READY`` handler after the audio seed, with
        ``_initialized`` already ``True`` — so the replayed ``start_turn`` /
        ``send_tts_audio`` calls take their normal path (the guards that buffered
        them no longer trip) and won't re-enqueue.
        """
        pending, self._preinit_inputs = self._preinit_inputs, []
        if not pending:
            return
        logger.info("Replaying %d buffered pre-init input op(s)", len(pending))
        for op in pending:
            if op[0] == "turn":
                await self.start_turn()
            else:  # ("audio", pcm, sample_rate, num_channels)
                await self.send_tts_audio(op[1], op[2], op[3])

    async def _flush_interrupt_deferred(self) -> None:
        """Replay input deferred during a barge-in, in arrival order, then resume.

        Called from the receive loop when the server's idle/fade frame clears the
        interruption window. Replays through the guard-free ``*_core`` seams so it
        never re-defers. Drains in a loop and only clears ``_deferring_input`` once
        the buffer is empty: any live ``send_tts_audio`` that lands mid-replay still
        appends (the flag is set) and is picked up in the next pass, so ordering is
        preserved and no frame slips ahead of the replayed backlog.
        """
        if self._interrupt_deferred:
            logger.info(
                "Replaying %d input op(s) deferred during barge-in",
                len(self._interrupt_deferred),
            )
        while self._interrupt_deferred:
            pending, self._interrupt_deferred = self._interrupt_deferred, []
            for op in pending:
                if op[0] == "turn":
                    await self._start_turn_core()
                else:  # ("audio", pcm, sample_rate, num_channels)
                    await self._send_tts_audio_core(op[1], op[2], op[3])
        self._deferring_input = False

    async def interrupt(self) -> bool:
        """Barge-in: fade the current turn and cancel it server-side if playing.

        No-op while an interruption is already in flight: the window stays open from
        the cancel until the server's first idle/fade-out frame acknowledges it (see
        :meth:`_on_interaction_response`), so a re-fire can't stack a second cancel.
        """
        if self._interruption_ongoing:
            self._tracer.instant("interruption", "interrupt_suppressed")
            return False
        if not self._initialized and self._buffer_preinit_tts_audio:
            # Barge-in before the buffered turn ever played: drop the queued input
            # so a not-yet-spoken line doesn't replay after the user interrupted.
            if self._preinit_inputs:
                self._preinit_inputs.clear()
                self._tracer.instant("interruption", "preinit_buffer_cleared")
            return False
        if self._synchronizer.interrupt():
            self._interruption_ongoing = True
            # OjinClient drops any audio still queued toward the server when it sees
            # this cancel, so no pre-cancel audio outlives the barge-in.
            await self._client.send_message(OjinCancelInteractionMessage())
            if self._config.server_feed_batching_enabled:
                self._batcher.reset()
            # Discard lead-gated payloads (they belong to the cancelled turn) and
            # reconcile the lead: the server flushes its input backlog on cancel,
            # so the feed restarts level with playback.
            self._feed_pending.clear()
            self._server_fed_ms = self._played_real_ms
            self._feed_wake.set()
            # Drop the cancelled turn's resampler tail so it can't bleed into the
            # next turn's feed; the next turn starts from a clean filter.
            self._flush_resampler()
            self._tr_interrupt_start = self._tracer.mark()
            self._tracer.instant("interruption", "cancel_sent")
            await self._events.emit(STVEvent.INTERRUPTED)
            return True
        return False

    # ------------------------------------------------------------------
    # Receive loop
    # ------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        """Pull server messages and dispatch them; never die on one bad message."""
        while self._initialized or self._receive_task is not None:
            try:
                message = await self._client.receive_message()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error receiving server message — continuing")
                continue
            if message is None:
                continue
            try:
                await self._handle_message(message)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Error handling %s — continuing", type(message).__name__
                )

    async def _handle_message(self, message: BaseModel) -> None:
        """Route one server message (session ready / interaction / error)."""
        if isinstance(message, OjinSessionReadyMessage):
            if message.parameters is not None:
                self._session_data = message.parameters
            self._initialized = True
            self._tracer.span("lifecycle", "connect", self._tr_connect_start)
            if self._playback_task is None:
                self._playback_task = asyncio.create_task(self._video_playback_loop())
            if (
                self._config.server_feed_batching_enabled
                and self._batch_flush_task is None
            ):
                self._batch_flush_task = asyncio.create_task(self._batch_flush_loop())
            if self._server_feed_max_lead_ms > 0 and self._feed_task is None:
                self._feed_task = asyncio.create_task(self._server_feed_loop())
            await self._events.emit(
                STVEvent.SESSION_READY, session_data=self._session_data
            )
            # Seed the server's audio timeline so it starts emitting silence frames.
            await self._client.send_message(
                OjinAudioInputMessage(audio_int16_bytes=b"\x00" * BYTES_PER_FRAME)
            )
            self._tracer.instant("to_server", "seed_sent")
            # Replay any input that arrived during the cold-start handshake.
            await self._flush_preinit_inputs()

        elif isinstance(message, OjinInteractionResponseMessage):
            was_interrupting = self._interruption_ongoing
            self._on_interaction_response(message)
            # The idle/fade frame that just closed the barge-in window is the server's
            # signal that it is ready for new input again — replay any turn that was
            # deferred while the cancel was settling (see start_turn / send_tts_audio).
            if (
                was_interrupting
                and not self._interruption_ongoing
                and self._deferring_input
            ):
                await self._flush_interrupt_deferred()

        elif isinstance(message, ErrorResponseMessage):
            code = str(message.payload.code)
            text = str(message.payload.message or code)
            self._tracer.instant(
                "lifecycle", "server_error", args={"code": code, "message": text}
            )
            await self._events.emit(STVEvent.ERROR, message=text, code=code, fatal=True)
            await self.close()

    def _validate_frame_type(self, frame_type: int) -> bool:
        """Raise if the frame type is not a known Ojin FrameType."""
        last_frame_type = (
            self._last_video_frame.frame_type
            if self._last_video_frame is not None
            else None
        )

        if (
            last_frame_type == FrameType.SPEECH
            and frame_type == FrameType.START_OF_SPEECH
        ):
            logger.warning(
                "Received BOOMERANG frame after SPEECH_FRAME frame; "
                "this is unexpected and may indicate a server-side issue."
            )
            return False

        return True

    def _on_interaction_response(self, message: OjinInteractionResponseMessage) -> None:
        """Build a VideoFrame, hand it to the decode worker, record recv trace."""
        frame_type = int(message.frame_type)
        if not self._validate_frame_type(frame_type):
            return  # Skip processing this frame due to invalid frame type

        volume = int(rms_int16(message.audio_frame_bytes) or 0)
        video_frame = VideoFrame(
            frame_type=frame_type,
            image_bytes=message.video_frame_bytes,
            audio_bytes=message.audio_frame_bytes,
            is_final=message.is_final_response,
            volume=volume,
        )
        self._last_video_frame = video_frame
        self._decode_in.put(video_frame)

        payload_bytes = len(message.video_frame_bytes) + len(message.audio_frame_bytes)
        self._tracer.instant(
            recv_lane_for_frame_type(frame_type),
            "frame_recv",
            cat=str(frame_type),
            args={"frame_type": frame_type, "volume": volume, "bytes": payload_bytes},
        )
        self._tracer.counter("recv_frame_type", frame_type)
        # Per-frame transit delay: local wall clock minus the server's send
        # hand-off stamp from the wire header. The bot↔server clock offset
        # shifts the whole lane by a constant — read the SHAPE, not the
        # absolute value: a flat lane is healthy pacing; a ramp is frames
        # aging in transit (network/proxy holding them); a step is a stall.
        if message.timestamp > 0:
            transit_ms = time.time() * 1000.0 - float(message.timestamp)
            self._tracer.counter("frame_transit_ms", round(transit_ms, 1))
        # End the barge-in window on the first idle/fade-out frame after a cancel —
        # the server's acknowledgement that the interruption took effect. Re-enables
        # interrupt() for the next turn.
        if self._interruption_ongoing and (
            video_frame.is_silence() or video_frame.is_fade_out()
        ):
            self._interruption_ongoing = False
            self._tracer.instant("interruption", "interrupt_ended")
        # Close the cancel→new-turn span on the first new-turn frame after barge-in.
        if (
            frame_type == FrameType.START_OF_SPEECH
            and self._tr_interrupt_start is not None
        ):
            self._tracer.span(
                "interruption", "interrupt→new_turn", self._tr_interrupt_start
            )
            self._tr_interrupt_start = None
        # First speech video frame from the server — the inference round-trip.
        if (
            self._awaiting_first_recv_video
            and self._tr_first_tts_audio_at is not None
            and not video_frame.is_silence()
            and not video_frame.is_fade_out()
        ):
            self._awaiting_first_recv_video = False
            self._tracer.record_response_latency("recv", self._tr_first_tts_audio_at)

    # ------------------------------------------------------------------
    # Playback loop (audio-as-clock)
    # ------------------------------------------------------------------

    async def _video_playback_loop(self) -> None:
        """Drive one synchronizer tick every frame interval and emit the result."""
        frame_duration = 1.0 / self._config.fps
        next_tick = time.perf_counter() + frame_duration
        while self._initialized:
            now = time.perf_counter()
            sleep_for = next_tick - now - 0.003
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            while time.perf_counter() < next_tick:
                pass
            next_tick += frame_duration
            self._diagnostics.note_tick()
            await self._emit_tick()

    async def _batch_flush_tick(self, idle: float) -> None:
        """Run one debounce tick: wait for audio or fire a tail flush on timeout.

        Extracted from the loop so the ``try``/``except`` does not sit directly
        inside ``while`` (avoids PERF203).  Raises ``asyncio.CancelledError``
        so the caller loop can exit cleanly; all other exceptions are swallowed
        and logged so one bad send never kills the loop.
        """
        try:
            await asyncio.wait_for(self._batch_added.wait(), timeout=idle)
            self._batch_added.clear()  # new audio arrived → restart the timer
        except asyncio.TimeoutError:
            if self._batcher.flush_due():
                pending = self._batcher.drain()
                if pending is not None:
                    await self._send_audio_message(pending)
        except asyncio.CancelledError:
            raise
        except Exception:  # never let one bad send kill the loop
            logger.exception("batch flush loop error — continuing")

    async def _batch_flush_loop(self) -> None:
        """Flush a sub-threshold audio tail after the idle gap, off the playback clock.

        Debounces on ``_batch_added``: each new chunk restarts the timer, so a
        flush only fires once TTS has been quiet for ``server_feed_flush_idle_ms``
        (turn end / short turn). Runs as its own task so a blocking websocket send
        never jitters the 25 fps playback loop. Re-arm of the initial threshold is
        NOT done here — that is tied to ``start_turn``.
        """
        idle = self._config.server_feed_flush_idle_ms / 1000.0
        while self._initialized:
            await self._batch_flush_tick(idle)

    async def _emit_tick(self) -> None:
        """Run one tick: drain decode output, step the synchronizer, emit frames.

        Always emits exactly one audio frame (real or silence) so the consumer's
        clock never starves, plus a video frame when one is available (repeating the
        last RGB otherwise). Dispatches started/stopped-speaking events and records
        per-tick trace. This is the testable seam the playback loop wraps in timing.
        """
        tick_start = time.perf_counter()
        lag_ms = (
            (tick_start - self._prev_tick_perf) * 1000.0 - 1000.0 / self._config.fps
            if self._prev_tick_perf > 0
            else 0.0
        )
        self._prev_tick_perf = tick_start

        with contextlib.suppress(queue.Empty):
            while True:
                self._synchronizer.enqueue_video(self._decode_out.get_nowait())

        result = self._synchronizer.tick()
        if result.warming_up:
            return

        pts = int(time.monotonic() * 1_000_000_000)
        await self._emit_video(result, pts)
        await self._emit_audio(result, pts)

        if result.audio_chunk:
            # Advance the playback position the server-feed lead cap gates on.
            # Faded (interrupted) audio counts too: fed/played are reconciled at
            # interrupt() anyway, and max(0, ...) absorbs any residue.
            sample_rate, num_channels = self._last_audio_shape
            self._played_real_ms += len(result.audio_chunk) / (
                sample_rate * num_channels * 2 / 1000.0
            )
            if self._feed_pending:
                self._feed_wake.set()

        if result.started_speaking:
            self._tr_speaking_start = self._tracer.mark()
            await self._events.emit(STVEvent.BOT_STARTED_SPEAKING)
        if result.stopped_speaking:
            if self._tr_speaking_start is not None:
                self._tracer.span("speaking", "bot_speaking", self._tr_speaking_start)
                self._tr_speaking_start = None
            await self._events.emit(STVEvent.BOT_STOPPED_SPEAKING)

        if self._tracing:
            self._trace_tick(result, tick_start, lag_ms)

    @staticmethod
    def _audio_kind(audio_chunk: Optional[bytes], interrupted: bool, has_buffer: bool):
        """Classify the tick's emitted audio (matches OjinVideoService labels)."""
        if audio_chunk is not None:
            return "silenced" if interrupted else "real"
        return "underrun" if has_buffer else "silence"

    def _trace_tick(self, result: TickResult, tick_start: float, lag_ms: float) -> None:
        """Emit OjinVideoService-parity per-tick lane markers and counters.

        Markers: played-audio kind (``play_audio``), held frames (``play:repeat``),
        buffer swaps (``buffers``), swap-time alignment trims (``lipsync``), overflow
        drops (``recv:idle``). Counters: buffer depth, pending frames, playback fps,
        played/bundled-audio RMS, idle skips, underruns, loop lag and tick cost — so
        the trace diffs cleanly against the bot's. Only called with a real tracer.
        """
        self._trace_markers(result)
        self._trace_counters(result, tick_start, lag_ms)

    def _trace_markers(self, result: TickResult) -> None:
        """Emit the per-tick instant markers on their Perfetto lanes."""
        tr = self._tracer
        if result.video_frame is not None:
            now_us = tr.now_us()
            self._tr_emit_times.append(now_us)
            while self._tr_emit_times and now_us - self._tr_emit_times[0] > _ONE_SEC_US:
                self._tr_emit_times.popleft()
        elif self._last_played_rgb is not None:
            tr.instant("play:repeat", "video_repeat")

        cur = self._synchronizer.current_buffer
        kind = self._audio_kind(
            result.audio_chunk, cur is not None and cur.interrupted, cur is not None
        )
        if kind == "underrun":
            self._tr_underruns += 1
        self._tr_idle_skips += result.idle_skipped
        tr.instant(
            "play_audio",
            "audio_emit",
            cat=kind,
            args={"kind": kind, "bytes": len(result.audio_chunk or b"")},
        )
        if result.swapped:
            tr.instant(
                "buffers",
                "swap",
                args={
                    "queue_len": len(self._synchronizer.audio_buffers),
                    "trim_frames": result.align_trim_frames,
                },
            )
        if result.align_trim_frames:
            # Positive = leading buffer audio trimmed (server dropped it);
            # negative = silence prepended (server emitted extra quiet head frames).
            tr.instant(
                "lipsync",
                "swap_align_trim",
                args={
                    "trim_frames": result.align_trim_frames,
                    "trim_ms": round(
                        result.align_trim_frames * 1000.0 / self._config.fps, 1
                    ),
                },
            )
        if result.catchup_dropped:
            self._tr_catchup_drops += result.catchup_dropped
            tr.instant(
                "lipsync",
                "repeat_catchup",
                args={"dropped": result.catchup_dropped},
            )
        if result.overflow_dropped:
            tr.instant(
                "recv:idle",
                "recv_overflow_drop",
                args={"dropped": result.overflow_dropped},
            )

    def _trace_counters(
        self, result: TickResult, tick_start: float, lag_ms: float
    ) -> None:
        """Emit the per-tick counter tracks (depth, fps, RMS, loop timing)."""
        tr = self._tracer
        cur = self._synchronizer.current_buffer
        cur_ms = 0.0
        if cur is not None and cur.sample_rate:
            cur_ms = len(cur.bytes_) / (cur.sample_rate * cur.num_channels * 2) * 1000.0
        tr.counter("current_buffer_ms", round(cur_ms, 1))
        tr.counter("server_feed_lead_ms", round(self._server_lead_ms(), 1))
        tr.counter("server_feed_gated_pending", len(self._feed_pending))
        tr.counter("queued_buffers", len(self._synchronizer.audio_buffers))
        tr.counter("pending_video_frames", len(self._synchronizer.video_frames))
        tr.counter("playback_fps", len(self._tr_emit_times))
        tr.counter("audio_underruns_total", self._tr_underruns)
        tr.counter("idle_backlog_skips_total", self._tr_idle_skips)
        tr.counter("repeat_catchup_drops_total", self._tr_catchup_drops)
        if result.video_frame is not None and result.audio_chunk:
            frame_rms = rms_int16(result.video_frame.audio_bytes)
            if frame_rms is not None:
                tr.counter("frame_audio_rms", round(frame_rms, 1))
        if result.audio_chunk:
            out_rms = rms_int16(result.audio_chunk)
            if out_rms is not None:
                tr.counter("output_audio_rms", round(out_rms, 1))
        tr.counter("loop_lag_ms", round(max(0.0, lag_ms), 1))
        work_ms = (time.perf_counter() - tick_start) * 1000.0
        tr.counter("tick_work_ms", round(work_ms, 1))
        self._trace_pipeline_depths(tr)

    def _trace_pipeline_depths(self, tr: Tracer) -> None:
        """Emit receive-pipeline depth gauges to localize where frames back up.

        When the fixed-25fps synchronizer starves (repeated frames / lipsync drift),
        exactly one stage is the bottleneck. These counters, read upstream→downstream,
        point straight at it:

        - ``recv_sock_bytes`` grows → the event loop can't read the socket fast enough
          (e.g. starved by the playback busy-wait); TCP is delivering but we're behind.
        - ``recv_ws_frames`` near 16 / ``recv_ws_paused`` = 1 → websockets read-
          backpressure engaged; the ``async for`` receive loop is the bottleneck.
        - ``recv_server_msgs`` grows → parsing keeps up but downstream consumption
          (decode/playback) lags; this unbounded queue is what hides upstream
          backpressure (why the proxy still egresses a steady 25fps).
        - ``recv_decode_in`` grows → the cv2 decode worker can't keep up.
        - ``recv_decode_out`` grows → the synchronizer drain is behind (rare; it drains
          fully each tick).
        - all ≈ 0 while frames still arrive < 25fps → frames simply aren't being
          delivered faster (upstream network/proxy), not a client-side backlog.

        The final stage, the synchronizer deque, is the pre-existing
        ``pending_video_frames`` counter.

        The ``tcp_*`` gauges (RTT/loss/window on the client end of the proxy→client TCP
        connection) further split that last case: a steady < 25fps with all queues ≈ 0
        but climbing ``tcp_retrans_total`` or high/jittery ``tcp_rtt_us`` indicates the
        *local* path (e.g. WSL2 NAT / WAN) is throttling delivery, not the proxy/server.
        """
        probe = getattr(self._client, "debug_queue_depths", None)
        depths = probe() if probe is not None else {}
        for name in ("sock_bytes", "ws_frames", "ws_paused", "server_msgs"):
            if name in depths:
                tr.counter(f"recv_{name}", depths[name])
        for name in (
            "tcp_rtt_us",
            "tcp_rttvar_us",
            "tcp_retrans_total",
            "tcp_retransmits",
            "tcp_snd_cwnd",
            "tcp_rcv_space",
            "tcp_bytes_received",
            "tcp_min_rtt_us",
            "tcp_rcv_ooopack",
        ):
            if name in depths:
                tr.counter(name, depths[name])
        # Ingress bandwidth: cumulative app-level bytes off the websocket plus
        # a per-tick rate lane. Compared with ``tcp_bytes_received`` (kernel
        # cumulative) this splits "network delivered late" from "read late".
        if "ingress_bytes_total" in depths:
            total = int(depths["ingress_bytes_total"])
            now = time.perf_counter()
            tr.counter("recv_ingress_bytes_total", total)
            prev = self._tr_prev_ingress
            if prev is not None and now > prev[1]:
                rate_mbps = (total - prev[0]) * 8.0 / (now - prev[1]) / 1e6
                tr.counter("recv_ingress_mbps", round(max(0.0, rate_mbps), 3))
            self._tr_prev_ingress = (total, now)
        tr.counter("recv_decode_in", self._decode_in.qsize())
        tr.counter("recv_decode_out", self._decode_out.qsize())

    async def _emit_video(self, result: TickResult, pts: int) -> None:
        """Emit a video frame for this tick (new, or a repeat of the last RGB)."""
        vframe = result.video_frame
        rgb = self._last_played_rgb
        size = self._last_played_size
        source_bytes = b""
        frame_type = 0
        volume = 0.0
        if vframe is not None:
            source_bytes = vframe.image_bytes
            frame_type = vframe.frame_type
            volume = float(vframe.volume)
            if vframe.out_rgb is not None:
                rgb = vframe.out_rgb
                size = vframe.out_size
                self._last_played_rgb = rgb
                self._last_played_size = size
            self._tracer.instant(
                play_lane_for_frame_type(vframe.frame_type),
                "video_emit",
                cat=str(vframe.frame_type),
                args={"frame_type": vframe.frame_type, "swapped": result.swapped},
            )
            if (
                self._awaiting_first_played_video
                and self._tr_first_tts_audio_at is not None
                and not vframe.is_silence()
                and not vframe.is_fade_out()
            ):
                self._awaiting_first_played_video = False
                latency_ms = self._tracer.record_response_latency(
                    "played", self._tr_first_tts_audio_at
                )
                self._tracer.span(
                    "latency",
                    "ojin",
                    self._tr_first_tts_audio_at,
                    args={"played_ms": latency_ms},
                )

        if rgb is None or size is None:
            return
        width, height = size  # the server's native frame size for this frame
        await self._output.write_video(
            STVVideoFrame(
                rgb=rgb,
                source_bytes=source_bytes,
                width=width,
                height=height,
                frame_type=frame_type,
                pts=pts,
                volume=volume,
            )
        )

    async def _emit_audio(self, result: TickResult, pts: int) -> None:
        """Emit exactly one audio frame this tick (real chunk or silence)."""
        sample_rate, num_channels = self._last_audio_shape
        pcm = result.audio_chunk
        if pcm is None:
            chunk_size = int(sample_rate / self._config.fps) * num_channels * 2
            pcm = b"\x00" * chunk_size
        await self._output.write_audio(
            STVAudioFrame(
                pcm=pcm, sample_rate=sample_rate, num_channels=num_channels, pts=pts
            )
        )

    # ------------------------------------------------------------------
    # Off-loop JPEG decode worker
    # ------------------------------------------------------------------

    def _start_decode_worker(self) -> None:
        """Start the background JPEG→RGB decode thread (idempotent)."""
        if self._decode_thread is not None:
            return
        thread = threading.Thread(
            target=self._decode_worker, name="ojin-stv-frame-decode", daemon=True
        )
        self._decode_thread = thread
        thread.start()

    def _stop_decode_worker(self) -> None:
        """Signal the decode thread to drain and exit, then join it."""
        thread = self._decode_thread
        if thread is None:
            return
        self._decode_thread = None
        self._decode_in.put(None)  # sentinel
        thread.join(timeout=1.0)

    def _decode_worker(self) -> None:
        """Decode queued JPEG frames to native-size RGB off the event loop, in order."""
        while True:
            frame = self._decode_in.get()
            if frame is None:  # shutdown sentinel
                break
            try:
                decoded = self._decoder.decode(frame.image_bytes)
                if decoded is not None:
                    frame.out_rgb, w, h = decoded
                    frame.out_size = (w, h)
            except Exception as exc:  # never let a bad frame kill the worker
                frame.out_rgb = None
                logger.warning("frame decode failed: %s", exc)
            self._decode_out.put(frame)
