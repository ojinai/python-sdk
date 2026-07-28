"""Shared outbound server-feed machinery for the STV clients.

Everything that shapes the client→server audio stream lives here: the
:class:`~ojin.stv.send_batcher.SendBatcher` wiring, the lead-cap gate that keeps
the server feed bounded ahead of playback, the idle-flush debounce, the
pre-initialization input buffer, the barge-in input deferral, and the streaming
resampler flush. :class:`~ojin.stv.ojin_stv_client.OjinSTVClient` and
:class:`~ojin.stv.ojin_stv_webrtc_client.OjinSTVWebRTCClient` mix this in and
provide the host-specific seams: ``_start_turn_core`` / ``_send_tts_audio_core``
(what a turn/audio op does once the guards pass), ``_feed_gate_open`` (when
input may flow to the server), and the playback-position source that calls
:meth:`OutboundFeedMixin._advance_played_real_ms`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from typing import Optional

from ojin.ojin_client_messages import IOjinClient, OjinAudioInputMessage
from ojin.stv.config import STVConfig
from ojin.stv.resampler import Resampler
from ojin.stv.send_batcher import SendBatcher
from ojin.stv.tracing import Tracer

logger = logging.getLogger(__name__)

# Legacy server-bound payloads are 16 kHz mono int16, so ms = bytes / 32. The
# direct-WebRTC client feeds at its declared native rate instead, so the byte↔ms
# conversion is an instance value (`_bytes_per_ms`) derived from the feed rate;
# 16 kHz stays the default so the legacy client is byte-identical.
FEED_SAMPLE_RATE = 16_000


class OutboundFeedMixin:
    """Batcher + lead-gate + preinit/deferral machinery shared by both clients."""

    # Provided by the host class before _init_outbound_feed runs.
    _client: IOjinClient
    _config: STVConfig
    _resampler: Resampler
    _tracer: Tracer
    _initialized: bool

    def _init_outbound_feed(
        self,
        buffer_preinit_tts_audio: bool,
        feed_sample_rate: int = FEED_SAMPLE_RATE,
    ) -> None:
        """Initialize the batcher, lead-cap, preinit and deferral state.

        ``feed_sample_rate`` is the mono int16 rate of the server-bound audio;
        it fixes the byte↔ms conversion (``bytes = ms * rate * 2 / 1000``). The
        legacy client leaves it at 16 kHz (32 B/ms); the direct-WebRTC client
        passes its declared native rate (e.g. 24 kHz → 48 B/ms).
        """
        # ms = bytes / _bytes_per_ms, for the batcher thresholds and the lead
        # clock. Derived from the feed rate so both clients share one formula.
        self._bytes_per_ms = feed_sample_rate * 2 / 1000.0
        self._batcher = SendBatcher(
            initial_chunk_bytes=int(
                self._config.server_feed_initial_chunk_ms * self._bytes_per_ms
            ),
            min_chunk_bytes=int(
                self._config.server_feed_min_chunk_ms * self._bytes_per_ms
            ),
            flush_idle_s=self._config.server_feed_flush_idle_ms / 1000.0,
        )
        self._batch_added = asyncio.Event()
        self._batch_flush_task: Optional[asyncio.Task] = None

        # Server-feed lead cap (see STVConfig.server_feed_max_lead_ms). The gate
        # compares how much audio has been shipped to the server (_server_fed_ms)
        # against how much playback has consumed (_played_real_ms); payloads that
        # would push the lead past the cap wait in _feed_pending and are released
        # by _server_feed_loop as playback advances.
        self._server_feed_max_lead_ms = float(
            max(0, self._config.server_feed_max_lead_ms)
        )
        self._server_fed_ms = 0.0
        self._played_real_ms = 0.0
        self._feed_pending: "deque[bytes]" = deque()
        self._feed_wake = asyncio.Event()
        self._feed_task: Optional[asyncio.Task] = None

        # Pre-initialization input buffer: when enabled, start_turn /
        # send_tts_audio calls that arrive before the feed gate opens are
        # recorded here (in order) and replayed once it does, instead of being
        # dropped. Entries are ("turn",) or ("audio", pcm, sample_rate,
        # num_channels).
        self._buffer_preinit_tts_audio = buffer_preinit_tts_audio
        self._preinit_inputs: list[tuple] = []

        # Barge-in in flight: set when a cancel is sent, cleared when the
        # server's first idle/fade-out frame acknowledges it. While set, further
        # interrupts are dropped so a re-fire can't stack a second cancel.
        self._interruption_ongoing = False

        # Deferred input during a barge-in. A NEW turn that opens while a cancel
        # is still settling server-side must not be fed yet — the server would
        # discard it, desyncing playback. Its start + audio are buffered here (in
        # order) and replayed once the ack frame clears the window. Deferral
        # begins at the new turn's start_turn (NOT at interrupt: the interrupted
        # turn's own trailing audio must still be dropped, not replayed).
        self._interrupt_deferred: list[tuple] = []
        self._deferring_input = False

    # ------------------------------------------------------------------
    # Host seams
    # ------------------------------------------------------------------

    def _feed_gate_open(self) -> bool:
        """Whether input may flow to the server (hosts may re-key this gate)."""
        return self._initialized

    async def _start_turn_core(self) -> None:
        """Open the next turn once the buffering/deferral guards have passed."""
        raise NotImplementedError

    async def _send_tts_audio_core(
        self, pcm: bytes, sample_rate: int, num_channels: int
    ) -> None:
        """Feed one TTS audio payload once the guards have passed."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Input — turn + audio (guard wrappers)
    # ------------------------------------------------------------------

    async def start_turn(self) -> None:
        """Open a new turn for the next utterance (≈ TTSStartedFrame)."""
        if not self._feed_gate_open() and self._buffer_preinit_tts_audio:
            # Defer until the gate opens so this turn boundary is replayed in
            # order with its audio (see _flush_preinit_inputs).
            self._preinit_inputs.append(("turn",))
            self._tracer.instant("tts_input", "tts_started_buffered")
            return
        if self._deferring_input or self._interruption_ongoing:
            # A new turn opened while a barge-in is still settling server-side.
            # Buffer its start (and everything that follows) instead of feeding
            # it now; replayed in order once the window clears.
            self._deferring_input = True
            self._interrupt_deferred.append(("turn",))
            self._tracer.instant("tts_input", "tts_started_deferred")
            return
        await self._start_turn_core()

    async def send_tts_audio(
        self, pcm: bytes, sample_rate: int, num_channels: int
    ) -> None:
        """Accept one TTS audio payload, honoring the preinit/deferral guards."""
        if not self._feed_gate_open():
            if self._buffer_preinit_tts_audio:
                # Queue verbatim; replayed in order once the gate opens.
                self._preinit_inputs.append(("audio", pcm, sample_rate, num_channels))
                self._tracer.instant(
                    "tts_input", "tts_audio_buffered", args={"bytes": len(pcm)}
                )
            else:
                logger.warning("send_tts_audio before session ready — dropping")
            return

        if self._deferring_input:
            # A turn opened during a barge-in (see start_turn); hold its audio
            # until the window clears so it replays in order with its boundary.
            self._interrupt_deferred.append(("audio", pcm, sample_rate, num_channels))
            self._tracer.instant(
                "tts_input", "tts_audio_deferred", args={"bytes": len(pcm)}
            )
            return

        await self._send_tts_audio_core(pcm, sample_rate, num_channels)

    async def _flush_preinit_inputs(self) -> None:
        """Replay input buffered before the feed gate opened, in arrival order.

        Called once from the gate-opening handler with the gate already open —
        so the replayed ``start_turn`` / ``send_tts_audio`` calls take their
        normal path and won't re-enqueue.
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

        Called when the server's ack frame clears the interruption window.
        Replays through the guard-free ``*_core`` seams so it never re-defers.
        Drains in a loop and only clears ``_deferring_input`` once the buffer is
        empty: any live ``send_tts_audio`` that lands mid-replay still appends
        (the flag is set) and is picked up in the next pass, so ordering is
        preserved.
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

    # ------------------------------------------------------------------
    # Batcher plumbing
    # ------------------------------------------------------------------

    def _flush_resampler(self) -> bytes:
        """Drain the streaming resampler's held tail at a turn boundary.

        Guarded so an injected resampler without ``flush`` degrades gracefully
        to the old drop-the-tail behaviour rather than raising.
        """
        flush = getattr(self._resampler, "flush", None)
        if not callable(flush):
            return b""
        tail = flush()
        return tail if isinstance(tail, bytes) else b""

    async def _drain_batch_at_turn_boundary(self) -> None:
        """Flush the previous turn's batched tail and re-arm the initial chunk."""
        if self._config.server_feed_batching_enabled:
            pending = self._batcher.drain()
            if pending:
                await self._send_audio_message(pending)
            self._batcher.rearm_initial()

    async def _feed_resampled_audio(self, resampled: bytes) -> None:
        """Route one resampled payload through the batcher (or straight out)."""
        if self._config.server_feed_batching_enabled:
            to_send = self._batcher.add(resampled)
            self._batch_added.set()  # any arrival resets the idle-flush timer
            if to_send is not None:
                await self._send_audio_message(to_send)
        else:
            await self._send_audio_message(resampled)

    # ------------------------------------------------------------------
    # Lead-cap gate
    # ------------------------------------------------------------------

    def _server_lead_ms(self) -> float:
        """How far the server feed is ahead of local playback, in ms (>= 0)."""
        return max(0.0, self._server_fed_ms - self._played_real_ms)

    def _advance_played_real_ms(self, delta_ms: float) -> None:
        """Advance the playback position the server-feed lead cap gates on."""
        self._played_real_ms += delta_ms
        if self._feed_pending:
            self._feed_wake.set()

    async def _send_audio_message(self, pcm: bytes) -> None:
        """Ship one server-bound audio payload, subject to the lead cap.

        Under the cap (or with it disabled) the payload goes straight out; over
        the cap it waits in ``_feed_pending`` for :meth:`_server_feed_loop` to
        release it as playback advances. A barge-in discards the pending queue.
        Wire chunking + time pacing live in ``OjinClient``; this layer only
        bounds how far AHEAD of playback the server is fed.
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
        self._server_fed_ms += len(pcm) / self._bytes_per_ms
        self._tracer.instant("to_server", "audio_sent", args={"bytes": len(pcm)})

    async def _server_feed_loop(self) -> None:
        """Release lead-gated payloads as playback advances (lead cap only).

        Wakes on new pending payloads and on playback advances; sends the head
        of ``_feed_pending`` whenever the lead is back under the cap. One
        payload may overshoot the cap by its own duration — the cap bounds the
        backlog, it is not a hard quantum.
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

    # ------------------------------------------------------------------
    # Idle-flush debounce
    # ------------------------------------------------------------------

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
        """Flush a sub-threshold audio tail after the idle gap.

        Debounces on ``_batch_added``: each new chunk restarts the timer, so a
        flush only fires once TTS has been quiet for
        ``server_feed_flush_idle_ms`` (turn end / short turn). Runs as its own
        task so a blocking websocket send never jitters the consumer's clock.
        Re-arm of the initial threshold is NOT done here — that is tied to
        ``start_turn``.
        """
        idle = self._config.server_feed_flush_idle_ms / 1000.0
        while self._initialized:
            await self._batch_flush_tick(idle)

    # ------------------------------------------------------------------
    # Task lifecycle + close-time flush
    # ------------------------------------------------------------------

    def _start_feed_tasks(self) -> None:
        """Start the batch-flush and lead-gate feeder tasks (idempotent)."""
        if self._config.server_feed_batching_enabled and self._batch_flush_task is None:
            self._batch_flush_task = asyncio.create_task(self._batch_flush_loop())
        if self._server_feed_max_lead_ms > 0 and self._feed_task is None:
            self._feed_task = asyncio.create_task(self._server_feed_loop())

    async def _stop_feed_tasks(self) -> None:
        """Cancel and await the batch-flush and feeder tasks."""
        for task_attr in ("_batch_flush_task", "_feed_task"):
            task = getattr(self, task_attr)
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
                setattr(self, task_attr, None)

    async def _flush_outbound_tail(self) -> None:
        """Best-effort flush of everything still owed to the server, in order.

        Lead-gated payloads first (older), then the batcher's sub-threshold
        tail. Straight sends — the session is ending, the lead cap no longer
        matters.
        """
        while self._feed_pending:
            with contextlib.suppress(Exception):
                await self._send_audio_now(self._feed_pending.popleft())
        if self._config.server_feed_batching_enabled:
            pending = self._batcher.drain()
            if pending is not None:
                with contextlib.suppress(Exception):
                    await self._send_audio_now(pending)
