"""Direct-WebRTC session engine (no local playback).

Prefer ``OjinSTVClient(webrtc=WebRTCSettings(...))``: this class is the engine
that client delegates to, kept public for backward compatibility.

``OjinSTVWebRTCClient`` negotiates direct WebRTC publishing with the inference
server at connect time (protocol v2, DR-006 as amended 2026-07-24): the
settings are declared in the WebSocket **upgrade request** (``webrtc_*`` query
params + the ``X-Ojin-Webrtc-Token`` header), the proxy merges them verbatim
into the ``sessionSetup.payload.parameters.webrtc`` it authors, and the server
reports the outcome inside ``sessionReady.payload.parameters.webrtc`` after
joining the room concurrently with model activation. One round trip, no
capability advertisement, no post-ready ack:

- ``status: "connected"`` — direct mode from the first frame: the server
  publishes A/V into the room and every ``interactionResponse`` is a 38-byte
  metadata frame. Those frames replace local playback as the signal layer:
  they advance the lead-gate clock, acknowledge barge-ins, and derive the
  speaking/first-frame events.
- ``status: "failed"`` — fatal (``WEBRTC_JOIN_FAILED``): credentials were
  supplied and a real error occurred; silent fallback would mask breakage.
- absent ``webrtc`` key, or an unrecognised status — fatal
  (``WEBRTC_UNSUPPORTED``): the server cannot publish this session into the
  room. The caller asked for WebRTC, so carrying on without it would only hide
  the problem — no avatar would ever reach the room.

Every fatal WebRTC error closes the session. ``webrtcStatus`` survives only as
an async post-connect notification (mid-session room drop: ``disconnected``
then ``failed(REJOIN_FAILED)``).

Typical use::

    client = OjinSTVClient(api_key=..., config_id=..., webrtc=WebRTCSettings(...))
    async with client:
        await client.start_turn()
        await client.send_tts_audio(pcm, 24000, 1)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from enum import Enum
from typing import Callable, Optional

from pydantic import BaseModel

from ojin.entities.interaction_messages import ErrorResponseMessage
from ojin.ojin_client import OjinClient
from ojin.ojin_client_messages import (
    FrameType,
    IOjinClient,
    OjinCancelInteractionMessage,
    OjinInteractionResponseMessage,
    OjinSessionReadyMessage,
    OjinWebRTCStatusMessage,
)
from ojin.stv._outbound_feed import OutboundFeedMixin
from ojin.stv.config import STVConfig, WebRTCSettings
from ojin.stv.events import EventEmitter, STVEvent
from ojin.stv.resampler import Resampler, default_resampler
from ojin.stv.tracing import NullTracer, Tracer, recv_lane_for_frame_type

logger = logging.getLogger(__name__)

_DEFAULT_WS_URL = "wss://models.ojin.ai/realtime"
_HALF_SECOND = 0.5
_HALF_SECOND_TOL = 0.01
_MS_PER_METADATA_FRAME = 40.0  # metadata frames are paced at 25/s
_METADATA_WATCHDOG_S = 5.0
_WATCHDOG_POLL_S = 1.0
_SPEECH_TYPES = (int(FrameType.SPEECH), int(FrameType.START_OF_SPEECH))
_SILENCE_TYPES = (int(FrameType.IDLE), int(FrameType.FADE_OUT))

WEBRTC_JOIN_FAILED = "WEBRTC_JOIN_FAILED"
WEBRTC_UNSUPPORTED = "WEBRTC_UNSUPPORTED"


class _DirectState(Enum):
    """Client-side direct-path states (protocol v2 — one negotiation, at setup).

    ``PENDING`` covers the whole ``sessionReady`` wait (the request rode the
    connect exchange); the outcome inside ``sessionReady`` resolves it to
    ``CONNECTED`` (direct mode) or ``FAILED`` (fatal — join failed, timed out,
    or the server does not support direct mode). A mid-session room loss also
    ends in ``FAILED``.
    """

    PENDING = "pending"
    CONNECTED = "connected"
    FAILED = "failed"


class OjinSTVWebRTCClient(OutboundFeedMixin):
    """Drives one direct-WebRTC STV session: TTS audio in, room-published A/V out."""

    def __init__(  # noqa: PLR0913 — injectable deps + flat field init
        self,
        *,
        webrtc_settings: WebRTCSettings,
        api_key: str = "",
        config_id: str = "",
        ws_url: str = _DEFAULT_WS_URL,
        resampler: Optional[Resampler] = None,
        tracer: Optional[Tracer] = None,
        client: Optional[IOjinClient] = None,
        config: Optional[STVConfig] = None,
        buffer_preinit_tts_audio: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a client; pass your own transport/resampler/tracer as needed.

        ``webrtc_settings`` carries the room credentials the server needs to
        join; ``token`` is never logged and never appears in traces or URLs.
        The settings are declared in the connection's upgrade request
        (``webrtc_*`` query params + token header, protocol v2) — the
        transport must support :meth:`OjinClient.set_webrtc_connect_settings`.
        Input sent via :meth:`start_turn` / :meth:`send_tts_audio` before the
        ``sessionReady`` outcome arrives is held client-side (when
        ``buffer_preinit_tts_audio`` is ``True``) and replayed at that moment;
        nothing is ever sent on the wire while the outcome is pending.
        ``clock`` is a monotonic seconds source, injectable for deterministic
        timer tests.
        """
        super().__init__()
        self._config = config or STVConfig()
        self._webrtc_settings = webrtc_settings
        self._clock = clock

        self._client: IOjinClient = client or OjinClient(
            ws_url=ws_url,
            api_key=api_key,
            config_id=config_id,
            mode=os.getenv("OJIN_MODE", ""),
            max_input_chunk_bytes=max(1, int(self._config.server_feed_max_chunk_bytes)),
            send_chunk_gap_s=self._config.server_feed_send_gap_ms / 1000.0,
        )
        self._resampler: Resampler = resampler or default_resampler()
        self._tracer: Tracer = tracer or NullTracer()
        self._events = EventEmitter()
        self._initialized = False
        # Feed the server at the declared native rate: the batcher thresholds
        # and lead clock follow it, and send_tts_audio resamples to it below.
        self._init_outbound_feed(
            buffer_preinit_tts_audio,
            feed_sample_rate=webrtc_settings.audio_sample_rate,
        )

        self._session_data: Optional[dict] = None
        self._state = _DirectState.PENDING
        self._receive_task: Optional[asyncio.Task] = None
        self._join_timer_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None
        # Teardown bookkeeping: a fatal WebRTC error closes the session from a
        # task of its own (so the failing receive/timer task never cancels
        # itself mid-teardown); close() is idempotent and joins that task.
        self._closed = False
        self._close_task: Optional[asyncio.Task] = None
        self._request_sent_at = 0.0
        self._last_metadata_at = 0.0
        self._first_frame_pending = False
        self._last_frame_type: Optional[int] = None

        # Turn tracker (replaces the playback Synchronizer's bookkeeping).
        self._turn_id = 0
        self._turn_active = False
        self._turn_audio_fed = False
        self._waiting_for_first_tts = False
        self._awaiting_first_recv_frame = False

        # Trace anchors (µs marks).
        self._tr_session_start = 0.0
        self._tr_connect_start = 0.0
        self._tr_negotiate_start: Optional[float] = None
        self._tr_speaking_start: Optional[float] = None
        self._tr_interrupt_start: Optional[float] = None
        self._tr_first_tts_audio_at: Optional[float] = None

        self._set_trace_other("producer", "ojin_stv_webrtc_client")
        self._set_trace_other(
            "recv_latency_semantics",
            "recv marks server publish time of the metadata frame, not media arrival",
        )
        register = getattr(self._client, "set_webrtc_status_callback", None)
        if callable(register):
            register(self._on_webrtc_status)
        else:
            logger.warning(
                "transport has no set_webrtc_status_callback — "
                "webrtcStatus notifications will never reach this client"
            )
        # Protocol v2 (amended): the webrtc request rides the WebSocket
        # upgrade request (webrtc_* query params + X-Ojin-Webrtc-Token
        # header), so the transport must know the settings before connect().
        # They contain the meeting token — the transport must never log them
        # or place the token in a URL.
        declare = getattr(self._client, "set_webrtc_connect_settings", None)
        if callable(declare):
            declare(self._webrtc_settings)
        else:
            logger.warning(
                "transport has no set_webrtc_connect_settings — the webrtc "
                "request cannot ride the connect exchange; the server will "
                "report the session as unsupported"
            )

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect_with_retry(self) -> bool:
        """Connect the transport, retrying up to the configured max attempts."""
        last_error: Optional[Exception] = None
        for attempt in range(self._config.client_connect_max_retries):
            try:
                await self._client.connect()
                logger.info("OjinSTVWebRTCClient connected")
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
        """Connect the control channel and run the receive loop.

        The webrtc request rides the connect exchange itself (declared on the
        transport at construction), so once the connection is up the
        negotiation is already in flight: the join timer armed here governs
        the whole ``sessionReady`` wait (``webrtc_join_timeout_s``).
        """
        self._tr_session_start = self._tracer.mark()
        self._tr_connect_start = self._tr_session_start
        if not await self.connect_with_retry():
            await self.close()
            return
        self._request_sent_at = self._clock()
        self._tr_negotiate_start = self._tracer.mark()
        self._tracer.instant("webrtc", "request_sent")
        self._join_timer_task = asyncio.create_task(self._join_timer())
        self._receive_task = asyncio.create_task(self._receive_loop())
        await self._client.start_interaction()

    async def __aenter__(self) -> "OjinSTVWebRTCClient":
        """Context-manager entry: :meth:`start`."""
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Context-manager exit: :meth:`close`."""
        await self.close()

    async def close(self) -> None:
        """Tear down timers, feed tasks, and the transport; record the session.

        Idempotent: a repeat call — or one racing the close a fatal WebRTC
        error scheduled — waits for the first teardown instead of redoing it.
        """
        pending = self._close_task
        if pending is not None and pending is not asyncio.current_task():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pending
            return
        if self._closed:
            return
        self._closed = True
        was_initialized = self._initialized
        self._initialized = False
        self._preinit_inputs.clear()
        self._cancel_join_timer()
        await self._cancel_task(self._watchdog_task)
        self._watchdog_task = None
        if was_initialized:
            self._tracer.span("lifecycle", "session", self._tr_session_start)
        await self._stop_feed_tasks()
        # Flush the owed tail only when direct mode actually opened; while the
        # outcome was pending (or after a failure) the wire must stay silent
        # even through teardown.
        if was_initialized and self._state is _DirectState.CONNECTED:
            await self._flush_outbound_tail()
        try:
            await self._client.close()
        except Exception as exc:  # never let teardown raise
            logger.warning("Error closing transport: %s", exc)
        receive_task, self._receive_task = self._receive_task, None
        await self._cancel_task(receive_task)
        await self._events.emit(STVEvent.CLOSED)

    @staticmethod
    async def _cancel_task(task: Optional[asyncio.Task]) -> None:
        """Cancel and await ``task`` unless it is the task running this code."""
        if task is None or task is asyncio.current_task():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    def _schedule_close(self) -> None:
        """Close the session from a fatal path, on a task of its own."""
        if self._closed or self._close_task is not None:
            return
        self._close_task = asyncio.get_running_loop().create_task(self.close())

    # ------------------------------------------------------------------
    # Input — turn + audio + interrupt (cores; guards live in the mixin)
    # ------------------------------------------------------------------

    def _feed_gate_open(self) -> bool:
        """Input flows only once sessionReady opened direct mode."""
        return self._initialized and self._state is _DirectState.CONNECTED

    async def _start_turn_core(self) -> None:
        """Open the next turn's bookkeeping and flush the prior turn's tail."""
        self._flush_resampler()
        self._turn_id += 1
        self._turn_active = True
        self._turn_audio_fed = False
        self._waiting_for_first_tts = True
        self._tracer.instant(
            "tts_input", "tts_started", args={"turn_id": self._turn_id}
        )
        await self._drain_batch_at_turn_boundary()

    async def _send_tts_audio_core(
        self, pcm: bytes, sample_rate: int, num_channels: int
    ) -> None:
        """Resample and feed one TTS payload toward the server."""
        duration = len(pcm) / (sample_rate * num_channels * 2)
        if abs(duration - _HALF_SECOND) < _HALF_SECOND_TOL and pcm == b"\x00" * len(
            pcm
        ):
            self._tracer.instant("tts_input", "tts_silence_discarded")
            return
        if not self._turn_active:
            logger.warning("TTS audio with no open turn — dropping %d bytes", len(pcm))
            return

        resampled = await self._resampler.resample(
            pcm, sample_rate, self._webrtc_settings.audio_sample_rate
        )

        if self._waiting_for_first_tts:
            self._waiting_for_first_tts = False
            self._tr_first_tts_audio_at = self._tracer.mark()
            self._awaiting_first_recv_frame = True
        self._turn_audio_fed = True
        self._tracer.instant(
            "tts_input",
            "tts_audio",
            args={"bytes": len(pcm), "duration_ms": round(duration * 1000, 1)},
        )
        await self._feed_resampled_audio(resampled)

    async def interrupt(self) -> bool:
        """Barge-in: cancel the current turn server-side if it is speaking.

        Before the ``sessionReady`` outcome nothing was ever sent, so a
        barge-in only clears the held input and reports INTERRUPTED locally —
        no wire cancel, no ack-suppression window. Once direct mode is open,
        this mirrors the WebSocket client: the window stays open from the
        cancel until the server's first idle/fade-out frame acknowledges it.
        """
        if self._interruption_ongoing:
            self._tracer.instant("interruption", "interrupt_suppressed")
            return False
        if not self._feed_gate_open():
            if self._preinit_inputs:
                self._preinit_inputs.clear()
                self._tracer.instant("interruption", "preinit_buffer_cleared")
            await self._events.emit(STVEvent.INTERRUPTED)
            return False
        if not self._is_speaking():
            return False
        self._interruption_ongoing = True
        self._turn_active = False
        self._turn_audio_fed = False
        await self._client.send_message(OjinCancelInteractionMessage())
        if self._config.server_feed_batching_enabled:
            self._batcher.reset()
        self._feed_pending.clear()
        self._server_fed_ms = self._played_real_ms
        self._feed_wake.set()
        self._flush_resampler()
        self._tr_interrupt_start = self._tracer.mark()
        self._tracer.instant("interruption", "cancel_sent")
        await self._events.emit(STVEvent.INTERRUPTED)
        return True

    def _is_speaking(self) -> bool:
        """Whether the current turn still has speech to cancel server-side."""
        return self._turn_active and (
            self._turn_audio_fed or self._last_frame_type in _SPEECH_TYPES
        )

    # ------------------------------------------------------------------
    # Receive loop + message routing
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
            await self._on_session_ready(message)
        elif isinstance(message, OjinInteractionResponseMessage):
            was_interrupting = self._interruption_ongoing
            await self._on_interaction_response(message)
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
            self._schedule_close()

    async def _on_session_ready(self, message: OjinSessionReadyMessage) -> None:
        """Emit SESSION_READY, then resolve the connect-time webrtc outcome.

        Protocol v2 (DR-006): the request rode the connect exchange; the server
        reports the join outcome inside ``sessionReady.parameters.webrtc`` —
        ``connected`` (direct mode), ``failed`` (fatal), or an absent key
        (fatal: the server does not support direct mode).
        """
        if self._state is _DirectState.FAILED:
            return  # e.g. arrived after the join timeout: the session is closing
        # Stop the join timer before anything awaits, so it cannot fire while
        # SESSION_READY handlers run.
        self._cancel_join_timer()
        if message.parameters is not None:
            self._session_data = message.parameters
        self._initialized = True
        self._tracer.span("lifecycle", "connect", self._tr_connect_start)
        await self._events.emit(STVEvent.SESSION_READY, session_data=self._session_data)
        if self._state is not _DirectState.PENDING:
            return  # one negotiation per connection — already resolved
        result = (message.parameters or {}).get("webrtc")
        if not isinstance(result, dict):
            await self._fail_unsupported("sessionReady carried no webrtc result")
            return
        status = str(result.get("status") or "")
        if status == "connected":
            await self._on_direct_connected(result)
        elif status == "failed":
            error = result.get("error") or {}
            await self._fail_direct(
                str(error.get("code") or ""),
                str(error.get("message") or ""),
                outcome="failed",
            )
        else:
            await self._fail_unsupported(f"unrecognised webrtc status {status!r}")

    async def _on_direct_connected(self, result: dict) -> None:
        """Open the direct path: start feed tasks and flush held input (no seed)."""
        self._state = _DirectState.CONNECTED
        join_ms = round((self._clock() - self._request_sent_at) * 1000.0, 1)
        participant_id = str(result.get("participant_id") or "")
        if self._tr_negotiate_start is not None:
            self._tracer.span(
                "webrtc",
                "negotiate",
                self._tr_negotiate_start,
                args={"outcome": "connected", "join_ms": join_ms},
            )
            self._tr_negotiate_start = None
        self._set_trace_other(
            "webrtc",
            {
                "mode": "direct",
                "provider": str(
                    result.get("provider") or self._webrtc_settings.provider
                ),
                "join_ms": join_ms,
                "participant_id": participant_id,
            },
        )
        self._first_frame_pending = True
        self._last_metadata_at = self._clock()
        self._start_feed_tasks()
        if self._watchdog_task is None:
            self._watchdog_task = asyncio.create_task(self._metadata_watchdog_loop())
        await self._events.emit(
            STVEvent.WEBRTC_CONNECTED, participant_id=participant_id or None
        )
        await self._flush_preinit_inputs()

    # ------------------------------------------------------------------
    # Failure paths + async post-connect notifications
    # ------------------------------------------------------------------

    async def _fail(self, code: str, message: str, negotiate_args: dict) -> None:
        """Terminal failure: emit a fatal ERROR, then close the session.

        Fail-fast by design — no rejoin, no fallback to another transport: the
        caller asked for WebRTC, and a silent downgrade would mask breakage.
        """
        if self._state is _DirectState.FAILED:
            return
        self._cancel_join_timer()
        self._state = _DirectState.FAILED
        self._discard_held_input()
        if self._tr_negotiate_start is not None:
            self._tracer.span(
                "webrtc", "negotiate", self._tr_negotiate_start, args=negotiate_args
            )
            self._tr_negotiate_start = None
        await self._events.emit(STVEvent.ERROR, message=message, code=code, fatal=True)
        self._schedule_close()

    async def _fail_direct(self, error_code: str, detail: str, outcome: str) -> None:
        """Report that the server could not join (or lost) the room — fatal."""
        await self._fail(
            WEBRTC_JOIN_FAILED,
            f"webrtc negotiation failed ({error_code or 'unknown'})"
            + (f": {detail}" if detail else ""),
            {"outcome": outcome, "error_code": error_code},
        )

    async def _fail_unsupported(self, reason: str) -> None:
        """Report that the server cannot publish this session into a room — fatal."""
        logger.error("Direct WebRTC is not available for this session: %s", reason)
        self._tracer.instant("webrtc", "unsupported", args={"reason": reason})
        await self._fail(
            WEBRTC_UNSUPPORTED,
            f"The server does not support direct WebRTC for this session ({reason})",
            {"outcome": "unsupported", "reason": reason},
        )

    async def _on_webrtc_status(self, status: OjinWebRTCStatusMessage) -> None:
        """Handle an async post-connect ``webrtcStatus`` notification.

        Protocol v2 demoted ``webrtcStatus`` to mid-session transitions only:
        ``disconnected`` (telemetry) then ``failed`` with ``REJOIN_FAILED``
        (fatal — the fail-fast, no-rejoin policy is unchanged). The
        negotiation-phase statuses (``connecting``/``connected``) no longer
        exist on the wire.
        """
        error = status.error or {}
        error_code = str(error.get("code") or "")
        args: dict = {"status": status.status}
        if error_code:
            args["error_code"] = error_code
        self._tracer.instant("webrtc", "webrtc_status", args=args)
        if status.status == "failed":
            await self._fail_direct(
                error_code, str(error.get("message") or ""), outcome="failed"
            )
        elif status.status == "disconnected":
            logger.warning(
                "webrtc transport disconnected (provider=%s)", status.provider
            )
        else:
            logger.warning(
                "Unexpected webrtcStatus %r in protocol v2 — ignored", status.status
            )

    async def _join_timer(self) -> None:
        """Fire the sessionReady timeout unless the outcome lands first."""
        await asyncio.sleep(self._webrtc_settings.webrtc_join_timeout_s)
        await self._handle_join_timeout()

    async def _handle_join_timeout(self) -> None:
        """Fail the session if no sessionReady arrived within the join timeout.

        In v2 the server joins the room during setup, so
        ``webrtc_join_timeout_s`` governs the whole ``sessionReady`` wait.
        """
        if self._state is not _DirectState.PENDING or self._initialized:
            return
        self._tracer.instant("webrtc", "join_timeout")
        await self._fail(
            WEBRTC_JOIN_FAILED,
            f"No sessionReady within {self._webrtc_settings.webrtc_join_timeout_s} s",
            {"outcome": "timeout"},
        )

    def _cancel_join_timer(self) -> None:
        """Stop the join timer without awaiting it (safe from any task)."""
        task = self._join_timer_task
        self._join_timer_task = None
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    def _discard_held_input(self) -> None:
        """Drop everything held for the never-opened (or now-dead) direct path."""
        if self._preinit_inputs:
            self._preinit_inputs.clear()
            self._tracer.instant("tts_input", "preinit_buffer_discarded")
        self._interrupt_deferred.clear()
        self._deferring_input = False

    # ------------------------------------------------------------------
    # Metadata frames — clock, events, ack window
    # ------------------------------------------------------------------

    def _validate_frame_type(self, frame_type: int) -> bool:
        """Reject the spurious boomerang START_OF_SPEECH right after SPEECH."""
        if (
            self._last_frame_type == FrameType.SPEECH
            and frame_type == FrameType.START_OF_SPEECH
        ):
            logger.warning(
                "Received BOOMERANG frame after SPEECH_FRAME frame; "
                "this is unexpected and may indicate a server-side issue."
            )
            return False
        return True

    async def _on_interaction_response(
        self, message: OjinInteractionResponseMessage
    ) -> None:
        """Process one frame: discard payloads, derive clock/events from metadata."""
        if self._state is not _DirectState.CONNECTED:
            return  # frames outside direct mode: parsed, discarded
        self._last_metadata_at = self._clock()
        frame_type = int(message.frame_type)
        if not self._validate_frame_type(frame_type):
            return
        if self._first_frame_pending:
            self._first_frame_pending = False
            self._tracer.instant(
                "webrtc", "first_metadata_frame", args={"frame_type": frame_type}
            )
            await self._events.emit(STVEvent.FIRST_FRAME, frame_type=frame_type)

        prev = self._last_frame_type
        self._last_frame_type = frame_type
        self._tracer.instant(
            recv_lane_for_frame_type(frame_type),
            "frame_recv",
            cat=str(frame_type),
            args={"frame_type": frame_type},
        )
        self._tracer.counter("recv_frame_type", frame_type)

        if frame_type in _SPEECH_TYPES:
            self._advance_played_real_ms(_MS_PER_METADATA_FRAME)

        if self._interruption_ongoing and frame_type in _SILENCE_TYPES:
            self._interruption_ongoing = False
            self._tracer.instant("interruption", "interrupt_ended")
        if (
            frame_type == FrameType.START_OF_SPEECH
            and self._tr_interrupt_start is not None
        ):
            self._tracer.span(
                "interruption", "interrupt→new_turn", self._tr_interrupt_start
            )
            self._tr_interrupt_start = None
        if (
            self._awaiting_first_recv_frame
            and self._tr_first_tts_audio_at is not None
            and frame_type in _SPEECH_TYPES
        ):
            self._awaiting_first_recv_frame = False
            self._tracer.record_response_latency("recv", self._tr_first_tts_audio_at)

        await self._emit_speaking_edges(prev, frame_type)

    async def _emit_speaking_edges(self, prev: Optional[int], frame_type: int) -> None:
        """Emit BOT_STARTED/STOPPED_SPEAKING on frame-type transitions."""
        prev_effective = prev if prev is not None else int(FrameType.IDLE)
        if prev_effective in _SILENCE_TYPES and frame_type in _SPEECH_TYPES:
            self._tr_speaking_start = self._tracer.mark()
            await self._events.emit(STVEvent.BOT_STARTED_SPEAKING)
        elif prev_effective in _SPEECH_TYPES and frame_type in _SILENCE_TYPES:
            if self._tr_speaking_start is not None:
                self._tracer.span("speaking", "bot_speaking", self._tr_speaking_start)
                self._tr_speaking_start = None
            # The fed audio has been consumed; a silent bot needs no cancel until
            # this turn feeds more audio.
            self._turn_audio_fed = False
            await self._events.emit(STVEvent.BOT_STOPPED_SPEAKING)

    # ------------------------------------------------------------------
    # Metadata watchdog (log-only liveness)
    # ------------------------------------------------------------------

    def _check_metadata_watchdog(self) -> bool:
        """Log when no metadata frame arrived within the watchdog window."""
        if self._state is not _DirectState.CONNECTED or not self._initialized:
            return False
        gap_s = self._clock() - self._last_metadata_at
        if gap_s < _METADATA_WATCHDOG_S:
            return False
        logger.warning(
            "No metadata frame for %.1f s — direct-webrtc signal layer may be stalled",
            gap_s,
        )
        self._tracer.instant(
            "webrtc", "metadata_watchdog", args={"gap_s": round(gap_s, 1)}
        )
        self._last_metadata_at = self._clock()  # re-arm instead of spamming
        return True

    async def _metadata_watchdog_loop(self) -> None:
        """Poll the metadata watchdog while the session is live."""
        while self._initialized:
            await asyncio.sleep(_WATCHDOG_POLL_S)
            self._check_metadata_watchdog()

    # ------------------------------------------------------------------
    # Trace helpers
    # ------------------------------------------------------------------

    def _set_trace_other(self, key: str, value: object) -> None:
        """Attach an otherData entry when the tracer supports it."""
        setter = getattr(self._tracer, "set_other_data", None)
        if callable(setter):
            setter(key, value)
