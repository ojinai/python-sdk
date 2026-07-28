"""WebSocket client for OJIN Speech-To-Video service."""

import array
import asyncio
import contextlib
import inspect
import json
import logging
import socket
import struct
import time
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type, TypeVar
from urllib.parse import urlencode

if TYPE_CHECKING:  # annotation only — avoid a runtime ojin -> ojin.stv cycle
    from ojin.stv.config import WebRTCSettings

try:  # FIONREAD socket probe is Unix-only; absent on Windows.
    import fcntl
    import termios
except ImportError:  # pragma: no cover - non-Unix platforms
    fcntl = None  # type: ignore[assignment]
    termios = None  # type: ignore[assignment]

import websockets
from pydantic import BaseModel
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import (
    ConnectionClosedError,
    ConnectionClosedOK,
    WebSocketException,
)
from websockets.protocol import State

from ojin.entities.interaction_messages import (
    CancelInteractionMessage,
    ErrorResponse,
    ErrorResponseMessage,
    InteractionInput,
    InteractionInputMessage,
    InteractionResponseMessage,
)
from ojin.entities.session_messages import SessionUpdateMessage
from ojin.ojin_client_messages import (
    IOjinClient,
    OjinAudioInputMessage,
    OjinCancelInteractionMessage,
    OjinEndInteractionMessage,
    OjinInteractionResponseMessage,
    OjinMessage,
    OjinSessionReadyMessage,
    OjinSessionReadyPing,
    OjinTextInputMessage,
    OjinWebRTCStatusMessage,
)

T = TypeVar("T", bound=OjinMessage)

logger = logging.getLogger(__name__)


def _configure_tcp_nodelay(ws: ClientConnection) -> None:
    """Enable TCP_NODELAY on the WebSocket transport socket for lower latency."""
    try:
        transport = ws.transport
        if transport is not None:
            sock = transport.get_extra_info("socket")
            if sock is not None:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception as e:
        logger.warning("Failed to set TCP_NODELAY: %s", e)


# ``struct tcp_info`` (linux/tcp.h) prefix: 8 x u8 then 24 x u32 (rto..total_retrans).
# Stable across kernels. Lets us see the *receive* side of the proxy->client TCP
# connection from inside the client (e.g. WSL2): RTT, loss/retransmits, and the
# receive-window estimate the client offers.
_TCP_INFO_FMT = "<8B24I"
_TCP_INFO_LEN = struct.calcsize(_TCP_INFO_FMT)  # 104 bytes
_TCP_INFO_BUF = 256
_TCP_INFO_FIELDS = {
    "tcp_retransmits": 2,  # u8: hard retransmit count
    "tcp_rtt_us": 23,  # smoothed RTT to the proxy (microseconds)
    "tcp_rttvar_us": 24,  # RTT variance (microseconds)
    "tcp_snd_cwnd": 26,  # send congestion window (MSS units)
    "tcp_rcv_space": 30,  # receive-window estimate this end advertises (bytes)
    "tcp_retrans_total": 31,  # cumulative segment retransmissions
}
# Extended tcp_info fields (appended by newer kernels at fixed byte offsets);
# each read is guarded by buffer length and simply absent on older kernels.
# Receive-side picks: kernel-level ingress byte count (advances even while the
# app is slow to read — splits "network delivered late" from "we read late"),
# out-of-order packet count (direct download-path loss/reorder evidence), and
# the path's floor RTT.
_TCP_INFO_EXT_FIELDS = {
    "tcp_bytes_received": (128, "<Q"),
    "tcp_min_rtt_us": (148, "<I"),
    "tcp_rcv_ooopack": (224, "<I"),
}


def _read_tcp_info(sock: socket.socket) -> dict[str, int]:
    """Return selected ``tcp_info`` fields for ``sock`` (``{}`` if unavailable)."""
    tcp_info_opt = getattr(socket, "TCP_INFO", None)
    if tcp_info_opt is None:
        return {}
    try:
        raw = sock.getsockopt(socket.IPPROTO_TCP, tcp_info_opt, _TCP_INFO_BUF)
        if len(raw) < _TCP_INFO_LEN:
            return {}
        t = struct.unpack(_TCP_INFO_FMT, raw[:_TCP_INFO_LEN])
        fields = {name: int(t[idx]) for name, idx in _TCP_INFO_FIELDS.items()}
        for name, (offset, fmt) in _TCP_INFO_EXT_FIELDS.items():
            if len(raw) >= offset + struct.calcsize(fmt):
                fields[name] = int(struct.unpack_from(fmt, raw, offset)[0])
        return fields
    except Exception:
        return {}


class OjinClient(IOjinClient):
    """WebSocket client for communicating with the OJIN STV service.

    This client handles the WebSocket connection, authentication, and message
    serialization/deserialization for the OJIN STV service.
    """

    def __init__(  # noqa: PLR0913
        self,
        ws_url: str,
        api_key: str,
        config_id: str,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
        mode: str | None = None,
        max_input_chunk_bytes: int = 1024 * 500,
        send_chunk_gap_s: float = 0.0,
    ):
        """Initialize the OJIN STV WebSocket client.

        Args:
            ws_url: WebSocket URL of the OJIN STV service
            api_key: API key for authentication
            config_id: Configuration ID for the persona
            reconnect_attempts: Number of reconnection attempts on failure
            reconnect_delay: Delay between reconnection attempts in seconds
            mode: Optional mode string for the connection (e.g., "dev" for
                development mode). When set to "dev", adds mode parameter to
                the WebSocket URL query string. Defaults to None.
            max_input_chunk_bytes: Cap on a single audio send; larger
                ``OjinAudioInputMessage`` payloads are split into this many bytes
                per WebSocket frame.
            send_chunk_gap_s: Minimum gap between consecutive audio sends while a
                backlog remains (split chunks of one message, or a queue of pending
                messages). Keeps a buffered burst from flooding the server; steady
                realtime (one message at a time, queue drains empty) is never
                delayed. Defaults to 0 (no pacing); ``OjinSTVClient`` passes its
                configured value.

        """
        super().__init__()
        self.ws_url = ws_url
        self.api_key = api_key
        self.config_id = config_id
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self._max_input_chunk_bytes = max(1, max_input_chunk_bytes)
        self._send_chunk_gap_s = max(0.0, send_chunk_gap_s)
        # Monotonic time of the last on-wire audio send; seeds far in the past so
        # the first send after connect/idle never waits.
        self._last_audio_send_t = float("-inf")

        self._ws: Optional[ClientConnection] = None
        self._available_response_messages_queue: asyncio.Queue[BaseModel] = (
            asyncio.Queue()
        )
        self._running = False
        self._receive_task: Optional[asyncio.Task] = None
        # Cumulative application-level bytes taken off the websocket; the
        # tracer derives an ingress-bandwidth lane from its slope.
        self._ingress_bytes_total: int = 0
        self._inference_server_ready: bool = False
        self._cancelled: bool = False
        self._active_interaction_id: str | None = None
        self._process_messages_task: Optional[asyncio.Task] = None
        self._pending_client_messages_queue: asyncio.Queue[OjinMessage] = (
            asyncio.Queue()
        )
        self._mode: str | None = mode
        self._pending_first_input: bool = False
        self._webrtc_status_callback: Optional[
            Callable[[OjinWebRTCStatusMessage], object]
        ] = None
        # Direct-WebRTC settings declared in the WebSocket upgrade request
        # (protocol v2, DR-006 as amended 2026-07-24): non-secret fields as
        # webrtc_* query params, the meeting token as the X-Ojin-Webrtc-Token
        # header. Carries a secret — never log it.
        self._webrtc_settings: Optional["WebRTCSettings"] = None

    def set_webrtc_connect_settings(self, settings: "WebRTCSettings") -> None:
        """Declare direct-WebRTC settings for the connection's upgrade request.

        Protocol v2 (DR-006, 2026-07-24 amendment): the proxy authors the
        ``sessionSetup`` the server sees, so the client's request travels in
        the WebSocket UPGRADE REQUEST itself — ``webrtc_*`` query params for
        the non-secret fields and the ``X-Ojin-Webrtc-Token`` header for the
        meeting token (same query/header split as ``config_id`` /
        ``Authorization``; headers stay out of access logs). Must be called
        before :meth:`connect`. The token is never placed in the URL and
        never logged.
        """
        self._webrtc_settings = settings

    def set_webrtc_status_callback(
        self, callback: Callable[[OjinWebRTCStatusMessage], object]
    ) -> None:
        """Register the handler invoked inline for each ``webrtcStatus`` message.

        The callback (sync or async) is called from the receive loop; statuses
        are never enqueued on the drainable response queue, so a cancel-time
        drain can never drop one.
        """
        self._webrtc_status_callback = callback

    async def connect(self) -> None:
        """Establish WebSocket connection and authenticate with the service."""
        if self._running:
            logger.warning("Client is already connected")
            return

        attempt = 0
        last_error = None

        while attempt < self.reconnect_attempts:
            try:
                url, headers = self._build_connect_request()
                self._ws = await websockets.connect(
                    url,
                    additional_headers=headers,
                    ping_interval=30,
                    ping_timeout=10,
                    # Disable permessage-deflate. The STV stream is JPEG video
                    # frames + PCM audio — JPEG is already compressed, so deflate
                    # saves ~nothing on size while its synchronous zlib INFLATE
                    # runs in the asyncio transport's ``data_received`` on the bot
                    # event-loop thread. A turn's frame burst was blocking the
                    # playback loop ~120ms there, stuttering audio. No compression
                    # ⇒ no inflate on the loop.
                    compression=None,
                )
                # Enable TCP_NODELAY to reduce latency for real-time frames
                _configure_tcp_nodelay(self._ws)
                self._running = True
                self._receive_task = asyncio.create_task(
                    self._receive_server_messages()
                )
                self._process_messages_task = asyncio.create_task(
                    self._process_client_messages()
                )
                return
            except WebSocketException as e:  # noqa: PERF203
                last_error = e
                attempt += 1
                if attempt < self.reconnect_attempts:
                    logger.warning(
                        "Connection attempt %d/%d failed. Retrying in %d seconds...",
                        attempt,
                        self.reconnect_attempts,
                        self.reconnect_delay,
                    )
                    await asyncio.sleep(self.reconnect_delay)

        logger.error("Failed to connect after %d attempts", self.reconnect_attempts)
        raise ConnectionError(f"Failed to connect to OJIN STV service: {last_error}")

    def _build_connect_request(self) -> tuple[str, Dict[str, str]]:
        """Build the WebSocket upgrade URL and headers.

        Direct-WebRTC settings (when declared) ride the upgrade request per
        protocol v2: non-secret fields as URL-encoded ``webrtc_*`` query
        params, the meeting token as the ``X-Ojin-Webrtc-Token`` header. The
        token never enters the URL, and neither the URL nor the headers are
        logged.
        """
        headers: Dict[str, str] = {"Authorization": f"{self.api_key}"}
        url = f"{self.ws_url}?config_id={self.config_id}"
        if self._mode == "dev":
            url = f"{url}&mode={self._mode}"
        if self._webrtc_settings is not None:
            url = f"{url}&{urlencode(self._webrtc_settings.to_connect_query_params())}"
            headers.update(self._webrtc_settings.to_connect_headers())
        return url, headers

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if not self._running:
            return

        self._running = False
        self._active_interaction_id = None

        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.error("Error closing WebSocket connection: %s", e)
            self._ws = None

        if self._process_messages_task:
            self._process_messages_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._process_messages_task
            self._process_messages_task = None

        if self._receive_task:
            self._receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receive_task
            self._receive_task = None

    async def _receive_server_messages(self) -> None:
        """Continuously receive and process incoming messages from the server."""
        if not self._ws:
            raise RuntimeError("WebSocket connection not established")

        try:
            async for message in self._ws:
                self._ingress_bytes_total += len(message)
                await self._handle_server_message(message)
        except (ConnectionClosedOK, ConnectionClosedError) as e:
            if self._running:  # Only log if we didn't initiate the close
                logger.error("WebSocket connection closed: %s", e)
        except Exception as e:
            logger.exception("Error in WebSocket receive loop: %s", e)
        finally:
            await self.close()

    async def _handle_server_message(self, message: str | bytes) -> None:
        """Handle an incoming WebSocket message.

        Args:
            message: Raw JSON message from WebSocket

        """
        try:
            if isinstance(message, bytes):
                try:
                    interaction_server_response = InteractionResponseMessage.from_bytes(
                        message
                    )
                    interaction_response = (
                        OjinInteractionResponseMessage.from_proxy_message(
                            interaction_server_response
                        )
                    )

                    if (
                        interaction_response.interaction_id
                        != self._active_interaction_id
                    ):
                        self._active_interaction_id = (
                            interaction_response.interaction_id
                        )

                    await self._available_response_messages_queue.put(
                        interaction_response
                    )
                    return
                except Exception as e:
                    logger.error(e)
                    raise

            if message == "No backend servers available. Please try again later.":
                # Proxy structured-log follow-up: oji-retv.
                logger.error(
                    "No backend servers available",
                    extra={
                        "component": "ojin_client",
                        "error_code": "no_backend_servers",
                        "error_message": message,
                        "config_id": self.config_id,
                        "active_interaction_id": self._active_interaction_id,
                    },
                )
                await self._available_response_messages_queue.put(
                    ErrorResponseMessage(
                        payload=ErrorResponse(
                            interaction_id=None,
                            code="NO_BACKEND_SERVER_AVAILABLE",
                            message=message,
                            timestamp=int(time.time() * 1000),
                            details=None,
                        )
                    )
                )
                raise RuntimeError(message)

            try:
                data = json.loads(message)
            except (json.JSONDecodeError, ValueError):
                # The proxy can send a plain-text error (e.g. "Model not found"
                # for an unresolvable model/config) rather than JSON. Surface it as
                # a clean ErrorResponseMessage so consumers see a proper error
                # instead of the receive loop dying on a JSON parse exception.
                logger.error(
                    "Server returned non-JSON message",
                    extra={
                        "component": "ojin_client",
                        "error_code": "server_text_error",
                        "error_message": str(message),
                        "config_id": self.config_id,
                        "active_interaction_id": self._active_interaction_id,
                    },
                )
                await self._available_response_messages_queue.put(
                    ErrorResponseMessage(
                        payload=ErrorResponse(
                            interaction_id=self._active_interaction_id,
                            code="SERVER_ERROR",
                            message=str(message),
                            timestamp=int(time.time() * 1000),
                            details=None,
                        )
                    )
                )
                return
            msg_type = data.get("type")

            if msg_type == "webrtcStatus":
                await self._dispatch_webrtc_status(data)
                return

            # Map message types to their corresponding classes
            message_types: Dict[str, Type[BaseModel]] = {
                "sessionReady": OjinSessionReadyMessage,
                "errorResponse": ErrorResponseMessage,
                "sessionPing": OjinSessionReadyPing,
            }

            if msg_type in message_types:
                msg_class = message_types[msg_type]
                # Convert the message data to the appropriate message class
                # Convert the raw data to the matching message class
                if msg_type == "sessionReady":
                    payload = data.get("payload", {})
                    session_ready = OjinSessionReadyMessage(
                        parameters=payload.get("parameters"),
                    )
                    self._inference_server_ready = True
                    await self._available_response_messages_queue.put(session_ready)
                    return

                msg = msg_class(**data)
                await self._available_response_messages_queue.put(msg)

                if isinstance(msg, ErrorResponseMessage):
                    raise RuntimeError(f"Error in Inference Server received: {msg}")

                if isinstance(msg, OjinSessionReadyPing):
                    # Discard session pings
                    pass
            else:
                logger.warning("Unknown message type: %s", msg_type)

        except Exception as e:
            logger.exception("Error handling message: %s", e)
            raise RuntimeError(e) from e

    async def _dispatch_webrtc_status(self, data: dict) -> None:
        """Parse a ``webrtcStatus`` message and deliver it to the callback.

        Parse and dispatch share one try/except: neither a malformed payload
        nor a throwing callback may escape into the receive loop's fatal
        error path.
        """
        try:
            status = OjinWebRTCStatusMessage(**(data.get("payload") or {}))
            callback = self._webrtc_status_callback
            if callback is None:
                logger.warning("webrtcStatus received with no callback registered")
                return
            result = callback(status)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("Error handling webrtcStatus message — dropped")

    async def start_interaction(self) -> None:
        """Drain any pending response messages before starting a new interaction."""
        self._drain_response_messages()

    def _drain_response_messages(self) -> None:
        """Remove stale queued response messages without blocking."""
        while not self._available_response_messages_queue.empty():
            self._available_response_messages_queue.get_nowait()
            with contextlib.suppress(ValueError):
                self._available_response_messages_queue.task_done()

    def _drain_client_messages(self) -> None:
        """Drop queued outbound client messages (e.g. on cancel) without blocking."""
        while not self._pending_client_messages_queue.empty():
            self._pending_client_messages_queue.get_nowait()
            with contextlib.suppress(ValueError):
                self._pending_client_messages_queue.task_done()

    async def send_message(self, message: BaseModel) -> None:
        """Send a message to the OJIN STV service.

        Args:
            message: The message to send

        Raises:
            ConnectionError: If not connected to the WebSocket

        """
        if not self._ws or not self._running:
            raise ConnectionError("Not connected to OJIN STV service")

        if self._inference_server_ready is not True:
            raise ConnectionError("Inference Server is not ready to receive messages")

        if isinstance(message, OjinCancelInteractionMessage):
            self._cancelled = True
            # Drop any audio still queued to be sent for the turn being cancelled:
            # the server discards post-cancel input, so forwarding it would only
            # desync the next turn. (The in-flight send loop also bails on
            # ``_cancelled`` between chunks — see _process_client_messages.)
            self._drain_client_messages()
            cancel_input = CancelInteractionMessage(payload=message.to_proxy_message())

            await self._ws.send(cancel_input.model_dump_json())

            self._drain_response_messages()

            self._cancelled = False

            return

        if isinstance(message, SessionUpdateMessage):
            await self._ws.send(message.model_dump_json())
            return

        if isinstance(
            message,
            (OjinAudioInputMessage, OjinTextInputMessage, OjinEndInteractionMessage),
        ):
            await self._pending_client_messages_queue.put(message)
            return

        logger.error("The message %s is Unknown", message)
        error = ErrorResponseMessage(
            payload=ErrorResponse(
                interaction_id=self._active_interaction_id,
                code="UNKNOWN",
                message="The message is Unknown",
                timestamp=int(time.monotonic() * 1000),
                details=None,
            )
        )
        raise RuntimeError(error)

    async def _process_client_messages(self) -> None:
        while self._running:
            if self._cancelled:
                await asyncio.sleep(0.01)  # Avoid busy-waiting
                continue

            if self._ws is None:
                await asyncio.sleep(1.0)
                continue

            message: OjinMessage = await self._pending_client_messages_queue.get()
            if isinstance(message, OjinAudioInputMessage):
                max_chunk_size = self._max_input_chunk_bytes
                audio_chunks = [
                    message.audio_int16_bytes[i : i + max_chunk_size]
                    for i in range(0, len(message.audio_int16_bytes), max_chunk_size)
                ]

                # NOTE(mouad): make sure we handle the case where the input is empty
                if len(audio_chunks) == 0:
                    audio_chunks.append(bytes())

                for chunk in audio_chunks:
                    if self._cancelled:
                        # A cancel landed mid-send: abandon the rest of this (now
                        # cancelled) turn's audio instead of finishing the burst.
                        break
                    # Pace the feed BY TIME: enforce a minimum interval between
                    # consecutive audio sends. Gating on queue depth misses the
                    # burst case entirely — a producer that enqueues one message
                    # per await (e.g. TTS buffered during a barge-in then replayed)
                    # interleaves with this consumer so the queue never LOOKS
                    # backlogged while the wire bursts many messages back-to-back
                    # (seen in the 2026-07-12 session-7 trace). Time-based spacing
                    # is race-free: bursts get spread out no matter how the queue
                    # interleaves, while steady realtime (batched emits already
                    # >= ~400 ms apart) never waits.
                    if self._send_chunk_gap_s > 0:
                        elapsed = time.monotonic() - self._last_audio_send_t
                        if elapsed < self._send_chunk_gap_s:
                            await asyncio.sleep(self._send_chunk_gap_s - elapsed)
                        if self._cancelled:
                            break  # a cancel landed during the pacing sleep
                    interaction_input = InteractionInput(
                        payload_type="audio",
                        payload=chunk,
                        timestamp=int(time.monotonic() * 1000),
                        params=message.params,
                    )
                    proxy_message = InteractionInputMessage(payload=interaction_input)

                    await self._ws.send(proxy_message.to_bytes())
                    self._last_audio_send_t = time.monotonic()

            elif isinstance(message, OjinTextInputMessage):
                text_message = message.to_proxy_message()
                await self._ws.send(text_message.to_bytes())

            elif isinstance(message, OjinEndInteractionMessage):
                end_interaction_message = message.to_proxy_message()
                await self._ws.send(end_interaction_message.model_dump_json())

    async def receive_message(self) -> BaseModel | None:
        """Receive the next message from the OJIN STV service.

        Returns:
            The next available message

        Raises:
            asyncio.QueueEmpty: If no messages are available

        """
        if self._cancelled:
            return None
        return await self._available_response_messages_queue.get()

    def is_connected(self) -> bool:
        """Check if the client is connected to the WebSocket."""
        return self._running and self._ws is not None and self._ws.state == State.OPEN

    def debug_queue_depths(self) -> dict[str, int]:
        """Best-effort receive-pipeline depth gauges (diagnostic only).

        Returns the depth of each buffering stage this client owns so a tracer can
        localize where frames back up between the wire and the consumer. Every probe
        is independently guarded: a websockets-internals change, a closed socket, or
        a non-Unix platform yields a missing key rather than an exception, so this is
        safe to call from a hot trace path.

        The keys, when present, are:

        - ``sock_bytes``: bytes ACKed by TCP but not yet read by us (kernel recv
          buffer, via ``FIONREAD``). Grows when the event loop can't read fast
          enough — e.g. starved by the playback busy-wait.
        - ``ws_frames``: frames assembled by ``websockets`` but not yet consumed by
          the ``async for`` receive loop. Capped at ``max_queue`` (default 16)
          before read-backpressure pauses the wire.
        - ``ws_paused``: ``1`` when ``websockets`` has paused reading (the cap was
          hit and TCP backpressure is engaged), else ``0``.
        - ``server_msgs``: parsed responses sitting in the (unbounded) queue awaiting
          :meth:`receive_message`. Grows when downstream consumption (decode/
          playback) lags — the absorber that hides backpressure upstream.
        - ``tcp_rtt_us`` / ``tcp_rttvar_us`` / ``tcp_retrans_total`` /
          ``tcp_snd_cwnd`` / ``tcp_rcv_space``: TCP path conditions (from
          ``TCP_INFO``) on the client end of the proxy→client connection. High
          ``tcp_rtt_us``/``tcp_rttvar_us`` or a climbing ``tcp_retrans_total``
          point to a lossy/throttled local path (e.g. WSL2 NAT / WAN) rather
          than the proxy or server. Linux-only; absent elsewhere.

        Returns:
            Mapping of gauge name to depth; a key is present only when readable.

        """
        depths: dict[str, int] = {}
        ws = self._ws
        if ws is not None:
            with contextlib.suppress(Exception):
                # websockets' asyncio Assembler buffers frames in a custom SimpleQueue
                # that supports len() but not qsize(); the stdlib queue.SimpleQueue is
                # the reverse. Try len() first, fall back to qsize().
                frames = ws.recv_messages.frames
                try:
                    depths["ws_frames"] = len(frames)
                except TypeError:
                    depths["ws_frames"] = frames.qsize()  # pyright: ignore[reportAttributeAccessIssue]
            with contextlib.suppress(Exception):
                depths["ws_paused"] = int(ws.recv_messages.paused)
            with contextlib.suppress(Exception):
                transport = ws.transport
                sock = transport.get_extra_info("socket") if transport else None
                if sock is not None:
                    if fcntl is not None and termios is not None:
                        buf = array.array("i", [0])
                        fcntl.ioctl(sock.fileno(), termios.FIONREAD, buf)
                        depths["sock_bytes"] = buf[0]
                    # TCP_INFO on the client end of the proxy->client conn:
                    # surfaces the *receive*-side path conditions (RTT/loss/
                    # window) so a sub-25fps delivery can be attributed to the
                    # local network (e.g. WSL2 NAT) vs the proxy. Empty on
                    # non-Linux / if the option is unavailable.
                    depths.update(_read_tcp_info(sock))
        with contextlib.suppress(Exception):
            depths["server_msgs"] = self._available_response_messages_queue.qsize()
        # Cumulative app-level ingress; paired with ``tcp_bytes_received`` it
        # splits "network delivered late" from "we read the socket late".
        depths["ingress_bytes_total"] = self._ingress_bytes_total
        return depths
