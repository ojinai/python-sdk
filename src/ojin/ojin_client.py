"""WebSocket client for OJIN Speech-To-Video service."""

import asyncio
import contextlib
import json
import logging
import socket
import time
from typing import Dict, Optional, Type, TypeVar

import websockets
from pydantic import BaseModel
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import (
    ConnectionClosedError,
    ConnectionClosedOK,
    WebSocketException,
)

from ojin.entities.interaction_messages import (
    CancelInteractionMessage,
    ErrorResponse,
    ErrorResponseMessage,
    InteractionInput,
    InteractionInputMessage,
    InteractionResponseMessage,
)
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
                logger.debug("TCP_NODELAY enabled on client websocket")
    except Exception as e:
        logger.warning("Failed to set TCP_NODELAY: %s", e)


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

        """
        super().__init__()
        self.ws_url = ws_url
        self.api_key = api_key
        self.config_id = config_id
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self._ws: Optional[ClientConnection] = None
        self._available_response_messages_queue: asyncio.Queue[BaseModel] = (
            asyncio.Queue()
        )
        self._running = False
        self._receive_task: Optional[asyncio.Task] = None
        self._inference_server_ready: bool = False
        self._cancelled: bool = False
        self._active_interaction_id: str | None = None
        self._process_messages_task: Optional[asyncio.Task] = None
        self._pending_client_messages_queue: asyncio.Queue[OjinMessage] = (
            asyncio.Queue()
        )
        self._mode: str | None = mode
        self._pending_first_input: bool = False

    async def connect(self) -> None:
        """Establish WebSocket connection and authenticate with the service."""
        if self._running:
            logger.warning("Client is already connected")
            return

        attempt = 0
        last_error = None

        while attempt < self.reconnect_attempts:
            try:
                headers = {"Authorization": f"{self.api_key}"}

                # Add query parameters for API key and config ID
                url = f"{self.ws_url}?config_id={self.config_id}"
                if self._mode == "dev":
                    url = f"{url}&mode={self._mode}"
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
                logger.info("Successfully connected to OJIN STV service")
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

        logger.info("Disconnected from OJIN STV service")

    async def _receive_server_messages(self) -> None:
        """Continuously receive and process incoming messages from the server."""
        if not self._ws:
            raise RuntimeError("WebSocket connection not established")

        try:
            async for message in self._ws:
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
                    logger.debug(
                        "Received InteractionResponse for id %s",
                        interaction_response.interaction_id,
                    )

                    if (
                        interaction_response.interaction_id
                        != self._active_interaction_id
                    ):
                        logger.debug("Interaction id changed.")
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

                logger.info("Received message: %s", msg)
            else:
                logger.warning("Unknown message type: %s", msg_type)

        except Exception as e:
            logger.exception("Error handling message: %s", e)
            raise RuntimeError(e) from e

    async def start_interaction(self) -> None:
        """Drain any pending response messages before starting a new interaction."""
        self._drain_response_messages()

    def _drain_response_messages(self) -> None:
        """Remove stale queued response messages without blocking."""
        while not self._available_response_messages_queue.empty():
            self._available_response_messages_queue.get_nowait()
            with contextlib.suppress(ValueError):
                self._available_response_messages_queue.task_done()

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
            logger.info("Interrupt")

            self._cancelled = True
            cancel_input = CancelInteractionMessage(payload=message.to_proxy_message())

            await self._ws.send(cancel_input.model_dump_json())

            self._drain_response_messages()

            self._cancelled = False

            return

        if isinstance(
            message,
            (OjinAudioInputMessage, OjinTextInputMessage, OjinEndInteractionMessage),
        ):
            logger.info("InteractionMessage")
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
                logger.debug("[_process_messages:] no websocket connection.")
                await asyncio.sleep(1.0)
                continue

            message: OjinMessage = await self._pending_client_messages_queue.get()
            if isinstance(message, OjinAudioInputMessage):
                max_chunk_size = 1024 * 500
                audio_chunks = [
                    message.audio_int16_bytes[i : i + max_chunk_size]
                    for i in range(0, len(message.audio_int16_bytes), max_chunk_size)
                ]
                logger.info(
                    "Split audio into %d chunks of max %d bytes",
                    len(audio_chunks),
                    max_chunk_size,
                )

                # NOTE(mouad): make sure we handle the case where the input is empty
                if len(audio_chunks) == 0:
                    audio_chunks.append(bytes())

                for _, chunk in enumerate(audio_chunks):
                    interaction_input = InteractionInput(
                        payload_type="audio",
                        payload=chunk,
                        timestamp=int(time.monotonic() * 1000),
                        params=message.params,
                    )
                    proxy_message = InteractionInputMessage(payload=interaction_input)

                    await self._ws.send(proxy_message.to_bytes())

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
        return self._running and self._ws is not None and self._ws.open
