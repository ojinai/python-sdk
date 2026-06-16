"""FastAPI WebSocket server for handling session messages."""

import asyncio
import json
import logging
import time
import uuid
from enum import Enum
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, ValidationError

from ojin.entities.interaction_messages import (
    InteractionInputMessage,
    InteractionResponse,
    InteractionResponseMessage,
    InteractionResponsePayload,
)

_INTERACTION_ID_DESC = "Client ID linking messages for a single logical interaction"
_TIMESTAMP_CLIENT_DESC = (
    "Timestamp (ms since Unix epoch) when message was sent by client"
)


class MessageType(str, Enum):
    """Enum for WebSocket message types."""

    # Client -> Proxy Messages
    START_INTERACTION = "StartInteraction"
    END_INTERACTION = "EndInteraction"
    INTERACTION_INPUT = "interactionInput"
    SESSION_UPDATE = "sessionUpdate"
    # Proxy -> Client Messages
    SESSION_READY = "sessionReady"
    INTERACTION_RESPONSE = "interactionResponse"
    ERROR_RESPONSE = "errorResponse"
    # Proxy -> Inference Server Messages
    SESSION_SETUP = "sessionSetup"
    INTERACTION_READY = "interactionReady"


class SessionSetupPayload(BaseModel):
    """Payload for session setup messages."""

    model_id: str = Field(..., description="Identifier for the target inference model")
    trace_id: str = Field(..., description="Identifier for the session")
    parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional model-specific parameters (key-value pairs)",
    )
    timestamp: Optional[int] = Field(
        None,
        description=(
            "Timestamp in milliseconds when the message was sent by the client"
        ),
    )


class SessionUpdatePayload(BaseModel):
    """Payload for session update messages."""

    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Updated model-specific parameters"
    )
    timestamp: Optional[int] = Field(
        None,
        description=(
            "Timestamp in milliseconds when the message was sent by the client"
        ),
    )


class SessionReadyPayload(BaseModel):
    """Payload for session ready messages."""

    trace_id: str = Field(
        ...,
        description=(
            "Unique identifier assigned to this WebSocket session by the proxy"
        ),
    )
    status: str = Field("success", description="Indicates successful setup")
    load: float = Field(
        0.5,
        description="Load of the inference server",
    )
    timestamp: Optional[int] = Field(
        None,
        description=(
            "Timestamp in milliseconds when the message was sent by the proxy"
        ),
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional model-specific parameters,"
            " contains details of the configured session."
        ),
    )


class InteractionInputPayload(BaseModel):
    """Payload for interaction input messages."""

    interaction_id: str = Field(
        ...,
        description=_INTERACTION_ID_DESC,
    )
    payload_type: str = Field(
        ...,
        description="Type of the data in the payload",
    )
    payload: bytes = Field(
        ...,
        description="The actual data.",
    )
    params: Optional[Dict[str, Any]] = Field(
        None,
        description=("Optional. Additional parameters for the interaction"),
    )
    timestamp: int = Field(
        ...,
        description=_TIMESTAMP_CLIENT_DESC,
    )


class ErrorResponsePayload(BaseModel):
    """Payload for error response messages."""

    interaction_id: Optional[str] = Field(
        None,
        description=(
            "Optional. The interaction_id related to the error, if applicable"
        ),
    )
    code: str = Field(
        ...,
        description=("Short error code (e.g., AUTH_FAILED, INVALID_MODEL_ID, TIMEOUT)"),
    )
    message: str = Field(
        ...,
        description="A human-readable description of the error",
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description=("Optional. Additional structured details about the error"),
    )
    timestamp: int = Field(
        ...,
        description=("Timestamp (ms since Unix epoch) when message was sent by proxy"),
    )


class SessionSetupMessage(BaseModel):
    """Session setup message sent from client to proxy."""

    type: MessageType = MessageType.SESSION_SETUP
    payload: SessionSetupPayload


class SessionUpdateMessage(BaseModel):
    """Session update message sent from client to proxy."""

    type: MessageType = MessageType.SESSION_UPDATE
    payload: SessionUpdatePayload


class SessionReadyMessage(BaseModel):
    """Session ready message sent from proxy to client."""

    type: MessageType = MessageType.SESSION_READY
    payload: SessionReadyPayload


class ErrorResponseMessage(BaseModel):
    """Error response message sent from proxy to client."""

    type: MessageType = MessageType.ERROR_RESPONSE
    payload: ErrorResponsePayload


class InteractionReady(BaseModel):
    """Interaction Ready."""

    interaction_id: str = Field(
        ...,
        description=_INTERACTION_ID_DESC,
    )
    timestamp: int = Field(
        ...,
        description=_TIMESTAMP_CLIENT_DESC,
    )


class InteractionReadyMessage(BaseModel):
    """Interaction input message sent from client to proxy."""

    type: MessageType = MessageType.INTERACTION_READY
    payload: InteractionReady


class StartInteraction(BaseModel):
    """Start interaction."""

    interaction_id: str = Field(
        ...,
        description=_INTERACTION_ID_DESC,
    )
    timestamp: int = Field(
        ...,
        description=_TIMESTAMP_CLIENT_DESC,
    )


class StartInteractionMessage(BaseModel):
    """Interaction input message sent from client to proxy."""

    type: MessageType = MessageType.START_INTERACTION
    payload: StartInteraction


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="WebSocket Mock Server", version="1.0.0")

# In-memory storage for active sessions
active_sessions: Dict[str, Dict[str, Any]] = {"1": {"trace_id": "1"}}


class WebSocketManager:
    """Manages WebSocket connections and sessions."""

    def __init__(self) -> None:  # noqa: D107
        self.connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Accept a WebSocket connection and store it."""
        await websocket.accept()
        self.connections[session_id] = websocket
        logger.info("WebSocket connected: %s", session_id)

    def disconnect(self, session_id: str) -> None:
        """Remove a WebSocket connection."""
        if session_id in self.connections:
            del self.connections[session_id]
            logger.info("WebSocket disconnected: %s", session_id)

    async def send_message(self, session_id: str, message: BaseModel) -> None:
        """Send a message to a specific WebSocket connection."""
        if session_id in self.connections:
            websocket = self.connections[session_id]
            await websocket.send_text(message.model_dump_json())


# WebSocket manager instance
ws_manager = WebSocketManager()


def get_current_timestamp() -> int:
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)


def generate_trace_id() -> str:
    """Generate a unique trace ID."""
    return str(uuid.uuid4())


async def handle_session_setup(
    _websocket: WebSocket,
    session_id: str,
    message_data: dict,
) -> None:
    """Handle session setup message."""
    try:
        setup_msg = SessionSetupMessage(**message_data)
        payload = setup_msg.payload

        # Store session info
        active_sessions[session_id] = {
            "model_id": payload.model_id,
            "trace_id": payload.trace_id,
            "parameters": payload.parameters or {},
            "status": "ready",
            "created_at": get_current_timestamp(),
        }

        # Send session ready response
        ready_payload = SessionReadyPayload(
            trace_id=payload.trace_id,
            status="success",
            load=0.5,
            timestamp=get_current_timestamp(),
            parameters={"model_id": payload.model_id},
        )
        ready_msg = SessionReadyMessage(payload=ready_payload)

        await ws_manager.send_message(session_id, ready_msg)
        logger.info(
            "Session setup completed for %s, trace_id: %s",
            session_id,
            payload.trace_id,
        )

    except ValidationError as e:
        await send_error_response(session_id, "Invalid session setup: %s" % e)
    except Exception as e:
        await send_error_response(session_id, "Session setup failed: %s" % e)


async def handle_session_update(
    _websocket: WebSocket,
    session_id: str,
    message_data: dict,
) -> None:
    """Handle session update message."""
    try:
        update_msg = SessionUpdateMessage(**message_data)
        payload = update_msg.payload

        logger.info("Session for %s %s", active_sessions, session_id)
        if session_id in active_sessions:
            # Update session parameters
            if payload.parameters:
                active_sessions[session_id]["parameters"].update(payload.parameters)

            logger.info("Session updated for %s", session_id)

            ready_payload = SessionReadyPayload(
                trace_id=active_sessions[session_id]["trace_id"],
                status="updated",
                load=0.5,
                timestamp=get_current_timestamp(),
                parameters=active_sessions[session_id]["parameters"],
            )
            ready_msg = SessionReadyMessage(payload=ready_payload)
            await ws_manager.send_message(session_id, ready_msg)
        else:
            await send_error_response(session_id, "Session not found")

    except ValidationError as e:
        await send_error_response(session_id, "Invalid session update: %s" % e)
    except Exception as e:
        await send_error_response(session_id, "Session update failed: %s" % e)


async def handle_interaction_input(
    _websocket: WebSocket,
    session_id: str,
    message_data: dict,
) -> None:
    """Handle interaction input message."""
    try:
        interaction_id = str(uuid.uuid4())
        InteractionInputMessage(**message_data)

        if session_id not in active_sessions:
            logger.info("idL: %s", session_id)
            await send_error_response(session_id, "Session not found")
            return

        # Mock processing delay
        await asyncio.sleep(0.1)

        # Send interaction response
        response_msg = InteractionResponseMessage(
            payload=InteractionResponse(
                interaction_id=interaction_id,
                payloads=[
                    InteractionResponsePayload(
                        payload_type="video",
                        data=b"frame_1",
                    )
                ],
                is_final_response=True,
                timestamp=int(time.time()),
                usage=1,
                index=0,
                frame_type=0,
            )
        )

        await ws_manager.send_message(session_id, response_msg)
        logger.info("Interaction processed for %s", response_msg)

    except ValidationError as e:
        await send_error_response(session_id, "Invalid interaction input: %s" % e)
    except Exception as e:
        await send_error_response(session_id, "Interaction failed: %s" % e)


async def _handle_ws_message(
    websocket: WebSocket,
    session_id: str,
    data: dict,
) -> None:
    """Process a single incoming WebSocket message."""
    if "bytes" in data:
        InteractionInputMessage.from_bytes(data["bytes"])
        test_video_bytes = b"\x00\x01\x02\x03\x04\x05" * 100
        interaction_response = InteractionResponseMessage(
            payload=InteractionResponse(
                interaction_id=str(uuid.uuid4()),
                payloads=[
                    InteractionResponsePayload(
                        payload_type="video",
                        data=test_video_bytes,
                    )
                ],
                is_final_response=False,
                timestamp=int(time.monotonic() * 1000),
                index=0,
                frame_type=0,
                usage=1,
            )
        )
        await websocket.send_bytes(interaction_response.to_bytes())
        return

    if "text" not in data:
        raise RuntimeError("unknown")

    text_payload = data["text"]
    if isinstance(text_payload, str) and text_payload:
        message_data = json.loads(text_payload)
    elif isinstance(text_payload, dict) or text_payload:
        message_data = text_payload
    else:
        message_data = data
    message_type = message_data.get("type")

    if message_type == MessageType.SESSION_SETUP:
        await handle_session_setup(websocket, session_id, message_data)
    elif message_type == MessageType.SESSION_UPDATE:
        await handle_session_update(websocket, session_id, message_data)
    elif message_type == MessageType.INTERACTION_INPUT:
        await handle_interaction_input(websocket, session_id, message_data)
    elif message_type == MessageType.START_INTERACTION:
        input_msg = StartInteractionMessage(**message_data)
        payload = input_msg.payload
        await asyncio.sleep(0.1)
        await ws_manager.send_message(
            session_id,
            InteractionReadyMessage(
                payload=InteractionReady(
                    interaction_id=str(payload.interaction_id),
                    timestamp=int(time.time() * 1000),
                )
            ),
        )
    elif message_type == MessageType.END_INTERACTION:
        test_video_bytes = b"\x00\x01\x02\x03\x04\x05" * 100
        interaction_response = InteractionResponseMessage(
            payload=InteractionResponse(
                interaction_id=str(uuid.uuid4()),
                payloads=[
                    InteractionResponsePayload(
                        payload_type="video",
                        data=test_video_bytes,
                    )
                ],
                is_final_response=True,
                timestamp=int(time.monotonic() * 1000),
                index=0,
                frame_type=0,
                usage=1,
            )
        )
        await websocket.send_bytes(interaction_response.to_bytes())
    else:
        await send_error_response(
            session_id,
            "Unknown message type: %s" % message_type,
        )


async def send_error_response(
    session_id: str,
    error_message: str,
    error_code: Optional[str] = None,
) -> None:
    """Send error response to client."""
    interaction_id = str(uuid.uuid4())
    error_payload = ErrorResponsePayload(
        code=error_code or "UNKNOWN_ERROR",
        message=error_message,
        timestamp=get_current_timestamp(),
        interaction_id=interaction_id,
        details=None,
    )
    error_msg = ErrorResponseMessage(payload=error_payload)
    await ws_manager.send_message(session_id, error_msg)
    logger.error("Error sent to %s: %s", session_id, error_message)


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
) -> None:
    """WebSocket endpoint for clients."""
    session_id = "1"

    try:
        # check for token header pretend it goes to the BE
        await ws_manager.connect(websocket, session_id)
        session_ready_payload = SessionReadyPayload(
            trace_id="1",
            status="success",
            load=0.5,
            timestamp=int(time.time()),
            parameters={"test": "mock_parameters"},
        )
        session_ready_message = SessionReadyMessage(payload=session_ready_payload)
        logger.info("message sent %s", session_ready_message)

        await ws_manager.send_message(session_id, session_ready_message)

        while True:
            # Wait for message from client
            data = await websocket.receive()

            if data["type"] != "websocket.receive":
                continue

            try:
                await _handle_ws_message(websocket, session_id, data)
            except json.JSONDecodeError:
                await send_error_response(session_id, "Invalid JSON format")
            except Exception as e:
                await send_error_response(
                    session_id,
                    "Message processing error: %s" % e,
                )

    except WebSocketDisconnect:
        logger.info("Client disconnected: %s", session_id)
    except Exception as e:
        logger.error("WebSocket error for %s: %s", session_id, e)
    finally:
        ws_manager.disconnect(session_id)
        active_sessions.pop(session_id, None)


@app.get("/")
async def root() -> dict:
    """Health check endpoint."""
    return {
        "message": "WebSocket Mock Server is running",
        "endpoint": "/ws",
    }


@app.get("/sessions")
async def get_active_sessions() -> dict:
    """Get information about active sessions."""
    return {
        "active_sessions": len(active_sessions),
        "sessions": active_sessions,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
