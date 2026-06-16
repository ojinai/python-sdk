"""Entities for session messages."""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Enum for WebSocket message types."""

    # Client -> Proxy Messages
    END_INTERACTION = "endInteraction"
    INTERACTION_INPUT = "interactionInput"
    SESSION_UPDATE = "sessionUpdate"
    CANCEL_INTERACTION = "cancelInteraction"

    # Proxy -> Client Messages
    SESSION_READY = "sessionReady"
    INTERACTION_RESPONSE = "interactionResponse"
    ERROR_RESPONSE = "errorResponse"

    # Proxy -> Inference Server Messages
    SESSION_SETUP = "sessionSetup"
    SESSION_PING = "sessionPing"


class SessionSetupPayload(BaseModel):
    """Payload for session setup messages."""

    model_id: str = Field(..., description="Identifier for the target inference model")
    trace_id: str = Field(..., description="Identifier for the session")
    model_config_id: str = Field(..., description="Identifier for the model config")
    api_key_id: str = Field(..., description="The client API key")
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Optional model-specific parameters (key-value pairs)"
    )
    timestamp: Optional[int] = Field(
        None,
        description="Timestamp in milliseconds when the message was sent by the client",
    )


class SessionUpdatePayload(BaseModel):
    """Payload for session update messages."""

    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Updated model-specific parameters"
    )
    timestamp: Optional[int] = Field(
        None,
        description="Timestamp in milliseconds when the message was sent by the client",
    )


class SessionReadyPayload(BaseModel):
    """Payload for session ready messages."""

    trace_id: str = Field(
        ...,
        description="Unique identifier assigned to this WebSocket session by the proxy",
    )
    status: str = Field("success", description="Indicates successful setup")
    load: float = Field(
        ...,
        description="Load of the inference server",
    )
    timestamp: Optional[int] = Field(
        None,
        description="Timestamp in milliseconds when the message was sent by the proxy",
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional model-specific parameters,"
            " contains details of the configured session."
        ),
    )
    num_clients: int = Field(
        default=0, description="Number of clients currently connected to the server."
    )
    max_capacity: int = Field(
        default=0, description="Maximum number of clients the server can handle."
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


class SessionSetupPing(BaseModel):
    """Session setup message sent from client to proxy."""

    type: MessageType = MessageType.SESSION_PING
