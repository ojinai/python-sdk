"""Unit tests for OjinClient server-message handling robustness."""

import json

from ojin.entities.interaction_messages import ErrorResponseMessage
from ojin.ojin_client import OjinClient
from ojin.ojin_client_messages import OjinSessionReadyMessage


def _client() -> OjinClient:
    """Return an OjinClient that is constructed but never connected (queue only)."""
    return OjinClient(ws_url="ws://test/realtime", api_key="k", config_id="c")


async def test_non_json_text_surfaces_clean_error() -> None:
    """A plain-text server error (e.g. 'Model not found') becomes an error message.

    Previously json.loads() raised on the text and killed the receive loop with no
    error surfaced to consumers; now it is queued as an ErrorResponseMessage.
    """
    client = _client()
    await client._handle_server_message("Model not found")  # must not raise
    msg = client._available_response_messages_queue.get_nowait()
    assert isinstance(msg, ErrorResponseMessage)
    assert msg.payload.message == "Model not found"


async def test_empty_text_surfaces_clean_error() -> None:
    """An empty text frame is surfaced as an error rather than crashing."""
    client = _client()
    await client._handle_server_message("")  # must not raise
    msg = client._available_response_messages_queue.get_nowait()
    assert isinstance(msg, ErrorResponseMessage)


async def test_session_ready_json_still_parsed() -> None:
    """Valid sessionReady JSON is still parsed into OjinSessionReadyMessage."""
    client = _client()
    payload = {"type": "sessionReady", "payload": {"parameters": {"persona": "x"}}}
    await client._handle_server_message(json.dumps(payload))
    msg = client._available_response_messages_queue.get_nowait()
    assert isinstance(msg, OjinSessionReadyMessage)
    assert msg.parameters == {"persona": "x"}
