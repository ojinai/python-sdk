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


def test_debug_queue_depths_disconnected_reports_only_parsed_queue() -> None:
    """With no websocket, only the always-readable parsed-queue depth is reported."""
    client = _client()
    assert client.debug_queue_depths() == {"server_msgs": 0}
    client._available_response_messages_queue.put_nowait(
        OjinSessionReadyMessage(parameters={})
    )
    assert client.debug_queue_depths()["server_msgs"] == 1


def test_debug_queue_depths_reads_len_based_frame_queue() -> None:
    """The websockets asyncio queue exposes len() (not qsize) — read it via len()."""

    class _LenFrames:
        """Mimics websockets' custom SimpleQueue: supports len(), not qsize()."""

        def __len__(self) -> int:
            return 4

    class _Assembler:
        frames = _LenFrames()
        paused = True

    class _WS:
        recv_messages = _Assembler()
        transport = None  # skip the Unix-only FIONREAD probe in this test

    client = _client()
    client._ws = _WS()  # type: ignore[assignment]
    depths = client.debug_queue_depths()
    assert depths["ws_frames"] == 4
    assert depths["ws_paused"] == 1
    assert depths["server_msgs"] == 0


def test_debug_queue_depths_reads_qsize_based_frame_queue() -> None:
    """A stdlib queue.SimpleQueue (qsize, no len) is read via the qsize fallback."""

    class _QSizeFrames:
        def qsize(self) -> int:
            return 7

    class _Assembler:
        frames = _QSizeFrames()
        paused = False

    class _WS:
        recv_messages = _Assembler()
        transport = None

    client = _client()
    client._ws = _WS()  # type: ignore[assignment]
    depths = client.debug_queue_depths()
    assert depths["ws_frames"] == 7
    assert depths["ws_paused"] == 0


def test_debug_queue_depths_never_raises_on_broken_internals() -> None:
    """A websockets-internals shape change degrades to a missing key, not a crash."""

    class _BadWS:
        """A websocket whose internals don't match what the probe expects."""

    client = _client()
    client._ws = _BadWS()  # type: ignore[assignment]
    # Must not raise; ws gauges are simply absent, server_msgs still present.
    assert client.debug_queue_depths() == {"server_msgs": 0}
