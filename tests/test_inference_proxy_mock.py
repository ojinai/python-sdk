"""Tests for the local inference proxy mock helpers."""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock

from ojin.entities.interaction_messages import (
    InteractionResponse,
    InteractionResponseMessage,
    InteractionResponsePayload,
)
from ojin.ojin_client_messages import FrameType, OjinInteractionResponseMessage
from tests.mock import inference_proxy_mock


def _response_message(frame_type: int, index: int) -> InteractionResponseMessage:
    return InteractionResponseMessage(
        payload=InteractionResponse(
            interaction_id=str(uuid.uuid4()),
            payloads=[InteractionResponsePayload(payload_type="image", data=b"jpg")],
            is_final_response=False,
            timestamp=1,
            usage=1,
            index=index,
            frame_type=frame_type,
        )
    )


def test_from_proxy_message_reads_frame_type_from_wire() -> None:
    """frame_type comes from the wire field, not the interaction_id heuristic."""
    for ft, idx in [(0, 0), (1, 1), (2, 0), (3, 1)]:
        msg = OjinInteractionResponseMessage.from_proxy_message(
            _response_message(frame_type=ft, index=idx)
        )
        assert msg.frame_type == FrameType(ft)


def test_from_proxy_message_legacy_index_fallback() -> None:
    """A legacy message (no trailing byte) decodes frame_type from index, incl. 2/3."""
    for legacy_index in (0, 1, 2, 3):
        wire = _response_message(frame_type=legacy_index, index=legacy_index).to_bytes()
        legacy_wire = wire[:-1]  # drop trailing frame_type byte → old server
        proxy_message = InteractionResponseMessage.from_bytes(legacy_wire)
        assert proxy_message.payload.frame_type == legacy_index
        msg = OjinInteractionResponseMessage.from_proxy_message(proxy_message)
        assert msg.frame_type == FrameType(legacy_index)


def test_frame_type_enum_has_four_values() -> None:
    """FrameType was extended from 2 to 4 values for the wire frame_type field."""
    assert FrameType.FADE_OUT == 2
    assert FrameType.START_OF_SPEECH == 3


def test_from_proxy_message_unknown_frame_type_falls_back_to_speech() -> None:
    """An unknown wire frame_type (e.g. a future value) falls back to SPEECH."""
    msg = OjinInteractionResponseMessage.from_proxy_message(
        _response_message(frame_type=255, index=0)
    )
    assert msg.frame_type == FrameType.SPEECH


_INTERACTION_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
_TIMESTAMP_MS = 123456


def _expected_final_response_bytes() -> bytes:
    test_video_bytes = b"\x00\x01\x02\x03\x04\x05" * 100
    return InteractionResponseMessage(
        payload=InteractionResponse(
            interaction_id=str(_INTERACTION_ID),
            payloads=[
                InteractionResponsePayload(
                    payload_type="video",
                    data=test_video_bytes,
                )
            ],
            is_final_response=True,
            timestamp=_TIMESTAMP_MS,
            index=0,
            frame_type=0,
            usage=1,
        )
    ).to_bytes()


async def test_end_interaction_text_message_sends_final_response(monkeypatch) -> None:
    """EndInteraction text messages should not be parsed as binary input."""
    websocket = AsyncMock()
    monkeypatch.setattr(inference_proxy_mock.uuid, "uuid4", lambda: _INTERACTION_ID)
    monkeypatch.setattr(inference_proxy_mock.time, "monotonic", lambda: 123.456)

    await inference_proxy_mock._handle_ws_message(
        websocket,
        "session-123",
        {
            "text": json.dumps(
                {"type": inference_proxy_mock.MessageType.END_INTERACTION}
            )
        },
    )

    websocket.send_bytes.assert_awaited_once_with(_expected_final_response_bytes())


async def test_preparsed_text_message_sends_final_response(monkeypatch) -> None:
    """Already parsed text payloads should be used without json.loads."""
    websocket = AsyncMock()
    monkeypatch.setattr(inference_proxy_mock.uuid, "uuid4", lambda: _INTERACTION_ID)
    monkeypatch.setattr(inference_proxy_mock.time, "monotonic", lambda: 123.456)

    await inference_proxy_mock._handle_ws_message(
        websocket,
        "session-123",
        {"text": {"type": inference_proxy_mock.MessageType.END_INTERACTION}},
    )

    websocket.send_bytes.assert_awaited_once_with(_expected_final_response_bytes())
