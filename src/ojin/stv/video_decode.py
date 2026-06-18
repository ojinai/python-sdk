"""Video-decoder protocol + default opencv implementation.

The client's decode worker turns the server's JPEG frames into RGB off the
playback loop (cv2 releases the GIL during imdecode, so it runs truly parallel),
**at whatever resolution the server sent** — the decoded size is reported back so
the client can tag the emitted frame with it. Injecting a custom decoder lets a
consumer change the pixel pipeline or skip decoding entirely
(``PassthroughDecoder``) and forward the raw JPEG carried on
``STVVideoFrame.source_bytes``.
"""

from __future__ import annotations

from typing import Optional, Protocol, Tuple, runtime_checkable

import cv2
import numpy as np

# (rgb_bytes, width, height) — RGB pixels at the server's native frame size.
DecodedFrame = Tuple[bytes, int, int]


def decode_to_rgb(jpeg: bytes) -> Optional[DecodedFrame]:
    """Decode a JPEG to native-size RGB bytes (pure, off-loop work).

    Pipeline: decode → BGR→RGB, preserving the server's frame dimensions (no
    resize/crop). Returns ``(rgb_bytes, width, height)``, or ``None`` for empty or
    undecodable input — the playback loop treats ``None`` as "no new frame" and
    repeats the last one.
    """
    if not jpeg:
        return None
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.tobytes(), int(w), int(h)


@runtime_checkable
class VideoDecoder(Protocol):
    """Decode a server JPEG to native-size RGB, or None to skip."""

    def decode(self, jpeg: bytes) -> Optional[DecodedFrame]:
        """Return ``(rgb, width, height)``, or ``None`` (passthrough / decode fail)."""
        ...


class OpenCVDecoder:
    """Default decoder using opencv (delegates to :func:`decode_to_rgb`)."""

    def decode(self, jpeg: bytes) -> Optional[DecodedFrame]:
        """Decode the JPEG to native-size RGB bytes + dimensions."""
        return decode_to_rgb(jpeg)


class PassthroughDecoder:
    """Decoder that never decodes — the consumer keeps the raw JPEG bytes."""

    def decode(self, jpeg: bytes) -> Optional[DecodedFrame]:  # noqa: ARG002
        """Return ``None`` always (no decode)."""
        return None
