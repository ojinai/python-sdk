"""Video-decoder protocol + default opencv implementation.

The client's decode worker turns the server's JPEG frames into target-sized RGB
off the playback loop (cv2 releases the GIL during imdecode/resize, so it runs
truly parallel). Injecting a custom decoder lets a consumer change the pixel
pipeline or skip decoding entirely (``PassthroughDecoder``) and forward the raw
JPEG carried on ``STVVideoFrame.source_bytes``.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import cv2
import numpy as np


def decode_to_rgb(jpeg: bytes, target_w: int, target_h: int) -> Optional[bytes]:
    """Decode a JPEG to cropped, target-sized RGB bytes (pure, off-loop work).

    Pipeline: decode → scale-to-cover → centre-crop → BGR→RGB. Returns ``None`` for
    empty input or an undecodable frame; the playback loop treats that as "no new
    frame" and repeats the last one. Channel/quality pipeline is identical to
    pipecat's ``OjinVideoService``.
    """
    if not jpeg:
        return None
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x = (new_w - target_w) // 2
    y = (new_h - target_h) // 2
    bgr = bgr[y : y + target_h, x : x + target_w]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.tobytes()


@runtime_checkable
class VideoDecoder(Protocol):
    """Decode a server JPEG to target-sized RGB bytes, or None to skip."""

    def decode(self, jpeg: bytes, target_w: int, target_h: int) -> Optional[bytes]:
        """Return decoded RGB bytes, or ``None`` (passthrough / decode failure)."""
        ...


class OpenCVDecoder:
    """Default decoder using opencv (delegates to :func:`decode_to_rgb`)."""

    def decode(self, jpeg: bytes, target_w: int, target_h: int) -> Optional[bytes]:
        """Decode the JPEG to cropped target-sized RGB bytes."""
        return decode_to_rgb(jpeg, target_w, target_h)


class PassthroughDecoder:
    """Decoder that never decodes — the consumer keeps the raw JPEG bytes."""

    def decode(self, jpeg: bytes, target_w: int, target_h: int) -> Optional[bytes]:  # noqa: ARG002
        """Return ``None`` always (no decode)."""
        return None
