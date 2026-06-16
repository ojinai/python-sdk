"""Unit tests for ojin.stv.video_decode."""

import cv2
import numpy as np

from ojin.stv.video_decode import OpenCVDecoder, PassthroughDecoder, decode_to_rgb


def _jpeg(w: int, h: int) -> bytes:
    """Encode a solid-green (BGR) image of size w x h as JPEG bytes."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 1] = 255  # green channel in BGR
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_decode_to_rgb_size_and_channel_order() -> None:
    """Decode produces target-sized RGB bytes with the channels swapped from BGR."""
    out = decode_to_rgb(_jpeg(64, 48), 32, 32)
    assert out is not None and len(out) == 32 * 32 * 3
    px = np.frombuffer(out, dtype=np.uint8).reshape(32, 32, 3)
    assert px[0, 0, 1] > 200 and px[0, 0, 0] < 50  # RGB: green high, red low


def test_decode_to_rgb_none_on_garbage() -> None:
    """Undecodable or empty input returns None."""
    assert decode_to_rgb(b"not a jpeg", 32, 32) is None
    assert decode_to_rgb(b"", 32, 32) is None


def test_opencv_decoder_delegates() -> None:
    """OpenCVDecoder.decode produces the same RGB as decode_to_rgb."""
    out = OpenCVDecoder().decode(_jpeg(16, 16), 8, 8)
    assert out is not None and len(out) == 8 * 8 * 3


def test_passthrough_decoder_returns_none() -> None:
    """PassthroughDecoder never decodes (consumer keeps the raw JPEG)."""
    assert PassthroughDecoder().decode(_jpeg(8, 8), 4, 4) is None
