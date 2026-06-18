"""Unit tests for ojin.stv.video_decode."""

import cv2
import numpy as np

from ojin.stv.video_decode import OpenCVDecoder, PassthroughDecoder, decode_to_rgb


def _jpeg(w: int, h: int) -> bytes:
    """Encode a solid-green (BGR) image of size w x h as JPEG bytes."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 1] = 255  # green channel in BGR
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_decode_to_rgb_native_size_and_channel_order() -> None:
    """Decode returns the server's native-size RGB + dims, channels swapped from BGR."""
    out = decode_to_rgb(_jpeg(64, 48))
    assert out is not None
    rgb, w, h = out
    assert (w, h) == (64, 48)  # native size preserved (no resize/crop)
    assert len(rgb) == 64 * 48 * 3
    px = np.frombuffer(rgb, dtype=np.uint8).reshape(48, 64, 3)
    assert px[0, 0, 1] > 200 and px[0, 0, 0] < 50  # RGB: green high, red low


def test_decode_to_rgb_none_on_garbage() -> None:
    """Undecodable or empty input returns None."""
    assert decode_to_rgb(b"not a jpeg") is None
    assert decode_to_rgb(b"") is None


def test_opencv_decoder_delegates() -> None:
    """OpenCVDecoder.decode returns native-size RGB + dims, like decode_to_rgb."""
    out = OpenCVDecoder().decode(_jpeg(16, 16))
    assert out is not None
    rgb, w, h = out
    assert (w, h) == (16, 16) and len(rgb) == 16 * 16 * 3


def test_passthrough_decoder_returns_none() -> None:
    """PassthroughDecoder never decodes (consumer keeps the raw JPEG)."""
    assert PassthroughDecoder().decode(_jpeg(8, 8)) is None
