"""Neutral, framework-agnostic output frame types for the STV client.

These replace pipecat's ``OutputAudioRawFrame`` / ``OutputImageRawFrame`` at the
client boundary so consumers depend only on ``ojin.stv``. ``FrameType`` is
re-exported from the wire message module so callers have one import site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ojin.ojin_client_messages import FrameType

__all__ = ["FrameType", "STVAudioFrame", "STVVideoFrame"]


@dataclass
class STVAudioFrame:
    """One tick of played audio (the original TTS audio, never the server's)."""

    pcm: bytes
    sample_rate: int
    num_channels: int
    pts: int  # nanoseconds


@dataclass
class STVVideoFrame:
    """One avatar video frame.

    ``rgb`` holds decoded RGB pixels when a decoder is active (``None`` for a
    passthrough decoder or a decode failure); ``source_bytes`` carries the raw JPEG
    from the server so a consumer can decode/forward it itself, but is **empty
    (``b""``) on a held/repeat tick** (no new server frame) — ``rgb`` repeats the
    last decoded frame on those ticks, so render from ``rgb`` for seamless video and
    skip empty ``source_bytes`` if you forward the JPEG instead.

    ``volume`` is the RMS amplitude of the audio the server bundled with this frame
    (the lip-sync target) — i.e. the amplitude of the audio timeline the video was
    generated for. It is ``0`` on a repeated/held frame (no new server frame this
    tick). Consumers use it to verify the played audio lines up with the video.
    """

    rgb: Optional[bytes]
    source_bytes: bytes
    width: int
    height: int
    frame_type: int
    pts: int  # nanoseconds
    format: str = "RGB"
    volume: float = 0.0
