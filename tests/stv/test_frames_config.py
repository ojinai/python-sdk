"""Unit tests for ojin.stv neutral frame types and STVConfig."""

import pytest

from ojin.stv.config import STVConfig, WebRTCSettings
from ojin.stv.frames import FrameType, STVAudioFrame, STVVideoFrame


def test_frames_construct() -> None:
    """The neutral output frames carry their payload + metadata."""
    a = STVAudioFrame(pcm=b"\x00\x00", sample_rate=16000, num_channels=1, pts=1)
    assert a.sample_rate == 16000 and a.pcm == b"\x00\x00"
    v = STVVideoFrame(
        rgb=None, source_bytes=b"jpg", width=2, height=2, frame_type=1, pts=1
    )
    assert v.format == "RGB" and v.frame_type == 1 and v.source_bytes == b"jpg"


def test_frametype_values() -> None:
    """FrameType mirrors the wire markers 0/1/2/3."""
    assert (
        FrameType.IDLE,
        FrameType.SPEECH,
        FrameType.FADE_OUT,
        FrameType.START_OF_SPEECH,
    ) == (0, 1, 2, 3)


def test_config_defaults() -> None:
    """STVConfig defaults match the ported OjinVideoSettings values."""
    c = STVConfig()
    assert c.fps == 25
    assert c.interrupt_audio_fade_s == 0.75
    assert c.align_audio_on_swap is True
    assert c.idle_buffer_target_frames == 6


def test_webrtc_settings_accepts_native_rates() -> None:
    """Common native TTS rates (all divisible by 25) are accepted."""
    for rate in (16000, 22050, 24000, 44100, 48000):
        assert WebRTCSettings(room_url="u", token="t", audio_sample_rate=rate)


@pytest.mark.parametrize("rate", [0, -16000, 16001, 7000, 50000])
def test_webrtc_settings_rejects_bad_rate(rate: int) -> None:
    """A rate that would misframe the 40 ms feed or divide-by-zero is rejected."""
    with pytest.raises(ValueError):
        WebRTCSettings(room_url="u", token="t", audio_sample_rate=rate)
