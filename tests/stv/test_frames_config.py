"""Unit tests for ojin.stv neutral frame types and STVConfig."""

import pytest

from ojin.stv.config import STVConfig, WebRTCProvider, WebRTCSettings
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


def test_webrtc_provider_defaults_to_daily() -> None:
    """Omitting the provider keeps the historical Daily default."""
    settings = WebRTCSettings(room_url="u", token="t")
    assert settings.provider is WebRTCProvider.DAILY
    assert settings.to_connect_query_params()["webrtc_provider"] == "daily"


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("livekit", WebRTCProvider.LIVEKIT),
        (" LiveKit ", WebRTCProvider.LIVEKIT),
        ("DAILY", WebRTCProvider.DAILY),
        (WebRTCProvider.LIVEKIT, WebRTCProvider.LIVEKIT),
    ],
)
def test_webrtc_provider_normalized(given: object, expected: WebRTCProvider) -> None:
    """Strings are normalized to the enum; the wire value stays the bare name."""
    settings = WebRTCSettings(room_url="u", token="t", provider=given)  # type: ignore[arg-type]
    assert settings.provider is expected
    assert settings.to_connect_query_params()["webrtc_provider"] == expected.value


def test_webrtc_provider_str_is_wire_value() -> None:
    """Logging/f-strings show the bare provider name on every Python version."""
    assert str(WebRTCProvider.DAILY) == "daily"
    assert f"{WebRTCProvider.LIVEKIT}" == "livekit"


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider": "zoom"},
        {"room_url": ""},
        {"room_url": "   "},
        {"token": ""},
        {"webrtc_join_timeout_s": 0},
        {"webrtc_join_timeout_s": -1.0},
    ],
)
def test_webrtc_settings_rejects_invalid_fields(overrides: dict) -> None:
    """Unknown providers, empty credentials and non-positive timeouts fail early."""
    kwargs = {"room_url": "u", "token": "t", **overrides}
    with pytest.raises(ValueError):
        WebRTCSettings(**kwargs)


def test_webrtc_settings_invalid_provider_message_lists_known() -> None:
    """The error names the supported providers."""
    with pytest.raises(ValueError, match="daily, livekit"):
        WebRTCSettings(room_url="u", token="t", provider="zoom")
