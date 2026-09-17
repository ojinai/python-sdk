"""Ojin Speech-To-Video high-level client (framework-agnostic).

``OjinSTVClient`` turns a stream of TTS audio into a lip-synced talking avatar.
Over the default WebSocket transport it hands you synced audio/video frames,
handling buffering, the audio-as-clock playback loop, and post-interruption
re-sync. Pass ``webrtc=WebRTCSettings(...)`` instead and the avatar is published
straight into your LiveKit or Daily room — same program, same events.
"""

from ojin.avatar_participant import (
    AVATAR_PARTICIPANT_USER_NAME,
    is_avatar_identity,
    is_avatar_participant,
)
from ojin.stv.config import STVConfig, WebRTCProvider, WebRTCSettings
from ojin.stv.events import STVEvent
from ojin.stv.frames import FrameType, STVAudioFrame, STVVideoFrame
from ojin.stv.ojin_stv_client import OjinSTVClient
from ojin.stv.ojin_stv_webrtc_client import OjinSTVWebRTCClient
from ojin.stv.output import QueueOutput, STVOutput
from ojin.stv.resampler import (
    NumpyLinearResampler,
    Resampler,
    SoxrResampler,
    SoxrStreamResampler,
    default_resampler,
)
from ojin.stv.session_trace import OjinSessionTrace
from ojin.stv.sync_check import (
    SyncReport,
    TickSample,
    cross_correlation_lag,
    luma_motion_rms,
    report_to_dict,
    summarize,
)
from ojin.stv.tracing import NullTracer, Tracer
from ojin.stv.video_decode import OpenCVDecoder, PassthroughDecoder, VideoDecoder

__all__ = [
    "AVATAR_PARTICIPANT_USER_NAME",
    "FrameType",
    "NullTracer",
    "NumpyLinearResampler",
    "OjinSTVClient",
    "OjinSTVWebRTCClient",
    "OjinSessionTrace",
    "OpenCVDecoder",
    "PassthroughDecoder",
    "QueueOutput",
    "Resampler",
    "STVAudioFrame",
    "STVConfig",
    "STVEvent",
    "STVOutput",
    "STVVideoFrame",
    "SoxrResampler",
    "SoxrStreamResampler",
    "SyncReport",
    "TickSample",
    "Tracer",
    "VideoDecoder",
    "WebRTCProvider",
    "WebRTCSettings",
    "cross_correlation_lag",
    "default_resampler",
    "is_avatar_identity",
    "is_avatar_participant",
    "luma_motion_rms",
    "report_to_dict",
    "summarize",
]
