"""Ojin Speech-To-Video high-level client (framework-agnostic).

``OjinSTVClient`` turns a stream of TTS audio into a lip-synced talking-avatar
audio/video stream, handling buffering, the audio-as-clock playback loop, and
post-interruption re-sync. It is pipecat-free and built on the swappable
``IOjinClient`` transport (WebSocket today, WebRTC later).
"""

from ojin.stv.config import STVConfig, WebRTCSettings
from ojin.stv.events import STVEvent
from ojin.stv.frames import FrameType, STVAudioFrame, STVVideoFrame
from ojin.stv.ojin_stv_client import OjinSTVClient
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
    "FrameType",
    "NullTracer",
    "NumpyLinearResampler",
    "OjinSTVClient",
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
    "WebRTCSettings",
    "cross_correlation_lag",
    "default_resampler",
    "luma_motion_rms",
    "report_to_dict",
    "summarize",
]
