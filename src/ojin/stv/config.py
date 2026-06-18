"""Tunable configuration for the STV client.

Parity with pipecat's ``OjinVideoSettings`` minus the pipecat-specific fields
(``start_frame_cls``, ``tts_audio_passthrough``), which belong in the adapter.
Connection identity (``api_key`` / ``config_id`` / ``ws_url``) is passed to the
client constructor, not here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class STVConfig:
    """Behavioral knobs for :class:`~ojin.stv.ojin_stv_client.OjinSTVClient`."""

    # Connection retry
    client_connect_max_retries: int = 3
    client_reconnect_delay: float = 3.0

    # Video framing — emitted frames carry the server's native resolution; only
    # the playback rate is configurable here.
    fps: int = 25

    # Video buffering
    max_buffered_video_frames: int = 700
    initial_buffer_frames: int = 6
    idle_buffer_target_frames: int = 6

    # Swap-time audio alignment
    align_audio_on_swap: bool = True
    align_audio_max_frames: int = 50

    # Barge-in
    interrupt_audio_fade_s: float = 0.75

    # Diagnostics — off by default; a published SDK should be quiet. Each tier
    # dumps every thread's stack to stderr when a playback tick stalls past its
    # threshold (ms); set a positive value to opt in, 0 disables that tier.
    lipsync_trace_enabled: bool = False
    loop_stall_watchdog_ms: float = 0.0  # hard-freeze tier (set e.g. 250 to enable)
    tick_warn_ms: float = 80.0  # slow-tick log line; active only with a tracer
    stall_probe_ms: float = 0.0  # small-stall probe (set e.g. 70 to enable)
