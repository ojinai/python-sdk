"""Tunable configuration for the STV client.

Parity with pipecat's ``OjinVideoSettings`` minus the pipecat-specific fields
(``start_frame_cls``, ``tts_audio_passthrough``), which belong in the adapter.
Connection identity (``api_key`` / ``config_id`` / ``ws_url``) is passed to the
client constructor, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WebRTCSettings:
    """Room credentials + knobs for the direct-WebRTC negotiation.

    ``token`` is excluded from ``repr`` and must never be logged.
    ``webrtc_join_timeout_s`` is client-local and is never serialized into the
    ``sessionUpdate`` payload.
    """

    room_url: str
    token: str = field(repr=False)
    provider: str = "daily"
    audio_sample_rate: int = 16000
    webrtc_join_timeout_s: float = 10.0


@dataclass
class STVConfig:
    """Behavioral knobs for :class:`~ojin.stv.ojin_stv_client.OjinSTVClient`."""

    # Connection retry
    client_connect_max_retries: int = 3
    client_reconnect_delay: float = 3.0

    # Video framing — emitted frames are always at 25 fps; only
    # the playback rate is configurable here.
    fps: int = 25

    # Video buffering
    max_buffered_video_frames: int = 700
    initial_buffer_frames: int = 6
    idle_buffer_target_frames: int = 6

    # Swap-time audio alignment
    align_audio_on_swap: bool = True
    align_audio_max_frames: int = 50

    # Mid-speech video-repeat catch-up. When speech audio drains on a tick with
    # no fresh video frame (a delivery stall), the shown frame repeats and the
    # video timeline falls 40 ms behind the audio — permanently, since only one
    # frame is popped per tick. When enabled, the owed frames are repaid by
    # skipping one extra plain-speech frame per tick once frames flow again.
    video_repeat_catchup_enabled: bool = True

    # Barge-in
    interrupt_audio_fade_s: float = 0.75

    # Server-feed audio batching. TTS providers that stream small (~40 ms)
    # chunks at realtime starve the inference server (no supply lead builds at
    # 25 fps-in == 25 fps-out) and churn sub-frame residues. Combine the
    # resampled server-bound copy into larger sends: a lead-establishing initial
    # chunk per turn, then a steady-state minimum. Disable to restore per-chunk
    # sends. (Playback of the original audio is unaffected.)
    server_feed_batching_enabled: bool = True
    server_feed_initial_chunk_ms: int = 1000
    server_feed_min_chunk_ms: int = 400
    server_feed_flush_idle_ms: int = 200
    # Server-feed send pacing (enforced by the transport, OjinClient). A backlog can
    # build when input is buffered (e.g. TTS deferred during a barge-in, then
    # replayed) or a large payload lands at once. To avoid flooding the inference
    # server, OjinClient caps a single send at ``server_feed_max_chunk_bytes`` (larger
    # payloads are split) and spaces consecutive sends ``server_feed_send_gap_ms``
    # apart — but ONLY while a backlog remains, so steady-state realtime streaming is
    # never delayed. Both are passed to OjinClient at construction.
    server_feed_max_chunk_bytes: int = 1024 * 50
    server_feed_send_gap_ms: int = 200
    # Server-feed lead cap: never ship audio more than this far ahead of local
    # playback. The server renders at ~1x realtime, so sending a long answer's
    # whole audio up front (measured: 272 s queued in ~25 s of wall clock,
    # session-7 trace 2026-07-12) buys nothing — it only builds a server-side
    # input backlog that must flush before a barge-in cancel is acknowledged
    # (ack grew 1.2 s -> 2.0 s over that session, and the deferred next turn
    # waits on it). With the cap, the backlog at cancel is bounded and constant
    # regardless of answer length. Payloads beyond the cap wait client-side and
    # are released as playback advances; a barge-in discards them outright.
    # 0 disables the cap (legacy: ship as fast as pacing allows).
    server_feed_max_lead_ms: int = 8000

    # Diagnostics — off by default; a published SDK should be quiet. Each tier
    # dumps every thread's stack to stderr when a playback tick stalls past its
    # threshold (ms); set a positive value to opt in, 0 disables that tier.
    lipsync_trace_enabled: bool = False
    loop_stall_watchdog_ms: float = 0.0  # hard-freeze tier (set e.g. 250 to enable)
    tick_warn_ms: float = 80.0  # slow-tick log line; active only with a tracer
    stall_probe_ms: float = 0.0  # small-stall probe (set e.g. 70 to enable)
