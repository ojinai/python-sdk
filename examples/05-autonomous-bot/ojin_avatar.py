"""Self-contained Pipecat adapter for an Ojin lip-synced talking-avatar (face).

`OjinAvatarService` is a Pipecat ``FrameProcessor`` that turns the TTS audio
stream into a lip-synced avatar. Place it in the pipeline **after your TTS
service and before ``transport.output()``** — the same slot Pipecat's built-in
avatar services (Simli, Tavus, HeyGen) use:

    transport.input() -> STT -> LLM -> TTS -> [OjinAvatarService] -> transport.output()

All avatar behaviour (connect/retry, audio-as-clock playback, A/V sync, re-sync
after barge-in) lives in ``ojin.stv.OjinSTVClient`` — this class is just the
mapping between Pipecat frames and the client's API. It needs only
``ojin-client[stv]`` and ``pipecat-ai``.

This module is the adapter from the ``ojin-stv-pipecat`` skill, kept here so the
example runs standalone.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ojin.stv import (
    OjinSessionTrace,
    OjinSTVClient,
    STVAudioFrame,
    STVEvent,
    STVVideoFrame,
)

logger = logging.getLogger(__name__)

# Per-session Perfetto trace output. On by default; set OJIN_STV_SESSION_TRACE to
# 0/false/no/off to disable. Files land under OJIN_STV_TRACE_DIR (default below) as
# <root>/<date>/<time>_<session_id>/session.json — the same layout the inference
# server's session trace uses, so the two diff cleanly in https://ui.perfetto.dev.
_TRACE_ROOT = os.getenv("OJIN_STV_TRACE_DIR", "/root/debug/sessions/stv-example")


def _trace_disabled() -> bool:
    """Return True if the per-session Perfetto trace is switched off via env."""
    return os.getenv("OJIN_STV_SESSION_TRACE", "").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }


def _is_trailing_silence(pcm: bytes, sample_rate: int, num_channels: int) -> bool:
    """Return True for the ~0.5 s all-zero sentinel some TTS engines emit at end-of-turn."""
    if not pcm:
        return False
    duration = len(pcm) / (sample_rate * num_channels * 2)
    return abs(duration - 0.5) < 0.01 and pcm == b"\x00" * len(pcm)


class _AvatarOutput:
    """STVOutput sink: pushes the client's synced A/V downstream as Pipecat frames."""

    def __init__(self, service: "OjinAvatarService") -> None:
        self._svc = service

    async def write_audio(self, frame: STVAudioFrame) -> None:
        # The client returns your ORIGINAL TTS audio to play (never the server's).
        await self._svc.push_frame(
            OutputAudioRawFrame(frame.pcm, frame.sample_rate, frame.num_channels)
        )

    async def write_video(self, frame: STVVideoFrame) -> None:
        # rgb is decoded pixels; it repeats the last frame on held ticks (smooth),
        # and is None only with a passthrough decoder (we use the default).
        if frame.rgb is not None:
            await self._svc.push_frame(
                OutputImageRawFrame(
                    image=frame.rgb,
                    size=(frame.width, frame.height),
                    format=frame.format,
                )
            )

    def on_event(self, event: STVEvent, **kwargs: object) -> None:
        pass  # lifecycle events are wired via the client's emitter (see __init__)


class OjinAvatarService(FrameProcessor):
    """Pipecat FrameProcessor that turns the TTS audio stream into a lip-synced avatar.

    Place it in the pipeline after your TTS service and before transport.output().
    All avatar behavior (connect/retry, audio-as-clock playback, A/V sync, re-sync
    after barge-in) lives in ojin.stv.OjinSTVClient; this class is just the mapping.
    """

    def __init__(
        self,
        *,
        api_key: str,
        config_id: str,
        image_size: tuple[int, int] = (512, 512),
        ws_url: str = "wss://models.ojin.foo/realtime",
    ) -> None:
        """Build the avatar service and wire the client's lifecycle events.

        Args:
            api_key: Ojin API key.
            config_id: Ojin Face-model config id (the avatar to drive).
            image_size: Avatar frame size; must match the transport's
                ``video_out_width``/``video_out_height``.
            ws_url: Ojin realtime websocket URL.

        """
        super().__init__(name="ojin-avatar")
        self._waiting_for_first_tts = False
        # Per-session Perfetto trace, injected as the client's tracer so every
        # audio/video tick, buffer swap, and barge-in lands on one timeline. It is
        # dumped to disk on close (see _write_trace). None => NullTracer (no-op).
        self._trace = (
            None
            if _trace_disabled()
            else OjinSessionTrace(session_id=uuid.uuid4().hex[:12], config_id=config_id)
        )
        self._stv = OjinSTVClient(
            api_key=api_key,
            config_id=config_id,
            ws_url=ws_url,
            image_size=image_size,  # must match the transport's video_out_width/height
            output=_AvatarOutput(self),  # push model — do NOT call output_stream()
            tracer=self._trace,  # None => the client falls back to a NullTracer
        )

        @self._stv.on(STVEvent.BOT_STARTED_SPEAKING)
        async def _started(**_: object) -> None:
            await self.stop_ttfb_metrics()  # first avatar frame => stop TTFB timer

        @self._stv.on(STVEvent.SESSION_READY)
        async def _ready(**_: object) -> None:
            # Bootstrap signal for the autonomous driver (05-autonomous-bot): the STV
            # session is now ready, so the avatar can accept TTS audio. Emit an
            # "idle & ready" BotStoppedSpeakingFrame upstream so the driver fires the
            # opening turn only now — never racing ahead of session-ready (which would
            # drop the first turn's audio and stall the loop). Harmless with a human
            # driver: nothing upstream reacts to a bot-stopped before any speech.
            await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

        @self._stv.on(STVEvent.ERROR)
        async def _error(message: str = "", fatal: bool = False, **_: object) -> None:
            await self.push_error(message, fatal=fatal)

    def can_generate_metrics(self) -> bool:
        """Report that this processor emits TTFB / usage metrics."""
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Map each inbound Pipecat frame to the matching OjinSTVClient call."""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self._stv.start()  # connect + run the playback loops
        elif isinstance(frame, TTSStartedFrame):
            self._waiting_for_first_tts = True
            await self._stv.start_turn()  # open a buffer for this utterance
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSAudioRawFrame):
            if _is_trailing_silence(frame.audio, frame.sample_rate, frame.num_channels):
                return  # drop the end-of-turn silence sentinel
            if self._waiting_for_first_tts:
                self._waiting_for_first_tts = False
                await self.start_ttfb_metrics()  # arm TTFB on the first real audio
            await self._stv.send_tts_audio(
                frame.audio, frame.sample_rate, frame.num_channels
            )
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._stv.interrupt()  # barge-in (no-op if the avatar is idle)
            await self.push_frame(frame, direction)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stv.close()
            self._write_trace()  # flush the session's Perfetto trace on teardown
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(
                frame, direction
            )  # stay transparent for everything else

    def _write_trace(self) -> None:
        """Dump the session's Perfetto trace to disk on close (best-effort, once)."""
        trace, self._trace = self._trace, None  # write at most once
        if trace is None:
            return
        now = datetime.now(timezone.utc)
        path = os.path.join(
            _TRACE_ROOT,
            now.strftime("%Y-%m-%d"),
            f"{now.strftime('%H-%M-%S')}_{trace.session_id}",
            "session.json",
        )
        try:
            logger.info("Ojin STV session trace written to %s", trace.dump(path))
        except Exception as exc:  # never let trace I/O break teardown
            logger.warning("Failed to write Ojin STV session trace: %s", exc)
