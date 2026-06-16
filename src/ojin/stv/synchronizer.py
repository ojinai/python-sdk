"""Pure, synchronous audio/video synchronization state machine.

This is the heart of the STV client, extracted from pipecat's ``OjinVideoService``
playback loop so every sync behavior — buffer swaps on the server's new-turn
boundary, barge-in fades, swap-time RMS alignment, idle-backlog draining, and
started/stopped-speaking edges — is unit-testable without asyncio, I/O, or decode.

The avatar plays the **original** TTS audio buffered here (not the audio returned
by the server); the client resamples a separate copy for the server. The
synchronizer therefore only ever stores/serves original PCM and never performs
I/O: it returns decisions via :class:`TickResult` and the client acts on them.
"""

from __future__ import annotations

import itertools
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from ojin.ojin_client_messages import FrameType
from ojin.stv.audio_utils import fade_chunk, rms_int16
from ojin.stv.config import STVConfig

OJIN_PERSONA_SAMPLE_RATE = 16_000
BYTES_PER_FRAME = int(OJIN_PERSONA_SAMPLE_RATE / 25 * 2)  # 40 ms @ 16 kHz int16

# Swap-time audio alignment (see Synchronizer.align_current_buffer_to_frame).
_ALIGN_ANCHOR_FRAMES = 6  # how many leading new-turn frames to match on
_ALIGN_MIN_RMS = 1.0  # below this the anchor is silence — skip aligning
_ALIGN_REL_TOL = 0.05  # match tolerance as a fraction of the anchor RMS
# OJIN FIX (lipsync_0609): minimum anchor frames before a non-zero trim is
# trusted. A 1-2 frame RMS signature cannot uniquely localize a position inside a
# multi-second buffer, producing a false trim that desyncs the whole turn.
_ALIGN_MIN_ANCHORS = 4

_AUDIO_BUFFER_IDS = itertools.count(1)


def _next_audio_buffer_id() -> int:
    return next(_AUDIO_BUFFER_IDS)


@dataclass
class AudioBuffer:
    """Holds the resampled-free **original** TTS audio for one logical utterance.

    Opened by ``open_turn`` (≈ ``TTSStartedFrame``), extended by ``add_audio``.
    ``interrupted`` is set on barge-in: the buffer keeps draining (so its in-flight
    video can pop in time) while audio is ramped to silence over the fade window;
    ``fade_samples_emitted`` tracks how far into that ramp we are (flat int16
    samples) so the gain is continuous across ticks.
    """

    sample_rate: int = OJIN_PERSONA_SAMPLE_RATE
    num_channels: int = 1
    bytes_: bytearray = field(default_factory=bytearray)
    started_at: float = field(default_factory=time.monotonic)
    buffer_id: int = field(default_factory=_next_audio_buffer_id)
    interrupted: bool = False
    fade_samples_emitted: int = 0


@dataclass
class VideoFrame:
    """One video frame from the inference server, with bundled audio.

    ``out_rgb`` is filled by the client's decode worker before the frame reaches
    :meth:`Synchronizer.tick`; ``None`` means "not decoded / decode failed" and the
    client repeats the last frame.
    """

    frame_type: int  # 0 idle / 1 speech / 2 fade / 3 new-turn
    image_bytes: bytes
    audio_bytes: bytes
    is_final: bool
    volume: int
    out_rgb: Optional[bytes] = None

    def is_silence(self) -> bool:
        """Whether this is an idle/silence frame (frame_type 0)."""
        return self.frame_type == FrameType.IDLE

    def is_fade_out(self) -> bool:
        """Whether this is a post-cancel fade-out frame (frame_type 2)."""
        return self.frame_type == FrameType.FADE_OUT

    def is_new_turn_start(self) -> bool:
        """Whether this is the first speech frame of a new turn (frame_type 3)."""
        return self.frame_type == FrameType.START_OF_SPEECH


@dataclass
class TickResult:
    """The decision a single :meth:`Synchronizer.tick` produced for the client."""

    video_frame: Optional[VideoFrame]
    audio_chunk: Optional[bytes]  # post-fade PCM; None on underrun/silence
    swapped: bool = False
    started_speaking: bool = False
    stopped_speaking: bool = False
    idle_skipped: int = 0
    align_trim_frames: int = 0
    overflow_dropped: int = 0  # oldest video frames dropped by the overflow cap
    warming_up: bool = False  # initial warm-up tick — client emits nothing


class Synchronizer:
    """Owns the audio-buffer queue and the decoded-video deque; steps per tick."""

    def __init__(self, config: STVConfig, clock=time.monotonic) -> None:
        """Create an idle synchronizer bound to ``config``.

        Args:
            config: behavioral knobs (fade, alignment, idle drain, warm-up).
            clock: monotonic seconds source (injectable for deterministic tests).

        """
        self._config = config
        self._clock = clock
        self._frame_duration = 1.0 / config.fps
        self.audio_buffers: Deque[AudioBuffer] = deque()
        self.current_buffer: Optional[AudioBuffer] = None
        self.swap_pending: bool = False
        self.video_frames: Deque[VideoFrame] = deque()
        self.was_speaking: bool = False
        self._warmup_remaining: int = config.initial_buffer_frames

    # ------------------------------------------------------------------
    # Audio buffering (≈ TTSStarted / TTSAudio target selection)
    # ------------------------------------------------------------------

    def open_turn(self) -> AudioBuffer:
        """Append a fresh buffer at the tail of the queue and return it."""
        buf = AudioBuffer()
        self.audio_buffers.append(buf)
        return buf

    def add_audio(
        self, pcm: bytes, sample_rate: int, num_channels: int
    ) -> Optional[AudioBuffer]:
        """Append original TTS bytes to the right buffer; return it (or None).

        Target selection mirrors ``OjinVideoService._on_tts_audio_frame``:
        the tail queued buffer if any; else the draining non-interrupted current
        buffer (in-turn streaming TTS); else ``None`` — straggler bytes from a
        cancelled turn that have no home and are dropped by the caller.
        """
        target: Optional[AudioBuffer]
        if self.audio_buffers:
            target = self.audio_buffers[-1]
        elif self.current_buffer is not None and not self.current_buffer.interrupted:
            target = self.current_buffer
        else:
            target = None

        if target is None:
            return None

        target.sample_rate = sample_rate
        target.num_channels = num_channels
        target.bytes_.extend(pcm)
        return target

    # ------------------------------------------------------------------
    # Interruption (≈ UserStartedSpeaking barge-in)
    # ------------------------------------------------------------------

    def can_interrupt(self) -> bool:
        """Whether a barge-in should take effect right now.

        True only when a current buffer exists, has not already been interrupted,
        and still has audio to fade — otherwise there is nothing to cancel.
        """
        return (
            self.current_buffer is not None
            and not self.current_buffer.interrupted
            and len(self.current_buffer.bytes_) > 0
        )

    def interrupt(self) -> bool:
        """Apply a barge-in; return whether the client must send a cancel.

        On a valid interrupt: mark the current buffer interrupted (it keeps
        draining while its audio is faded to silence), drop any pending-swap
        intent, and discard the queued buffers of the cancelled turn so the next
        server new-turn boundary lands on the genuinely-new turn's fresh buffer
        (keeping client and server symmetric — both discard everything
        pre-interrupt). Returns ``False`` when idle/already-interrupted.
        """
        if not self.can_interrupt():
            return False
        assert self.current_buffer is not None
        self.current_buffer.interrupted = True
        self.swap_pending = False
        self.audio_buffers.clear()
        return True

    # ------------------------------------------------------------------
    # Buffer swap + swap-time alignment
    # ------------------------------------------------------------------

    def current_replaceable(self) -> bool:
        """Whether a queued buffer may be promoted over the current one this tick.

        Normally only when there is no current buffer, or it has fully drained and
        was not interrupted (the ``not interrupted`` guard stops a stale old-turn
        SPEECH frame from triggering a premature swap during a post-cancel fadeout).
        Once a new-turn boundary HAS passed without a buffer (``swap_pending``), the
        current interrupted/empty buffer is the orphaned prior turn, so allow
        replacing it as soon as the next buffer is available.
        """
        cur = self.current_buffer
        if cur is None:
            return True
        if self.swap_pending:
            return cur.interrupted or len(cur.bytes_) == 0
        return not cur.interrupted and len(cur.bytes_) == 0

    def swap_to_next_buffer(self, align_to_frame: Optional[VideoFrame] = None) -> int:
        """Promote the head of the buffer queue to current; return frames trimmed.

        Discards whatever remains in the current buffer (in-flight audio of a
        cancelled turn, or audio buffered ahead of playback). Skips empty head
        buffers. If the queue is empty, records ``swap_pending`` so the playback
        loop can promote the replacement the moment it lands. When the trigger is a
        true new-turn frame and alignment is enabled, lines the new buffer's head up
        with the server's first new-turn frame and returns the number of leading
        40 ms frames trimmed (0 otherwise).
        """
        if not self.audio_buffers:
            self.swap_pending = True
            return 0

        new_buffer: Optional[AudioBuffer] = None
        while self.audio_buffers:
            candidate = self.audio_buffers.popleft()
            if len(candidate.bytes_) == 0:
                continue
            new_buffer = candidate
            break

        if new_buffer is None:
            self.current_buffer = None
            self.swap_pending = True
            return 0

        self.current_buffer = new_buffer
        self.swap_pending = False

        # OJIN FIX (lipsync_0609): only the server-signalled new-turn boundary can
        # drop leading speech, so it is the only path that needs alignment. A
        # natural turn end has no fade and no drop — the correct trim is always 0,
        # so aligning there can only inject error.
        if (
            self._config.align_audio_on_swap
            and align_to_frame is not None
            and align_to_frame.is_new_turn_start()
        ):
            return self.align_current_buffer_to_frame(align_to_frame)
        return 0

    def _anchor_rms(self, align_to_frame: VideoFrame) -> Optional[list[float]]:
        """Build the RMS anchor signature for alignment, or None if unusable.

        Anchors are the swap-triggering frame plus the leading non-silence frames
        already buffered. Returns ``None`` when there is nothing to anchor on
        (empty/silent audio) or when fewer than ``_ALIGN_MIN_ANCHORS`` real anchors
        exist — the lipsync_0609 guard against a false trim from a too-short
        signature.
        """
        if not align_to_frame.audio_bytes:
            return None
        anchors_src = [align_to_frame] + [
            f for f in list(self.video_frames) if not f.is_silence()
        ]
        anchor_rms: list[float] = []
        for f in anchors_src[:_ALIGN_ANCHOR_FRAMES]:
            r = rms_int16(f.audio_bytes)
            if r is None:
                break
            anchor_rms.append(r)
        if not anchor_rms or max(anchor_rms) < _ALIGN_MIN_RMS:
            return None
        if len(anchor_rms) < _ALIGN_MIN_ANCHORS:
            return None
        return anchor_rms

    def align_current_buffer_to_frame(self, align_to_frame: VideoFrame) -> int:
        """Trim leading 40 ms frames so the buffer head matches the server frame.

        The server can drop leading speech of a new turn (e.g. the post-fade
        SPEECH-drop window), so the first video frame can correspond to audio
        further into the buffer than byte 0. We recover that offset by matching the
        bundled audio of the first server frame(s) against successive 40 ms windows
        using an RMS amplitude-envelope signature — robust to the server (16 kHz)
        vs buffer (TTS rate) sample-rate differences since RMS is rate-independent.
        Conservative: trims only on a confident non-zero match, so the steady-state
        (offset 0) case is untouched. Returns the number of frames trimmed.
        """
        buf = self.current_buffer
        if buf is None or align_to_frame.is_silence() or align_to_frame.is_fade_out():
            return 0

        anchor_rms = self._anchor_rms(align_to_frame)
        if anchor_rms is None:
            return 0

        buf_frame_bytes = (
            int(buf.sample_rate * self._frame_duration) * buf.num_channels * 2
        )
        if buf_frame_bytes <= 0:
            return 0
        max_d = min(
            self._config.align_audio_max_frames,
            len(buf.bytes_) // buf_frame_bytes - len(anchor_rms),
        )
        if max_d <= 0:
            return 0

        best_d = self._best_alignment_offset(buf, anchor_rms, buf_frame_bytes, max_d)
        if best_d > 0:
            del buf.bytes_[: best_d * buf_frame_bytes]
        return best_d

    @staticmethod
    def _best_alignment_offset(
        buf: AudioBuffer, anchor_rms: list[float], buf_frame_bytes: int, max_d: int
    ) -> int:
        """Return the leading-frame offset whose RMS window best matches anchors.

        0 when no offset beats the head (d=0) within tolerance — i.e. the server
        dropped nothing. The tolerance scales with anchor energy so it works at any
        volume; only a confident, better-than-head match trims.
        """

        def window_err(d: int) -> Optional[float]:
            err = 0.0
            for j, a in enumerate(anchor_rms):
                start = (d + j) * buf_frame_bytes
                r = rms_int16(bytes(buf.bytes_[start : start + buf_frame_bytes]))
                if r is None:
                    return None
                err += (r - a) * (r - a)
            return err / len(anchor_rms)

        base_err = window_err(0)
        if base_err is None:
            return 0
        best_d, best_err = 0, base_err
        for d in range(1, max_d + 1):
            e = window_err(d)
            if e is None:
                break
            if e < best_err:
                best_d, best_err = d, e

        tol = (_ALIGN_REL_TOL * (sum(anchor_rms) / len(anchor_rms))) ** 2
        if best_d > 0 and best_err <= tol and best_err < base_err:
            return best_d
        return 0

    # ------------------------------------------------------------------
    # Idle backlog drain
    # ------------------------------------------------------------------

    def drain_idle_backlog(self, popped: Optional[VideoFrame]) -> int:
        """Drop extra leading silence frames to shrink an idle video backlog.

        The loop pops one frame per tick — the rate the server produces them — so a
        backlog that built up while the consumer wasn't draining is otherwise
        carried for the whole session, delaying every reply. When idle (the popped
        frame is silence and no speech audio is draining) and the pending buffer is
        over ``idle_buffer_target_frames``, drop the next silence frame(s) so it
        shrinks back toward the target: one normally, two when a reply's speech is
        already queued behind the silence. Only silence is ever dropped; audio is
        untouched. Returns the number dropped (0 in the steady state).
        """
        target = self._config.idle_buffer_target_frames
        if target <= 0 or popped is None or not popped.is_silence():
            return 0
        if len(self.video_frames) <= target:
            return 0
        # Never advance video past audio: only drain while idle (no live buffer).
        if self.current_buffer is not None and len(self.current_buffer.bytes_) > 0:
            return 0
        speech_pending = any(not f.is_silence() for f in self.video_frames)
        max_skip = 2 if speech_pending else 1
        skipped = 0
        while (
            skipped < max_skip
            and len(self.video_frames) > target
            and self.video_frames[0].is_silence()
        ):
            self.video_frames.popleft()
            skipped += 1
        return skipped

    # ------------------------------------------------------------------
    # Per-tick step (audio-as-clock, no I/O)
    # ------------------------------------------------------------------

    def enqueue_video(self, frame: VideoFrame) -> None:
        """Append a decoded video frame for the next tick to consume."""
        self.video_frames.append(frame)

    def is_currently_speaking(self) -> bool:
        """Whether the avatar is producing real (non-faded) speech audio now."""
        return (
            self.current_buffer is not None
            and not self.current_buffer.interrupted
            and len(self.current_buffer.bytes_) > 0
        )

    def tick(self) -> TickResult:
        """Advance one 40 ms frame; return what the client should emit.

        Order matches ``OjinVideoService``'s playback loop: overflow cap → warm-up
        hold → pop one video frame → swap decision (new-turn boundary, else natural
        turn end / deferred-swap recovery) → idle backlog drain → audio drain (with
        barge-in fade) → started/stopped-speaking edge detection. No sleeping, no
        I/O: the client performs the resulting sends/emits/trace.
        """
        result = TickResult(video_frame=None, audio_chunk=None)

        # Overflow backstop (unbounded-growth guard).
        cap = self._config.max_buffered_video_frames
        if len(self.video_frames) > cap:
            result.overflow_dropped = len(self.video_frames) - cap
            for _ in range(result.overflow_dropped):
                self.video_frames.popleft()

        # Initial warm-up: hold the first frames so playback has a jitter cushion.
        if self.video_frames and self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            result.warming_up = True
            return result

        # Pop one video frame and decide on a buffer swap BEFORE draining audio so
        # the popped frame pairs with the new buffer's first chunk.
        video_frame: Optional[VideoFrame] = None
        if self.video_frames:
            video_frame = self.video_frames.popleft()
            # Two swap triggers: (1) the server's new-turn boundary (frame_type=3),
            # and (2) a natural turn end / deferred-swap recovery — a SPEECH frame
            # popped while the current buffer is replaceable and a buffer is queued.
            # Only (1) actually aligns (handled inside swap_to_next_buffer).
            new_turn = video_frame.is_new_turn_start()
            natural_end = (
                not video_frame.is_silence()
                and not video_frame.is_fade_out()
                and bool(self.audio_buffers)
                and self.current_replaceable()
            )
            if new_turn or natural_end:
                result.align_trim_frames = self.swap_to_next_buffer(video_frame)
                result.swapped = True
        result.video_frame = video_frame

        result.idle_skipped = self.drain_idle_backlog(video_frame)
        result.audio_chunk = self._drain_audio_chunk()

        speaking = self.is_currently_speaking()
        if speaking and not self.was_speaking:
            result.started_speaking = True
        elif not speaking and self.was_speaking:
            result.stopped_speaking = True
        self.was_speaking = speaking

        return result

    def _drain_audio_chunk(self) -> Optional[bytes]:
        """Drain one chunk of output audio from the current buffer (or None).

        Returns the played PCM for this tick: the raw chunk when speaking; a faded
        chunk during a barge-in (ramped to silence over ``interrupt_audio_fade_s``,
        keyed to samples emitted so it is smooth across tick jitter); ``None`` on
        underrun or once the fade has completed. Bytes are always consumed when
        present (even when silenced) so the buffer drains in time with the clock.
        """
        cur = self.current_buffer
        if cur is None:
            return None

        chunk_size = int(cur.sample_rate * self._frame_duration) * cur.num_channels * 2
        buf = cur.bytes_
        if len(buf) >= chunk_size:
            chunk = bytes(buf[:chunk_size])
            del buf[:chunk_size]
        elif len(buf) > 0:
            chunk = bytes(buf)
            buf.clear()
        else:
            return None  # underrun — caller emits silence

        if not cur.interrupted:
            return chunk

        # Barge-in fade: ramp this turn's audio to silence instead of hard-cutting.
        fade_total = int(
            self._config.interrupt_audio_fade_s * cur.sample_rate * cur.num_channels
        )
        if fade_total <= 0 or cur.fade_samples_emitted >= fade_total:
            return None  # ramp disabled or complete → silence
        out = fade_chunk(chunk, cur.fade_samples_emitted, fade_total)
        cur.fade_samples_emitted += len(chunk) // 2
        return out
