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
from typing import Callable, Deque, Optional

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

# Video-repeat catch-up: cap on how many owed frames are remembered. A stall
# longer than this (1 s at 25 fps) looks broken regardless and the next
# new-turn swap re-anchors anyway, so bounding the debt bounds the
# fast-motion catch-up window that follows a stall.
_REPEAT_DEBT_MAX = 25

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

    ``out_rgb`` / ``out_size`` are filled together by the client's decode worker
    before the frame reaches :meth:`Synchronizer.tick`: ``out_rgb`` is the decoded
    RGB bytes and ``out_size`` is its ``(width, height)`` at the server's native
    resolution. ``out_rgb`` ``None`` means "not decoded / decode failed" and the
    client repeats the last frame.
    """

    frame_type: int  # 0 idle / 1 speech / 2 fade / 3 new-turn
    image_bytes: bytes
    audio_bytes: bytes
    is_final: bool
    volume: int
    out_rgb: Optional[bytes] = None
    out_size: Optional[tuple[int, int]] = None

    def is_silence(self) -> bool:
        """Whether this is an idle/silence frame (frame_type 0)."""
        return self.frame_type == FrameType.IDLE

    def is_speech(self) -> bool:
        """Whether this is a plain continuation-speech frame (frame_type 1)."""
        return self.frame_type == FrameType.SPEECH

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
    # Swap-time alignment shift in 40 ms frames: positive = leading audio trimmed
    # (the server dropped it), negative = silence prepended (the server emitted
    # extra leading quiet frames the buffer does not have).
    align_trim_frames: int = 0
    overflow_dropped: int = 0  # oldest video frames dropped by the overflow cap
    catchup_dropped: int = 0  # speech frames skipped to repay video-repeat debt
    warming_up: bool = False  # initial warm-up tick — client emits nothing


class Synchronizer:
    """Owns the audio-buffer queue and the decoded-video deque; steps per tick."""

    def __init__(
        self, config: STVConfig, clock: Callable[[], float] = time.monotonic
    ) -> None:
        """Create an idle synchronizer bound to ``config``.

        Args:
            config: behavioral knobs (fade, alignment, idle drain, warm-up).
            clock: monotonic seconds source (injectable for deterministic tests).

        """
        super().__init__()
        self._config = config
        self._clock = clock
        self._frame_duration = 1.0 / config.fps
        self.audio_buffers: Deque[AudioBuffer] = deque()
        self.current_buffer: Optional[AudioBuffer] = None
        self.swap_pending: bool = False
        self.video_frames: Deque[VideoFrame] = deque()
        self.was_speaking: bool = False
        self._warmup_remaining: int = config.initial_buffer_frames
        # Frames of video owed after mid-speech repeats (audio drained while no
        # frame was available). Repaid by skipping one extra speech frame per
        # tick once frames flow again; reset at any turn boundary.
        self._repeat_debt: int = 0

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
        self._repeat_debt = 0  # the cancelled turn's owed video is moot
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

    def _anchor_rms(
        self, align_to_frame: VideoFrame
    ) -> Optional[tuple[list[float], int]]:
        """Build the RMS anchor signature for alignment, or None if unusable.

        Anchors are the swap-triggering frame plus the leading non-silence frames
        already buffered, **starting at the first audible frame**: a turn can open
        with speech-typed frames whose audio is near-zero (e.g. the server padded
        the turn head while starved of TTS), and matching on those both pollutes
        the signature and — worse — used to abort alignment entirely because the
        anchor looked silent. The count of skipped near-silent head frames is
        returned alongside the signature so the caller can convert the matched
        buffer offset into a signed shift (see ``align_current_buffer_to_frame``).

        Returns ``(anchor_rms, lead_silent)``, or ``None`` when there is nothing
        to anchor on (empty/silent audio), the silent head is implausibly long
        (> ``align_audio_max_frames``), or fewer than ``_ALIGN_MIN_ANCHORS``
        audible anchors exist — the lipsync_0609 guard against a false trim from
        a too-short signature.
        """
        if not align_to_frame.audio_bytes:
            return None
        anchors_src = [align_to_frame] + [
            f for f in list(self.video_frames) if not f.is_silence()
        ]
        lead_silent = 0
        anchor_rms: list[float] = []
        for f in anchors_src:
            r = rms_int16(f.audio_bytes)
            if r is None:
                break
            if not anchor_rms and r < _ALIGN_MIN_RMS:
                lead_silent += 1
                if lead_silent > self._config.align_audio_max_frames:
                    return None
                continue
            anchor_rms.append(r)
            if len(anchor_rms) >= _ALIGN_ANCHOR_FRAMES:
                break
        if not anchor_rms or max(anchor_rms) < _ALIGN_MIN_RMS:
            return None
        if len(anchor_rms) < _ALIGN_MIN_ANCHORS:
            return None
        return anchor_rms, lead_silent

    def align_current_buffer_to_frame(self, align_to_frame: VideoFrame) -> int:
        """Shift the buffer head so it lines up with the server's frame timeline.

        The server's turn head can disagree with the local buffer in either
        direction: it can drop leading speech of a new turn (e.g. the post-fade
        SPEECH-drop window) — the first video frame then corresponds to audio
        further into the buffer than byte 0 — or it can emit extra leading
        near-silent speech frames the buffer does not have (e.g. padding minted
        while the server was starved of TTS in the turn's first-fragment gap).
        We recover the audible onset on both sides by matching the bundled audio
        of the first server frame(s) against successive 40 ms windows using an
        RMS amplitude-envelope signature — robust to the server (16 kHz) vs
        buffer (TTS rate) sample-rate differences since RMS is rate-independent —
        then apply the signed difference: trim leading buffer audio (positive
        shift) or prepend that much silence (negative shift) so the audio waits
        for the video. Conservative: only a confident envelope match shifts, so
        the steady-state (offset 0, no silent head) case is untouched. Returns
        the signed shift in frames (0 when nothing was done).
        """
        buf = self.current_buffer
        if buf is None or align_to_frame.is_silence() or align_to_frame.is_fade_out():
            return 0

        anchor = self._anchor_rms(align_to_frame)
        if anchor is None:
            return 0
        anchor_rms, lead_silent = anchor

        buf_frame_bytes = (
            int(buf.sample_rate * self._frame_duration) * buf.num_channels * 2
        )
        if buf_frame_bytes <= 0:
            return 0
        max_d = min(
            self._config.align_audio_max_frames,
            len(buf.bytes_) // buf_frame_bytes - len(anchor_rms),
        )
        if max_d < 0:
            return 0

        best_d = self._best_alignment_offset(buf, anchor_rms, buf_frame_bytes, max_d)
        if best_d is None:
            return 0
        shift = best_d - lead_silent
        if shift > 0:
            del buf.bytes_[: shift * buf_frame_bytes]
        elif shift < 0:
            buf.bytes_[:0] = bytes(-shift * buf_frame_bytes)
        return shift

    @staticmethod
    def _best_alignment_offset(
        buf: AudioBuffer, anchor_rms: list[float], buf_frame_bytes: int, max_d: int
    ) -> Optional[int]:
        """Return the leading-frame offset whose RMS window matches the anchors.

        ``None`` when no offset matches within tolerance — the caller must not
        shift anything. ``0`` is a *confident* match at the head (distinct from
        the old sentinel meaning "give up"): with a silent-head lead the caller
        turns it into a negative shift. The head is preferred whenever it is
        within tolerance, so the steady-state case never wanders to a later
        window. The tolerance scales with anchor energy so it works at any
        volume.
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

        tol = (_ALIGN_REL_TOL * (sum(anchor_rms) / len(anchor_rms))) ** 2
        base_err = window_err(0)
        if base_err is None:
            return None
        if base_err <= tol:
            return 0
        best_d, best_err = 0, base_err
        for d in range(1, max_d + 1):
            e = window_err(d)
            if e is None:
                break
            if e < best_err:
                best_d, best_err = d, e
        if best_d > 0 and best_err <= tol:
            return best_d
        return None

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
            # Repay video-repeat debt: while mid-speech repeats left the video
            # timeline behind the audio, skip one extra plain-speech frame per
            # tick (video runs 2x for that tick) until the streams line up again.
            # Never skips across a turn boundary — both the shown and the skipped
            # frame must be plain continuation speech.
            if (
                self._repeat_debt > 0
                and self._config.video_repeat_catchup_enabled
                and video_frame.is_speech()
                and self.video_frames
                and self.video_frames[0].is_speech()
            ):
                video_frame = self.video_frames.popleft()
                self._repeat_debt -= 1
                result.catchup_dropped = 1
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

        # A turn boundary re-anchors the streams: owed video from the previous
        # turn must not bleed into the next one.
        if result.swapped or (
            video_frame is not None
            and (video_frame.is_silence() or video_frame.is_fade_out())
        ):
            self._repeat_debt = 0

        result.idle_skipped = self.drain_idle_backlog(video_frame)
        result.audio_chunk = self._drain_audio_chunk()

        # Accrue video-repeat debt: real (non-faded) speech audio advanced this
        # tick but no video frame was available — the shown frame repeated, so
        # the video timeline is now one frame behind the audio.
        if (
            self._config.video_repeat_catchup_enabled
            and video_frame is None
            and result.audio_chunk is not None
            and self.current_buffer is not None
            and not self.current_buffer.interrupted
        ):
            self._repeat_debt = min(self._repeat_debt + 1, _REPEAT_DEBT_MAX)

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
