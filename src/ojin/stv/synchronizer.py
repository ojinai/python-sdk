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
# Loose sanity cap for a non-head match, as a fraction of anchor energy. The
# strict `_ALIGN_REL_TOL` stays as the head short-circuit, but real cross-rate
# envelopes (16 kHz server frames vs 24 kHz TTS buffer) measured err ratios up
# to ~5x the strict tolerance at the TRUE offset, while wrong offsets measured
# 87-374x — so confidence comes from the margin below, not the absolute error.
_ALIGN_MAX_REL_TOL = 0.25
# A non-head match is only trusted when it beats every other offset (head
# included) by this factor. Measured margins at true offsets: 47-3000x.
_ALIGN_MARGIN = 8.0
# Deferred alignment: at a real turn entry the video queue is only 1-2 frames
# deep (just-in-time delivery), so the swap tick rarely has _ALIGN_MIN_ANCHORS
# audible frames to anchor on. Rather than skip alignment (which left ±1-frame
# skews uncorrected for whole turns), gather anchors from the frames popped on
# the following ticks and align late — a shift is a constant timeline offset,
# so applying it k ticks in still aligns the remainder of the turn. Give up
# after this many ticks (the anchor window plus slack).
_ALIGN_DEFER_MAX_TICKS = 8

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
    # Capped head of the 16 kHz server-bound copy of this buffer's audio —
    # the exact bytes the server's audible-onset gate classified. Filled via
    # note_resampled_audio; read once by the swap-time onset-gate mirror.
    resampled_head: bytearray = field(default_factory=bytearray)


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
class _PendingAlign:
    """Deferred swap-alignment state while anchors are still arriving.

    Created when a swap fires with too few audible frames queued to build a
    trustworthy anchor signature. ``head_snapshot`` preserves the buffer head as
    it was at the swap (the live buffer drains one chunk per tick, but the match
    offset is defined against the original head); ``rms_seq`` accumulates the
    per-frame bundled-audio RMS of the trigger and every speech frame popped
    since.
    """

    buffer_id: int
    head_snapshot: bytes
    buf_frame_bytes: int
    rms_seq: list[float]
    ticks_waited: int = 0


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
        # Swap alignment waiting for enough audible frames to anchor on.
        self._pending_align: Optional[_PendingAlign] = None

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

    def note_resampled_audio(self, buf: AudioBuffer, resampled: bytes) -> None:
        """Record a buffer's 16 kHz head copy for the onset-gate mirror (capped)."""
        cap = (self._config.onset_gate_max_frames + 4) * BYTES_PER_FRAME
        room = cap - len(buf.resampled_head)
        if room > 0 and resampled:
            buf.resampled_head.extend(resampled[:room])

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
        self._pending_align = None  # so is its unfinished alignment
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
        # Whatever alignment the outgoing buffer still owed is moot now.
        self._pending_align = None
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

        # Align on every speech-triggered swap, natural turn ends included. The
        # lipsync_0609 guard restricted this to the server-signalled new-turn
        # boundary because the old matcher could only trim and a false trim on a
        # natural end (where the correct shift is usually 0) injected error. Two
        # things changed: the matcher is now confident-match-or-nothing with a
        # head preference (steady state resolves to shift 0 and is untouched),
        # and natural-turn entries were measured carrying real head anomalies
        # (near-zero speech frames the buffer lacks) that ONLY this path can
        # self-heal — the server never marks them, so waiting for frame_type 3
        # means never correcting them.
        if self._config.align_audio_on_swap and align_to_frame is not None:
            return self.align_current_buffer_to_frame(align_to_frame)
        return 0

    def _collect_anchor(
        self, rms_seq: list[float]
    ) -> tuple[Optional[list[float]], int, bool]:
        """Split an ordered per-frame RMS sequence into (anchor, lead, starved).

        The anchor signature starts at the first audible frame: a turn can open
        with speech-typed frames whose audio is near-zero (e.g. the server
        padded the turn head while starved of TTS), and matching on those both
        pollutes the signature and used to abort alignment entirely because the
        anchor looked silent. ``lead`` counts those skipped near-silent head
        frames so the caller can convert a matched buffer offset into a signed
        shift.

        ``starved`` is True when the sequence simply ran out before
        ``_ALIGN_MIN_ANCHORS`` audible frames were seen (the lipsync_0609 guard
        against a too-short signature) — more frames may still arrive, so the
        caller can defer instead of giving up. Anchor is None in that case, and
        when the silent head is implausibly long (> ``align_audio_max_frames``;
        ``starved`` False — unusable).
        """
        lead_silent = 0
        anchor_rms: list[float] = []
        for r in rms_seq:
            if not anchor_rms and r < _ALIGN_MIN_RMS:
                lead_silent += 1
                if lead_silent > self._config.align_audio_max_frames:
                    return None, lead_silent, False
                continue
            anchor_rms.append(r)
            if len(anchor_rms) >= _ALIGN_ANCHOR_FRAMES:
                break
        if len(anchor_rms) >= _ALIGN_MIN_ANCHORS:
            return anchor_rms, lead_silent, False
        return None, lead_silent, True

    def _buf_frame_bytes(self, buf: AudioBuffer) -> int:
        """Bytes per 40 ms frame at the buffer's sample shape."""
        return int(buf.sample_rate * self._frame_duration) * buf.num_channels * 2

    def _match_and_apply(
        self,
        anchor_rms: list[float],
        lead_silent: int,
        head: bytes,
        buf_frame_bytes: int,
    ) -> int:
        """Match the anchor against ``head`` and apply the signed shift.

        ``head`` is the buffer head as it was at the swap (the live buffer may
        have drained since; a shift is a constant timeline offset, so applying
        it to the current head still aligns the remainder of the turn). Returns
        the applied shift in frames: positive = leading audio trimmed (the
        server dropped it), negative = silence prepended so the audio waits for
        the video (the server emitted extra near-silent head frames).
        """
        buf = self.current_buffer
        if buf is None or buf_frame_bytes <= 0:
            return 0
        max_d = min(
            self._config.align_audio_max_frames,
            len(head) // buf_frame_bytes - len(anchor_rms),
        )
        if max_d < 0:
            return 0
        best_d = self._best_alignment_offset(head, anchor_rms, buf_frame_bytes, max_d)
        if best_d is None:
            return 0
        shift = best_d - lead_silent
        if shift > 0:
            del buf.bytes_[: min(shift * buf_frame_bytes, len(buf.bytes_))]
        elif shift < 0:
            buf.bytes_[:0] = bytes(-shift * buf_frame_bytes)
        return shift

    def _feed_pending_align(self, frame: VideoFrame) -> int:
        """Grow a deferred alignment with this tick's popped frame; maybe apply.

        Returns the signed shift applied this tick (0 otherwise). The pending
        state is dropped when the turn moves on (silence/fade popped, buffer
        replaced), when the window fills without enough audible anchors, or
        after ``_ALIGN_DEFER_MAX_TICKS``.
        """
        st = self._pending_align
        if st is None:
            return 0
        cur = self.current_buffer
        if (
            cur is None
            or cur.buffer_id != st.buffer_id
            or frame.is_silence()
            or frame.is_fade_out()
        ):
            self._pending_align = None
            return 0
        st.ticks_waited += 1
        r = rms_int16(frame.audio_bytes)
        if r is not None:
            st.rms_seq.append(r)
        anchor_rms, lead_silent, starved = self._collect_anchor(st.rms_seq)
        if anchor_rms is not None:
            self._pending_align = None
            return self._match_and_apply(
                anchor_rms, lead_silent, st.head_snapshot, st.buf_frame_bytes
            )
        if not starved or st.ticks_waited >= _ALIGN_DEFER_MAX_TICKS:
            self._pending_align = None
        return 0

    def _mirror_onset_gate_trim(self, buf: AudioBuffer, trigger: VideoFrame) -> int:
        """Trim the head frames the server's audible-onset gate dropped.

        The server retypes a turn's leading sub-threshold SPEECH frames to IDLE
        and drops them, so its speech timeline starts at the first audible
        frame; the locally played buffer must lose the same frames or the whole
        turn plays with video ahead of audio by 40 ms per gated frame. Replays
        the gate's exact predicate (RMS below ``onset_gate_min_rms`` of int16
        full scale, at most ``onset_gate_max_frames``, stop at the first
        audible frame) over the 16 kHz resampled head — the same bytes the
        server classified — falling back to the original bytes when no
        resampled copy was recorded. Skipped when the trigger frame is itself
        sub-threshold: then the server did not gate (its first speech frame IS
        the quiet head) and trimming would skew the other way. Returns the
        number of 40 ms frames trimmed.
        """
        if not self._config.onset_gate_mirror_enabled:
            return 0
        threshold = self._config.onset_gate_min_rms * 32768.0
        trigger_rms = rms_int16(trigger.audio_bytes)
        if trigger_rms is None or trigger_rms < threshold:
            return 0
        use_resampled = len(buf.resampled_head) >= BYTES_PER_FRAME
        src: "bytearray | bytes" = buf.resampled_head if use_resampled else buf.bytes_
        frame_bytes = BYTES_PER_FRAME if use_resampled else self._buf_frame_bytes(buf)
        if frame_bytes <= 0:
            return 0
        dropped = 0
        while dropped < self._config.onset_gate_max_frames:
            window = bytes(src[dropped * frame_bytes : (dropped + 1) * frame_bytes])
            if len(window) < frame_bytes:
                break
            window_rms = rms_int16(window)
            if window_rms is None or window_rms >= threshold:
                break
            dropped += 1
        if dropped:
            buf_frame_bytes = self._buf_frame_bytes(buf)
            del buf.bytes_[: min(dropped * buf_frame_bytes, len(buf.bytes_))]
            del buf.resampled_head[: dropped * BYTES_PER_FRAME]
        return dropped

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
        then apply the signed difference (see ``_match_and_apply``).

        At a real turn entry the video queue is typically only 1-2 frames deep
        (just-in-time delivery), so the swap tick rarely has enough audible
        frames for a trustworthy signature. In that case alignment is DEFERRED:
        the buffer head is snapshotted and the anchor keeps growing from the
        frames popped on the following ticks (``_feed_pending_align``), applying
        the shift a few ticks late — still a whole-turn fix, since the skew is a
        constant offset. Returns the shift applied NOW (0 when deferred/skipped).
        """
        buf = self.current_buffer
        if buf is None or align_to_frame.is_silence() or align_to_frame.is_fade_out():
            return 0
        if not align_to_frame.audio_bytes:
            return 0
        buf_frame_bytes = self._buf_frame_bytes(buf)
        if buf_frame_bytes <= 0:
            return 0

        # Deterministic onset-gate mirror first; the envelope matcher below
        # then only resolves residual skew on the trimmed head. Ordering also
        # means a deferred alignment snapshots the post-trim head.
        mirror_trim = self._mirror_onset_gate_trim(buf, align_to_frame)

        rms_seq: list[float] = []
        for f in [align_to_frame] + [
            f for f in list(self.video_frames) if not f.is_silence()
        ]:
            r = rms_int16(f.audio_bytes)
            if r is None:
                break
            rms_seq.append(r)

        anchor_rms, lead_silent, starved = self._collect_anchor(rms_seq)
        # The match only ever looks at the first max_d + anchor (+ lead) frames.
        snap_frames = (
            self._config.align_audio_max_frames + _ALIGN_ANCHOR_FRAMES + lead_silent + 1
        )
        head = bytes(buf.bytes_[: snap_frames * buf_frame_bytes])
        if anchor_rms is not None:
            return mirror_trim + self._match_and_apply(
                anchor_rms, lead_silent, head, buf_frame_bytes
            )
        if starved and rms_seq:
            # Not enough audible frames yet — snapshot the head and gather
            # anchors from the next ticks' pops. Seed with the trigger only:
            # the frames peeked above are still queued and will pop on the
            # following ticks, re-entering the sequence in order (seeding both
            # would double-count them).
            self._pending_align = _PendingAlign(
                buffer_id=buf.buffer_id,
                head_snapshot=head,
                buf_frame_bytes=buf_frame_bytes,
                rms_seq=rms_seq[:1],
            )
        return mirror_trim

    @staticmethod
    def _best_alignment_offset(
        head: bytes, anchor_rms: list[float], buf_frame_bytes: int, max_d: int
    ) -> Optional[int]:
        """Return the leading-frame offset whose RMS window matches the anchors.

        ``None`` when no offset is a confident match — the caller must not shift
        anything. ``0`` is a *confident* match at the head (distinct from the
        old sentinel meaning "give up"): with a silent-head lead the caller
        turns it into a negative shift.

        Confidence rules, calibrated against real staging envelopes (session
        4e35f6826fa5: true offsets scored err ratios 0.03-4.5x the strict
        tolerance while wrong offsets scored 87-374x, with best-vs-second-best
        margins of 47-3000x):
        - The head is preferred whenever it is within the strict tolerance
          (``_ALIGN_REL_TOL``), so the steady-state case never wanders.
        - A non-head match must pass a loose sanity cap (``_ALIGN_MAX_REL_TOL``
          of anchor energy) AND beat every other offset, head included, by
          ``_ALIGN_MARGIN`` — the discriminative signal is the margin, not the
          absolute error, which cross-sample-rate envelopes can't keep tight.
        """

        def window_err(d: int) -> Optional[float]:
            err = 0.0
            for j, a in enumerate(anchor_rms):
                start = (d + j) * buf_frame_bytes
                r = rms_int16(bytes(head[start : start + buf_frame_bytes]))
                if r is None:
                    return None
                err += (r - a) * (r - a)
            return err / len(anchor_rms)

        mean_rms = sum(anchor_rms) / len(anchor_rms)
        tol = (_ALIGN_REL_TOL * mean_rms) ** 2
        base_err = window_err(0)
        if base_err is None:
            return None
        if base_err <= tol:
            return 0
        best_d, best_err = 0, base_err
        second_err = float("inf")
        for d in range(1, max_d + 1):
            e = window_err(d)
            if e is None:
                break
            if e < best_err:
                best_d, best_err, second_err = d, e, best_err
            elif e < second_err:
                second_err = e
        tol_cap = (_ALIGN_MAX_REL_TOL * mean_rms) ** 2
        # Strictly-less: a periodic envelope matching two offsets equally well
        # (both errors 0) is ambiguous and must not shift.
        if best_d > 0 and best_err <= tol_cap and best_err * _ALIGN_MARGIN < second_err:
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
                and self._pending_align is None  # a skip would corrupt the anchor
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
            else:
                # A swap that couldn't align yet (anchor-starved) keeps gathering
                # anchors from the popped frames; the shift lands a few ticks in.
                deferred_shift = self._feed_pending_align(video_frame)
                if deferred_shift:
                    result.align_trim_frames = deferred_shift
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
