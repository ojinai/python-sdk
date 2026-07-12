"""Unit tests for the pure synchronizer (audio/video sync state machine)."""

from ojin.stv.config import STVConfig
from ojin.stv.synchronizer import AudioBuffer, Synchronizer, VideoFrame


def make_sync(**cfg) -> Synchronizer:
    """Return a synchronizer with a frozen clock for deterministic tests."""
    return Synchronizer(STVConfig(**cfg), clock=lambda: 0.0)


def vf(ft: int, audio: bytes = b"", img: bytes = b"x") -> VideoFrame:
    """Build a VideoFrame of the given wire frame_type."""
    return VideoFrame(
        frame_type=ft, image_bytes=img, audio_bytes=audio, is_final=False, volume=0
    )


# ----------------------------------------------------------------------
# Task 3: buffering
# ----------------------------------------------------------------------


def test_open_turn_appends_buffer() -> None:
    """open_turn appends a fresh empty buffer at the tail."""
    s = make_sync()
    b = s.open_turn()
    assert s.audio_buffers and s.audio_buffers[-1] is b and not b.bytes_


def test_add_audio_extends_tail_buffer() -> None:
    """add_audio extends the tail buffer and records its sample shape."""
    s = make_sync()
    s.open_turn()
    t = s.add_audio(b"\x01\x02" * 10, 24000, 1)
    assert t is s.audio_buffers[-1] and len(t.bytes_) == 20 and t.sample_rate == 24000


def test_add_audio_to_draining_current_when_no_queue() -> None:
    """With no queued buffer, in-turn audio extends the draining current buffer."""
    s = make_sync()
    s.open_turn()
    s.add_audio(b"\x00" * 8, 16000, 1)
    s.current_buffer = s.audio_buffers.popleft()  # promote
    t = s.add_audio(b"\x01" * 8, 16000, 1)
    assert t is s.current_buffer and len(t.bytes_) == 16


def test_add_audio_dropped_when_current_interrupted_and_no_queue() -> None:
    """Straggler audio after a cancel (interrupted current, empty queue) is dropped."""
    s = make_sync()
    s.open_turn()
    s.add_audio(b"\x00" * 8, 16000, 1)
    s.current_buffer = s.audio_buffers.popleft()
    s.current_buffer.interrupted = True
    assert s.add_audio(b"\x01" * 8, 16000, 1) is None


# ----------------------------------------------------------------------
# Task 4: interrupt / can_interrupt / queue discard
# ----------------------------------------------------------------------


def test_can_interrupt_requires_current_with_bytes() -> None:
    """Interruptible only when a non-interrupted current buffer has audio left."""
    s = make_sync()
    assert s.can_interrupt() is False
    s.current_buffer = AudioBuffer()
    s.current_buffer.bytes_.extend(b"\x01\x02")
    assert s.can_interrupt() is True
    s.current_buffer.interrupted = True
    assert s.can_interrupt() is False


def test_interrupt_marks_and_discards_queue_and_returns_true() -> None:
    """A valid barge-in marks current interrupted and drops queued buffers."""
    s = make_sync()
    s.current_buffer = AudioBuffer()
    s.current_buffer.bytes_.extend(b"\x01\x02")
    s.open_turn()
    s.open_turn()  # two queued buffers from the cancelled turn
    s.swap_pending = True
    assert s.interrupt() is True
    assert s.current_buffer.interrupted is True
    assert len(s.audio_buffers) == 0 and s.swap_pending is False


def test_interrupt_noop_when_idle() -> None:
    """No current buffer → interrupt is a no-op returning False (no cancel)."""
    s = make_sync()
    assert s.interrupt() is False


# ----------------------------------------------------------------------
# Task 5: swap + RMS alignment
# ----------------------------------------------------------------------


def test_current_replaceable_rules() -> None:
    """A current buffer is replaceable only when drained and not interrupted."""
    s = make_sync()
    assert s.current_replaceable() is True  # no current
    s.current_buffer = AudioBuffer()
    s.current_buffer.bytes_.extend(b"\x00\x00")
    assert s.current_replaceable() is False  # has bytes, not interrupted
    s.current_buffer.bytes_.clear()
    assert s.current_replaceable() is True  # drained, not interrupted


def test_current_replaceable_when_swap_pending_and_interrupted() -> None:
    """Once a boundary passed with no buffer, an interrupted remnant is replaceable."""
    s = make_sync()
    s.current_buffer = AudioBuffer()
    s.current_buffer.bytes_.extend(b"\x00\x00")
    s.current_buffer.interrupted = True
    s.swap_pending = True
    assert s.current_replaceable() is True


def test_swap_promotes_head_skipping_empty() -> None:
    """Swap skips empty head buffers and promotes the first with audio."""
    s = make_sync()
    s.open_turn()  # empty buffer (skipped)
    s.open_turn()
    s.add_audio(b"\x01" * 8, 16000, 1)  # non-empty
    s.swap_to_next_buffer(align_to_frame=None)
    assert s.current_buffer is not None and len(s.current_buffer.bytes_) == 8
    assert s.swap_pending is False


def test_swap_sets_pending_when_queue_empty() -> None:
    """A new-turn boundary with no queued buffer defers the swap."""
    s = make_sync()
    s.swap_to_next_buffer(align_to_frame=vf(3))
    assert s.current_buffer is None and s.swap_pending is True


def test_align_no_trim_below_min_anchors() -> None:
    """A single-frame RMS signature never trims (lipsync_0609 guard)."""
    s = make_sync()
    s.current_buffer = AudioBuffer(sample_rate=16000)
    s.current_buffer.bytes_.extend((b"\x00\x05" * 320) * 10)  # 5 frames of audio
    before = len(s.current_buffer.bytes_)
    loud = b"\x00\x40" * 320  # one loud 40 ms frame, no other anchors queued
    trimmed = s.align_current_buffer_to_frame(vf(3, audio=loud))
    assert trimmed == 0 and len(s.current_buffer.bytes_) == before


def test_align_trims_dropped_leading_speech() -> None:
    """With enough anchors, align trims leading audio to match the server's frame."""
    s = make_sync()
    fb = int(16000 * (1 / 25)) * 2  # 1280 bytes per 40 ms frame

    def frame_of(v: int) -> bytes:
        return v.to_bytes(2, "little", signed=True) * (fb // 2)

    # Buffer = 3 quiet frames then a varying loud region (a flat envelope would
    # match at several offsets and be rejected as ambiguous — real speech varies).
    loud_seq = [16000, 9000, 13000, 6000, 11000, 15000, 8000, 12000]
    quiet = frame_of(1280) * 3
    s.current_buffer = AudioBuffer(sample_rate=16000)
    s.current_buffer.bytes_.extend(quiet + b"".join(frame_of(v) for v in loud_seq))
    # The server dropped the 3 quiet frames: its frames start at the loud onset.
    for v in loud_seq[1:6]:
        s.video_frames.append(vf(1, audio=frame_of(v)))
    trigger = vf(3, audio=frame_of(loud_seq[0]))
    trimmed = s.align_current_buffer_to_frame(trigger)
    assert trimmed == 3  # the 3 quiet frames were trimmed


# ----------------------------------------------------------------------
# Task 6: idle backlog drain
# ----------------------------------------------------------------------


def test_drain_idle_skips_one_silence_over_target() -> None:
    """Over target while idle, one extra silence frame is dropped per tick."""
    s = make_sync()  # idle_buffer_target_frames == 6
    for _ in range(9):
        s.video_frames.append(vf(0))
    popped = s.video_frames.popleft()  # simulate the tick's pop → 8 remain
    assert s.drain_idle_backlog(popped) == 1 and len(s.video_frames) == 7


def test_drain_idle_two_when_speech_pending() -> None:
    """Two silence frames are dropped when a reply's speech waits behind them."""
    s = make_sync()
    for _ in range(8):
        s.video_frames.append(vf(0))
    s.video_frames.append(vf(1))  # speech behind the silence
    popped = s.video_frames.popleft()
    assert s.drain_idle_backlog(popped) == 2


def test_drain_idle_noop_when_audio_draining() -> None:
    """Never advance video past audio: no drain while a buffer has bytes."""
    s = make_sync()
    s.current_buffer = AudioBuffer()
    s.current_buffer.bytes_.extend(b"\x01" * 16)
    for _ in range(9):
        s.video_frames.append(vf(0))
    popped = s.video_frames.popleft()
    assert s.drain_idle_backlog(popped) == 0


def test_drain_idle_noop_on_non_silence_pop() -> None:
    """The drain only runs when the popped frame is silence."""
    s = make_sync()
    for _ in range(9):
        s.video_frames.append(vf(0))
    assert s.drain_idle_backlog(vf(1)) == 0


# ----------------------------------------------------------------------
# Task 7: tick() + speaking edges
# ----------------------------------------------------------------------

CHUNK = int(16000 * (1 / 25)) * 2  # 1280 bytes per 40 ms @ 16 kHz mono


def test_tick_warmup_holds_frames_then_emits() -> None:
    """During warm-up tick reports warming_up and emits nothing."""
    s = make_sync(initial_buffer_frames=3)
    for _ in range(5):
        s.video_frames.append(vf(0))
    r = s.tick()
    assert r.warming_up is True and r.video_frame is None and r.audio_chunk is None


def test_tick_drains_audio_chunk_when_speaking() -> None:
    """A populated current buffer drains one chunk and raises the speaking edge."""
    s = make_sync(initial_buffer_frames=0)
    buf = AudioBuffer(sample_rate=16000, num_channels=1)
    buf.bytes_.extend(b"\x01\x02" * CHUNK)  # plenty of audio
    s.current_buffer = buf
    s.video_frames.append(vf(1))
    r = s.tick()
    assert r.audio_chunk is not None and len(r.audio_chunk) == CHUNK
    assert r.started_speaking is True and r.video_frame is not None


def test_tick_underrun_returns_none_chunk() -> None:
    """An empty (non-interrupted) current buffer underruns to a None chunk."""
    s = make_sync(initial_buffer_frames=0)
    s.current_buffer = AudioBuffer(sample_rate=16000)
    r = s.tick()
    assert r.audio_chunk is None and r.started_speaking is False


def test_tick_stopped_speaking_edge_fires_once() -> None:
    """The stopped-speaking edge fires the tick the buffer drains, then not again."""
    s = make_sync(initial_buffer_frames=0)
    buf = AudioBuffer(sample_rate=16000)
    buf.bytes_.extend(b"\x01\x02" * CHUNK)  # two chunks
    s.current_buffer = buf
    first = s.tick()  # drains chunk 1, still has audio → started this tick
    assert first.started_speaking is True
    second = s.tick()  # drains chunk 2 → empty → stopped edge
    assert second.stopped_speaking is True
    third = s.tick()  # still empty → no repeat
    assert third.stopped_speaking is False


def test_tick_swaps_on_frame_type_3() -> None:
    """A new-turn frame promotes the queued buffer before draining audio."""
    s = make_sync(initial_buffer_frames=0)
    s.open_turn()
    s.add_audio(b"\x01" * CHUNK, 16000, 1)
    s.video_frames.append(vf(3, audio=b"\x01" * CHUNK))
    r = s.tick()
    assert r.swapped is True and s.current_buffer is not None


def test_tick_natural_turn_end_swap() -> None:
    """A speech frame while current is drained+queued triggers a natural swap."""
    s = make_sync(initial_buffer_frames=0)
    s.current_buffer = AudioBuffer(sample_rate=16000)  # drained, not interrupted
    s.open_turn()
    s.add_audio(b"\x02" * (CHUNK * 2), 16000, 1)  # two chunks queued
    s.video_frames.append(vf(1, audio=b"\x02" * 16))
    r = s.tick()
    assert r.swapped is True and len(s.current_buffer.bytes_) > 0


def test_tick_interrupt_fade_then_silence() -> None:
    """Barge-in fades audio for the window, then emits silence (None chunk)."""
    s = make_sync(initial_buffer_frames=0)
    buf = AudioBuffer(sample_rate=16000)
    buf.bytes_.extend(b"\x40\x40" * 100_000)  # long buffer so the fade completes first
    s.current_buffer = buf
    s.interrupt()
    first = s.tick()
    assert first.audio_chunk is not None  # faded, not yet silent
    for _ in range(200):  # exhaust the fade window
        s.tick()
    assert s.tick().audio_chunk is None  # ramp complete → silence


# ----------------------------------------------------------------------
# Mid-speech video-repeat catch-up (repeat debt)
# ----------------------------------------------------------------------


def make_speaking_sync(chunks: int = 4) -> Synchronizer:
    """Return a mid-turn synchronizer whose current buffer holds `chunks` chunks."""
    s = make_sync(initial_buffer_frames=0)
    buf = AudioBuffer(sample_rate=16000)
    buf.bytes_.extend(b"\x01\x02" * (CHUNK // 2) * chunks)
    s.current_buffer = buf
    return s


def test_tick_repeat_stall_accrues_debt_and_catches_up() -> None:
    """A mid-speech stall (audio drains, no frame) is repaid by skipping a frame."""
    s = make_speaking_sync()
    s.video_frames.append(vf(1))
    s.tick()  # normal paired tick
    stalled = s.tick()  # no video queued: audio drains, video repeats → debt 1
    assert stalled.video_frame is None and stalled.audio_chunk is not None
    f1, f2, f3 = vf(1), vf(1), vf(1)
    for f in (f1, f2, f3):
        s.video_frames.append(f)
    caught_up = s.tick()
    assert caught_up.catchup_dropped == 1
    assert caught_up.video_frame is f2  # f1 skipped: video jumps 80 ms this tick
    assert len(s.video_frames) == 1
    assert s.tick().catchup_dropped == 0  # debt repaid — back to one per tick


def test_tick_repeat_catchup_never_skips_across_turn_boundary() -> None:
    """Catch-up only skips plain speech: a new-turn head frame is never consumed."""
    s = make_speaking_sync()
    s.video_frames.append(vf(1))
    s.tick()
    s.tick()  # stall → debt 1
    s.video_frames.append(vf(1))
    s.video_frames.append(vf(3, audio=b"\x00\x40" * (CHUNK // 2)))
    r = s.tick()
    assert r.catchup_dropped == 0 and r.video_frame is not None
    assert r.video_frame.frame_type == 1  # the new-turn frame stays queued


def test_tick_repeat_debt_reset_at_turn_end() -> None:
    """Owed video from one turn does not bleed into the next (idle pop resets)."""
    s = make_speaking_sync(chunks=2)
    s.video_frames.append(vf(1))
    s.tick()
    s.tick()  # stall while last chunk drains → debt 1
    s.video_frames.append(vf(0))
    s.tick()  # idle frame pops → turn over → debt reset
    s.current_buffer = AudioBuffer(sample_rate=16000)
    s.current_buffer.bytes_.extend(b"\x01\x02" * CHUNK)
    s.video_frames.append(vf(1))
    s.video_frames.append(vf(1))
    r = s.tick()
    assert r.catchup_dropped == 0 and len(s.video_frames) == 1


def test_tick_idle_stall_accrues_no_debt() -> None:
    """A stall with no speech draining (idle) owes nothing."""
    s = make_sync(initial_buffer_frames=0)
    s.tick()  # idle, no video, no audio
    buf = AudioBuffer(sample_rate=16000)
    buf.bytes_.extend(b"\x01\x02" * CHUNK)
    s.current_buffer = buf
    s.video_frames.append(vf(1))
    s.video_frames.append(vf(1))
    r = s.tick()
    assert r.catchup_dropped == 0 and len(s.video_frames) == 1


def test_tick_repeat_catchup_disabled_by_config() -> None:
    """With the knob off, a stall shifts video permanently (legacy behavior)."""
    s = make_sync(initial_buffer_frames=0, video_repeat_catchup_enabled=False)
    buf = AudioBuffer(sample_rate=16000)
    buf.bytes_.extend(b"\x01\x02" * CHUNK * 2)
    s.current_buffer = buf
    s.video_frames.append(vf(1))
    s.tick()
    s.tick()  # stall
    s.video_frames.append(vf(1))
    s.video_frames.append(vf(1))
    r = s.tick()
    assert r.catchup_dropped == 0 and len(s.video_frames) == 1


def test_interrupt_resets_repeat_debt() -> None:
    """A barge-in cancels the turn's owed video."""
    s = make_speaking_sync()
    s.video_frames.append(vf(1))
    s.tick()
    s.tick()  # stall → debt 1
    assert s.interrupt() is True
    s.current_buffer = AudioBuffer(sample_rate=16000)
    s.current_buffer.bytes_.extend(b"\x01\x02" * CHUNK)
    s.video_frames.append(vf(1))
    s.video_frames.append(vf(1))
    r = s.tick()
    assert r.catchup_dropped == 0 and len(s.video_frames) == 1


# ----------------------------------------------------------------------
# Bidirectional swap alignment (silent-head anchor)
# ----------------------------------------------------------------------

FB = int(16000 * (1 / 25)) * 2  # 1280 bytes per 40 ms frame @ 16 kHz


def test_align_prepends_silence_for_server_padded_head() -> None:
    """Extra near-zero server head frames delay the audio instead of aborting.

    The server can open a turn with speech-typed frames whose audio is ~zero
    (padding minted while starved of TTS). The buffer has no such frames, so the
    audio must WAIT: alignment prepends that much silence and reports a negative
    shift. The old code skipped alignment entirely (silent anchor) and the whole
    turn played 40 ms x N out of sync.
    """
    s = make_sync()
    s.current_buffer = AudioBuffer(sample_rate=16000)
    s.current_buffer.bytes_.extend((b"\x00\x40" * (FB // 2)) * 8)  # loud at head
    before = len(s.current_buffer.bytes_)
    zero = b"\x00\x00" * (FB // 2)
    for _ in range(2):  # two more near-zero speech frames queued behind the trigger
        s.video_frames.append(vf(1, audio=zero))
    for _ in range(5):  # then the audible onset
        s.video_frames.append(vf(1, audio=b"\x00\x40" * (FB // 2)))
    shift = s.align_current_buffer_to_frame(vf(3, audio=zero))
    assert shift == -3
    assert len(s.current_buffer.bytes_) == before + 3 * FB
    assert s.current_buffer.bytes_[: 3 * FB] == bytes(3 * FB)


def test_align_silent_head_and_buffer_lead_nets_out() -> None:
    """Server silent head and buffer quiet head cancel into a net signed shift."""
    s = make_sync()
    s.current_buffer = AudioBuffer(sample_rate=16000)
    zero_frames = bytes(3 * FB)  # 3 near-zero frames the server never rendered...
    loud_seq = [16000, 9000, 13000, 6000, 11000, 15000, 8000, 12000]  # varying
    loud = b"".join(v.to_bytes(2, "little", signed=True) * (FB // 2) for v in loud_seq)
    s.current_buffer.bytes_.extend(zero_frames + loud)
    before = len(s.current_buffer.bytes_)
    for v in loud_seq[:5]:
        s.video_frames.append(
            vf(1, audio=v.to_bytes(2, "little", signed=True) * (FB // 2))
        )
    # ...while the server emits 1 near-zero head frame: net trim = 3 - 1 = 2.
    shift = s.align_current_buffer_to_frame(vf(3, audio=b"\x00\x00" * (FB // 2)))
    assert shift == 2
    assert len(s.current_buffer.bytes_) == before - 2 * FB


def test_align_no_shift_without_confident_match() -> None:
    """An anchor that matches nowhere in the buffer shifts nothing."""
    s = make_sync()
    s.current_buffer = AudioBuffer(sample_rate=16000)
    s.current_buffer.bytes_.extend((b"\x00\x05" * (FB // 2)) * 10)  # quiet only
    before = len(s.current_buffer.bytes_)
    for _ in range(5):
        s.video_frames.append(vf(1, audio=b"\x00\x40" * (FB // 2)))
    shift = s.align_current_buffer_to_frame(vf(3, audio=b"\x00\x40" * (FB // 2)))
    assert shift == 0 and len(s.current_buffer.bytes_) == before


def test_tick_natural_end_swap_aligns_padded_head() -> None:
    """Natural-end swaps run the aligner too — server head anomalies self-heal.

    The server never marks a natural turn entry with frame_type 3, so anomalies
    there (e.g. stale near-zero bytes rendered at the head) used to stick for
    the whole turn. The swap triggered by a plain SPEECH frame must align.
    """
    s = make_sync(initial_buffer_frames=0)
    s.current_buffer = AudioBuffer(sample_rate=16000)  # drained → replaceable
    s.open_turn()
    s.add_audio(bytes(b"\x00\x40" * (FB // 2)) * 8, 16000, 1)  # loud from byte 0
    zero = b"\x00\x00" * (FB // 2)
    for _ in range(2):  # two more near-zero speech frames behind the trigger
        s.video_frames.append(vf(1, audio=zero))
    for _ in range(5):
        s.video_frames.append(vf(1, audio=b"\x00\x40" * (FB // 2)))
    s.video_frames.appendleft(vf(1, audio=zero))  # the swap-triggering frame
    r = s.tick()
    assert r.swapped is True
    assert r.align_trim_frames == -3  # audio delayed: 3 frames of silence prepended
    assert r.audio_chunk == bytes(FB)  # first drained chunk is the prepended silence


# ----------------------------------------------------------------------
# Deferred swap alignment (anchor starvation at just-in-time turn entry)
# ----------------------------------------------------------------------


def pcm_frame(rms: float, fb: int = FB) -> bytes:
    """One 40 ms frame of constant int16 samples whose RMS equals `rms`."""
    v = round(rms)
    return v.to_bytes(2, "little", signed=True) * (fb // 2)


def test_deferred_align_fires_after_anchors_arrive() -> None:
    """A starved swap defers alignment and applies the shift a few ticks later.

    At a real turn entry the video queue is 1-2 frames deep, so the swap tick
    cannot build a 4-frame anchor. The old guard silently skipped alignment,
    leaving ±1-frame skews for the whole turn. Now the head is snapshotted and
    the shift lands once enough frames have popped.
    """
    s = make_sync(initial_buffer_frames=0)
    s.current_buffer = AudioBuffer(sample_rate=16000)  # drained → replaceable
    s.open_turn()
    # Buffer: 1 near-zero frame the server dropped, then distinct loud frames.
    rms_buf = [0, 5000, 12000, 15000, 9000, 6000, 13000, 8000, 11000, 7000]
    s.audio_buffers[-1].bytes_.extend(b"".join(pcm_frame(v) for v in rms_buf))
    before = len(s.audio_buffers[-1].bytes_)
    # Server frames arrive one per tick, starting at the buffer's 2nd frame
    # (the server dropped the near-zero head): trigger swap with only it queued.
    server_rms = rms_buf[1:]
    s.video_frames.append(vf(1, audio=pcm_frame(server_rms[0])))
    r = s.tick()  # swap fires, anchor starved (1 audible frame) → deferred
    assert r.swapped is True and r.align_trim_frames == 0
    shifts = []
    for v in server_rms[1:5]:
        s.video_frames.append(vf(1, audio=pcm_frame(v)))
        shifts.append(s.tick().align_trim_frames)
    # 4th audible anchor completes on the 3rd follow-up tick → trim of 1 frame.
    assert +1 in shifts
    cur = s.current_buffer
    assert cur is not None
    drained = 5 * FB  # 5 ticks drained one chunk each
    assert len(cur.bytes_) == before - drained - 1 * FB  # plus the 1-frame trim


def test_deferred_align_gives_up_after_window() -> None:
    """Deferred alignment stops waiting after _ALIGN_DEFER_MAX_TICKS."""
    s = make_sync(initial_buffer_frames=0)
    s.current_buffer = AudioBuffer(sample_rate=16000)
    s.open_turn()
    s.audio_buffers[-1].bytes_.extend(pcm_frame(9000) * 30)
    s.video_frames.append(vf(1, audio=pcm_frame(9000)))
    s.tick()  # swap, starved → pending
    assert s._pending_align is not None
    for _ in range(9):  # feed only near-silent speech frames: never enough anchors
        s.video_frames.append(vf(1, audio=pcm_frame(0)))
        s.tick()
    assert s._pending_align is None  # gave up quietly


def test_deferred_align_reset_on_interrupt() -> None:
    """A barge-in cancels any pending alignment of the cancelled turn."""
    s = make_sync(initial_buffer_frames=0)
    s.current_buffer = AudioBuffer(sample_rate=16000)
    s.open_turn()
    s.audio_buffers[-1].bytes_.extend(pcm_frame(9000) * 30)
    s.video_frames.append(vf(1, audio=pcm_frame(9000)))
    s.tick()
    assert s._pending_align is not None
    assert s.interrupt() is True
    assert s._pending_align is None


# ----------------------------------------------------------------------
# Regression fixtures: REAL RMS envelopes from staging session 4e35f6826fa5
# (the two turn entries that stayed ±40 ms off because alignment never fired)
# ----------------------------------------------------------------------

# Run 4 (+40 ms, post-barge-in): the server dropped the buffer's near-zero head
# frame; SOS arrives loud immediately. True shift: trim +1.
RUN4_BUFFER = [
    0.7,
    5454.8,
    12847.3,
    15372.4,
    12332.1,
    6281.4,
    12637.9,
    13634.0,
    11398.3,
    14984.5,
    13269.9,
    8580.5,
]
RUN4_FRAMES = [5413.9, 12845.9, 15372.0, 12330.6, 6090.4, 12529.3, 13624.6]

# Run 2 (-40 ms, natural entry): the server emitted 2 near-zero speech head
# frames while the buffer has 1. True shift: prepend 1 (net -1).
RUN2_BUFFER = [
    0.8,
    6117.4,
    13390.6,
    577.2,
    7061.3,
    10377.8,
    15811.8,
    10566.7,
    795.0,
    3629.6,
    1189.5,
    17878.1,
]
RUN2_FRAMES = [0.4, 0.3, 7782.9, 12476.1, 448.1, 7094.6, 11730.0, 15376.1]


def _run_session_entry(buffer_rms, frame_rms, trigger_type=1):
    """Replay a turn entry: starved queue, frames arriving one per tick."""
    s = make_sync(initial_buffer_frames=0)
    s.current_buffer = AudioBuffer(sample_rate=16000)
    s.open_turn()
    s.audio_buffers[-1].bytes_.extend(b"".join(pcm_frame(v) for v in buffer_rms))
    s.video_frames.append(vf(trigger_type, audio=pcm_frame(frame_rms[0])))
    first = s.tick()
    assert first.swapped is True
    shifts = [first.align_trim_frames]
    for v in frame_rms[1:]:
        s.video_frames.append(vf(1, audio=pcm_frame(v)))
        shifts.append(s.tick().align_trim_frames)
    return shifts


def test_real_envelope_run4_post_barge_in_trims_one() -> None:
    """Session 4e35f6826fa5 run 4: +40 ms skew → deferred trim of +1 frame."""
    shifts = _run_session_entry(RUN4_BUFFER, RUN4_FRAMES, trigger_type=3)
    assert +1 in shifts and -1 not in shifts


def test_real_envelope_run2_natural_entry_prepends_one() -> None:
    """Session 4e35f6826fa5 run 2: -40 ms skew → deferred prepend of 1 frame."""
    shifts = _run_session_entry(RUN2_BUFFER, RUN2_FRAMES, trigger_type=1)
    assert -1 in shifts and +1 not in shifts


def test_ambiguous_match_shifts_nothing() -> None:
    """Two equally-good windows (periodic envelope) fail the margin → no shift."""
    s = make_sync(initial_buffer_frames=0)
    s.current_buffer = AudioBuffer(sample_rate=16000)
    s.open_turn()
    # Perfectly periodic buffer: the anchor matches at d=2 AND d=4 equally.
    pattern = [9000.0, 2000.0]
    s.audio_buffers[-1].bytes_.extend(b"".join(pcm_frame(v) for v in pattern * 8))
    before = len(s.audio_buffers[-1].bytes_)
    frames = [2000.0, 9000.0, 2000.0, 9000.0, 2000.0]  # off-phase vs head
    s.video_frames.append(vf(1, audio=pcm_frame(frames[0])))
    s.tick()
    for v in frames[1:]:
        s.video_frames.append(vf(1, audio=pcm_frame(v)))
        r = s.tick()
        assert r.align_trim_frames == 0
    cur = s.current_buffer
    assert cur is not None
    assert len(cur.bytes_) == before - 5 * FB  # only the 5 drained chunks
