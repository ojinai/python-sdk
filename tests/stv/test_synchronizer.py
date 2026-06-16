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
    # Buffer = 3 quiet frames then a loud region; server dropped the 3 quiet frames.
    quiet = (b"\x00\x05" * (fb // 2)) * 3
    loud = (b"\x00\x40" * (fb // 2)) * 8
    s.current_buffer = AudioBuffer(sample_rate=16000)
    s.current_buffer.bytes_.extend(quiet + loud)
    # 5 loud anchor frames already buffered as upcoming speech frames.
    for _ in range(5):
        s.video_frames.append(vf(1, audio=b"\x00\x40" * (fb // 2)))
    trigger = vf(3, audio=b"\x00\x40" * (fb // 2))
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
