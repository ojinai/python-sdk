"""Unit tests for the pure SendBatcher state machine."""

from typing import Callable, Optional

from ojin.stv.send_batcher import SendBatcher


class FakeClock:
    """A controllable monotonic clock for deterministic timing tests."""

    def __init__(self) -> None:
        """Start the clock at t=0."""
        self.t = 0.0

    def __call__(self) -> float:
        """Return the current fake time."""
        return self.t

    def advance(self, dt: float) -> None:
        """Move the fake clock forward by ``dt`` seconds."""
        self.t += dt


def make_batcher(
    initial: int = 1000,
    min_: int = 400,
    idle: float = 0.2,
    clock: Optional[Callable[[], float]] = None,
) -> SendBatcher:
    """Build a batcher sized in raw bytes (1 byte == 1 unit for easy math)."""
    return SendBatcher(
        initial_chunk_bytes=initial,
        min_chunk_bytes=min_,
        flush_idle_s=idle,
        clock=clock or FakeClock(),
    )


def test_below_initial_threshold_returns_none() -> None:
    """Below the initial threshold, add buffers and returns nothing."""
    b = make_batcher(initial=1000, min_=400)
    assert b.add(b"\x00" * 600) is None
    assert b.pending_bytes == 600


def test_initial_threshold_emits_all_buffered() -> None:
    """Reaching the initial threshold drains ALL buffered bytes, not just threshold."""
    b = make_batcher(initial=1000, min_=400)
    assert b.add(b"\x01" * 600) is None
    out = b.add(b"\x02" * 500)  # total 1100 >= 1000
    assert out == b"\x01" * 600 + b"\x02" * 500
    assert b.pending_bytes == 0


def test_subsequent_emits_use_min_threshold() -> None:
    """After the first emit, the smaller min threshold applies."""
    b = make_batcher(initial=1000, min_=400)
    b.add(b"\x00" * 1000)  # first emit (initial)
    assert b.add(b"\x00" * 300) is None  # 300 < 400 min
    out = b.add(b"\x00" * 100)  # 400 >= 400 min
    assert out is not None and len(out) == 400


def test_flush_due_only_after_idle_gap() -> None:
    """flush_due is true only once the buffer has sat idle for flush_idle_s."""
    clk = FakeClock()
    b = make_batcher(initial=1000, min_=400, idle=0.2, clock=clk)
    b.add(b"\x00" * 100)
    assert b.flush_due() is False
    clk.advance(0.1)
    assert b.flush_due() is False
    clk.advance(0.15)  # total 0.25 >= 0.2
    assert b.flush_due() is True


def test_flush_due_false_when_empty() -> None:
    """An empty buffer is never flush-due, no matter how much time passes."""
    clk = FakeClock()
    b = make_batcher(idle=0.2, clock=clk)
    clk.advance(10.0)
    assert b.flush_due() is False


def test_drain_returns_all_and_clears() -> None:
    """Drain returns everything buffered and empties the buffer."""
    b = make_batcher(initial=1000)
    b.add(b"\x07" * 250)
    assert b.drain() == b"\x07" * 250
    assert b.pending_bytes == 0
    assert b.drain() is None


def test_rearm_initial_restores_initial_threshold() -> None:
    """rearm_initial makes the next emit wait for the initial threshold again."""
    b = make_batcher(initial=1000, min_=400)
    b.add(b"\x00" * 1000)  # first emit → now on min threshold
    b.rearm_initial()
    assert b.add(b"\x00" * 400) is None  # 400 < 1000 again
    out = b.add(b"\x00" * 600)  # 1000 >= initial
    assert out is not None and len(out) == 1000


def test_reset_discards_pending_and_rearms_initial() -> None:
    """Reset throws away buffered bytes and re-arms the initial threshold."""
    b = make_batcher(initial=1000, min_=400)
    b.add(b"\x00" * 1000)  # first emit → min threshold next
    b.add(b"\x00" * 200)
    b.reset()
    assert b.pending_bytes == 0
    assert b.add(b"\x00" * 400) is None  # re-armed to initial (400 < 1000)


def test_empty_add_is_noop() -> None:
    """Adding empty bytes does nothing."""
    b = make_batcher()
    assert b.add(b"") is None
    assert b.pending_bytes == 0


def test_drain_does_not_change_initial_flag() -> None:
    """Drain does not touch the initial flag — min threshold survives a drain."""
    b = make_batcher(initial=1000, min_=400)
    b.add(b"\x00" * 1000)  # emit → now on min threshold
    b.add(b"\x00" * 200)  # sub-min tail
    b.drain()  # flush tail, should leave the flag alone
    assert b.add(b"\x00" * 400) is not None  # 400 >= min, not initial
