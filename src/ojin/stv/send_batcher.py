"""Server-feed audio batcher: combine TTS chunks into larger sends.

A pure, synchronous state machine (no I/O, no asyncio) — the testable seam for
``OjinSTVClient``'s server-bound audio. TTS providers stream small (~40 ms)
chunks; sending each immediately starves the inference server's 25 fps timeline
(input rate == output rate, so no supply lead ever builds) and churns sub-frame
residues that disrupt buffer swaps and lip-sync. This batcher accumulates the
resampled 16 kHz bytes and emits them in larger chunks: a big initial chunk per
turn to establish the lead, then a steady-state minimum to suppress residue
churn. A separate idle timeout (driven by the client) flushes a sub-threshold
tail when TTS goes quiet.

All sizes are in bytes of mono int16 PCM at the server feed rate (16 kHz). The
client converts its millisecond config to bytes and injects them here.
"""

from __future__ import annotations

import time
from typing import Callable, Optional


class SendBatcher:
    """Accumulate resampled TTS bytes; emit when a size or idle threshold is met."""

    def __init__(
        self,
        initial_chunk_bytes: int,
        min_chunk_bytes: int,
        flush_idle_s: float,
        clock: Callable[[], float] = time.monotonic,
        fixed_chunk_size: bool = False,
    ) -> None:
        """Create an empty batcher armed for an initial-threshold emit.

        Args:
            initial_chunk_bytes: size of the first emit after each (re)arm — the
                lead-establishing chunk.
            min_chunk_bytes: steady-state minimum emit size.
            flush_idle_s: quiet seconds after the last add before a tail is
                flush-due.
            clock: monotonic seconds source (injectable for deterministic tests).
            fixed_chunk_size: emit exactly the active threshold, retaining any
                overflow for subsequent :meth:`pop_ready` calls.

        """
        super().__init__()
        self._initial_chunk_bytes = initial_chunk_bytes
        self._min_chunk_bytes = min_chunk_bytes
        self._flush_idle_s = flush_idle_s
        self._clock = clock
        self._fixed_chunk_size = fixed_chunk_size
        if fixed_chunk_size and min(initial_chunk_bytes, min_chunk_bytes) <= 0:
            raise ValueError("Fixed audio chunk sizes must be positive")
        self._buf = bytearray()
        self._next_is_initial = True
        self._last_add_ts = clock()

    @property
    def pending_bytes(self) -> int:
        """Number of buffered bytes not yet emitted."""
        return len(self._buf)

    def add(self, pcm16k: bytes) -> Optional[bytes]:
        """Append bytes; return a batch to send when the size threshold is met.

        The threshold is ``initial_chunk_bytes`` until the first emit after each
        (re)arm, then ``min_chunk_bytes``. By default it returns all buffered
        bytes when the threshold is reached. In fixed-size mode it returns
        exactly the threshold; call :meth:`pop_ready` until it returns ``None``
        to release any other complete packets. Empty input is a no-op.
        """
        if not pcm16k:
            return None
        self._buf.extend(pcm16k)
        self._last_add_ts = self._clock()
        return self.pop_ready()

    def pop_ready(self) -> Optional[bytes]:
        """Return the next complete packet without resetting the idle timer."""
        if not self._buf:
            return None
        threshold = (
            self._initial_chunk_bytes
            if self._next_is_initial
            else self._min_chunk_bytes
        )
        if len(self._buf) >= threshold:
            self._next_is_initial = False
            if self._fixed_chunk_size:
                out = bytes(self._buf[:threshold])
                del self._buf[:threshold]
                return out
            return self._take()
        return None

    def flush_due(self) -> bool:
        """Return whether buffered bytes have sat idle for ``flush_idle_s``."""
        return (
            len(self._buf) > 0
            and (self._clock() - self._last_add_ts) >= self._flush_idle_s
        )

    def drain(self) -> Optional[bytes]:
        """Return all buffered bytes (or ``None`` if empty) and clear.

        Does not touch the initial flag — used for the idle-timeout tail flush,
        the start-of-turn previous-tail flush, and the close-time final flush.
        """
        if not self._buf:
            return None
        return self._take()

    def rearm_initial(self) -> None:
        """Make the next emit use the initial (lead) threshold again."""
        self._next_is_initial = True

    def reset(self) -> None:
        """Discard buffered bytes and re-arm the initial threshold (barge-in)."""
        self._buf.clear()
        self._next_is_initial = True

    def _take(self) -> bytes:
        """Drain and return the whole buffer as immutable bytes."""
        out = bytes(self._buf)
        self._buf.clear()
        return out
