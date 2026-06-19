# OjinSTVClient Audio Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `OjinSTVClient` coalesce server-bound TTS audio into larger chunks (a ~1000 ms lead per turn, ≥400 ms steady-state, with an idle-timeout tail flush) so providers that stream 40 ms chunks no longer starve the inference server or churn sub-frame residues.

**Architecture:** A new pure, synchronous `SendBatcher` state machine owns the resampled 16 kHz server-bound byte buffer and decides *when* a batch is ready. `OjinSTVClient` resamples per arriving chunk (keeping the streaming resampler warm), feeds the resampled bytes to the batcher, sends whatever the batcher emits, and runs a small debounce task to flush a sub-threshold tail when TTS goes quiet. No `inference-server` changes; the avatar playback path is untouched.

**Tech Stack:** Python 3.12+, asyncio, `uv` for deps/commands, `pytest` (+ `pytest-asyncio`), `ruff` for format/lint, `pyright` (advisory).

## Global Constraints

- Scope is `python-sdk` ONLY. No changes to `inference-server` or any other service.
- The avatar playback path is unchanged: the **original** TTS audio is still buffered verbatim in the `Synchronizer` per arrival; only the resampled **server-bound copy** is batched.
- Resample **per arriving chunk** (never re-resample a batch) so the streaming `SoxrStreamResampler` keeps its filter history warm.
- Server feed audio is mono int16 little-endian PCM at `OJIN_PERSONA_SAMPLE_RATE` = 16000 Hz ⇒ 32 bytes/ms, 1280 bytes/frame (40 ms).
- Defaults: `server_feed_batching_enabled=True`, `server_feed_initial_chunk_ms=1000`, `server_feed_min_chunk_ms=400`, `server_feed_flush_idle_ms=200`. The enabled flag is an escape hatch that restores today's per-chunk sends.
- Type hints required on all function signatures. Public modules/classes/functions carry docstrings (Ruff `D` rules). New async tests follow the existing file's `asyncio.run(run())` wrapper style for consistency within `tests/stv/test_ojin_stv_client.py`.
- Conventional Commits. Never use `--no-verify`. Never commit to `main` (work stays on `feat/stv-client-audio-batching`).
- Run from the `python-sdk` repo root. Tests: `uv run pytest tests/stv` (do NOT run `tests/integration/` — it starts a real server). Format/lint: `uv run ruff format .` and `uv run ruff check --fix .`.

---

### Task 1: `SendBatcher` pure state machine

**Files:**
- Create: `src/ojin/stv/send_batcher.py`
- Test: `tests/stv/test_send_batcher.py`

**Interfaces:**
- Consumes: nothing (pure, stdlib only).
- Produces:
  - `class SendBatcher`
  - `SendBatcher(initial_chunk_bytes: int, min_chunk_bytes: int, flush_idle_s: float, clock: Callable[[], float] = time.monotonic)`
  - `add(self, pcm16k: bytes) -> Optional[bytes]` — append; return all buffered bytes when the (initial-or-min) size threshold is met, else `None`; empty input is a no-op
  - `flush_due(self) -> bool` — buffer non-empty AND idle ≥ `flush_idle_s`
  - `drain(self) -> Optional[bytes]` — return all buffered bytes (or `None`) and clear; does NOT change the initial flag
  - `rearm_initial(self) -> None` — next emit uses the initial threshold again
  - `reset(self) -> None` — discard buffered bytes and re-arm initial
  - `pending_bytes` (property) `-> int`

- [ ] **Step 1: Write the failing tests**

Create `tests/stv/test_send_batcher.py`:

```python
"""Unit tests for the pure SendBatcher state machine."""

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
    initial: int = 1000, min_: int = 400, idle: float = 0.2, clock=None
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
    """Reaching the initial threshold drains ALL buffered bytes, not just the threshold."""
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
    """drain returns everything buffered and empties the buffer."""
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
    """reset throws away buffered bytes and re-arms the initial threshold."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/stv/test_send_batcher.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ojin.stv.send_batcher'`.

- [ ] **Step 3: Write the implementation**

Create `src/ojin/stv/send_batcher.py`:

```python
"""Server-feed audio batcher: coalesce TTS chunks into larger sends.

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
    ) -> None:
        """Create an empty batcher armed for an initial-threshold emit.

        Args:
            initial_chunk_bytes: size of the first emit after each (re)arm — the
                lead-establishing chunk.
            min_chunk_bytes: steady-state minimum emit size.
            flush_idle_s: quiet seconds after the last add before a tail is
                flush-due.
            clock: monotonic seconds source (injectable for deterministic tests).

        """
        super().__init__()
        self._initial_chunk_bytes = initial_chunk_bytes
        self._min_chunk_bytes = min_chunk_bytes
        self._flush_idle_s = flush_idle_s
        self._clock = clock
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
        (re)arm, then ``min_chunk_bytes``. On reaching it, drains and returns ALL
        buffered bytes (the threshold is a minimum, not a quantum) and clears the
        initial flag. Returns ``None`` below threshold. Empty input is a no-op.
        """
        if not pcm16k:
            return None
        self._buf.extend(pcm16k)
        self._last_add_ts = self._clock()
        threshold = (
            self._initial_chunk_bytes
            if self._next_is_initial
            else self._min_chunk_bytes
        )
        if len(self._buf) >= threshold:
            self._next_is_initial = False
            return self._take()
        return None

    def flush_due(self) -> bool:
        """True when buffered bytes have sat idle for ``flush_idle_s``."""
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/stv/test_send_batcher.py -v`
Expected: PASS (9 passed).

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format src/ojin/stv/send_batcher.py tests/stv/test_send_batcher.py
uv run ruff check --fix src/ojin/stv/send_batcher.py tests/stv/test_send_batcher.py
git add src/ojin/stv/send_batcher.py tests/stv/test_send_batcher.py
git commit -m "feat: add SendBatcher state machine for server-feed audio batching"
```

---

### Task 2: Config fields + batch on size threshold in `send_tts_audio`

**Files:**
- Modify: `src/ojin/stv/config.py`
- Modify: `src/ojin/stv/ojin_stv_client.py`
- Modify: `tests/stv/test_ojin_stv_client.py`

**Interfaces:**
- Consumes: `SendBatcher` (Task 1); `OJIN_PERSONA_SAMPLE_RATE` (already imported in the client from `ojin.stv.synchronizer`).
- Produces (on `OjinSTVClient`):
  - `self._batcher: SendBatcher`
  - `self._batch_added: asyncio.Event`
  - `self._batch_flush_task: Optional[asyncio.Task]` (declared here; used in Task 3)
  - `async def _send_audio_message(self, pcm: bytes) -> None`
- Produces (on `STVConfig`): `server_feed_batching_enabled: bool`, `server_feed_initial_chunk_ms: int`, `server_feed_min_chunk_ms: int`, `server_feed_flush_idle_ms: int`.

- [ ] **Step 1: Add the config fields**

In `src/ojin/stv/config.py`, add to the `STVConfig` dataclass (after the `Barge-in` block, before `Diagnostics`):

```python
    # Server-feed audio batching. TTS providers that stream small (~40 ms)
    # chunks at realtime starve the inference server (no supply lead builds at
    # 25 fps-in == 25 fps-out) and churn sub-frame residues. Coalesce the
    # resampled server-bound copy into larger sends: a lead-establishing initial
    # chunk per turn, then a steady-state minimum. Disable to restore per-chunk
    # sends. (Playback of the original audio is unaffected.)
    server_feed_batching_enabled: bool = True
    server_feed_initial_chunk_ms: int = 1000
    server_feed_min_chunk_ms: int = 400
    server_feed_flush_idle_ms: int = 200
```

- [ ] **Step 2: Write the failing tests**

In `tests/stv/test_ojin_stv_client.py`: (a) update `make_client` to accept config overrides; (b) replace `test_send_tts_audio_buffers_original_and_sends_resampled` with a deterministic batching-disabled version; (c) add a coalesce test.

Replace the existing `make_client` (lines 22-32) with:

```python
def make_client(**config_overrides):
    """Build a client wired to in-memory fakes (config overrides optional)."""
    fc = FakeOjinClient()
    out = ListOutput()
    c = OjinSTVClient(
        client=fc,
        output=out,
        tracer=RecordingTracer(),
        config=STVConfig(
            loop_stall_watchdog_ms=0, stall_probe_ms=0, **config_overrides
        ),
    )
    return c, fc, out
```

Replace `test_send_tts_audio_buffers_original_and_sends_resampled` (lines 61-75) with:

```python
def test_send_tts_audio_buffers_original_and_sends_resampled() -> None:
    """With batching off, send_tts_audio buffers the original and sends each chunk."""

    async def run() -> None:
        c, fc, _out = make_client(server_feed_batching_enabled=False)
        await c.start()
        await asyncio.sleep(0.02)
        fc.sent.clear()
        await c.start_turn()
        await c.send_tts_audio(b"\x01\x02" * 640, 16000, 1)  # 40 ms @ 16 kHz (identity)
        assert c._synchronizer.audio_buffers[-1].bytes_  # original buffered
        audio = [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        assert len(audio) == 1 and len(audio[0].audio_int16_bytes) == 1280
        await c.close()

    asyncio.run(run())
```

Add this new test below it:

```python
def test_batching_coalesces_to_initial_then_min() -> None:
    """Batching emits one ~1000 ms initial chunk, then ~400 ms min chunks."""

    async def run() -> None:
        c, fc, _out = make_client(
            server_feed_initial_chunk_ms=1000, server_feed_min_chunk_ms=400
        )
        await c.start()
        await asyncio.sleep(0.02)
        fc.sent.clear()
        await c.start_turn()
        frame = b"\x01\x02" * 640  # 40 ms @ 16 kHz = 1280 bytes, non-silent
        for _ in range(25):  # 25 * 40 ms = 1000 ms = initial threshold
            await c.send_tts_audio(frame, 16000, 1)
        audio = [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        assert len(audio) == 1
        assert len(audio[0].audio_int16_bytes) == 32000  # one 1000 ms batch
        for _ in range(10):  # 10 * 40 ms = 400 ms = min threshold
            await c.send_tts_audio(frame, 16000, 1)
        audio = [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        assert len(audio) == 2
        assert len(audio[1].audio_int16_bytes) == 12800  # one 400 ms batch
        await c.close()

    asyncio.run(run())
```

- [ ] **Step 3: Run the new/changed tests to verify they fail**

Run: `uv run pytest tests/stv/test_ojin_stv_client.py -k "buffers_original or coalesces" -v`
Expected: `test_batching_coalesces_to_initial_then_min` FAILS — batching isn't wired yet, so 25 chunks produce 25 sends (or 0 if treated differently), not 1 batch of 32000.

- [ ] **Step 4: Wire the batcher into the client**

In `src/ojin/stv/ojin_stv_client.py`:

(a) Add the import near the other `ojin.stv` imports:

```python
from ojin.stv.send_batcher import SendBatcher
```

(b) Add a module-level constant near the other constants (after `_ONE_SEC_US`):

```python
_BYTES_PER_MS_16K = OJIN_PERSONA_SAMPLE_RATE * 2 / 1000.0  # 32 B/ms mono int16
```

(c) In `__init__`, after `self._synchronizer = Synchronizer(self._config)`, construct the batcher and its task state:

```python
        self._batcher = SendBatcher(
            initial_chunk_bytes=int(
                self._config.server_feed_initial_chunk_ms * _BYTES_PER_MS_16K
            ),
            min_chunk_bytes=int(
                self._config.server_feed_min_chunk_ms * _BYTES_PER_MS_16K
            ),
            flush_idle_s=self._config.server_feed_flush_idle_ms / 1000.0,
        )
        self._batch_added = asyncio.Event()
        self._batch_flush_task: Optional[asyncio.Task] = None
```

(d) Add the send helper (place it just above `send_tts_audio`):

```python
    async def _send_audio_message(self, pcm: bytes) -> None:
        """Send one server-bound audio payload and record the to_server trace."""
        await self._client.send_message(OjinAudioInputMessage(audio_int16_bytes=pcm))
        self._tracer.instant("to_server", "audio_sent", args={"bytes": len(pcm)})
```

(e) In `send_tts_audio`, replace the final two lines (the direct send + its instant):

```python
        await self._client.send_message(
            OjinAudioInputMessage(audio_int16_bytes=resampled)
        )
        self._tracer.instant("to_server", "audio_sent", args={"bytes": len(resampled)})
```

with:

```python
        if self._config.server_feed_batching_enabled:
            to_send = self._batcher.add(resampled)
            self._batch_added.set()  # reset the idle-flush debounce timer
            if to_send is not None:
                await self._send_audio_message(to_send)
        else:
            await self._send_audio_message(resampled)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/stv/test_ojin_stv_client.py -k "buffers_original or coalesces" -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Run the full stv suite to catch regressions**

Run: `uv run pytest tests/stv -q`
Expected: PASS (the pre-existing send/seed/silence/rms tests still pass — batching is transparent to them because the input-lane trace fires per arrival and the seed bypasses the batcher).

- [ ] **Step 7: Format, lint, commit**

```bash
uv run ruff format src/ojin/stv/config.py src/ojin/stv/ojin_stv_client.py tests/stv/test_ojin_stv_client.py
uv run ruff check --fix src/ojin/stv/config.py src/ojin/stv/ojin_stv_client.py tests/stv/test_ojin_stv_client.py
git add src/ojin/stv/config.py src/ojin/stv/ojin_stv_client.py tests/stv/test_ojin_stv_client.py
git commit -m "feat: batch server-bound TTS audio by size in OjinSTVClient"
```

---

### Task 3: Idle-timeout tail flush task

**Files:**
- Modify: `src/ojin/stv/ojin_stv_client.py`
- Modify: `tests/stv/test_ojin_stv_client.py`

**Interfaces:**
- Consumes: `self._batcher` (`flush_due`, `drain`), `self._batch_added`, `self._batch_flush_task`, `self._send_audio_message` (Task 2).
- Produces: `async def _batch_flush_loop(self) -> None`; the task is created in the `SESSION_READY` handler and cancelled in `close()`.

- [ ] **Step 1: Write the failing test**

Add to `tests/stv/test_ojin_stv_client.py`:

```python
def test_idle_timeout_flushes_subthreshold_tail() -> None:
    """A turn shorter than the initial threshold flushes its tail after idle."""

    async def run() -> None:
        c, fc, _out = make_client(
            server_feed_initial_chunk_ms=1000,
            server_feed_min_chunk_ms=400,
            server_feed_flush_idle_ms=60,
        )
        await c.start()
        await asyncio.sleep(0.02)
        fc.sent.clear()
        await c.start_turn()
        frame = b"\x01\x02" * 640  # 40 ms @ 16 kHz
        for _ in range(5):  # 200 ms < 1000 ms initial → no size send
            await c.send_tts_audio(frame, 16000, 1)
        assert not [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        await asyncio.sleep(0.15)  # > flush_idle (60 ms) → idle flush fires
        audio = [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        assert len(audio) == 1
        assert len(audio[0].audio_int16_bytes) == 6400  # 5 * 1280 = 200 ms tail
        await c.close()

    asyncio.run(run())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/stv/test_ojin_stv_client.py -k idle_timeout_flushes -v`
Expected: FAIL — no flush task exists, so the 200 ms tail is never sent (`len(audio) == 0`).

- [ ] **Step 3: Implement the flush loop and wire it**

In `src/ojin/stv/ojin_stv_client.py`:

(a) Add the loop method (place it just below `_video_playback_loop`):

```python
    async def _batch_flush_loop(self) -> None:
        """Flush a sub-threshold audio tail after the idle gap, off the playback clock.

        Debounces on ``_batch_added``: each new chunk restarts the timer, so a
        flush only fires once TTS has been quiet for ``server_feed_flush_idle_ms``
        (turn end / short turn). Runs as its own task so a blocking websocket send
        never jitters the 25 fps playback loop. Re-arm of the initial threshold is
        NOT done here — that is tied to ``start_turn``.
        """
        idle = self._config.server_feed_flush_idle_ms / 1000.0
        while self._initialized:
            try:
                await asyncio.wait_for(self._batch_added.wait(), timeout=idle)
                self._batch_added.clear()  # new audio arrived → restart the timer
            except asyncio.TimeoutError:
                if self._batcher.flush_due():
                    pending = self._batcher.drain()
                    if pending is not None:
                        await self._send_audio_message(pending)
            except asyncio.CancelledError:
                raise
            except Exception:  # never let one bad send kill the loop
                logger.exception("batch flush loop error — continuing")
```

(b) In `_handle_message`, in the `OjinSessionReadyMessage` branch, right after the block that creates `self._playback_task`, add:

```python
            if (
                self._config.server_feed_batching_enabled
                and self._batch_flush_task is None
            ):
                self._batch_flush_task = asyncio.create_task(self._batch_flush_loop())
```

(c) In `close()`, cancel the flush task before the transport closes. Locate the `try:` / `await self._client.close()` block and insert this immediately BEFORE it. Leave the existing `for task in (self._receive_task, self._playback_task):` loop unchanged.

```python
        # Stop the batch-flush task before the transport closes.
        if self._batch_flush_task is not None:
            self._batch_flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._batch_flush_task
            self._batch_flush_task = None
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/stv/test_ojin_stv_client.py -k idle_timeout_flushes -v`
Expected: PASS.

- [ ] **Step 5: Run the full stv suite**

Run: `uv run pytest tests/stv -q`
Expected: PASS.

- [ ] **Step 6: Format, lint, commit**

```bash
uv run ruff format src/ojin/stv/ojin_stv_client.py tests/stv/test_ojin_stv_client.py
uv run ruff check --fix src/ojin/stv/ojin_stv_client.py tests/stv/test_ojin_stv_client.py
git add src/ojin/stv/ojin_stv_client.py tests/stv/test_ojin_stv_client.py
git commit -m "feat: flush sub-threshold audio tail on idle timeout"
```

---

### Task 4: Re-arm on `start_turn`, discard on `interrupt`, flush on `close`

**Files:**
- Modify: `src/ojin/stv/ojin_stv_client.py`
- Modify: `tests/stv/test_ojin_stv_client.py`

**Interfaces:**
- Consumes: `self._batcher` (`drain`, `rearm_initial`, `reset`, `pending_bytes`), `self._send_audio_message`, `self._batch_flush_task`.
- Produces: behavioral changes only (no new public symbols).

- [ ] **Step 1: Write the failing tests**

Add to `tests/stv/test_ojin_stv_client.py`:

```python
def test_start_turn_flushes_previous_tail_and_rearms_initial() -> None:
    """start_turn sends the prior turn's tail, then requires a fresh initial chunk."""

    async def run() -> None:
        c, fc, _out = make_client(
            server_feed_initial_chunk_ms=1000,
            server_feed_min_chunk_ms=400,
            server_feed_flush_idle_ms=10000,  # large: isolate from the idle flush
        )
        await c.start()
        await asyncio.sleep(0.02)
        fc.sent.clear()
        await c.start_turn()
        frame = b"\x01\x02" * 640
        for _ in range(5):  # 200 ms tail, below initial → not size-sent
            await c.send_tts_audio(frame, 16000, 1)
        assert not [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        await c.start_turn()  # flushes the 200 ms tail, re-arms initial
        audio = [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        assert len(audio) == 1 and len(audio[0].audio_int16_bytes) == 6400
        for _ in range(10):  # 400 ms — would hit min, but initial is armed → no send
            await c.send_tts_audio(frame, 16000, 1)
        audio = [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        assert len(audio) == 1  # still just the flushed tail
        await c.close()

    asyncio.run(run())


def test_interrupt_discards_pending_batch() -> None:
    """Barge-in throws away un-sent audio of the cancelled turn."""

    async def run() -> None:
        c, fc, _out = make_client(
            server_feed_initial_chunk_ms=1000, server_feed_flush_idle_ms=10000
        )
        await c.start()
        await asyncio.sleep(0.02)
        c._synchronizer.current_buffer = AudioBuffer(sample_rate=16000)
        c._synchronizer.current_buffer.bytes_.extend(b"\x01\x02" * 100)  # interruptible
        fc.sent.clear()
        await c.start_turn()
        frame = b"\x01\x02" * 640
        for _ in range(5):  # 200 ms pending in the batcher
            await c.send_tts_audio(frame, 16000, 1)
        assert c._batcher.pending_bytes == 6400
        await c.interrupt()
        assert c._batcher.pending_bytes == 0  # discarded
        audio = [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        assert not audio  # the cancelled turn's audio was never sent
        await c.close()

    asyncio.run(run())


def test_close_flushes_final_tail() -> None:
    """close best-effort flushes a buffered tail before the transport closes."""

    async def run() -> None:
        c, fc, _out = make_client(
            server_feed_initial_chunk_ms=1000, server_feed_flush_idle_ms=10000
        )
        await c.start()
        await asyncio.sleep(0.02)
        fc.sent.clear()
        await c.start_turn()
        frame = b"\x01\x02" * 640
        for _ in range(3):  # 120 ms tail, below initial and idle window won't fire
            await c.send_tts_audio(frame, 16000, 1)
        assert not [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        await c.close()
        audio = [m for m in fc.sent if isinstance(m, OjinAudioInputMessage)]
        assert len(audio) == 1 and len(audio[0].audio_int16_bytes) == 3840  # 3 * 1280

    asyncio.run(run())
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/stv/test_ojin_stv_client.py -k "start_turn_flushes or interrupt_discards or close_flushes" -v`
Expected: FAIL — start_turn doesn't flush/re-arm, interrupt doesn't reset the batcher, close doesn't flush.

- [ ] **Step 3: Implement the three hooks**

In `src/ojin/stv/ojin_stv_client.py`:

(a) `start_turn` — in the initialized path, after the existing `self._tracer.instant("tts_input", "tts_started", ...)` line, add:

```python
        if self._config.server_feed_batching_enabled:
            pending = self._batcher.drain()
            if pending is not None:
                await self._send_audio_message(pending)
            self._batcher.rearm_initial()
```

(b) `interrupt` — inside the `if self._synchronizer.interrupt():` block, after `await self._client.send_message(OjinCancelInteractionMessage())`, add:

```python
            if self._config.server_feed_batching_enabled:
                self._batcher.reset()
```

(c) `close` — add a best-effort final flush of any buffered tail. Insert it immediately AFTER the flush-task-cancel block added in Task 3 (still BEFORE the `try:` / `await self._client.close()` block):

```python
        if was_initialized and self._config.server_feed_batching_enabled:
            pending = self._batcher.drain()
            if pending is not None:
                with contextlib.suppress(Exception):
                    await self._send_audio_message(pending)
```

This is purely additive — Task 3 already cancels the flush task here; this just drains and sends the remainder before the transport goes down.

- [ ] **Step 4: Run them to verify they pass**

Run: `uv run pytest tests/stv/test_ojin_stv_client.py -k "start_turn_flushes or interrupt_discards or close_flushes" -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the full stv suite**

Run: `uv run pytest tests/stv -q`
Expected: PASS.

- [ ] **Step 6: Format, lint, commit**

```bash
uv run ruff format src/ojin/stv/ojin_stv_client.py tests/stv/test_ojin_stv_client.py
uv run ruff check --fix src/ojin/stv/ojin_stv_client.py tests/stv/test_ojin_stv_client.py
git add src/ojin/stv/ojin_stv_client.py tests/stv/test_ojin_stv_client.py
git commit -m "feat: re-arm, flush, and discard the audio batch on turn/interrupt/close"
```

---

### Task 5: Changelog + version bump (release gate)

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `pyproject.toml`

**Interfaces:** none (release metadata).

- [ ] **Step 1: Bump the version**

In `pyproject.toml`, change:

```toml
version = "0.7.1"
```

to:

```toml
version = "0.8.0"
```

- [ ] **Step 2: Add the changelog entry**

In `CHANGELOG.md`, insert a new section directly above the `## 0.7.0 - 2026-06-18` heading (keep the existing entries below it):

```markdown
## 0.8.0 - 2026-06-19

### Added
- `OjinSTVClient` now batches the audio it sends to the inference server instead
  of forwarding every TTS chunk immediately. Providers that stream small (~40 ms)
  chunks at realtime previously starved the server's 25 fps timeline (no supply
  lead builds when input rate equals output rate) and churned sub-frame residues
  that disrupted buffer swaps and lip-sync. The client now accumulates a
  lead-establishing initial chunk per `start_turn` (`server_feed_initial_chunk_ms`,
  default 1000 ms), then steady-state chunks of at least `server_feed_min_chunk_ms`
  (default 400 ms), and flushes a sub-threshold tail after
  `server_feed_flush_idle_ms` of quiet (default 200 ms). A barge-in discards the
  un-sent tail; `close` flushes it best-effort. Set
  `server_feed_batching_enabled=False` to restore per-chunk sends. The avatar
  playback path is unchanged — only the server-bound resampled copy is batched.
```

- [ ] **Step 3: Verify the package still imports and the suite is green**

```bash
uv run python -c "import ojin.stv.send_batcher, ojin.stv.ojin_stv_client; print('ok')"
uv run pytest tests/stv -q
```
Expected: `ok`, then all stv tests PASS.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md pyproject.toml
git commit -m "chore: release 0.8.0 — server-feed audio batching"
```

---

## Final verification (run after Task 5)

- [ ] **Full check (format, lint, tests) — the reviewer-equivalent gate:**

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest tests/stv -q
```
Expected: format clean, lint clean, all `tests/stv` PASS. (`tests/integration/` is intentionally skipped — it starts a real server.)

- [ ] **Confirm the spec is satisfied:** re-read `docs/superpowers/specs/2026-06-19-ojinstvclient-audio-batching-design.md` and tick off each requirement against the tasks above.
