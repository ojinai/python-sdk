# OjinSTVClient server-feed audio batching

**Date:** 2026-06-19
**Status:** Design — approved parameters, pending spec review
**Scope:** `python-sdk` only (client-side). No `inference-server` changes.

## Problem

Some TTS providers (notably Cartesia) stream audio as **40 ms chunks in realtime**.
Today [`OjinSTVClient.send_tts_audio`](../../../src/ojin/stv/ojin_stv_client.py)
resamples each arriving chunk to 16 kHz and sends it to the inference server
**immediately**, one `OjinAudioInputMessage` per chunk (~25 sends/second). This
produces two failure modes downstream, both of which disappear when the client
sends larger chunks:

1. **No supply lead.** The server runs a 25 fps virtual timeline and needs the
   *input* to lead the pacer by roughly one inference chunk + inference time
   (~1.4 s for FlashHead) to play gaplessly. With 40 ms chunks arriving at
   realtime, input rate (25 fps) equals output rate (25 fps), so the server's
   virtual clock never gets ahead of the wall clock and the lead never grows.
   This is documented as Round 2 of the server's own
   `docs/issues/audio_starvation_on_start` investigation:
   *"at 25 fps in == 25 fps out that lead never grows"* → permanent
   near-starvation; the pacer underruns on any jitter.

2. **Sub-frame residue churn.** The default resampler
   ([`SoxrStreamResampler`](../../../src/ojin/stv/resampler.py)) is *streaming* (it
   carries filter history across calls), so cross-chunk audio artifacts are
   already solved. But it emits ~640 ± δ samples per 40 ms input, so each
   arrival lands a sub-frame residue on the server's frame-aligned buffer. At
   ~25 arrivals/second this trips the server's residue / end-of-speech flush
   machinery continuously, producing phantom speech→idle→speech transitions
   that disrupt the client's buffer swaps and lip-sync alignment.

Both are fixed on the **client** by accumulating audio into larger sends: a big
first chunk establishes the supply lead, and ≥ 400 ms steady-state chunks cut
residue churn ~10×. 99% of users go through `OjinSTVClient`, so fixing it here
covers the real-world traffic without touching the server.

## Goals

- Accumulate an **initial chunk** (default 1000 ms) before the first send of a
  turn, re-armed on every `start_turn()`, to establish the server's supply lead.
- Combine steady-state sends to a **minimum of 400 ms** each.
- Flush a sub-threshold **tail** promptly when TTS goes quiet (turn ends / short
  turns), so the end of an utterance is never stranded.
- Preserve audio/video sync and the existing playback path.
- Be configurable, with an escape hatch to restore today's per-chunk sends.

## Non-goals

- No changes to `inference-server` (the server's existing flush/idle machinery
  already handles the now-batched tails from SDK clients).
- No change to the avatar playback path: the **original** TTS audio is still
  buffered verbatim in the `Synchronizer` per arrival. Only the resampled
  *server-bound copy* is batched.
- No change to the resampler. We continue resampling **per arriving chunk** to
  keep the streaming resampler warm; we accumulate the resampled *output*.

## Design

### New component: `SendBatcher` (pure state machine)

New file `src/ojin/stv/send_batcher.py`. A pure, synchronous, I/O-free state
machine — same testability pattern as `Synchronizer`. It owns the 16 kHz
server-bound byte buffer and decides *when* a batch is ready; the client performs
the actual `send_message`.

```
SendBatcher(initial_chunk_bytes: int,
            min_chunk_bytes: int,
            flush_idle_s: float,
            clock: Callable[[], float] = time.monotonic)
```

State:

- `_buf: bytearray` — accumulated resampled 16 kHz PCM.
- `_next_is_initial: bool = True` — whether the next emit uses the initial
  threshold (set True at construction and by `rearm_initial()`).
- `_last_add_ts: float` — monotonic time of the most recent `add`.

Methods:

- `add(pcm16k: bytes, now: float) -> Optional[bytes]`
  Append `pcm16k`; stamp `_last_add_ts = now`. Threshold =
  `initial_chunk_bytes` if `_next_is_initial` else `min_chunk_bytes`. If
  `len(_buf) >= threshold`: **drain all** accumulated bytes, set
  `_next_is_initial = False`, return them. Otherwise return `None`. Empty input
  is a no-op. (Drain-all, not drain-threshold: a single fat arrival sends as-is;
  the threshold is a *minimum*, not a quantum.)
- `flush_due(now: float) -> bool`
  `len(_buf) > 0 and now - _last_add_ts >= flush_idle_s`.
- `drain() -> Optional[bytes]`
  Unconditionally return all buffered bytes (or `None` if empty) and clear. Does
  **not** touch `_next_is_initial`. Used by the idle-flush loop (after
  `flush_due`), by `start_turn` (flush the previous turn's tail), and by `close`.
- `rearm_initial() -> None`
  Set `_next_is_initial = True`. Called by `start_turn` after draining.
- `reset() -> None`
  Clear `_buf` and set `_next_is_initial = True`. Used by `interrupt` to
  **discard** the cancelled turn's un-sent audio.
- `pending_bytes -> int` (property) — for tests/metrics.

Re-arm is tied to `start_turn()` **only** (per decision). The idle-flush sends a
tail without re-arming, so a mid-turn TTS stall that flushes a partial does not
force the rest of the same turn back to the initial threshold.

### Configuration (`STVConfig`)

New fields (with byte conversions done in the client against
`OJIN_PERSONA_SAMPLE_RATE` = 16 000, mono int16 ⇒ 32 bytes/ms):

| Field | Default | Meaning | Bytes |
|---|---|---|---|
| `server_feed_batching_enabled` | `True` | Master switch; `False` restores per-chunk sends | — |
| `server_feed_initial_chunk_ms` | `1000` | First send after each `start_turn()` (the lead) | 32 000 (25 frames) |
| `server_feed_min_chunk_ms` | `400` | Steady-state minimum send size | 12 800 (10 frames) |
| `server_feed_flush_idle_ms` | `200` | Quiet time after last chunk before flushing a sub-threshold tail | — |

`flush_idle = 200 ms` sits above Cartesia's ~40 ms inter-chunk gap (so a
streaming turn is never fragmented) and matches the server's existing
`_RESIDUE_FLUSH_S`.

### Wiring into `OjinSTVClient`

A small `_send_audio_message(pcm: bytes)` helper centralises the actual send:
build `OjinAudioInputMessage(audio_int16_bytes=pcm)`, `await
client.send_message(...)`, and emit the `to_server / audio_sent` tracer instant
with the **batch** byte count.

**`send_tts_audio`** — unchanged up to and including the resample (per-chunk, to
keep the streaming resampler warm and the playback buffering immediate). Replace
the direct send with:

```python
if self._config.server_feed_batching_enabled:
    to_send = self._batcher.add(resampled, time.monotonic())
    if to_send is not None:
        await self._send_audio_message(to_send)
    self._batch_added.set()  # wake the idle-flush loop's debounce timer
else:
    await self._send_audio_message(resampled)
```

The `tts_audio` input-lane instant and `input_audio_rms` counter still fire per
arrival (they describe TTS input, not the send). The `audio_sent` instant moves
to the actual send (per batch). The TTFB anchor (`_waiting_for_first_tts` →
`_tr_first_tts_audio_at`) stays keyed to first-arrival, unchanged — it measures
user-perceived latency, not send time.

**`start_turn`** (initialized path) — after `open_turn()` / setting
`_waiting_for_first_tts`, flush the previous turn's tail and re-arm:

```python
if self._config.server_feed_batching_enabled:
    pending = self._batcher.drain()
    if pending is not None:
        await self._send_audio_message(pending)
    self._batcher.rearm_initial()
```

The pre-init path is unchanged (the `("turn",)` op is recorded and replayed; the
replay takes this normal path after `SESSION_READY`).

**`interrupt`** (initialized path) — after the synchronizer interrupt + cancel
send, `self._batcher.reset()` to discard the cancelled turn's un-sent audio.

**`close`** — best-effort final flush before tearing down the transport: drain
and try-send the remainder, suppressing send errors (teardown must not raise).

**Idle-flush task** `_batch_flush_loop` — created alongside the playback task in
the `SESSION_READY` handler (only when batching is enabled), cancelled in
`close()` with the other tasks. Debounce on an `asyncio.Event` (`_batch_added`,
set in `send_tts_audio`):

```python
async def _batch_flush_loop(self) -> None:
    idle = self._config.server_feed_flush_idle_ms / 1000.0
    while self._initialized:
        try:
            await asyncio.wait_for(self._batch_added.wait(), timeout=idle)
            self._batch_added.clear()   # new audio arrived → restart the timer
        except asyncio.TimeoutError:
            if self._batcher.flush_due(time.monotonic()):
                pending = self._batcher.drain()
                if pending is not None:
                    await self._send_audio_message(pending)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("batch flush loop error — continuing")
```

The send happens off the timing-critical `_video_playback_loop`, so a blocking
websocket send can never jitter the 25 fps playback clock.

### Audio/video sync rationale

Avatar playback is gated on video arrival: the `Synchronizer` only drains audio
after a video frame triggers a buffer swap
([`swap_to_next_buffer`](../../../src/ojin/stv/synchronizer.py)). Holding the
server-bound copy therefore cannot push audio ahead of video — it only delays
when the corresponding video returns, which the synchronizer already absorbs. Net
effect is purely added latency.

### Latency

When TTS generates **faster than realtime** (the common case — providers stream
small frames but generate ahead), the larger first send *lowers* TTFB versus
dribbling: the server receives a full inference chunk at once instead of
accumulating it over ~1 s of wall time. When TTS is delivered at **exactly
realtime** (the reproduction case), the initial 1000 ms accumulation costs ~1 s
to fill — comparable to the ~0.96 s the server already needed to assemble the
first FlashHead chunk from dribbles — so there is no meaningful TTFB regression,
and the lead is gained. Short turns below the initial threshold are bounded by
`flush_idle` (≤ ~200 ms tail latency).

## Edge cases & error handling

- **Trailing-silence sentinel.** The existing 0.5 s all-zero sentinel discard
  stays *before* the batcher — sentinels are never batched or sent.
- **Empty resample output.** Filter delay can make a tiny input resample to 0
  bytes; `add` treats empty as a no-op.
- **Pre-init replay.** The flush task starts at `SESSION_READY` *before*
  `_flush_preinit_inputs`, so replayed audio batches normally and any final
  partial flushes after idle.
- **Interrupt before init.** Existing `_preinit_inputs.clear()` path is
  unchanged; the batcher is empty (nothing sent yet) and `reset()` is harmless.
- **Send failure.** `send_tts_audio` propagates send errors as today. The
  idle-flush loop and `close` swallow + log send errors so they never kill the
  loop or teardown.
- **Batching disabled.** No batcher use, no flush task — byte-for-byte today's
  behavior (regression-tested).
- **Seed frame.** The `SESSION_READY` timeline seed (`b"\x00" *
  BYTES_PER_FRAME`) is a control message and continues to bypass the batcher.

## Testing

**`tests/stv/test_send_batcher.py`** (pure, injected clock):

- Below initial threshold: feeding 40 ms chunks emits nothing until ≥ 1000 ms,
  then emits all accumulated.
- After the first emit, subsequent emits use the 400 ms threshold.
- Drain-all semantics (emit returns everything buffered, not threshold-worth).
- `flush_due` true only after `flush_idle_s` of no `add`, false while empty.
- `drain()` unconditional and `None` when empty.
- `rearm_initial()` restores the initial threshold for the next emit.
- `reset()` discards buffered bytes and re-arms initial.
- Empty-input no-op.

**`tests/stv/test_ojin_stv_client.py`** (additions, `FakeOjinClient.sent`):

- Batching on: `start_turn` then stream 25 × 40 ms chunks → every
  `OjinAudioInputMessage` (excluding the seed) is ≥ 400 ms and the first is
  ≥ 1000 ms.
- Short turn (e.g. 5 × 40 ms = 200 ms) then quiet → the tail is flushed after
  `flush_idle`.
- `start_turn` flushes the previous turn's pending tail and re-arms initial.
- `interrupt` discards pending audio (no stray send of the cancelled turn).
- Batching off → one send per chunk (today's behavior preserved).
- The `SESSION_READY` seed send is unchanged.

## Files

- New: `src/ojin/stv/send_batcher.py`
- New: `tests/stv/test_send_batcher.py`
- Modified: `src/ojin/stv/ojin_stv_client.py` (wiring, helper, flush task)
- Modified: `src/ojin/stv/config.py` (four new fields)
- Modified: `tests/stv/test_ojin_stv_client.py` (integration tests)
- Modified: `CHANGELOG.md` (Keep a Changelog entry) + `pyproject.toml` version
  bump (release gate; `0.7.1` → `0.8.0` — additive, default-on behavior change)
