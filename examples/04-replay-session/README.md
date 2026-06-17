# 04 · Replay a recorded session (barge-in cancel-latency repro)

A diagnosis harness, not a live demo. It loads a **recorded pair of Perfetto
session traces** — the client trace written by `ojin.stv.OjinSessionTrace` (wired
into `03-pipecat-example`) and the inference server's `session.json` — and
reconstructs the interruption timeline from them, the way you'd compare the two
side by side in <https://ui.perfetto.dev>.

## The bug it reproduces

Session `8b00aa4e27e2` (2026-06-16): the user barges in, the client **fades its
local audio and sends a Cancel immediately**, but the avatar keeps "speaking" for
**~9 seconds** before the server's fade-out frames arrive.

Root cause (confirmed by reading the server): the inference server eagerly drains
the websocket into a **single unbounded FIFO message queue** with **no priority
path for Cancel**. The client streams a whole turn's audio at ~5× realtime, so the
Cancel — though it reaches the server within milliseconds — sits behind ~8s of
buffered audio and is only acted on once that backlog drains. The server records
`interrupt_requested` the instant it *dequeues* the Cancel, and the fade itself is
fast after that (~346ms). So the latency is **delivery/queueing**, not the fade.

## What it measures

| signal | meaning |
|---|---|
| `client_fade_lag_ms` | within the client trace, cancel → first FADE frame. Alignment-free; this *is* the user-visible "audio faded but the face kept talking" lag (~9s). |
| `cancel_to_server_interrupt_ms` | clocks aligned on the first audio chunk; cancel → server `interrupt_requested`. Localises the lag to the server FIFO (~8.4s). |
| `server_fade_latency_ms` | server `interrupt_requested` → `fade_out_emitted` (~346ms — healthy). |

## Run

```bash
python replay_session.py                      # bundled fixtures under traces/
python replay_session.py CLIENT.json SERVER.json
pytest test_replay_session.py
```

`traces/` holds trimmed, Perfetto-loadable copies of the two traces (only the
lanes needed to show the bug: interruption / to_server / recv on the client;
audio_input / interrupt_requested / fade_out on the server).

## The fix (separate, server-side)

This harness characterizes the broken behaviour. The fix lives in the inference
server and has two parts:

1. **Cancel fast-path** — detect `CancelInteractionMessage` out-of-band in the
   reader so it doesn't queue behind buffered audio (kills the ~8s latency).
2. **Interrupting-until-idle guard** — once cancelling, hold (don't drop) incoming
   audio so it can't start a new speech segment until fade-out + idle have been
   emitted to the client.

Both are validated by inference-server unit tests; a fresh trace after the fix
should show `client_fade_lag_ms` collapse to ~1.4s.
