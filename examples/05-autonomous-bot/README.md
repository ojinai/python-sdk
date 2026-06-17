# 05 · Autonomous Ojin avatar bot (no mic, no STT)

A hands-free variant of [`03-pipecat-example`](../03-pipecat-example) that makes the
avatar **talk to itself**, turn after turn, with **no microphone and no speech-to-text**.
It exists to reproduce the avatar's continuous-speech behaviour (lip-sync / repeated
frames) deterministically, without a human driving the conversation.

Everything else — the Ojin Face model, the LLM (Gemini), the TTS (ElevenLabs), the
`wss://models.ojin.foo/realtime` endpoint, the avatar size — is **identical to example
03**, so a trace captured here is directly comparable.

## What's different from 03

1. **STT removed.** No Deepgram, no mic. The `deepgram` pipecat extra is dropped.
2. **The user side is synthesized.** [`autonomous_driver.py`](./autonomous_driver.py)'s
   `AutonomousUserDriver` emits a canned user utterance as **transcription frames**
   after every bot turn ends (`BotStoppedSpeakingFrame`), which drives the next LLM
   turn. The bot greets/answers, finishes, gets the next synthetic prompt, and so on —
   indefinitely.
3. **External turn strategies.** The user aggregator uses
   `ExternalUserTurnStrategies()`, so turn boundaries come from the driver's
   `UserStartedSpeakingFrame` / `UserStoppedSpeakingFrame` (not from VAD/STT on live
   audio, of which there is none).

```
transport.input() -> [AutonomousUserDriver] -> user_agg -> LLM -> TTS -> [OjinAvatarService] -> transport.output()
```

Each synthetic turn is the exact frame sequence real STT + VAD would produce, so
nothing downstream knows the difference:

```
UserStartedSpeakingFrame
TranscriptionFrame(text="Tell me an interesting fact about the ocean.", finalized=True)
UserStoppedSpeakingFrame   # ExternalUserTurnStopStrategy finalizes -> the LLM runs
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -e "../..[stv]" + pipecat extras
cp .env.example .env                      # then fill in OJIN/GEMINI/ELEVENLABS keys

python bot.py -t daily     # bot joins a Daily room and self-talks; join later to watch
python bot.py -t webrtc    # open http://localhost:7860/client to watch (no mic used)
```

The conversation starts on its own a few seconds after the avatar connects — you never
speak. With `-t daily` it runs fully headless (the bot joins the room itself); with
`-t webrtc` open the page to watch (the page's mic is ignored).

Tuning knobs live on `AutonomousUserDriver(...)` in [`bot.py`](./bot.py): `utterances`,
`inter_turn_delay_s`, and `max_turns` (default: run forever). The first turn is gated on
the avatar's STV session-ready signal, so it never races startup.

## Reading the trace

Every avatar session writes a Perfetto trace (see
[`ojin_avatar.py`](./ojin_avatar.py)) to:

```
/root/debug/sessions/stv-example/<date>/<time>_<session_id>/session.json
```

Open it at [ui.perfetto.dev](https://ui.perfetto.dev). The repeat/lip-sync symptom is
a `play:repeat` marker fired when the fixed-25fps playback finds no fresh frame; the
per-tick **receive-pipeline depth gauges** tell you *why*:

| Counter | What a rising value means |
|---|---|
| `pending_video_frames` | the synchronizer's frame cushion; **draining to 0 ⇒ repeats** |
| `recv_decode_in` / `recv_decode_out` | the cv2 decode worker is behind (it isn't, at 1024×1024) |
| `recv_ws_frames` / `recv_ws_paused` | the websocket read loop is behind / backpressure engaged |
| `recv_server_msgs` | parsed frames awaiting consumption (the unbounded absorber) |
| `recv_sock_bytes` | kernel socket bytes unread (event loop starved) |

If `pending_video_frames` drains to 0 while every `recv_*` gauge sits at ~0, frames
simply aren't arriving at 25fps (upstream), not a client-side backlog. Compare the
proxy's own trace (same `trace_id`) for the other half of the picture.
