---
name: ojin-stv-integration
description: >-
  Use when integrating the Ojin Speech-To-Video client (ojin.stv.OjinSTVClient
  from the ojin-client[stv] package) into an async agent pipeline or media
  transport — feeding TTS audio in and getting a lip-synced talking-avatar
  audio+video stream out, handling lifecycle events, barge-in/interruption, and
  the audio-as-clock model. Framework-agnostic foundation; for Pipecat
  specifically also use the ojin-stv-pipecat skill.
---

# Integrating the Ojin Speech-To-Video client

`ojin.stv.OjinSTVClient` turns a stream of **TTS audio** into a **lip-synced
talking-avatar** audio+video stream. You feed it the audio your agent already
produces; it returns the avatar's video frames plus the audio to play, kept in
A/V sync for you. It's framework-agnostic (no Pipecat/LiveKit/web framework
required) and drops into any async program.

In a typical agent it's one stage: `STT → LLM → TTS → OjinSTVClient → transport`.

## Install & import

```bash
pip install "ojin-client[stv]"     # or: uv add "ojin-client[stv]"
```

```python
from ojin import resolve_credentials                 # credential helper (optional)
from ojin.stv import (
    OjinSTVClient, STVConfig, STVEvent,
    STVAudioFrame, STVVideoFrame, FrameType,
)
```

## The mental model — read this first

Five facts that explain every API decision:

1. **You feed TTS audio; the avatar lip-syncs to it.** Call `send_tts_audio(pcm,
   sample_rate, num_channels)` with whatever your TTS engine emits (any rate /
   channel count).
2. **Audio is the clock.** After a brief initial warm-up (it buffers
   `config.initial_buffer_frames`, default 6 ticks), the client emits exactly one
   audio frame per tick (25 fps) so the consumer's clock never starves, plus a
   video frame alongside it.
3. **The avatar plays your *original* audio.** Only a 16 kHz mono `int16` *copy*
   is sent to the server for lip-sync inference — the `STVAudioFrame.pcm` you get
   back is your original audio, byte-for-byte. You never play the server's audio.
4. **It's stateful and session-based.** Connect → `SESSION_READY` → send turns →
   close. Audio sent before `SESSION_READY` is dropped.
5. **A "turn" is one utterance.** `start_turn()` opens a buffer, `send_tts_audio()`
   appends to it (call repeatedly as TTS streams), `interrupt()` barges in.

## Minimal end-to-end

```python
import asyncio
from ojin import resolve_credentials
from ojin.stv import OjinSTVClient, STVAudioFrame, STVVideoFrame, STVEvent

async def main():
    creds = resolve_credentials()  # reads OJIN_API_KEY / OJIN_CONFIG_ID from env/.env
    client = OjinSTVClient(
        api_key=creds.api_key,
        config_id=creds.config_id,
        image_size=(512, 512),     # match your face model's output size
    )

    @client.on(STVEvent.SESSION_READY)
    async def _ready(session_data=None, **_):
        # Session is live — now it's safe to send audio.
        tts_pcm = b"..."                       # your TTS engine's PCM bytes
        await client.say(tts_pcm, sample_rate=24000, num_channels=1)

    async with client:                         # connects + runs the playback loop
        async for frame in client.output_stream():
            if isinstance(frame, STVVideoFrame):
                if frame.rgb is not None:
                    ...                         # render/forward frame.rgb (raw RGB pixels)
            elif isinstance(frame, STVAudioFrame):
                ...                             # play frame.pcm at frame.sample_rate

asyncio.run(main())
```

`async with client:` is `start()` + `close()`. `output_stream()` ends when you
`close()` (the `async for` returns).

## The integration contract

### Input — what you call

| Method | When | Notes |
|---|---|---|
| `await client.start()` | once, to connect | or use `async with client:`; runs the receive + playback loops |
| `await client.start_turn()` | at the start of an utterance | opens a fresh audio buffer (≈ "TTS started") |
| `await client.send_tts_audio(pcm, sample_rate, num_channels)` | for each TTS chunk | buffers your audio for playback, sends a 16 kHz copy for lip-sync. **No-op before `SESSION_READY`.** |
| `await client.say(pcm, sample_rate, num_channels)` | one-shot | = `start_turn()` + one `send_tts_audio()` |
| `await client.interrupt()` | user barges in | fades the current turn and cancels it server-side; emits `INTERRUPTED` |
| `await client.close()` | shutdown | tears down loops + transport; ends `output_stream()` |

Stream TTS by calling `start_turn()` once, then `send_tts_audio()` for every chunk
as it arrives. Use `say()` only when you have the whole utterance up front.

### Output — what you receive

The client emits frames through an **`STVOutput`** sink. By default that's a
`QueueOutput` you drain with `async for frame in client.output_stream()`. Frames:

**`STVAudioFrame`** — one tick of audio to play (your original TTS audio, never the
server's):
| Field | Meaning |
|---|---|
| `pcm` | audio bytes to play this tick |
| `sample_rate`, `num_channels` | format of `pcm` (matches what you sent) |
| `pts` | presentation timestamp, nanoseconds |

**`STVVideoFrame`** — one avatar frame:
| Field | Meaning |
|---|---|
| `rgb` | decoded RGB pixels (`bytes`), or `None` — **use this to render** |
| `source_bytes` | the raw JPEG from the server; **empty (`b""`) on held/repeat ticks** |
| `width`, `height` | frame size (from `config.image_size`) |
| `format` | pixel format, `"RGB"` |
| `frame_type` | `FrameType`: `IDLE=0`, `SPEECH=1`, `FADE_OUT=2`, `START_OF_SPEECH=3` |
| `volume` | RMS of the audio this frame was generated for; `0.0` on a held tick (a verification aid, **not** a playback gain) |
| `pts` | presentation timestamp, nanoseconds |

### Events — what you handle

Register with `@client.on(STVEvent.X)` (decorator) or `client.add_listener(event, cb)`.
Handlers may be sync or async; one failing handler never breaks the loops.

| Event | Fires when | Kwargs |
|---|---|---|
| `SESSION_READY` | session is live (safe to send audio) | `session_data` |
| `BOT_STARTED_SPEAKING` | a buffer is promoted to current | — |
| `BOT_STOPPED_SPEAKING` | current buffer drains, none queued | — |
| `INTERRUPTED` | `interrupt()` took effect | — |
| `ERROR` | server/connection error | `message`, `fatal` (+ `code` on server-side errors) |
| `CLOSED` | session torn down | — |

> Write handlers with defaults + `**_`, e.g. `def _err(message="", code=None,
> fatal=False, **_):`. Kwargs vary by event (and `code` is only present on
> server-side errors, not connect failures), so a handler with required positional
> params would raise on some events.

## Two ways to consume the output

**A. `QueueOutput` + `output_stream()` (default).** Easiest. You pull frames in an
`async for` and route them wherever you like. Good for files, custom rendering, web
sockets, batch jobs. (See example `02-realtime-speech-to-video` for a streaming
web consumer; `01-speech-to-video-mp4-generator` for an offline one.)

**B. Inject a custom `STVOutput` sink (push model).** Implement the protocol and
the client pushes into it as frames are produced — ideal when your transport wants
to be *called* rather than polled (this is how the Pipecat adapter works):

```python
from ojin.stv import STVAudioFrame, STVVideoFrame, STVEvent, STVOutput

class MyTransportSink:                          # structurally an STVOutput
    async def write_audio(self, frame: STVAudioFrame) -> None:
        await my_transport.send_audio(frame.pcm, frame.sample_rate, frame.num_channels)

    async def write_video(self, frame: STVVideoFrame) -> None:
        if frame.rgb is not None:
            await my_transport.send_video(frame.rgb, frame.width, frame.height)

    def on_event(self, event: STVEvent, **kwargs) -> None:
        ...                                     # optional; may be a no-op

client = OjinSTVClient(api_key=..., config_id=..., output=MyTransportSink())
# NOTE: with a custom sink, do NOT call output_stream() — consume your sink directly.
```

## Barge-in (interruption)

When the user starts talking over the avatar, call `await client.interrupt()`. It
fades the in-flight turn (so audio doesn't cut harshly), cancels generation
server-side, and emits `INTERRUPTED`. It's a no-op if the avatar is idle, so it's
safe to call on every "user started speaking" signal. Start the next turn normally
with `start_turn()` / `send_tts_audio()`.

## Credentials

`resolve_credentials()` reads `OJIN_API_KEY` and `OJIN_CONFIG_ID` (from the
environment or a `.env`) and raises `MissingCredentialsError` with setup steps if
either is absent — fail fast with a helpful message instead of a vague auth error:

```python
from ojin import resolve_credentials, MissingCredentialsError
try:
    creds = resolve_credentials()
except MissingCredentialsError as e:
    print(e); raise SystemExit(1)
```

Get an API key and a Face-model **config id** at https://ojin.ai. Never hardcode
them — read from env/`.env`.

## Gotchas (each one bites at least once)

- **Don't send audio before `SESSION_READY`.** `send_tts_audio()` silently drops if
  the session isn't ready. Send from the `SESSION_READY` handler, or guard with
  `if client.is_connected:`.
- **Two valid video paths — `rgb` or `source_bytes`.** Render decoded pixels from
  `rgb` (it repeats the last frame on held ticks, so playback stays smooth), *or*
  forward the raw JPEG in `source_bytes` (as example `02` does, streaming to a
  browser). In the JPEG path you **must skip held ticks**, where `source_bytes` is
  empty (`b""`) — `rgb` has no such caveat.
- **Don't use `PassthroughDecoder` if you need pixels.** With it `rgb` is `None`, and
  when `rgb` is `None` the client emits **no** video frame that tick. The default
  `OpenCVDecoder` gives you `rgb`.
- **`output_stream()` requires the default `QueueOutput`.** If you injected a custom
  `STVOutput`, it raises — consume your sink instead.
- **Feed natural TTS chunks, not tiny one-frame messages.** The server needs a
  steady audio timeline; very small chunks make it return idle (still-mouth) frames.
  Let your TTS stream in its native chunk sizes.
- **Set `image_size` to your face model's output.** Emitted frames are tagged with
  `config.image_size`; a mismatch produces stretched/garbled video downstream.

## Verify against your installed version

These tables were written against a specific release. Before relying on a symbol,
confirm it in *your* install:

```python
import inspect, dataclasses, ojin.stv as stv
print(inspect.signature(stv.OjinSTVClient.send_tts_audio))
print([e.name for e in stv.STVEvent])
print([f.name for f in dataclasses.fields(stv.STVVideoFrame)])
```

If something differs, trust the installed source and open an issue:
https://github.com/ojinai/python-sdk/issues
